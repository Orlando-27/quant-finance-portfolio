"""
M52 -- Reinforcement Learning for Trading
==========================================
CQF Concepts Explained | Project 19 | Quantitative Finance Portfolio

Theory
------
Reinforcement Learning frames trading as a Markov Decision Process (MDP):

    (S, A, P, R, gamma)

  S  -- state space   : features describing market conditions at time t
  A  -- action space  : {-1 short, 0 flat, 1 long}
  P  -- transition    : P(s'|s,a) -- unknown, learned through interaction
  R  -- reward        : log-return * position - transaction_cost * |a_t - a_{t-1}|
  gamma               : discount factor for future rewards

Bellman Optimality Equation
---------------------------
    Q*(s,a) = E[ r + gamma * max_{a'} Q*(s',a') | s,a ]

Q-Learning (off-policy TD control):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_{a'} Q(s',a') - Q(s,a) ]

Deep Q-Network (DQN)
--------------------
Replaces tabular Q with neural network Q(s,a;theta) and minimises:
    L(theta) = E[ (y - Q(s,a;theta))^2 ]
    y = r + gamma * max_{a'} Q(s',a'; theta^-)   (target network theta^-)

Key stabilisation tricks:
  * Experience replay  : sample mini-batches from replay buffer
  * Target network     : frozen copy updated every C steps
  * Epsilon-greedy     : eps-greedy exploration decaying over time

References
----------
Watkins & Dayan (1992) "Q-learning", Machine Learning 8:279-292
Mnih et al. (2015) "Human-level control through deep RL", Nature 518
Moody & Saffell (2001) "Learning to trade via direct RL", IEEE TNN
"""

import os
import sys
import warnings
import random
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# =============================================================================
# STYLE
# =============================================================================
DARK   = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#c9d1d9"
GREEN  = "#3fb950"
RED    = "#f85149"
ACCENT = "#58a6ff"
GOLD   = "#d29922"
PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  TEXT,
})

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "m52_rl_trading")
os.makedirs(FIGS, exist_ok=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print()
print("=" * 65)
print("  MODULE 52: REINFORCEMENT LEARNING FOR TRADING")
print("  MDP | Q-Learning | DQN | Epsilon-Greedy | Backtest")
print("=" * 65)

# =============================================================================
# 1. SYNTHETIC PRICE SERIES WITH REGIME STRUCTURE
# =============================================================================
N = 1500
dt = 1 / 252

# Two-state hidden Markov-like drift
drift_bull  =  0.15 * dt          # annualised 15 %
drift_bear  = -0.10 * dt          # annualised -10 %
vol_bull    =  0.12 * np.sqrt(dt)
vol_bear    =  0.22 * np.sqrt(dt)

p_bull_stay = 0.97
p_bear_stay = 0.93

regime = np.zeros(N, dtype=int)   # 0=bull, 1=bear
regime[0] = 0
for t in range(1, N):
    if regime[t-1] == 0:
        regime[t] = 0 if np.random.rand() < p_bull_stay else 1
    else:
        regime[t] = 1 if np.random.rand() < p_bear_stay else 0

ret = np.where(
    regime == 0,
    drift_bull + vol_bull  * np.random.randn(N),
    drift_bear + vol_bear  * np.random.randn(N),
)
price = 100.0 * np.exp(np.cumsum(ret))

print(f"  [01] Price series: N={N}  bull={regime.sum()==0} days")
print(f"       Bull days={np.sum(regime==0)}  Bear days={np.sum(regime==1)}")

# =============================================================================
# 2. STATE FEATURES
# =============================================================================
# State vector at time t (observable):
#   s_t = [ ret_t, ret_{t-1}, ret_{t-2}, zscore_5, zscore_20, vol_ratio ]
LOOKBACK = 20   # minimum history before trading

def compute_states(ret: np.ndarray) -> np.ndarray:
    """Build state matrix from return series."""
    n = len(ret)
    states = np.zeros((n, 6))
    for t in range(n):
        r0 = ret[t]
        r1 = ret[t-1] if t >= 1 else 0.0
        r2 = ret[t-2] if t >= 2 else 0.0
        # 5-day z-score
        if t >= 5:
            mu5 = ret[max(0,t-5):t].mean()
            sd5 = ret[max(0,t-5):t].std() + 1e-8
            zs5 = (r0 - mu5) / sd5
        else:
            zs5 = 0.0
        # 20-day z-score
        if t >= 20:
            mu20 = ret[max(0,t-20):t].mean()
            sd20 = ret[max(0,t-20):t].std() + 1e-8
            zs20 = (r0 - mu20) / sd20
        else:
            zs20 = 0.0
        # vol ratio 5/20
        if t >= 20:
            vr = (ret[t-5:t].std() + 1e-8) / (ret[t-20:t].std() + 1e-8)
        else:
            vr = 1.0
        states[t] = [r0, r1, r2, zs5, zs20, vr]
    return states

states_all = compute_states(ret)
print(f"  [02] State features: {states_all.shape[1]}-dim  "
      f"(returns x3, z-score x2, vol-ratio x1)")

# =============================================================================
# 3. TRADING ENVIRONMENT
# =============================================================================
ACTIONS   = [-1, 0, 1]   # short, flat, long
N_ACTIONS = len(ACTIONS)
TC        = 0.0005        # one-way transaction cost (5 bps)
GAMMA     = 0.99

class TradingEnv:
    """
    Minimal trading environment.
    Episode = one full trajectory through the price series.
    """
    def __init__(self, ret: np.ndarray, states: np.ndarray,
                 start: int = LOOKBACK):
        self.ret    = ret
        self.states = states
        self.start  = start
        self.n      = len(ret)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t      = self.start
        self.pos    = 0          # current position
        self.pnl    = 0.0
        return self.states[self.t].copy()

    def step(self, action_idx: int):
        a   = ACTIONS[action_idx]
        r_t = self.ret[self.t]

        # reward = position * return - TC * |change in position|
        reward = a * r_t - TC * abs(a - self.pos)
        self.pos  = a
        self.pnl += reward
        self.t   += 1
        done = (self.t >= self.n - 1)
        next_state = self.states[self.t].copy() if not done else np.zeros(6)
        return next_state, reward, done

env = TradingEnv(ret, states_all)
print(f"  [03] TradingEnv: actions={ACTIONS}  TC={TC*1e4:.0f}bps  gamma={GAMMA}")

# =============================================================================
# 4. Q-LEARNING (TABULAR) -- discretised state
# =============================================================================
def discretise(state: np.ndarray, bins: int = 5) -> tuple:
    """Clip and bin each feature into `bins` discrete values."""
    clipped = np.clip(state, -3, 3)
    idx = np.floor((clipped + 3) / 6 * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    return tuple(idx)

# Tabular Q: dict of (discrete_state) -> [Q_short, Q_flat, Q_long]
Q_table = {}

def q_get(s_disc):
    if s_disc not in Q_table:
        Q_table[s_disc] = np.zeros(N_ACTIONS)
    return Q_table[s_disc]

def q_update(s, a, r, s_next, done, alpha=0.05):
    q_s  = q_get(s)
    q_ns = q_get(s_next)
    target = r if done else r + GAMMA * np.max(q_ns)
    q_s[a] += alpha * (target - q_s[a])

# --- Training Q-Learning ---
N_EPISODES_Q = 300
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 0.98

eps         = EPS_START
rewards_q   = []

for ep in range(N_EPISODES_Q):
    state      = env.reset()
    s_disc     = discretise(state)
    ep_reward  = 0.0
    done       = False
    while not done:
        # epsilon-greedy
        if np.random.rand() < eps:
            a_idx = np.random.randint(N_ACTIONS)
        else:
            a_idx = np.argmax(q_get(s_disc))
        ns, reward, done = env.step(a_idx)
        ns_disc = discretise(ns)
        q_update(s_disc, a_idx, reward, ns_disc, done)
        s_disc    = ns_disc
        ep_reward += reward
    eps = max(EPS_END, eps * EPS_DECAY)
    rewards_q.append(ep_reward)

print(f"  [04] Q-Learning: {N_EPISODES_Q} episodes  "
      f"final_eps={eps:.3f}  "
      f"Q-table entries={len(Q_table)}")
print(f"       Last-50 avg reward: {np.mean(rewards_q[-50:]):.4f}")

# =============================================================================
# 5. DEEP Q-NETWORK (DQN)
# =============================================================================
STATE_DIM  = 6

# --- Lightweight NumPy DQN (no external DL framework) ---
def relu(x):    return np.maximum(0, x)
def relu_d(x):  return (x > 0).astype(float)

class DenseLayer:
    """Single dense layer with He initialisation."""
    def __init__(self, n_in: int, n_out: int, lr: float = 1e-3):
        self.W  = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b  = np.zeros(n_out)
        self.lr = lr
        # Adam moments
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, x):
        self.x_in = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        dW = self.x_in.T @ grad_out
        db = grad_out.sum(axis=0)
        dx = grad_out @ self.W.T
        # Adam update
        self.t  += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * dW**2
        mW_hat  = self.mW / (1 - beta1**self.t)
        vW_hat  = self.vW / (1 - beta2**self.t)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * db**2
        mb_hat  = self.mb / (1 - beta1**self.t)
        vb_hat  = self.vb / (1 - beta2**self.t)
        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
        return dx

class QNetwork:
    """Two-hidden-layer Q-network: state_dim -> 64 -> 32 -> n_actions."""
    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3):
        self.l1 = DenseLayer(state_dim, 64, lr)
        self.l2 = DenseLayer(64,        32, lr)
        self.l3 = DenseLayer(32,        n_actions, lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, state_dim) -> (batch, n_actions)"""
        h1 = relu(self.l1.forward(x))
        self.h1_pre = self.l1.x_in @ self.l1.W + self.l1.b
        h2 = relu(self.l2.forward(h1))
        self.h2_pre = self.l2.x_in @ self.l2.W + self.l2.b
        out = self.l3.forward(h2)
        # cache for backward
        self._h1 = h1
        self._h2 = h2
        return out

    def backward(self, grad: np.ndarray):
        g = self.l3.backward(grad)
        g = g * relu_d(self.h2_pre)
        g = self.l2.backward(g)
        g = g * relu_d(self.h1_pre)
        self.l1.backward(g)

    def copy_weights_to(self, target):
        """Copy weights to target network (hard update)."""
        import copy
        for src_l, tgt_l in [(self.l1, target.l1),
                              (self.l2, target.l2),
                              (self.l3, target.l3)]:
            tgt_l.W = src_l.W.copy()
            tgt_l.b = src_l.b.copy()

LR_DQN       = 5e-4
BATCH_SIZE   = 64
REPLAY_CAP   = 5_000
TARGET_UPD   = 200      # steps between hard target-network updates
N_EPISODES_D = 400

online_net = QNetwork(STATE_DIM, N_ACTIONS, lr=LR_DQN)
target_net = QNetwork(STATE_DIM, N_ACTIONS, lr=LR_DQN)
online_net.copy_weights_to(target_net)

replay = deque(maxlen=REPLAY_CAP)
eps_d    = 1.0
step_cnt = 0
rewards_dqn = []
losses_dqn  = []

def sample_batch():
    batch = random.sample(replay, BATCH_SIZE)
    s  = np.array([b[0] for b in batch])
    a  = np.array([b[1] for b in batch])
    r  = np.array([b[2] for b in batch])
    ns = np.array([b[3] for b in batch])
    d  = np.array([b[4] for b in batch], dtype=float)
    return s, a, r, ns, d

for ep in range(N_EPISODES_D):
    state     = env.reset()
    ep_reward = 0.0
    done      = False
    while not done:
        # epsilon-greedy
        if np.random.rand() < eps_d:
            a_idx = np.random.randint(N_ACTIONS)
        else:
            q_vals = online_net.forward(state[np.newaxis, :])[0]
            a_idx  = int(np.argmax(q_vals))

        ns, reward, done = env.step(a_idx)
        replay.append((state, a_idx, reward, ns, float(done)))
        state      = ns
        ep_reward += reward
        step_cnt  += 1

        # --- training step ---
        if len(replay) >= BATCH_SIZE:
            s, a, r, ns_b, d = sample_batch()
            # target Q-values
            q_online = online_net.forward(s)
            q_target_ns = target_net.forward(ns_b)
            y = r + GAMMA * np.max(q_target_ns, axis=1) * (1 - d)
            # compute TD error only for taken actions
            q_pred = q_online[np.arange(BATCH_SIZE), a]
            td_err = q_pred - y
            loss   = float(np.mean(td_err**2))
            losses_dqn.append(loss)
            # backprop: gradient only at taken action index
            grad = np.zeros_like(q_online)
            grad[np.arange(BATCH_SIZE), a] = 2 * td_err / BATCH_SIZE
            online_net.backward(grad)

        # hard update target network
        if step_cnt % TARGET_UPD == 0:
            online_net.copy_weights_to(target_net)

    eps_d = max(0.05, eps_d * 0.992)
    rewards_dqn.append(ep_reward)

print(f"  [05] DQN: {N_EPISODES_D} episodes  "
      f"replay_cap={REPLAY_CAP}  batch={BATCH_SIZE}  "
      f"target_upd={TARGET_UPD}")
print(f"       Last-50 avg reward : {np.mean(rewards_dqn[-50:]):.4f}")
print(f"       Final TD loss (avg): {np.mean(losses_dqn[-200:]):.6f}")

# =============================================================================
# 6. BACKTEST: DQN vs Q-LEARNING vs BUY-AND-HOLD vs MOMENTUM
# =============================================================================
def greedy_episode(net_or_table, use_dqn: bool):
    """Run one greedy episode, return action sequence and cumulative PnL."""
    state  = env.reset()
    pos    = 0
    cum    = [0.0]
    acts   = []
    done   = False
    while not done:
        if use_dqn:
            q_vals = net_or_table.forward(state[np.newaxis, :])[0]
            a_idx  = int(np.argmax(q_vals))
        else:
            s_disc = discretise(state)
            a_idx  = int(np.argmax(q_get(s_disc)))
        a = ACTIONS[a_idx]
        ns, reward, done = env.step(a_idx)
        cum.append(cum[-1] + reward)
        acts.append(a)
        state = ns
    return np.array(acts), np.array(cum)

acts_dqn, pnl_dqn = greedy_episode(online_net, use_dqn=True)
acts_q,   pnl_q   = greedy_episode(None,       use_dqn=False)

# Buy-and-hold
n_bt     = N - LOOKBACK - 1
ret_bt   = ret[LOOKBACK:LOOKBACK + n_bt]
pnl_bnh  = np.cumsum(ret_bt)
pnl_bnh  = np.concatenate([[0], pnl_bnh])

# Momentum: long if MA5 > MA20 else short
ma5  = np.array([ret[max(0,t-5):t].mean()  for t in range(LOOKBACK, LOOKBACK+n_bt)])
ma20 = np.array([ret[max(0,t-20):t].mean() for t in range(LOOKBACK, LOOKBACK+n_bt)])
mom_pos = np.sign(ma5 - ma20)
mom_pos[mom_pos == 0] = 1
pnl_mom_raw = mom_pos * ret_bt - TC * np.abs(np.diff(np.concatenate([[0], mom_pos])))
pnl_mom = np.concatenate([[0], np.cumsum(pnl_mom_raw)])

def sharpe(pnl_seq, freq=252):
    d = np.diff(pnl_seq)
    return float(np.mean(d) / (np.std(d) + 1e-9) * np.sqrt(freq))

def max_dd(pnl_seq):
    peak = np.maximum.accumulate(pnl_seq)
    dd   = peak - pnl_seq
    return float(np.max(dd))

strategies = {
    "DQN":          pnl_dqn,
    "Q-Learning":   pnl_q,
    "Momentum":     pnl_mom,
    "Buy-and-Hold": pnl_bnh,
}

print(f"  [06] Backtest Results (greedy / fixed policy):")
print(f"       {'Strategy':<15} {'Total PnL':>10} {'Sharpe':>8} {'Max DD':>8}")
for name, pnl in strategies.items():
    print(f"       {name:<15} {pnl[-1]:>10.4f} "
          f"{sharpe(pnl):>8.3f} {max_dd(pnl):>8.4f}")

# =============================================================================
# 7. FIGURE 1 -- Learning Curves
# =============================================================================
def smooth(arr, w=20):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=DARK)
fig.suptitle("M52 -- Reinforcement Learning for Trading: Learning Curves",
             color=TEXT, fontsize=11, y=1.01)

# 7A: Q-Learning episode rewards
ax = axes[0]
ax.plot(smooth(rewards_q, 20), color=GREEN, lw=1.2, label="Smoothed (w=20)")
ax.plot(rewards_q, color=GREEN, lw=0.3, alpha=0.3)
ax.axhline(0, color=TEXT, lw=0.5, ls="--")
ax.set_title("Q-Learning: Episode Reward")
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Reward")
ax.legend(fontsize=7)
ax.grid(True)

# 7B: DQN episode rewards
ax = axes[1]
ax.plot(smooth(rewards_dqn, 20), color=ACCENT, lw=1.2, label="Smoothed (w=20)")
ax.plot(rewards_dqn, color=ACCENT, lw=0.3, alpha=0.3)
ax.axhline(0, color=TEXT, lw=0.5, ls="--")
ax.set_title("DQN: Episode Reward")
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Reward")
ax.legend(fontsize=7)
ax.grid(True)

# 7C: DQN TD loss
ax = axes[2]
ax.semilogy(smooth(losses_dqn, 50), color=GOLD, lw=1.2, label="TD Loss (log)")
ax.set_title("DQN: TD Loss Convergence")
ax.set_xlabel("Training Step")
ax.set_ylabel("MSE Loss (log)")
ax.legend(fontsize=7)
ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m52_fig1_learning_curves.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [07] Fig 1 saved: learning curves")

# =============================================================================
# 8. FIGURE 2 -- Backtest PnL & Position Analysis
# =============================================================================
t_bt = np.arange(len(pnl_dqn))
colors_bt = [ACCENT, GREEN, GOLD, RED]
names_bt  = ["DQN", "Q-Learning", "Momentum", "Buy-and-Hold"]
pnls_bt   = [pnl_dqn, pnl_q, pnl_mom, pnl_bnh]

fig = plt.figure(figsize=(15, 9), facecolor=DARK)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("M52 -- Backtest: Cumulative PnL & Position Analysis",
             color=TEXT, fontsize=11)

# 8A: Cumulative PnL all strategies
ax = fig.add_subplot(gs[0, :])
for pnl, name, col in zip(pnls_bt, names_bt, colors_bt):
    ax.plot(t_bt[:len(pnl)], pnl, label=name, color=col, lw=1.2)
ax.axhline(0, color=TEXT, lw=0.5, ls="--")
ax.set_title("Cumulative PnL: All Strategies")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative Log-Return")
ax.legend(fontsize=8)
ax.set_facecolor(PANEL)
ax.grid(True)

# 8B: DQN drawdown
ax = fig.add_subplot(gs[1, 0])
peak_dqn = np.maximum.accumulate(pnl_dqn)
dd_dqn   = peak_dqn - pnl_dqn
ax.fill_between(t_bt, -dd_dqn, 0, color=ACCENT, alpha=0.6)
ax.plot(t_bt, -dd_dqn, color=ACCENT, lw=0.8)
ax.set_title("DQN Drawdown")
ax.set_xlabel("Day")
ax.set_ylabel("Drawdown")
ax.set_facecolor(PANEL)
ax.grid(True)

# 8C: BnH drawdown
ax = fig.add_subplot(gs[1, 1])
peak_bnh = np.maximum.accumulate(pnl_bnh)
dd_bnh   = peak_bnh - pnl_bnh
ax.fill_between(np.arange(len(pnl_bnh)), -dd_bnh, 0, color=RED, alpha=0.6)
ax.plot(np.arange(len(pnl_bnh)), -dd_bnh, color=RED, lw=0.8)
ax.set_title("Buy-and-Hold Drawdown")
ax.set_xlabel("Day")
ax.set_ylabel("Drawdown")
ax.set_facecolor(PANEL)
ax.grid(True)

# 8D: DQN action distribution
ax = fig.add_subplot(gs[2, 0])
uniq, cnt = np.unique(acts_dqn, return_counts=True)
labels_a  = {-1: "Short", 0: "Flat", 1: "Long"}
bar_cols   = [RED if u == -1 else TEXT if u == 0 else GREEN for u in uniq]
ax.bar([labels_a[u] for u in uniq], cnt / cnt.sum() * 100,
       color=bar_cols, edgecolor=DARK, linewidth=0.5)
ax.set_title("DQN Action Distribution (%)")
ax.set_ylabel("Frequency (%)")
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")

# 8E: Sharpe bar chart
ax = fig.add_subplot(gs[2, 1])
sharpes = [sharpe(p) for p in pnls_bt]
bar_c   = [ACCENT, GREEN, GOLD, RED]
bars    = ax.bar(names_bt, sharpes, color=bar_c,
                 edgecolor=DARK, linewidth=0.5)
ax.axhline(0, color=TEXT, lw=0.5, ls="--")
for bar, v in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_title("Sharpe Ratio Comparison")
ax.set_ylabel("Sharpe Ratio")
ax.set_facecolor(PANEL)
ax.grid(True, axis="y")

fig.savefig(os.path.join(FIGS, "m52_fig2_backtest_pnl.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [08] Fig 2 saved: backtest PnL & drawdown")

# =============================================================================
# 9. FIGURE 3 -- Q-Value Landscape & Epsilon Decay
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=DARK)
fig.suptitle("M52 -- Q-Value Landscape & Policy Diagnostics",
             color=TEXT, fontsize=11, y=1.01)

# 9A: Q-value heatmap over z-score 5/20 grid
ax = axes[0]
n_grid = 30
zs5_g  = np.linspace(-3, 3, n_grid)
zs20_g = np.linspace(-3, 3, n_grid)
q_map  = np.zeros((n_grid, n_grid))
for i, z5 in enumerate(zs5_g):
    for j, z20 in enumerate(zs20_g):
        s = np.array([0, 0, 0, z5, z20, 1.0])
        qv = online_net.forward(s[np.newaxis, :])[0]
        q_map[i, j] = ACTIONS[int(np.argmax(qv))]
im = ax.imshow(q_map, origin="lower", aspect="auto",
               extent=[-3, 3, -3, 3],
               cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("DQN Optimal Action\n(z-score 5d vs 20d)")
ax.set_xlabel("Z-score 20d")
ax.set_ylabel("Z-score 5d")
fig.colorbar(im, ax=ax, label="-1=Short  0=Flat  1=Long",
             orientation="vertical", fraction=0.046, pad=0.04)

# 9B: Q-values for a sample trajectory (DQN)
ax = axes[1]
n_show = 200
sample_states = states_all[LOOKBACK:LOOKBACK + n_show]
q_traj = online_net.forward(sample_states)
for a_i, (col, lab) in enumerate(zip([RED, TEXT, GREEN],
                                      ["Q(short)", "Q(flat)", "Q(long)"])):
    ax.plot(q_traj[:, a_i], color=col, lw=0.8, label=lab, alpha=0.85)
ax.set_title("Q-Values Along Sample Trajectory")
ax.set_xlabel("Day")
ax.set_ylabel("Q-Value")
ax.legend(fontsize=7)
ax.grid(True)

# 9C: Epsilon decay schedule
ax = axes[2]
eps_sched = [max(0.05, 1.0 * (0.992 ** ep)) for ep in range(N_EPISODES_D)]
ax.plot(eps_sched, color=PURPLE, lw=1.5)
ax.axhline(0.05, color=TEXT, lw=0.8, ls="--", label="eps_min=0.05")
ax.fill_between(range(N_EPISODES_D), eps_sched, 0.05,
                color=PURPLE, alpha=0.2, label="Exploration region")
ax.set_title("Epsilon-Greedy Decay Schedule")
ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon")
ax.legend(fontsize=7)
ax.grid(True)

for ax in axes:
    ax.set_facecolor(PANEL)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "m52_fig3_qvalue_policy.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print("  [09] Fig 3 saved: Q-value landscape & policy diagnostics")

# =============================================================================
# SUMMARY
# =============================================================================
dqn_sharpe  = sharpe(pnl_dqn)
bnh_sharpe  = sharpe(pnl_bnh)
dqn_mdd     = max_dd(pnl_dqn)
bnh_mdd     = max_dd(pnl_bnh)
long_pct    = float(np.sum(acts_dqn == 1)  / len(acts_dqn) * 100)
short_pct   = float(np.sum(acts_dqn == -1) / len(acts_dqn) * 100)
flat_pct    = float(np.sum(acts_dqn == 0)  / len(acts_dqn) * 100)

print()
print("  MODULE 52 COMPLETE -- 3 figures saved")
print("  Key Concepts:")
print("  [1] MDP: (S,A,P,R,gamma) -- state, action, transition, reward, discount")
print("  [2] Bellman: Q*(s,a)=E[r+gamma*max_a' Q*(s',a') | s,a]")
print("  [3] Q-Learning: off-policy TD control, tabular with discretised states")
print("  [4] DQN: neural Q-approximator + experience replay + target network")
print("  [5] Epsilon-greedy: exploration-exploitation trade-off")
print(f"  [6] DQN Sharpe={dqn_sharpe:.3f}  BnH Sharpe={bnh_sharpe:.3f}  "
      f"improvement={dqn_sharpe-bnh_sharpe:+.3f}")
print(f"  [7] DQN MDD={dqn_mdd:.4f}  BnH MDD={bnh_mdd:.4f}")
print(f"  [8] DQN positions: Long={long_pct:.1f}%  "
      f"Short={short_pct:.1f}%  Flat={flat_pct:.1f}%")
print("  NEXT: M53 -- Natural Language Processing for Finance (Sentiment)")
print()
