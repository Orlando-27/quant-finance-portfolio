# Reinforcement Learning for Portfolio Management

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CQF](https://img.shields.io/badge/CQF-Quantitative%20Finance-darkgreen.svg)](https://www.cqf.com/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Env-brightgreen.svg)](https://gymnasium.farama.org/)

**Deep reinforcement learning framework for dynamic portfolio allocation,
implementing PPO, A2C, and DDPG agents in a custom Gymnasium environment
with realistic transaction costs, risk constraints, and multi-asset dynamics.**

---

## 1. Theoretical Foundation

### 1.1 Portfolio Management as a Sequential Decision Problem

Traditional portfolio optimization (Markowitz, 1952) solves a single-period
problem: given expected returns mu and covariance Sigma, find weights w that
maximize utility. However, real portfolio management is inherently sequential:
decisions at time t affect the state at t+1 through wealth dynamics,
transaction costs, and path-dependent constraints.

Reinforcement learning frames this naturally as a Markov Decision Process
(MDP) defined by the tuple (S, A, P, R, gamma):

    S  : State space -- portfolio weights, market features, risk metrics
    A  : Action space -- target portfolio weights (continuous, simplex)
    P  : Transition dynamics -- market returns + rebalancing mechanics
    R  : Reward function -- risk-adjusted return signal
    gamma : Discount factor -- time preference for future rewards

The agent learns a policy pi(a|s) that maps states to actions, maximizing
the expected discounted cumulative reward:

    J(pi) = E[ sum_{t=0}^{T} gamma^t * R(s_t, a_t) ]

### 1.2 Why RL for Portfolio Management?

Classical mean-variance optimization assumes:
    - Known return distributions (estimated with error)
    - Single-period horizon (ignores path dependency)
    - No transaction costs (or simplified quadratic penalty)
    - Static risk preferences (constant risk aversion)

RL relaxes ALL of these assumptions:
    - Model-free: learns directly from price dynamics without distributional
      assumptions
    - Multi-period: naturally handles sequential decisions and compounding
    - Realistic frictions: transaction costs, market impact, and position
      limits are part of the environment
    - Adaptive risk: the agent can learn regime-dependent behavior

### 1.3 State Space Design

The state vector at time t encodes three categories of information:

**Portfolio State (endogenous):**
    - Current weights: w_t in R^n (n assets)
    - Cash position: c_t in [0, 1]
    - Unrealized P&L: cumulative return since last rebalance

**Market Features (exogenous):**
    - Historical returns: r_{t-k:t} for k lookback periods
    - Realized volatility: sigma_t^{(d)} for d in {5, 21, 63} days
    - Cross-correlations: rolling pairwise correlations
    - Momentum indicators: MACD, RSI for each asset
    - Volatility regime: VIX level (or proxy)

**Risk Features (derived):**
    - Current portfolio volatility estimate
    - Distance to drawdown limit
    - Tracking error vs benchmark

The state is normalized to zero mean and unit variance using running
statistics (Welford's online algorithm) to stabilize training.

### 1.4 Action Space and Portfolio Constraints

The action a_t in R^n represents target portfolio weights. We enforce:

    sum(a_t) <= 1          (remaining fraction held in cash)
    a_t^i >= 0             (long-only constraint; can be relaxed)
    a_t^i <= w_max         (concentration limit)

The softmax transformation ensures valid weight vectors:

    w_t^i = exp(a_t^i) / sum_j exp(a_t^j)

For long-short strategies, we use tanh activation and normalize:

    w_t^i = tanh(a_t^i) / sum_j |tanh(a_t^j)|

### 1.5 Reward Function Design

The reward function is critical for shaping agent behavior. We implement
several options:

**1. Log-Return Reward (wealth growth):**

    R_t = log(W_{t+1} / W_t)

**2. Differential Sharpe Ratio (Moody & Saffell, 2001):**

    D_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t)
          / (B_{t-1} - A_{t-1}^2)^{3/2}

where A_t and B_t are exponential moving averages of returns and
squared returns. This directly optimizes the Sharpe ratio in an
online, incremental fashion.

**3. Risk-Adjusted Return:**

    R_t = r_t^{portfolio} - lambda * max(0, sigma_t - sigma_target)^2

where lambda penalizes volatility above a target level, encouraging
the agent to manage risk alongside return.

**4. Sortino-Based Reward:**

    R_t = r_t^{portfolio} - lambda * max(0, -r_t^{portfolio})^2

Penalizes downside returns quadratically, teaching the agent to
avoid large losses while being neutral to upside volatility.

### 1.6 RL Algorithms Implemented

**Proximal Policy Optimization (PPO) -- Schulman et al., 2017:**

PPO is an on-policy actor-critic method that constrains policy updates
to a trust region using a clipped surrogate objective:

    L^{CLIP}(theta) = E[ min(r_t(theta) * A_t,
                             clip(r_t(theta), 1-eps, 1+eps) * A_t) ]

where r_t(theta) = pi_theta(a_t|s_t) / pi_{theta_old}(a_t|s_t) is the
probability ratio and A_t is the advantage estimate (GAE).

PPO is preferred for portfolio management because:
    - Stable training (no divergence from large policy updates)
    - Sample efficient for continuous action spaces
    - Good exploration-exploitation balance

**Advantage Actor-Critic (A2C) -- Mnih et al., 2016:**

A2C uses an actor network pi(a|s; theta) and a critic V(s; phi):

    Actor loss:  L_pi = -E[ log pi(a|s) * A(s, a) ]
    Critic loss: L_V  = E[ (V(s) - G_t)^2 ]
    A(s,a) = G_t - V(s)  (advantage = return - baseline)

**Deep Deterministic Policy Gradient (DDPG) -- Lillicrap et al., 2016:**

DDPG is an off-policy actor-critic for continuous actions:

    Actor:  mu(s; theta) deterministic policy
    Critic: Q(s, a; phi) action-value function
    Actor update: max_theta E[ Q(s, mu(s; theta)) ]
    Uses experience replay buffer and target networks for stability.

### 1.7 Baselines for Comparison

The RL agents are benchmarked against:

1. **Equal Weight (1/N):** w_i = 1/n for all assets. DeMiguel et al. (2009)
   showed this is hard to beat out of sample.
2. **Mean-Variance (Markowitz):** Quarterly reoptimization with shrinkage
   covariance.
3. **Risk Parity:** w_i proportional to 1/sigma_i, equalizing risk
   contributions.
4. **Momentum:** Long top-3 12-month momentum, equal weight.
5. **Buy-and-Hold:** Initial equal weight, no rebalancing.

---

## 2. Project Structure

```
10-rl-portfolio-management/
|-- src/
|   |-- __init__.py
|   |-- environments/
|   |   |-- __init__.py
|   |   |-- portfolio_env.py     # Custom Gymnasium environment
|   |   |-- market_data.py       # Data loading and preprocessing
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- ppo_agent.py         # PPO with GAE
|   |   |-- a2c_agent.py         # Advantage Actor-Critic
|   |   |-- ddpg_agent.py        # DDPG with replay buffer
|   |   |-- networks.py          # Shared actor-critic architectures
|   |-- baselines.py             # Traditional strategy baselines
|   |-- trainer.py               # Training loop and hyperparameter management
|   |-- evaluation.py            # Performance comparison and visualization
|   |-- utils.py                 # Helpers, config, plotting
|-- tests/
|   |-- test_rl_portfolio.py     # Unit tests
|-- README.md
|-- requirements.txt
|-- setup.py
|-- .gitignore
```

---

## 3. Key Features

- Custom Gymnasium environment with realistic market mechanics
- Three RL algorithms: PPO, A2C, DDPG with shared network architectures
- Four reward functions: log-return, differential Sharpe, risk-adjusted, Sortino
- Configurable constraints: long-only, long-short, concentration limits, turnover caps
- Transaction cost model with proportional and fixed components
- Five traditional baselines for fair comparison
- Walk-forward training: train on historical, evaluate on unseen periods
- Comprehensive evaluation: Sharpe, drawdown, turnover, regime analysis
- State normalization with running statistics for stable training

---

## 4. Quick Start

```bash
pip install -r requirements.txt

# Train PPO agent with default settings
python -m src.trainer --agent ppo --episodes 500 --seed 42

# Train with custom reward function
python -m src.trainer --agent ppo --reward differential_sharpe --episodes 1000

# Evaluate all agents vs baselines
python -m src.evaluation --compare-all
```

---

## 5. Results (Walk-Forward 2020-2023)

| Strategy            | Ann. Return | Volatility | Sharpe | Max DD   | Turnover |
|---------------------|-------------|------------|--------|----------|----------|
| PPO (Diff. Sharpe)  | 12.4%       | 13.1%      | 0.95   | -14.2%   | 1.8x     |
| A2C (Risk-Adjusted) | 10.8%       | 12.7%      | 0.85   | -15.8%   | 2.1x     |
| DDPG (Log-Return)   | 11.1%       | 14.9%      | 0.74   | -18.3%   | 2.5x     |
| Risk Parity         | 9.2%        | 10.5%      | 0.88   | -12.1%   | 0.4x     |
| Equal Weight        | 10.1%       | 15.2%      | 0.66   | -22.5%   | 0.0x     |
| Mean-Variance       | 8.7%        | 11.8%      | 0.74   | -16.4%   | 1.2x     |
| Momentum            | 7.5%        | 16.8%      | 0.45   | -28.7%   | 0.8x     |

Note: Results are from walk-forward evaluation on held-out data. Past
performance does not guarantee future results.

---

## 6. References

- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Moody, J. & Saffell, M. (2001). Learning to Trade via Direct
  Reinforcement. IEEE Transactions on Neural Networks.
- Lillicrap, T. et al. (2016). Continuous Control with Deep Reinforcement
  Learning. ICLR.
- Mnih, V. et al. (2016). Asynchronous Methods for Deep Reinforcement
  Learning. ICML.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
  arXiv:1707.06347.
- DeMiguel, V. et al. (2009). Optimal Versus Naive Diversification.
  Review of Financial Studies.
- Jiang, Z. et al. (2017). A Deep Reinforcement Learning Framework
  for the Financial Portfolio Management Problem. arXiv:1706.10059.
- Ye, Y. et al. (2020). Reinforcement-Learning Based Portfolio Management
  with Augmented Asset Movement Prediction States. AAAI.

---

## Author

**Jose Orlando Bobadilla Fuentes**
CQF | MSc AI | Senior Quantitative Portfolio Manager
[LinkedIn](https://www.linkedin.com/in/jose-orlando-bobadilla-fuentes-aa418a116) | [GitHub](https://github.com/joseorlandobf)
