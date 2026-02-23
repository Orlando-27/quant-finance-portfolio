#!/usr/bin/env python3
# MODULE 42: RNN / LSTM -- LONG-TERM MEMORY IN SEQUENCES
# CQF Concepts Explained | Group 8: Deep Learning & NLP (2/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI
#
# THEORETICAL SCOPE
# -----------------
# Recurrent Neural Networks (RNN) process sequential data by
# maintaining a hidden state h_t that summarizes past information:
#
#   h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
#   y_t = W_hy * h_t + b_y
#
# Vanishing gradient problem (Bengio et al. 1994):
#   dL/dh_t = dL/dh_T * prod_{k=t+1}^{T} dh_k/dh_{k-1}
#   ||dh_k/dh_{k-1}|| = ||W_hh^T * diag(tanh'(z_k))||
#   If spectral_radius(W_hh) < 1: gradients vanish exponentially
#   If spectral_radius(W_hh) > 1: gradients explode
#
# LSTM (Hochreiter & Schmidhuber 1997):
#   Solves vanishing gradient via gated cell state c_t:
#
#   f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)  [forget gate]
#   i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)  [input gate]
#   g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)   [cell candidate]
#   o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)  [output gate]
#   c_t = f_t * c_{t-1} + i_t * g_t           [cell state update]
#   h_t = o_t * tanh(c_t)                     [hidden state]
#
# Key insight: c_t flows through time with only multiplicative
# (forget gate) and additive updates -- gradient highway.
#
# GRU (Cho et al. 2014) -- simplified LSTM:
#   z_t = sigma(W_z * [h_{t-1}, x_t])  [update gate]
#   r_t = sigma(W_r * [h_{t-1}, x_t])  [reset gate]
#   h_t_tilde = tanh(W * [r_t*h_{t-1}, x_t])
#   h_t = (1-z_t)*h_{t-1} + z_t*h_t_tilde
#
# Financial application:
#   Time-series prediction: given last L=20 daily returns,
#   predict sign of next-period return (binary classification).
#   Features: single asset return series (univariate LSTM).
#   Compared against: naive persistence, MLP, LSTM.
#
# SECTIONS
# --------
# 01  Vanishing gradient: RNN gradient norm vs sequence length
# 02  LSTM from scratch: all 4 gates in NumPy
# 03  Sequence dataset: sliding window on return series
# 04  sklearn MLPClassifier as MLP baseline
# 05  sklearn no native LSTM -- implement via manual unrolling
# 06  LSTM walk-forward: rolling train/test on return sequences
# 07  Gate activation visualization: f,i,o,g over time
# 08  Sequence length sensitivity: L=5,10,20,40
# 09  Return autocorrelation: justification for sequential model
# 10  Comparison: persistence vs MLP vs LSTM
# 11  Regime detection: hidden state clustering
# 12  Summary dashboard
#
# REFERENCES
# ----------
# Hochreiter & Schmidhuber (1997). Long Short-Term Memory.
#   Neural Computation 9(8), 1735-1780.
# Cho et al. (2014). Learning Phrase Representations using RNN
#   Encoder-Decoder. EMNLP 2014.
# Bengio, Simard & Frasconi (1994). Learning Long-Term Dependencies
#   with Gradient Descent is Difficult. IEEE Trans. NN 5(2).
# Fischer & Krauss (2018). Deep learning with LSTM networks for
#   financial market predictions. EJOR 270(2), 654-669.

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

DARK='#0D1117'; PANEL='#161B22'; WHITE='#E6EDF3'; GRAY='#8B949E'
GREEN='#3FB950'; RED='#F85149'; BLUE='#58A6FF'; ORANGE='#D29922'
PURPLE='#BC8CFF'; CYAN='#39D353'; YELLOW='#E3B341'
ACCENT=[BLUE,GREEN,ORANGE,RED,PURPLE,CYAN,YELLOW,GRAY]

plt.rcParams.update({
    'figure.facecolor':DARK,'axes.facecolor':PANEL,'axes.edgecolor':GRAY,
    'axes.labelcolor':WHITE,'text.color':WHITE,'xtick.color':GRAY,
    'ytick.color':GRAY,'grid.color':'#21262D','grid.linewidth':0.6,
    'legend.facecolor':PANEL,'legend.edgecolor':GRAY,
    'font.size':9,'axes.titlesize':10,'axes.titlecolor':WHITE,
})

def wm(ax, text='Jose O. Bobadilla | CQF', alpha=0.10):
    ax.text(0.5,0.5,text,transform=ax.transAxes,fontsize=14,
            color=WHITE,alpha=alpha,ha='center',va='center',
            rotation=30,fontweight='bold')

print('='*65)
print('  MODULE 42: RNN / LSTM')
print('  Vanishing Gradient | LSTM Gates | Sequence Prediction')
print('='*65)
t0=time.perf_counter()

# ================================================================
# SECTION 01: SYNTHETIC RETURN SERIES WITH REGIMES
# ================================================================
# Two-state Markov regime model:
# State 0 (bull): mu=+0.04%, sigma=0.8% daily
# State 1 (bear): mu=-0.06%, sigma=1.6% daily
# Transition: P(0->1)=0.02, P(1->0)=0.05 per day
np.random.seed(42)
T_days=252*6
P_trans=np.array([[0.98,0.02],[0.05,0.95]])
mu_reg=np.array([0.0004,-0.0006])
sig_reg=np.array([0.008,0.016])

states=np.zeros(T_days,dtype=int)
for t in range(1,T_days):
    states[t]=np.random.choice(2,p=P_trans[states[t-1]])

raw_ret=np.random.normal(
    mu_reg[states],sig_reg[states])

dates=pd.date_range('2018-01-02',periods=T_days,freq='B')
ret_series=pd.Series(raw_ret,index=dates,name='ret')
price_series=np.exp(ret_series.cumsum())

print('  [01] Return series: {} days  Bull={:.0f}d  Bear={:.0f}d'.format(
      T_days,(states==0).sum(),(states==1).sum()))

# ================================================================
# SECTION 02: VANISHING GRADIENT DEMO
# ================================================================
# For simple RNN h_t = tanh(W*h_{t-1} + U*x_t),
# the gradient dh_T/dh_t = prod_{k=t+1}^{T} W*diag(tanh'(z_k))
# Approximate by tracking ||dh_T/dh_t|| vs (T-t) for W near 1
lengths=list(range(1,51))
grad_norms_rnn=[]
grad_norms_lstm=[]

W_rnn=0.95  # slightly below 1 => vanishing
for L in lengths:
    # RNN: product of W * tanh'(z) ~ W * 0.5 (avg saturation)
    norm_rnn=(W_rnn*0.5)**L
    grad_norms_rnn.append(norm_rnn)
    # LSTM: forget gate ~ 0.9 (learned to remember)
    norm_lstm=0.9**L
    grad_norms_lstm.append(norm_lstm)

print('  [02] Vanishing gradient computed over T=1..50')

# ================================================================
# SECTION 03: LSTM FROM SCRATCH (NUMPY)
# ================================================================
class LSTMCell:
    '''
    Single LSTM cell. Processes one time step.
    Input size = input_dim, hidden size = hidden_dim.
    '''
    def __init__(self, input_dim, hidden_dim):
        self.n=input_dim; self.h=hidden_dim
        d=input_dim+hidden_dim
        # Xavier init
        s=np.sqrt(2.0/d)
        self.Wf=np.random.randn(hidden_dim,d)*s; self.bf=np.ones(hidden_dim)*0.5
        self.Wi=np.random.randn(hidden_dim,d)*s; self.bi=np.zeros(hidden_dim)
        self.Wg=np.random.randn(hidden_dim,d)*s; self.bg=np.zeros(hidden_dim)
        self.Wo=np.random.randn(hidden_dim,d)*s; self.bo=np.zeros(hidden_dim)
        self.Why=np.random.randn(1,hidden_dim)*s; self.by=np.zeros(1)

    @staticmethod
    def sig(z): return 1/(1+np.exp(-np.clip(z,-15,15)))

    def step(self, x, h_prev, c_prev):
        xh=np.concatenate([x,h_prev])
        f=self.sig(self.Wf@xh+self.bf)
        i=self.sig(self.Wi@xh+self.bi)
        g=np.tanh(self.Wg@xh+self.bg)
        o=self.sig(self.Wo@xh+self.bo)
        c=f*c_prev+i*g
        h=o*np.tanh(c)
        return h,c,{'f':f,'i':i,'g':g,'o':o,'c':c}

    def forward_sequence(self, X_seq):
        '''X_seq: (T, input_dim)'''
        T=len(X_seq)
        h=np.zeros(self.h); c=np.zeros(self.h)
        hs=np.zeros((T,self.h)); gates=[]
        for t,x in enumerate(X_seq):
            h,c,gate=self.step(x,h,c)
            hs[t]=h; gates.append(gate)
        y=hs@self.Why.T+self.by
        return y.ravel(),hs,gates

lstm_cell=LSTMCell(1,32)
# Run on a short sequence to visualize gates
x_demo=raw_ret[:100].reshape(-1,1)
y_demo,hs_demo,gates_demo=lstm_cell.forward_sequence(x_demo)
print('  [03] LSTM cell defined and tested on 100-step sequence')

# ================================================================
# SECTION 04: SLIDING WINDOW DATASET
# ================================================================
L=20  # lookback window
FWDD=5  # predict next 5-day return sign

# Build sequences: X shape (n_samples, L), y binary
rets=raw_ret
n_seq=len(rets)-L-FWDD
X_seq=np.zeros((n_seq,L))
y_seq=np.zeros(n_seq,dtype=int)
y_ret_seq=np.zeros(n_seq)

for i in range(n_seq):
    X_seq[i]=rets[i:i+L]
    fwd_r=rets[i+L:i+L+FWDD].sum()
    y_seq[i]=int(fwd_r>0)
    y_ret_seq[i]=fwd_r

# Train/test split: 70/30 in time
split=int(0.70*n_seq)
Xtr,Xte=X_seq[:split],X_seq[split:]
ytr_b,yte_b=y_seq[:split],y_seq[split:]
ytr_r,yte_r=y_ret_seq[:split],y_ret_seq[split:]
sc=StandardScaler().fit(Xtr)
Xtr_sc=sc.transform(Xtr); Xte_sc=sc.transform(Xte)
print('  [04] Sliding window: L={} seq  FWDD={}d  train={} test={}'.format(
      L,FWDD,len(Xtr),len(Xte)))

# ================================================================
# SECTION 05: MLP BASELINE + LSTM VIA MANUAL FEATURES
# ================================================================
# sklearn has no native LSTM; approximate LSTM with:
# (a) MLP on raw window features
# (b) MLP on engineered sequence features (mean, std, trend, momentum)
def seq_features(X):
    '''Extract sequence statistics: substitute for LSTM hidden state.'''
    mu=X.mean(axis=1,keepdims=True)
    sg=X.std(axis=1,keepdims=True)+1e-9
    trend=X[:,-1]-X[:,0]  # linear trend proxy
    lag=min(5, X.shape[1]-1); mom=X[:,-1]-X[:,-(lag+1)]   # 5-day momentum (adaptive)
    sharpe=mu.ravel()*np.sqrt(252/5)/(sg.ravel())
    acf1=np.array([np.corrcoef(x[:-1],x[1:])[0,1] for x in X])
    return np.column_stack([mu.ravel(),sg.ravel(),trend,mom,sharpe,acf1])

Xtr_feat=np.hstack([Xtr_sc,seq_features(Xtr)])
Xte_feat=np.hstack([Xte_sc,seq_features(Xte)])

# MLP on raw window
mlp_raw=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',
                      solver='adam',max_iter=300,random_state=42)
mlp_raw.fit(Xtr_sc,ytr_b)
prob_raw=mlp_raw.predict_proba(Xte_sc)[:,1]
acc_raw=accuracy_score(yte_b,mlp_raw.predict(Xte_sc))
auc_raw=roc_auc_score(yte_b,prob_raw)

# MLP on engineered features (LSTM proxy)
mlp_feat=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',
                       solver='adam',max_iter=300,random_state=42)
mlp_feat.fit(Xtr_feat,ytr_b)
prob_feat=mlp_feat.predict_proba(Xte_feat)[:,1]
acc_feat=accuracy_score(yte_b,mlp_feat.predict(Xte_feat))
auc_feat=roc_auc_score(yte_b,prob_feat)

# Persistence baseline: predict sign of last return
persist_pred=(Xte[:,-1]>0).astype(int)
acc_persist=accuracy_score(yte_b,persist_pred)

print('  [05] MLP raw: Acc={:.3f} AUC={:.3f}'.format(acc_raw,auc_raw))
print('       MLP+feat: Acc={:.3f} AUC={:.3f}'.format(acc_feat,auc_feat))
print('       Persistence: Acc={:.3f}'.format(acc_persist))

# ================================================================
# SECTION 06: SEQUENCE LENGTH SENSITIVITY
# ================================================================
print('  [06] Sequence length sensitivity ...')
L_grid=[5,10,20,40]
L_acc,L_auc=[],[]
for Lg in L_grid:
    n_=len(rets)-Lg-FWDD
    Xs=np.array([rets[i:i+Lg] for i in range(n_)])
    ys=np.array([int(rets[i+Lg:i+Lg+FWDD].sum()>0) for i in range(n_)])
    sp_=int(0.70*n_)
    Xtr_,Xte_=Xs[:sp_],Xs[sp_:]
    ytr_,yte_=ys[:sp_],ys[sp_:]
    sc_=StandardScaler().fit(Xtr_)
    m_=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',
                     solver='adam',max_iter=200,random_state=42)
    Xftr=np.hstack([sc_.transform(Xtr_),seq_features(Xtr_)])
    Xfte=np.hstack([sc_.transform(Xte_),seq_features(Xte_)])
    m_.fit(Xftr,ytr_)
    p_=m_.predict_proba(Xfte)[:,1]
    L_acc.append(accuracy_score(yte_,m_.predict(Xfte)))
    L_auc.append(roc_auc_score(yte_,p_))
    print('    L={}: Acc={:.3f} AUC={:.3f}'.format(Lg,L_acc[-1],L_auc[-1]))

# ================================================================
# SECTION 07: REGIME DETECTION VIA HIDDEN STATES
# ================================================================
# Run LSTM cell on full series, cluster hidden states
from sklearn.cluster import KMeans
x_full=raw_ret.reshape(-1,1)
_,hs_full,_=lstm_cell.forward_sequence(x_full)
# PCA to 2D for visualization
hs_centered=hs_full-hs_full.mean(axis=0)
U,S,Vt=np.linalg.svd(hs_centered,full_matrices=False)
hs_2d=U[:,:2]*S[:2]
# K-means: 2 clusters (bull/bear)
km=KMeans(n_clusters=2,random_state=42,n_init=10).fit(hs_2d)
pred_regime=km.labels_
# Align clusters to true states by majority
overlap0=np.mean(states[:len(pred_regime)][pred_regime==0]==0)
overlap1=np.mean(states[:len(pred_regime)][pred_regime==0]==1)
if overlap1>overlap0: pred_regime=1-pred_regime
regime_acc=np.mean(pred_regime==states[:len(pred_regime)])
print('  [07] Regime detection accuracy: {:.3f}'.format(regime_acc))

# ================================================================
# FIGURE 1: VANISHING GRADIENT + GATES
# ================================================================
print()
print('  Generating figures ...')
fig=plt.figure(figsize=(16,12),facecolor=DARK)
fig.suptitle('M42 -- RNN/LSTM: Vanishing Gradient & Gate Dynamics',
             color=WHITE,fontsize=14,fontweight='bold',y=0.98)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

# (a) Vanishing gradient
ax=fig.add_subplot(gs[0,0])
ax.semilogy(lengths,grad_norms_rnn,color=RED,lw=2,marker='o',ms=3,
            label='RNN (W=0.95)')
ax.semilogy(lengths,grad_norms_lstm,color=GREEN,lw=2,marker='s',ms=3,
            label='LSTM (forget=0.9)')
ax.set_xlabel('Sequence length T'); ax.set_ylabel('Gradient norm (log)')
ax.set_title('Vanishing Gradient:\nRNN vs LSTM over T steps')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# (b) LSTM gate activations over time
ax=fig.add_subplot(gs[0,1])
gate_f=np.array([g['f'].mean() for g in gates_demo])
gate_i=np.array([g['i'].mean() for g in gates_demo])
gate_o=np.array([g['o'].mean() for g in gates_demo])
gate_g=np.array([np.tanh(g['g']).mean() for g in gates_demo])
ax.plot(gate_f,color=BLUE,lw=1.5,label='Forget gate f_t')
ax.plot(gate_i,color=GREEN,lw=1.5,label='Input gate i_t')
ax.plot(gate_o,color=ORANGE,lw=1.5,label='Output gate o_t')
ax.plot(gate_g,color=PURPLE,lw=1.5,label='Cell cand. g_t (tanh)')
ax.set_xlabel('Time step'); ax.set_ylabel('Gate activation')
ax.set_title('LSTM Gate Activations\n(100-step return sequence)')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# (c) Return series + regime
ax=fig.add_subplot(gs[0,2])
cum=np.exp(np.cumsum(raw_ret[:500]))
ax.plot(cum,color=BLUE,lw=1.5,label='Price (first 500d)')
ax2b=ax.twinx()
ax2b.fill_between(range(500),states[:500],0,
                  color=RED,alpha=0.20,label='Bear regime')
ax2b.set_ylabel('Regime',color=RED)
ax2b.tick_params(axis='y',colors=RED)
ax.set_xlabel('Day'); ax.set_ylabel('Price')
ax.set_title('Synthetic Price: Bull/Bear Regimes')
ax.legend(fontsize=7,loc='upper left'); wm(ax)

# (d) Hidden state PCA + regime clusters
ax=fig.add_subplot(gs[1,:2])
colors_reg=[GREEN if r==0 else RED for r in pred_regime[:500]]
ax.scatter(hs_2d[:500,0],hs_2d[:500,1],c=colors_reg,s=4,alpha=0.6)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('LSTM Hidden State PCA\nRegime clustering (K=2)  acc={:.3f}'.format(regime_acc))
# Legend patches
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN,label='Bull (pred)'),
                   Patch(color=RED,label='Bear (pred)')],fontsize=8)
ax.grid(True); wm(ax)

# (e) Sequence length sensitivity
ax=fig.add_subplot(gs[1,2])
ax.plot(L_grid,L_auc,color=BLUE,lw=2,marker='o',ms=6,label='AUC')
ax.plot(L_grid,L_acc,color=GREEN,lw=2,marker='s',ms=6,label='Accuracy')
ax.axhline(0.5,color=RED,lw=1,ls='--',label='Random')
ax.set_xlabel('Lookback window L'); ax.set_ylabel('Score')
ax.set_title('Sequence Length Sensitivity\n(MLP+seq features)')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

p1=os.path.join(OUT_DIR,'m42_01_lstm_gates_regimes.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# ================================================================
# FIGURE 2: AUTOCORRELATION + MODEL COMPARISON
# ================================================================
fig,axes=plt.subplots(2,2,figsize=(16,10),facecolor=DARK)
fig.suptitle('M42 -- Return Autocorrelation & Model Comparison',
             color=WHITE,fontsize=14,fontweight='bold')

# ACF of returns
ax=axes[0,0]
max_lag=40
acf_vals=[]
for lag in range(1,max_lag+1):
    acf_vals.append(pearsonr(raw_ret[:-lag],raw_ret[lag:])[0])
ax.bar(range(1,max_lag+1),acf_vals,
       color=[GREEN if abs(v)>1.96/np.sqrt(T_days) else BLUE for v in acf_vals],
       alpha=0.8,width=0.7)
conf=1.96/np.sqrt(T_days)
ax.axhline(conf,color=RED,lw=1,ls='--',label='+/-95% CI')
ax.axhline(-conf,color=RED,lw=1,ls='--')
ax.axhline(0,color=GRAY,lw=0.5)
ax.set_xlabel('Lag (days)'); ax.set_ylabel('Autocorrelation')
ax.set_title('Return ACF: Justification for Sequential Model')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# ACF of absolute returns (volatility clustering)
ax=axes[0,1]
acf_abs=[]
for lag in range(1,max_lag+1):
    acf_abs.append(pearsonr(np.abs(raw_ret[:-lag]),np.abs(raw_ret[lag:]))[0])
ax.bar(range(1,max_lag+1),acf_abs,color=ORANGE,alpha=0.8,width=0.7)
ax.axhline(conf,color=RED,lw=1,ls='--',label='+/-95% CI')
ax.axhline(-conf,color=RED,lw=1,ls='--')
ax.axhline(0,color=GRAY,lw=0.5)
ax.set_xlabel('Lag (days)'); ax.set_ylabel('Autocorrelation')
ax.set_title('|Return| ACF: Volatility Clustering\n(ARCH effect)')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Model comparison bar
ax=axes[1,0]
models=['Persistence','MLP (raw)','MLP+SeqFeat']
accs=[acc_persist,acc_raw,acc_feat]
aucs=[0.5,auc_raw,auc_feat]
x=np.arange(len(models))
ax.bar(x-0.2,accs,width=0.35,color=BLUE,alpha=0.8,label='Accuracy')
ax.bar(x+0.2,aucs,width=0.35,color=GREEN,alpha=0.8,label='AUC')
ax.axhline(0.5,color=RED,lw=1,ls='--',label='Random=0.5')
ax.set_xticks(x); ax.set_xticklabels(models,fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Model Comparison: Persistence vs MLP vs MLP+SeqFeat')
ax.legend(fontsize=8); ax.grid(True,axis='y'); wm(ax)

# Summary table
ax=axes[1,1]
ax.set_facecolor(DARK); ax.axis('off')
tdata=[
    ['Persistence (last sign)','{:.3f}'.format(acc_persist),'0.500','--','trivial'],
    ['MLP raw window (L=20)','{:.3f}'.format(acc_raw),'{:.3f}'.format(auc_raw),
     '(64,32)','raw returns'],
    ['MLP + seq features','{:.3f}'.format(acc_feat),'{:.3f}'.format(auc_feat),
     '(64,32)','mean,std,trend,acf'],
    ['Best L (sensitivity)','L={}'.format(L_grid[np.argmax(L_auc)]),
     '{:.3f}'.format(max(L_auc)),'(64,32)','optimized window'],
]
cols=['Model','Accuracy','AUC','Architecture','Features']
tbl=ax.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax.set_title('Sequence Model Comparison Summary',color=WHITE)

p2=os.path.join(OUT_DIR,'m42_02_autocorrelation_comparison.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# ================================================================
# FIGURE 3: LSTM CELL DIAGRAM + CELL STATE DYNAMICS
# ================================================================
fig=plt.figure(figsize=(16,8),facecolor=DARK)
fig.suptitle('M42 -- LSTM Cell State Dynamics & Architecture Diagram',
             color=WHITE,fontsize=14,fontweight='bold')
gs3=gridspec.GridSpec(1,2,figure=fig,wspace=0.32)

# Cell state & hidden state over time
ax=fig.add_subplot(gs3[0,0])
c_vals=np.array([g['c'].mean() for g in gates_demo])
h_vals=hs_demo.mean(axis=1)
ax.plot(c_vals,color=BLUE,lw=2,label='Cell state c_t (mean)')
ax.plot(h_vals,color=GREEN,lw=2,label='Hidden state h_t (mean)')
ax2b=ax.twinx()
ax2b.plot(x_demo.ravel()*100,color=GRAY,lw=0.8,alpha=0.5,label='Return (%)')
ax2b.set_ylabel('Return (%)',color=GRAY)
ax2b.tick_params(axis='y',colors=GRAY)
ax.set_xlabel('Time step'); ax.set_ylabel('State value')
ax.set_title('LSTM Cell State vs Hidden State\n(100-step return sequence)')
ax.legend(fontsize=8,loc='upper left'); ax.grid(True); wm(ax)

# Forget gate heatmap: which past inputs are remembered
ax=fig.add_subplot(gs3[0,1])
f_vals=np.array([g['f'] for g in gates_demo])  # (100, 32)
im=ax.imshow(f_vals[:,:16].T,cmap='RdYlGn',aspect='auto',vmin=0,vmax=1)
plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04).ax.tick_params(colors=WHITE)
ax.set_xlabel('Time step'); ax.set_ylabel('Hidden unit index')
ax.set_title('Forget Gate f_t Heatmap\n(1=remember, 0=forget)')
wm(ax)

p3=os.path.join(OUT_DIR,'m42_03_cell_state_dynamics.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# ================================================================
# SUMMARY
# ================================================================
print()
print('='*65)
print('  MODULE 42 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] RNN vanishing: ||dh_T/dh_t|| ~ (W*tanh_prime)^(T-t)')
print('  [2] LSTM forget gate: c_t = f_t*c_{t-1} + i_t*g_t')
print('  [3] Cell state: additive gradient highway prevents vanish')
print('  [4] GRU: 2 gates vs LSTM 3 gates, competitive performance')
print('  [5] Seq features (mean,std,trend,acf) proxy LSTM hidden state')
print('  [6] |Return| ACF > return ACF: volatility clustering (ARCH)')
print('  [7] Regime detection via hidden state clustering (PCA+KMeans)')
print()
print('  MLP raw: Acc={:.3f} AUC={:.3f}'.format(acc_raw,auc_raw))
print('  MLP+SeqFeat: Acc={:.3f} AUC={:.3f}'.format(acc_feat,auc_feat))
print('  Persistence baseline: Acc={:.3f}'.format(acc_persist))
print('  Regime detection acc: {:.3f}'.format(regime_acc))
print('  Best window L={} AUC={:.3f}'.format(
      L_grid[np.argmax(L_auc)],max(L_auc)))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)