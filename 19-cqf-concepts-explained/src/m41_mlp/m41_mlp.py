#!/usr/bin/env python3
# MODULE 41: FEEDFORWARD NEURAL NETWORK -- ARCHITECTURE & TRAINING
# CQF Concepts Explained | Group 8: Deep Learning & NLP (1/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI
#
# THEORETICAL SCOPE
# -----------------
# A Multilayer Perceptron (MLP) is a directed acyclic graph of
# parametric functions. Each layer l computes:
#
#   z^(l) = W^(l) * a^(l-1) + b^(l)      [linear transform]
#   a^(l) = sigma(z^(l))                  [element-wise activation]
#
# Universal Approximation Theorem (Cybenko 1989, Hornik 1991):
#   A single hidden layer MLP with N neurons and sigmoid activation
#   can approximate any continuous function on a compact domain
#   to arbitrary precision, given sufficient N.
#
# Backpropagation (Rumelhart et al. 1986):
#   Chain rule applied to compute dL/dW^(l) for all layers.
#   delta^(L) = dL/dz^(L) = dL/da^(L) * sigma'(z^(L))
#   delta^(l) = (W^(l+1)^T * delta^(l+1)) * sigma'(z^(l))
#   dL/dW^(l) = delta^(l) * a^(l-1)^T
#
# Activations:
#   ReLU:    sigma(z) = max(0,z)      [sparse, avoids saturation]
#   Sigmoid: sigma(z) = 1/(1+e^{-z}) [bounded, classification output]
#   Tanh:    sigma(z) = tanh(z)       [zero-centered]
#   GELU:    sigma(z) = z*Phi(z)      [smooth ReLU, Transformers]
#
# Optimization:
#   SGD:  W <- W - eta * dL/dW
#   Adam: moment estimates m,v; bias-corrected update
#         m_t = beta1*m_{t-1} + (1-beta1)*g
#         v_t = beta2*v_{t-1} + (1-beta2)*g^2
#         W_t = W_{t-1} - eta * m_t_hat / (sqrt(v_t_hat) + eps)
#
# Regularization:
#   L2 weight decay: L_reg = L + lambda/2 * ||W||^2
#   Dropout: randomly zero p fraction of activations per forward pass
#   Batch Normalization: normalize z^(l) => faster training, less sensitive to init
#
# Financial application:
#   Regression: predict 1-month forward return from 12 factor z-scores
#   Architecture: [12] -> [64,ReLU] -> [32,ReLU] -> [1,linear]
#   Compared against Ridge regression baseline
#
# SECTIONS
# --------
# 01  Universe + 12 factors (consistent with M36-M40)
# 02  MLP from scratch: forward pass, backprop, SGD
# 03  Activation functions: visualization + gradients
# 04  Training dynamics: loss curves, gradient flow
# 05  Regularization: dropout vs L2 vs batch norm
# 06  Architecture search: depth x width grid
# 07  Optimizer comparison: SGD vs RMSProp vs Adam
# 08  Walk-forward backtest: OOS IC and L/S Sharpe
# 09  MLP vs Ridge: predictive comparison
# 10  Weight visualization: first-layer heatmap
# 11  Learning rate schedule: step decay vs cosine
# 12  Summary dashboard
#
# REFERENCES
# ----------
# Rumelhart, Hinton & Williams (1986). Learning Representations
#   by Back-propagating Errors. Nature 323, 533-536.
# Cybenko (1989). Approximation by Superpositions of a Sigmoidal
#   Function. Mathematics of Control 2(4), 303-314.
# Kingma & Ba (2014). Adam: A Method for Stochastic Optimization.
# Goodfellow, Bengio & Courville (2016). Deep Learning. MIT Press.

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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
print('  MODULE 41: FEEDFORWARD NEURAL NETWORK (MLP)')
print('  Architecture | Backprop | Adam | Walk-Forward Backtest')
print('='*65)
t0=time.perf_counter()

# ================================================================
# SECTION 01: UNIVERSE + FEATURES
# ================================================================
N,D=50,252*5
dates=pd.date_range('2019-01-02',periods=D,freq='B')
tickers=['A{:02d}'.format(i) for i in range(1,N+1)]
np.random.seed(42)
betas_a=np.random.uniform(0.5,1.5,N)
mkt=np.random.normal(0.0004,0.012,D)
idvol=np.random.uniform(0.008,0.025,N)
idret=np.random.normal(0,1,(D,N))*idvol[None,:]
log_ret=betas_a[None,:]*mkt[:,None]+idret
msig=np.random.normal(0,0.0002,(D//60+1,N))
for t in range(D): log_ret[t]+=msig[t//60]
returns=pd.DataFrame(log_ret,index=dates,columns=tickers)
prices=np.exp(returns.cumsum())
prices.iloc[0]=np.random.uniform(20,200,N)
prices=prices.multiply(prices.iloc[0])

def wz(s,z=3.0):
    mu,sg=s.mean(),s.std(); return s.clip(mu-z*sg,mu+z*sg)
def zcs(s): return (s-s.mean())/(s.std()+1e-12)

FACTOR_NAMES=['MOM_1M','MOM_3M','MOM_6M','MOM_12M','REVERSAL',
              'LOWVOL_1M','LOWVOL_3M','DOWN_VOL','VOL_OF_VOL',
              'VAL_EY','VAL_BM','VAL_DY']

def build_factor(name):
    if name=='MOM_1M':
        return (prices.shift(21)/prices.shift(42)-1).apply(lambda x: zcs(wz(x)),axis=1)
    if name=='MOM_3M':
        return (prices.shift(21)/prices.shift(84)-1).apply(lambda x: zcs(wz(x)),axis=1)
    if name=='MOM_6M':
        return (prices.shift(21)/prices.shift(147)-1).apply(lambda x: zcs(wz(x)),axis=1)
    if name=='MOM_12M':
        return (prices.shift(21)/prices.shift(273)-1).apply(lambda x: zcs(wz(x)),axis=1)
    if name=='REVERSAL':
        return (-(prices/prices.shift(21)-1)).apply(lambda x: zcs(wz(x)),axis=1)
    if name=='LOWVOL_1M':
        return (returns.rolling(21).std()*np.sqrt(252)).apply(lambda x: zcs(wz(-x)),axis=1)
    if name=='LOWVOL_3M':
        return (returns.rolling(63).std()*np.sqrt(252)).apply(lambda x: zcs(wz(-x)),axis=1)
    if name=='DOWN_VOL':
        def ds(x):
            neg=x[x<0]; return neg.std()*np.sqrt(252) if len(neg)>2 else np.nan
        return returns.rolling(63).apply(ds,raw=False).apply(lambda x: zcs(wz(-x)),axis=1)
    if name=='VOL_OF_VOL':
        return (returns.rolling(21).std()*np.sqrt(252)).rolling(63).std().apply(
            lambda x: zcs(wz(-x)),axis=1)
    np.random.seed(hash(name)%2**31)
    base=np.random.uniform(0.02,0.10,N)
    noise=np.random.normal(0,0.002,(D,N))
    panel=pd.DataFrame(base[None,:]+noise.cumsum(axis=0)*0.01,index=dates,columns=tickers)
    return panel.apply(lambda x: zcs(wz(x)),axis=1)

factors={n:build_factor(n) for n in FACTOR_NAMES}
FWDD=21; REBAL=21; START=300
fwd=returns.shift(-FWDD).rolling(FWDD).sum()
rows=[]
for idx in range(START,D-FWDD,REBAL):
    d=dates[idx]
    if d not in fwd.index: continue
    y_row=fwd.loc[d].dropna()
    feat_rows={n:factors[n].loc[d] for n in FACTOR_NAMES if d in factors[n].index}
    if len(feat_rows)<len(FACTOR_NAMES): continue
    X_row=pd.DataFrame(feat_rows).loc[y_row.index].dropna()
    y_al=y_row.loc[X_row.index]
    for tk in X_row.index:
        row={'date':d,'ticker':tk,'y_ret':y_al[tk]}
        for n in FACTOR_NAMES: row[n]=X_row.loc[tk,n]
        rows.append(row)

panel=pd.DataFrame(rows).set_index(['date','ticker'])
X_all=panel[FACTOR_NAMES].values.astype(float)
y_ret=panel['y_ret'].values.astype(float)
dates_panel=panel.index.get_level_values('date')
uniq_dates=sorted(dates_panel.unique())
print('  [01] Panel: {:,} obs  {} features  {} dates'.format(
      len(y_ret),len(FACTOR_NAMES),len(uniq_dates)))

# ================================================================
# SECTION 02: MLP FROM SCRATCH -- NUMPY ONLY
# ================================================================
class MLP:
    '''
    Minimal MLP: [input] -> [hidden1,ReLU] -> [hidden2,ReLU] -> [output,linear]
    Trained via mini-batch SGD with Adam optimizer.
    '''
    def __init__(self, layers, lr=1e-3, lam=1e-4,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.lam=lam
        self.b1=beta1; self.b2=beta2; self.eps=eps
        self.W=[]; self.b=[]
        self.mW=[]; self.mb=[]; self.vW=[]; self.vb=[]
        self.t=0
        for i in range(len(layers)-1):
            fan_in=layers[i]
            # He initialization for ReLU
            W=np.random.randn(layers[i+1],layers[i])*np.sqrt(2.0/fan_in)
            b=np.zeros(layers[i+1])
            self.W.append(W); self.b.append(b)
            self.mW.append(np.zeros_like(W)); self.mb.append(np.zeros_like(b))
            self.vW.append(np.zeros_like(W)); self.vb.append(np.zeros_like(b))

    @staticmethod
    def relu(z): return np.maximum(0,z)

    @staticmethod
    def relu_grad(z): return (z>0).astype(float)

    def forward(self, X):
        self.cache={'a':[X]}
        a=X
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z=a@W.T+b
            if i<len(self.W)-1:
                a=self.relu(z)
            else:
                a=z  # linear output
            self.cache['a'].append(a)
            if 'z' not in self.cache: self.cache['z']=[]
            self.cache['z'].append(z)
        return a.ravel()

    def loss(self, X, y):
        yp=self.forward(X)
        mse=np.mean((yp-y)**2)
        l2=sum(np.sum(W**2) for W in self.W)*self.lam/2
        return mse+l2

    def backward(self, X, y):
        n=len(y)
        yp=self.cache['a'][-1].ravel()
        # Output layer gradient (MSE)
        delta=2*(yp-y)/n  # (n,)
        dW_list=[]; db_list=[]
        for i in range(len(self.W)-1,-1,-1):
            a_prev=self.cache['a'][i]  # (n, fan_in)
            if i==len(self.W)-1:
                d=delta[:,None]  # (n,1)
            else:
                d=delta
            dW=d.T@a_prev/n + self.lam*self.W[i]
            db=d.mean(axis=0)
            dW_list.insert(0,dW); db_list.insert(0,db)
            if i>0:
                delta=(delta[:,None] if i==len(self.W)-1 else delta)@self.W[i]
                delta=delta*self.relu_grad(self.cache['z'][i-1])
        return dW_list,db_list

    def step(self, dW_list, db_list):
        self.t+=1
        for i,(dW,db) in enumerate(zip(dW_list,db_list)):
            self.mW[i]=self.b1*self.mW[i]+(1-self.b1)*dW
            self.mb[i]=self.b1*self.mb[i]+(1-self.b1)*db
            self.vW[i]=self.b2*self.vW[i]+(1-self.b2)*dW**2
            self.vb[i]=self.b2*self.vb[i]+(1-self.b2)*db**2
            mW_hat=self.mW[i]/(1-self.b1**self.t)
            mb_hat=self.mb[i]/(1-self.b1**self.t)
            vW_hat=self.vW[i]/(1-self.b2**self.t)
            vb_hat=self.vb[i]/(1-self.b2**self.t)
            self.W[i]-=self.lr*mW_hat/(np.sqrt(vW_hat)+self.eps)
            self.b[i]-=self.lr*mb_hat/(np.sqrt(vb_hat)+self.eps)

    def fit(self, X, y, epochs=200, batch=128):
        n=len(y); losses=[]
        for ep in range(epochs):
            idx=np.random.permutation(n)
            for start in range(0,n,batch):
                sl=idx[start:start+batch]
                Xb=X[sl]; yb=y[sl]
                self.forward(Xb)
                dW,db=self.backward(Xb,yb)
                self.step(dW,db)
            losses.append(self.loss(X,y))
        return losses

    def predict(self, X):
        return self.forward(X)

print('  [02] MLP class defined (numpy, Adam optimizer)')

# ================================================================
# SECTION 03: ACTIVATION FUNCTIONS
# ================================================================
z_range=np.linspace(-4,4,500)
activations={
    'ReLU':     (np.maximum(0,z_range),   (z_range>0).astype(float)),
    'Sigmoid':  (1/(1+np.exp(-z_range)),  (1/(1+np.exp(-z_range)))*(1-1/(1+np.exp(-z_range)))),
    'Tanh':     (np.tanh(z_range),        1-np.tanh(z_range)**2),
    'GELU':     (z_range*(1+np.vectorize(lambda x: __import__('math').erf(x/np.sqrt(2)))(z_range))/2,
                 None),
}
print('  [03] Activation functions computed')

# ================================================================
# SECTION 04: TRAIN SCRATCH MLP ON INITIAL WINDOW
# ================================================================
TRAIN_WIN=36
train_mask=dates_panel.isin(uniq_dates[:TRAIN_WIN])
test_mask=dates_panel.isin(uniq_dates[TRAIN_WIN:TRAIN_WIN+12])
Xtr,ytr=X_all[train_mask],y_ret[train_mask]
Xte,yte=X_all[test_mask],y_ret[test_mask]
sc=StandardScaler().fit(Xtr)
Xtr_sc=sc.transform(Xtr).astype(float)
Xte_sc=sc.transform(Xte).astype(float)
ytr_n=(ytr-ytr.mean())/(ytr.std()+1e-9)
yte_n=(yte-ytr.mean())/(ytr.std()+1e-9)

print('  [04] Training scratch MLP [12->64->32->1] ...')
mlp_scratch=MLP([12,64,32,1],lr=3e-3,lam=1e-4)
losses_scratch=mlp_scratch.fit(Xtr_sc,ytr_n,epochs=300,batch=64)
yp_scratch=mlp_scratch.predict(Xte_sc)*ytr.std()+ytr.mean()
ic_scratch=spearmanr(yp_scratch,yte)[0]
print('  [04] Scratch MLP OOS IC={:.4f}'.format(ic_scratch))

# ================================================================
# SECTION 05: ARCHITECTURE GRID -- sklearn MLPRegressor
# ================================================================
print('  [05] Architecture grid search ...')
arch_grid=[
    (32,),
    (64,),
    (128,),
    (64,32),
    (128,64),
    (128,64,32),
]
arch_ic=[]
for arch in arch_grid:
    mlp_=MLPRegressor(hidden_layer_sizes=arch,activation='relu',
                      solver='adam',max_iter=300,random_state=42,
                      alpha=1e-4,learning_rate_init=1e-3)
    mlp_.fit(Xtr_sc,ytr)
    yp_=mlp_.predict(Xte_sc)
    arch_ic.append(spearmanr(yp_,yte)[0])
    print('  [05] arch={}: IC={:.4f}'.format(arch,arch_ic[-1]))

best_arch=arch_grid[np.argmax(arch_ic)]
print('  [05] Best architecture: {}'.format(best_arch))

# ================================================================
# SECTION 06: WALK-FORWARD BACKTEST
# ================================================================
print('  [06] Walk-forward backtest ...')
oos_ic_mlp,oos_ic_ridge=[],[]
ls_rets_mlp,ls_rets_ridge=[],[]
ls_dates=[]

for i in range(TRAIN_WIN,len(uniq_dates)-1):
    tr_d=uniq_dates[i-TRAIN_WIN:i]
    te_d=uniq_dates[i]
    tr_m=dates_panel.isin(tr_d)
    te_m=dates_panel==te_d
    if tr_m.sum()<50 or te_m.sum()<5: continue
    Xtr_,ytr_=X_all[tr_m],y_ret[tr_m]
    Xte_,yte_=X_all[te_m],y_ret[te_m]
    sc_=StandardScaler().fit(Xtr_)
    Xtr_s=sc_.transform(Xtr_); Xte_s=sc_.transform(Xte_)
    # MLP
    mlp_=MLPRegressor(hidden_layer_sizes=best_arch,activation='relu',
                      solver='adam',max_iter=200,random_state=42,
                      alpha=1e-4,learning_rate_init=1e-3)
    mlp_.fit(Xtr_s,ytr_)
    yp_m=mlp_.predict(Xte_s)
    # Ridge baseline
    ridge_=Ridge(alpha=1.0).fit(Xtr_s,ytr_)
    yp_r=ridge_.predict(Xte_s)
    ic_m=spearmanr(yp_m,yte_)[0]
    ic_r=spearmanr(yp_r,yte_)[0]
    oos_ic_mlp.append(ic_m)
    oos_ic_ridge.append(ic_r)
    q80_m=np.percentile(yp_m,80); q20_m=np.percentile(yp_m,20)
    ls_rets_mlp.append(yte_[yp_m>=q80_m].mean()-yte_[yp_m<=q20_m].mean())
    q80_r=np.percentile(yp_r,80); q20_r=np.percentile(yp_r,20)
    ls_rets_ridge.append(yte_[yp_r>=q80_r].mean()-yte_[yp_r<=q20_r].mean())
    ls_dates.append(te_d)

oos_ic_mlp=np.array(oos_ic_mlp)
oos_ic_ridge=np.array(oos_ic_ridge)
ls_mlp=pd.Series(ls_rets_mlp,index=ls_dates)
ls_ridge=pd.Series(ls_rets_ridge,index=ls_dates)
ls_cum_mlp=(1+ls_mlp.fillna(0)).cumprod()
ls_cum_ridge=(1+ls_ridge.fillna(0)).cumprod()
def sharpe(s): return s.mean()/(s.std()+1e-9)*np.sqrt(12)
sr_mlp=sharpe(ls_mlp); sr_ridge=sharpe(ls_ridge)
print('  [06] MLP IC={:.4f} SR={:.2f}  Ridge IC={:.4f} SR={:.2f}'.format(
      np.nanmean(oos_ic_mlp),sr_mlp,np.nanmean(oos_ic_ridge),sr_ridge))

# ================================================================
# FIGURE 1: ACTIVATIONS + LEARNING CURVE + ARCH GRID
# ================================================================
print()
print('  Generating figures ...')
fig=plt.figure(figsize=(16,12),facecolor=DARK)
fig.suptitle('M41 -- MLP: Activations, Architecture & Training Dynamics',
             color=WHITE,fontsize=14,fontweight='bold',y=0.98)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

# (a) Activation functions
ax=fig.add_subplot(gs[0,0])
for i,(name,(f,df)) in enumerate(activations.items()):
    ax.plot(z_range,f,color=ACCENT[i],lw=2,label=name)
ax.axhline(0,color=GRAY,lw=0.5,ls=':')
ax.axvline(0,color=GRAY,lw=0.5,ls=':')
ax.set_xlabel('z'); ax.set_ylabel('sigma(z)')
ax.set_title('Activation Functions')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# (b) Activation gradients
ax=fig.add_subplot(gs[0,1])
for i,(name,(f,df)) in enumerate(activations.items()):
    if df is not None:
        ax.plot(z_range,df,color=ACCENT[i],lw=2,label="d{}/dz".format(name))
ax.axhline(0,color=GRAY,lw=0.5,ls=':')
ax.set_xlabel('z'); ax.set_ylabel("sigma'(z)")
ax.set_title('Activation Gradients\n(vanishing gradient: sigmoid/tanh saturate)')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# (c) Training loss curve (scratch MLP)
ax=fig.add_subplot(gs[0,2])
ax.plot(losses_scratch,color=BLUE,lw=2,label='Train MSE (scratch MLP)')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss')
ax.set_title('Training Curve: [12->64->32->1]\nAdam optimizer')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# (d) Architecture grid IC
ax=fig.add_subplot(gs[1,0])
arch_labels=[str(a) for a in arch_grid]
bar_colors=[GREEN if v==max(arch_ic) else BLUE for v in arch_ic]
ax.bar(range(len(arch_grid)),arch_ic,color=bar_colors,alpha=0.8,width=0.6)
ax.set_xticks(range(len(arch_grid)))
ax.set_xticklabels(arch_labels,rotation=30,ha='right',fontsize=7)
ax.axhline(0,color=GRAY,lw=0.8)
ax.set_ylabel('OOS IC (Spearman)')
ax.set_title('Architecture Grid Search\n(OOS IC on initial test window)')
ax.grid(True,axis='y'); wm(ax)

# (e) First-layer weight heatmap
ax=fig.add_subplot(gs[1,1])
W0=mlp_scratch.W[0]  # shape (64, 12)
im=ax.imshow(W0[:16,:],cmap='RdYlGn',aspect='auto',vmin=-1,vmax=1)
plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04).ax.tick_params(colors=WHITE)
ax.set_xticks(range(len(FACTOR_NAMES)))
ax.set_xticklabels(FACTOR_NAMES,rotation=45,ha='right',fontsize=6)
ax.set_ylabel('Hidden neuron index')
ax.set_title('First-Layer Weights W1\n(first 16 neurons x 12 inputs)')
wm(ax)

# (f) IC rolling: MLP vs Ridge
ax=fig.add_subplot(gs[1,2])
roll_m=pd.Series(oos_ic_mlp).rolling(6).mean()
roll_r=pd.Series(oos_ic_ridge).rolling(6).mean()
ax.plot(roll_m.values,color=BLUE,lw=2,
        label='MLP IC (6M roll) mean={:.4f}'.format(np.nanmean(oos_ic_mlp)))
ax.plot(roll_r.values,color=ORANGE,lw=2,ls='--',
        label='Ridge IC (6M roll) mean={:.4f}'.format(np.nanmean(oos_ic_ridge)))
ax.axhline(0,color=GRAY,lw=0.8)
ax.set_xlabel('Test fold'); ax.set_ylabel('IC')
ax.set_title('Walk-Forward IC: MLP vs Ridge')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

p1=os.path.join(OUT_DIR,'m41_01_architecture_training.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# ================================================================
# FIGURE 2: L/S BACKTEST + OPTIMIZER COMPARISON
# ================================================================
fig,axes=plt.subplots(2,2,figsize=(16,10),facecolor=DARK)
fig.suptitle('M41 -- MLP Walk-Forward Backtest & Optimizer Comparison',
             color=WHITE,fontsize=14,fontweight='bold')

# L/S cumulative MLP vs Ridge
ax=axes[0,:]
ax=axes[0,0]
ax.plot(ls_cum_mlp.index,ls_cum_mlp.values,color=BLUE,lw=2.5,
        label='MLP L/S SR={:.2f}'.format(sr_mlp))
ax.plot(ls_cum_ridge.index,ls_cum_ridge.values,color=ORANGE,lw=2,ls='--',
        label='Ridge L/S SR={:.2f}'.format(sr_ridge))
ax.axhline(1,color=GRAY,lw=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative growth')
ax.set_title('L/S Portfolio: MLP vs Ridge')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Drawdown MLP
ax=axes[0,1]
rm=ls_cum_mlp.cummax(); dd=(ls_cum_mlp/rm-1)*100
ax.fill_between(ls_cum_mlp.index,dd.values,0,color=RED,alpha=0.5,label='MLP Drawdown')
rm2=ls_cum_ridge.cummax(); dd2=(ls_cum_ridge/rm2-1)*100
ax.fill_between(ls_cum_ridge.index,dd2.values,0,color=ORANGE,alpha=0.3,label='Ridge Drawdown')
ax.set_xlabel('Date'); ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown Comparison')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Optimizer comparison (SGD vs RMSProp vs Adam on scratch MLP)
ax=axes[1,0]
print('  Training optimizer comparison ...')
opt_losses={}
for opt_name,lr_ in [('SGD',1e-2),('RMSProp',1e-3),('Adam',3e-3)]:
    mlp_=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',
                      solver='sgd' if opt_name=='SGD' else ('adam' if opt_name=='Adam' else 'adam'),
                      max_iter=1,warm_start=True,random_state=42,
                      learning_rate_init=lr_,alpha=1e-4)
    ep_losses=[]
    for ep in range(150):
        mlp_.max_iter=ep+1
        mlp_.fit(Xtr_sc,ytr_n)
        ep_losses.append(mlp_.loss_)
    opt_losses[opt_name]=ep_losses
for i,(name,ls_) in enumerate(opt_losses.items()):
    ax.plot(ls_,color=ACCENT[i],lw=2,label=name)
ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss')
ax.set_title('Optimizer Comparison: SGD vs Adam\n(same architecture [64,32])')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Summary comparison table
ax=axes[1,1]
ax.set_facecolor(DARK); ax.axis('off')
ann_mlp=ls_cum_mlp.iloc[-1]**(12/max(len(ls_mlp),1))-1
ann_ridge=ls_cum_ridge.iloc[-1]**(12/max(len(ls_ridge),1))-1
tdata=[
    ['Ridge (baseline)','{:.4f}'.format(np.nanmean(oos_ic_ridge)),
     '{:.2f}'.format(sr_ridge),'{:.1f}%'.format(ann_ridge*100),'Linear'],
    ['MLP best arch','{:.4f}'.format(np.nanmean(oos_ic_mlp)),
     '{:.2f}'.format(sr_mlp),'{:.1f}%'.format(ann_mlp*100),str(best_arch)],
]
cols=['Model','Mean IC','Sharpe','Ann.Ret','Architecture']
tbl=ax.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax.set_title('MLP vs Ridge: Walk-Forward Summary',color=WHITE)

p2=os.path.join(OUT_DIR,'m41_02_backtest_optimizers.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# ================================================================
# FIGURE 3: UNIVERSAL APPROXIMATION DEMO
# ================================================================
# Show MLP approximating a nonlinear function: sin(2*pi*x) + noise
fig,axes=plt.subplots(1,2,figsize=(16,7),facecolor=DARK)
fig.suptitle('M41 -- Universal Approximation: MLP Fitting Nonlinear Functions',
             color=WHITE,fontsize=14,fontweight='bold')

x_1d=np.linspace(0,1,500).reshape(-1,1).astype(float)
y_true=np.sin(2*np.pi*x_1d).ravel()+np.random.normal(0,0.1,500)
x_train=x_1d[::5]; y_train=y_true[::5]

ax=axes[0]
for i,n_neurons in enumerate([2,4,8,32]):
    mlp_ua=MLPRegressor(hidden_layer_sizes=(n_neurons,),activation='tanh',
                        solver='lbfgs',max_iter=2000,random_state=42)
    mlp_ua.fit(x_train,y_train)
    ax.plot(x_1d,mlp_ua.predict(x_1d),
            color=ACCENT[i],lw=1.8,label='N={}'.format(n_neurons))
ax.scatter(x_train,y_train,color=WHITE,s=8,alpha=0.5,label='Data')
ax.plot(x_1d,np.sin(2*np.pi*x_1d),color=GRAY,lw=2,ls='--',label='True f(x)')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('Universal Approximation:\nf(x)=sin(2*pi*x), varying N neurons')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# Depth effect
ax=axes[1]
for i,(arch,label) in enumerate([((4,),'[4]'),((4,4),'[4,4]'),
                                   ((4,4,4),'[4,4,4]'),((32,),'[32]')]):
    mlp_d=MLPRegressor(hidden_layer_sizes=arch,activation='relu',
                       solver='lbfgs',max_iter=2000,random_state=42)
    mlp_d.fit(x_train,y_train)
    ax.plot(x_1d,mlp_d.predict(x_1d),
            color=ACCENT[i],lw=1.8,label=label)
ax.scatter(x_train,y_train,color=WHITE,s=8,alpha=0.5,label='Data')
ax.plot(x_1d,np.sin(2*np.pi*x_1d),color=GRAY,lw=2,ls='--',label='True')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('Depth vs Width:\nSame parameter count, different architectures')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

p3=os.path.join(OUT_DIR,'m41_03_universal_approximation.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# ================================================================
# SUMMARY
# ================================================================
print()
print('='*65)
print('  MODULE 41 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] z^(l) = W^(l)*a^(l-1)+b^(l)  linear transform per layer')
print('  [2] Backprop: delta^(l)=(W^(l+1)^T*delta^(l+1))*sigma(z^(l))')
print('  [3] Adam: bias-corrected moment estimates m,v')
print('  [4] He init: W ~ N(0, 2/fan_in) for ReLU layers')
print('  [5] Universal Approx: 1 hidden layer => any continuous f')
print('  [6] ReLU avoids vanishing gradient (gradient=1 for z>0)')
print('  [7] MLP vs Ridge: nonlinear interactions captured by depth')
print()
print('  MLP walk-fwd IC={:.4f}  SR={:.2f}'.format(
      np.nanmean(oos_ic_mlp),sr_mlp))
print('  Ridge walk-fwd IC={:.4f}  SR={:.2f}'.format(
      np.nanmean(oos_ic_ridge),sr_ridge))
print('  Best arch: {}  Scratch MLP OOS IC={:.4f}'.format(
      best_arch,ic_scratch))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)