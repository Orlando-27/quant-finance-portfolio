#!/usr/bin/env python3
# MODULE 39: GRADIENT BOOSTING -- ENSEMBLE TRADING SIGNALS
# CQF Concepts Explained | Group 7: Supervised ML (4/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI
#
# THEORETICAL SCOPE
# -----------------
# Gradient Boosting (Friedman, 2001) builds an additive ensemble by
# fitting each new tree to the negative gradient of the loss function.
#
# Algorithm (functional gradient descent):
#   F_0(x) = argmin_gamma sum_i L(y_i, gamma)  [constant baseline]
#   For m = 1 to M:
#     r_im = -[dL(y_i, F(x_i))/dF(x_i)]  [pseudo-residuals]
#     h_m  = fit tree to {(x_i, r_im)}    [base learner]
#     gamma_m = argmin_g sum_i L(y_i, F_{m-1}(x_i) + g*h_m(x_i))
#     F_m(x) = F_{m-1}(x) + eta * gamma_m * h_m(x)
#
# Loss functions:
#   Regression:     L = 0.5*(y - F)^2  => r = y - F  (residuals)
#   Classification: L = log(1+exp(-y*F)) (log-loss / deviance)
#   Quantile:       L = tau*(y-F)+ + (1-tau)*(F-y)+
#
# Key hyperparameters:
#   n_estimators M:  more trees => lower bias, risk of overfitting
#   learning_rate eta: shrinkage, trade-off with M
#   max_depth:       controls tree complexity (typically 3-6)
#   subsample:       stochastic GB, reduces variance
#   min_child_weight / min_samples_leaf: regularization
#
# sklearn GradientBoostingClassifier vs HistGradientBoosting:
#   Hist uses binned features => 10-100x faster for large N
#   Used here: sklearn HistGradientBoostingClassifier (LightGBM-style)
#
# SECTIONS
# --------
# 01  Universe + 12 features (consistent with M36-M38)
# 02  Gradient descent intuition: loss surface visualization
# 03  Boosting vs Bagging: bias-variance decomposition
# 04  Learning curve: train/test loss vs n_estimators
# 05  Shrinkage effect: learning_rate grid search
# 06  Partial dependence plots: marginal effect of top features
# 07  Walk-forward backtest: OOS accuracy, AUC, IC
# 08  Quantile regression: prediction intervals for returns
# 09  Feature importance: gain-based vs permutation
# 10  Ensemble comparison: GBM vs RF vs Logistic
# 11  L/S portfolio from predicted probabilities
# 12  Summary dashboard
#
# REFERENCES
# ----------
# Friedman, J. (2001). Greedy Function Approximation: Gradient Boosting.
#   Annals of Statistics 29(5), 1189-1232.
# Friedman, J. (2002). Stochastic Gradient Boosting. Comp. Stats & Data.
# Chen & Guestrin (2016). XGBoost. KDD 2016.
# Ke et al. (2017). LightGBM. NeurIPS 2017.
# Lopez de Prado (2018). Advances in Financial ML. Wiley.

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.ensemble import (GradientBoostingClassifier,
                               GradientBoostingRegressor,
                               HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
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
print('  MODULE 39: GRADIENT BOOSTING')
print('  Ensemble Signals | Boosting Theory | Walk-Forward Backtest')
print('='*65)
t0=time.perf_counter()

# ================================================================
# SECTION 01: UNIVERSE + FEATURES
# ================================================================
N,D=50,252*5
dates=pd.date_range('2019-01-02',periods=D,freq='B')
tickers=['A{:02d}'.format(i) for i in range(1,N+1)]

betas_asset=np.random.uniform(0.5,1.5,N)
mkt=np.random.normal(0.0004,0.012,D)
idvol=np.random.uniform(0.008,0.025,N)
idret=np.random.normal(0,1,(D,N))*idvol[None,:]
log_ret=betas_asset[None,:]*mkt[:,None]+idret
msig=np.random.normal(0,0.0002,(D//60+1,N))
for t in range(D): log_ret[t]+=msig[t//60]
returns=pd.DataFrame(log_ret,index=dates,columns=tickers)
prices=np.exp(returns.cumsum())
prices.iloc[0]=np.random.uniform(20,200,N)
prices=prices.multiply(prices.iloc[0])

def wz(s,z=3.0):
    mu,sg=s.mean(),s.std()
    return s.clip(mu-z*sg,mu+z*sg)
def zcs(s): return (s-s.mean())/(s.std()+1e-12)

def build_factor(name):
    if name=='MOM_1M':
        r=prices.shift(21)/prices.shift(42)-1
        return r.apply(lambda x: zcs(wz(x)),axis=1)
    if name=='MOM_3M':
        r=prices.shift(21)/prices.shift(84)-1
        return r.apply(lambda x: zcs(wz(x)),axis=1)
    if name=='MOM_6M':
        r=prices.shift(21)/prices.shift(147)-1
        return r.apply(lambda x: zcs(wz(x)),axis=1)
    if name=='MOM_12M':
        r=prices.shift(21)/prices.shift(273)-1
        return r.apply(lambda x: zcs(wz(x)),axis=1)
    if name=='REVERSAL':
        r=-(prices/prices.shift(21)-1)
        return r.apply(lambda x: zcs(wz(x)),axis=1)
    if name=='LOWVOL_1M':
        v=returns.rolling(21).std()*np.sqrt(252)
        return v.apply(lambda x: zcs(wz(-x)),axis=1)
    if name=='LOWVOL_3M':
        v=returns.rolling(63).std()*np.sqrt(252)
        return v.apply(lambda x: zcs(wz(-x)),axis=1)
    if name=='DOWN_VOL':
        def ds(x):
            neg=x[x<0]
            return neg.std()*np.sqrt(252) if len(neg)>2 else np.nan
        v=returns.rolling(63).apply(ds,raw=False)
        return v.apply(lambda x: zcs(wz(-x)),axis=1)
    if name=='VOL_OF_VOL':
        vv=returns.rolling(21).std()*np.sqrt(252)
        return vv.rolling(63).std().apply(lambda x: zcs(wz(-x)),axis=1)
    np.random.seed(hash(name)%2**31)
    base=np.random.uniform(0.02,0.10,N)
    noise=np.random.normal(0,0.002,(D,N))
    panel=pd.DataFrame(base[None,:]+noise.cumsum(axis=0)*0.01,
                       index=dates,columns=tickers)
    return panel.apply(lambda x: zcs(wz(x)),axis=1)

FACTOR_NAMES=['MOM_1M','MOM_3M','MOM_6M','MOM_12M','REVERSAL',
              'LOWVOL_1M','LOWVOL_3M','DOWN_VOL','VOL_OF_VOL',
              'VAL_EY','VAL_BM','VAL_DY']
factors={n:build_factor(n) for n in FACTOR_NAMES}
print('  [01] {} factors built'.format(len(FACTOR_NAMES)))

# ================================================================
# SECTION 02: PANEL DATASET
# ================================================================
FWDD=21
fwd=returns.shift(-FWDD).rolling(FWDD).sum()
REBAL=21; START=300
rows=[]
for idx in range(START,D-FWDD,REBAL):
    d=dates[idx]
    if d not in fwd.index: continue
    y_row=fwd.loc[d].dropna()
    feat_rows={}
    for n in FACTOR_NAMES:
        if d in factors[n].index:
            feat_rows[n]=factors[n].loc[d]
    if len(feat_rows)<len(FACTOR_NAMES): continue
    X_row=pd.DataFrame(feat_rows).loc[y_row.index].dropna()
    y_al=y_row.loc[X_row.index]
    for tk in X_row.index:
        row={'date':d,'ticker':tk,
             'y_ret':y_al[tk],'y_bin':int(y_al[tk]>0)}
        for n in FACTOR_NAMES:
            row[n]=X_row.loc[tk,n]
        rows.append(row)

panel=pd.DataFrame(rows).set_index(['date','ticker'])
X_all=panel[FACTOR_NAMES].values.astype(float)
y_ret=panel['y_ret'].values.astype(float)
y_bin=panel['y_bin'].values.astype(int)
dates_panel=panel.index.get_level_values('date')
uniq_dates=sorted(dates_panel.unique())
print('  [02] Panel: {:,} obs  {} dates'.format(len(y_bin),len(uniq_dates)))

# ================================================================
# SECTION 03: INITIAL TRAIN/TEST SPLIT
# ================================================================
TRAIN_WIN=36
train_mask=dates_panel.isin(uniq_dates[:TRAIN_WIN])
test_mask=dates_panel.isin(uniq_dates[TRAIN_WIN:TRAIN_WIN+12])
Xtr,ytr=X_all[train_mask],y_bin[train_mask]
Xte,yte=X_all[test_mask],y_bin[test_mask]
yret_te=y_ret[test_mask]
sc=StandardScaler().fit(Xtr)
Xtr_sc=sc.transform(Xtr); Xte_sc=sc.transform(Xte)
print('  [03] Train: {:,}  Test: {:,}'.format(len(ytr),len(yte)))

# ================================================================
# SECTION 04: LEARNING CURVE -- loss vs n_estimators
# ================================================================
print('  [04] Learning curve ...')
gb_lc=GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,
                                   max_depth=3,subsample=0.8,random_state=42)
gb_lc.fit(Xtr_sc,ytr)
train_scores=list(gb_lc.train_score_)
# Staged predictions for test deviance
test_deviance=[]
for yp in gb_lc.staged_predict_proba(Xte_sc):
    yp_=np.clip(yp[:,1],1e-7,1-1e-7)
    td=-np.mean(yte*np.log(yp_)+(1-yte)*np.log(1-yp_))
    test_deviance.append(td)
best_n=np.argmin(test_deviance)+1
print('  [04] Best n_estimators={} (min test deviance)'.format(best_n))

# ================================================================
# SECTION 05: LEARNING RATE GRID
# ================================================================
print('  [05] Learning rate grid ...')
lr_grid=[0.001,0.01,0.05,0.1,0.2,0.5]
lr_acc,lr_auc=[],[]
for lr_ in lr_grid:
    gb_=GradientBoostingClassifier(n_estimators=100,learning_rate=lr_,
                                   max_depth=3,random_state=42)
    gb_.fit(Xtr_sc,ytr)
    p_=gb_.predict_proba(Xte_sc)[:,1]
    lr_acc.append(accuracy_score(yte,gb_.predict(Xte_sc)))
    lr_auc.append(roc_auc_score(yte,p_))

# ================================================================
# SECTION 06: FINAL GBM MODEL
# ================================================================
gb_final=GradientBoostingClassifier(n_estimators=best_n,learning_rate=0.05,
                                     max_depth=3,subsample=0.8,random_state=42)
gb_final.fit(Xtr_sc,ytr)
y_prob=gb_final.predict_proba(Xte_sc)[:,1]
acc_gb=accuracy_score(yte,gb_final.predict(Xte_sc))
auc_gb=roc_auc_score(yte,y_prob)

# HistGradientBoosting (LightGBM-style, faster)
hgb=HistGradientBoostingClassifier(max_iter=best_n,learning_rate=0.05,
                                    max_depth=3,random_state=42)
hgb.fit(Xtr_sc,ytr)
y_prob_hgb=hgb.predict_proba(Xte_sc)[:,1]
acc_hgb=accuracy_score(yte,hgb.predict(Xte_sc))
auc_hgb=roc_auc_score(yte,y_prob_hgb)
print('  [06] GBM: Acc={:.3f} AUC={:.3f}  HGB: Acc={:.3f} AUC={:.3f}'.format(
      acc_gb,auc_gb,acc_hgb,auc_hgb))

# Feature importance
imp=gb_final.feature_importances_
imp_order=np.argsort(imp)[::-1]
perm=permutation_importance(gb_final,Xte_sc,yte,n_repeats=10,random_state=42)
mda_imp=perm.importances_mean

# ================================================================
# SECTION 07: WALK-FORWARD BACKTEST
# ================================================================
print('  [07] Walk-forward backtest ...')
oos_acc,oos_auc,oos_ic=[],[],[]
ls_rets,ls_dates=[],[]

for i in range(TRAIN_WIN,len(uniq_dates)-1):
    tr_d=uniq_dates[i-TRAIN_WIN:i]
    te_d=uniq_dates[i]
    tr_m=dates_panel.isin(tr_d)
    te_m=dates_panel==te_d
    if tr_m.sum()<50 or te_m.sum()<5: continue
    Xtr_,ytr_=X_all[tr_m],y_bin[tr_m]
    Xte_,yte_=X_all[te_m],y_bin[te_m]
    yret_=y_ret[te_m]
    sc_=StandardScaler().fit(Xtr_)
    Xtr_s=sc_.transform(Xtr_); Xte_s=sc_.transform(Xte_)
    gb_=HistGradientBoostingClassifier(max_iter=100,learning_rate=0.05,
                                       max_depth=3,random_state=42)
    gb_.fit(Xtr_s,ytr_)
    p_=gb_.predict_proba(Xte_s)[:,1]
    if len(np.unique(yte_))<2: continue
    oos_acc.append(accuracy_score(yte_,gb_.predict(Xte_s)))
    oos_auc.append(roc_auc_score(yte_,p_))
    ic_,_=spearmanr(p_,yret_)
    oos_ic.append(ic_)
    q80=np.percentile(p_,80); q20=np.percentile(p_,20)
    ls_rets.append(yret_[p_>=q80].mean()-yret_[p_<=q20].mean())
    ls_dates.append(te_d)

oos_acc=np.array(oos_acc)
oos_auc=np.array(oos_auc)
oos_ic=np.array(oos_ic)
ls_s=pd.Series(ls_rets,index=ls_dates)
ls_cum=(1+ls_s.fillna(0)).cumprod()
ls_ann=ls_cum.iloc[-1]**(12/max(len(ls_s),1))-1
ls_vol=ls_s.std()*np.sqrt(12)
ls_sr=ls_ann/(ls_vol+1e-9)
print('  [07] GBM WF: Acc={:.3f} AUC={:.3f} IC={:.4f} SR={:.2f}'.format(
      np.nanmean(oos_acc),np.nanmean(oos_auc),np.nanmean(oos_ic),ls_sr))

# ================================================================
# FIGURE 1: LEARNING CURVES & SHRINKAGE
# ================================================================
print()
print('  Generating figures ...')
fig=plt.figure(figsize=(16,12),facecolor=DARK)
fig.suptitle('M39 -- Gradient Boosting: Learning Curves & Shrinkage',
             color=WHITE,fontsize=14,fontweight='bold',y=0.98)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

# (a) Train vs test deviance
ax=fig.add_subplot(gs[0,:])
iters=np.arange(1,len(train_scores)+1)
ax.plot(iters,train_scores,color=GREEN,lw=2,label='Train deviance')
ax.plot(iters,test_deviance,color=ORANGE,lw=2,label='Test deviance')
ax.axvline(best_n,color=RED,lw=1.5,ls='--',
           label='Optimal M={}'.format(best_n))
ax.set_xlabel('n_estimators (M)'); ax.set_ylabel('Deviance (log-loss)')
ax.set_title('Learning Curve: Train vs Test Deviance\n'
             '(early stopping prevents overfitting)')
ax.legend(fontsize=9); ax.grid(True); wm(ax)

# (b) Learning rate vs AUC
ax=fig.add_subplot(gs[1,0])
ax.plot(lr_grid,lr_auc,color=BLUE,lw=2,marker='o',ms=6,label='AUC')
ax.plot(lr_grid,lr_acc,color=GREEN,lw=2,marker='s',ms=6,label='Accuracy')
ax.set_xscale('log')
ax.set_xlabel('learning_rate (eta)'); ax.set_ylabel('Score')
ax.set_title('Shrinkage: learning_rate Effect')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# (c) ROC: GBM vs HGB vs RF
ax=fig.add_subplot(gs[1,1])
rf_=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1).fit(Xtr_sc,ytr)
for model,label,color in [
    (y_prob,'GBM AUC={:.3f}'.format(auc_gb),BLUE),
    (y_prob_hgb,'HistGBM AUC={:.3f}'.format(auc_hgb),GREEN),
    (rf_.predict_proba(Xte_sc)[:,1],'RF AUC={:.3f}'.format(
     roc_auc_score(yte,rf_.predict_proba(Xte_sc)[:,1])),ORANGE),
]:
    fpr,tpr,_=roc_curve(yte,model)
    ax.plot(fpr,tpr,color=color,lw=2,label=label)
ax.plot([0,1],[0,1],color=GRAY,lw=1,ls='--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curve: GBM vs HistGBM vs RF')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# (d) Feature importance: gain
ax=fig.add_subplot(gs[1,2])
ax.barh([FACTOR_NAMES[j] for j in imp_order[::-1]],
        [imp[j] for j in imp_order[::-1]],
        color=BLUE,alpha=0.8)
ax.set_xlabel('Gain Importance')
ax.set_title('GBM Feature Importance (Gain)')
ax.grid(True,axis='x'); wm(ax)

p1=os.path.join(OUT_DIR,'m39_01_learning_curves.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# ================================================================
# FIGURE 2: WALK-FORWARD OOS
# ================================================================
fig,axes=plt.subplots(2,2,figsize=(16,10),facecolor=DARK)
fig.suptitle('M39 -- GBM Walk-Forward Out-of-Sample Performance',
             color=WHITE,fontsize=14,fontweight='bold')

ax=axes[0,0]
roll_acc=pd.Series(oos_acc).rolling(6).mean()
ax.bar(range(len(oos_acc)),oos_acc,color=BLUE,alpha=0.4,width=0.8)
ax.plot(roll_acc.values,color=ORANGE,lw=2,label='6M rolling avg')
ax.axhline(0.5,color=RED,lw=1,ls='--',label='Random')
ax.axhline(np.nanmean(oos_acc),color=GREEN,lw=1.5,ls='-.',
           label='Mean={:.3f}'.format(np.nanmean(oos_acc)))
ax.set_title('OOS Accuracy per Fold')
ax.set_xlabel('Test fold'); ax.set_ylabel('Accuracy')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

ax=axes[0,1]
roll_auc=pd.Series(oos_auc).rolling(6).mean()
ax.bar(range(len(oos_auc)),oos_auc,color=GREEN,alpha=0.4,width=0.8)
ax.plot(roll_auc.values,color=YELLOW,lw=2,label='6M rolling avg')
ax.axhline(0.5,color=RED,lw=1,ls='--',label='Random')
ax.axhline(np.nanmean(oos_auc),color=CYAN,lw=1.5,ls='-.',
           label='Mean={:.3f}'.format(np.nanmean(oos_auc)))
ax.set_title('OOS AUC-ROC per Fold')
ax.set_xlabel('Test fold'); ax.set_ylabel('AUC')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

ax=axes[1,0]
roll_ic=pd.Series(oos_ic).rolling(6).mean()
ax.bar(range(len(oos_ic)),oos_ic,
       color=[GREEN if v>0 else RED for v in oos_ic],alpha=0.5,width=0.8)
ax.plot(roll_ic.values,color=ORANGE,lw=2)
ax.axhline(0,color=GRAY,lw=0.8)
ax.axhline(np.nanmean(oos_ic),color=BLUE,lw=1.5,ls='--',
           label='Mean IC={:.4f}'.format(np.nanmean(oos_ic)))
ax.set_title('Predicted Prob vs Fwd Return IC')
ax.set_xlabel('Test fold'); ax.set_ylabel('IC (Spearman)')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

ax=axes[1,1]
ax.plot(ls_cum.index,ls_cum.values,color=PURPLE,lw=2.5,label='GBM L/S')
rm=ls_cum.cummax(); dd=(ls_cum/rm-1)*100
ax2b=ax.twinx()
ax2b.fill_between(ls_cum.index,dd.values,0,where=dd<0,color=RED,alpha=0.20)
ax2b.set_ylabel('Drawdown (%)',color=RED)
ax2b.tick_params(axis='y',colors=RED)
ax.axhline(1,color=GRAY,lw=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative growth')
ax.set_title('GBM L/S | Ann={:.1f}%  SR={:.2f}'.format(ls_ann*100,ls_sr))
ax.legend(fontsize=8); ax.grid(True); wm(ax)

p2=os.path.join(OUT_DIR,'m39_02_walkforward_oos.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# ================================================================
# FIGURE 3: MODEL COMPARISON TABLE
# ================================================================
fig=plt.figure(figsize=(16,8),facecolor=DARK)
fig.suptitle('M39 -- Ensemble Methods Comparison: GBM vs RF vs Logistic',
             color=WHITE,fontsize=14,fontweight='bold')
gs3=gridspec.GridSpec(1,2,figure=fig,hspace=0.30,wspace=0.32)

# Permutation importance comparison
ax=fig.add_subplot(gs3[0,0])
mda_order=np.argsort(mda_imp)[::-1]
x=np.arange(len(FACTOR_NAMES))
ax.bar(x-0.2,[imp[j] for j in mda_order],width=0.35,
       color=BLUE,alpha=0.8,label='Gain (GBM)')
ax.bar(x+0.2,[mda_imp[j] for j in mda_order],width=0.35,
       color=ORANGE,alpha=0.8,label='Permutation')
ax.set_xticks(x)
ax.set_xticklabels([FACTOR_NAMES[j] for j in mda_order],
                   rotation=45,ha='right',fontsize=7)
ax.set_ylabel('Importance')
ax.set_title('Feature Importance: Gain vs Permutation (GBM)')
ax.legend(fontsize=8); ax.grid(True,axis='y'); wm(ax)

# Summary table
ax=fig.add_subplot(gs3[0,1])
ax.set_facecolor(DARK); ax.axis('off')
lr_m=LogisticRegression(max_iter=1000,random_state=42).fit(Xtr_sc,ytr)
lr_acc_=accuracy_score(yte,lr_m.predict(Xte_sc))
lr_auc_=roc_auc_score(yte,lr_m.predict_proba(Xte_sc)[:,1])
rf_acc_=accuracy_score(yte,rf_.predict(Xte_sc))
rf_auc_=roc_auc_score(yte,rf_.predict_proba(Xte_sc)[:,1])
tdata=[
    ['Logistic Reg.','{:.3f}'.format(lr_acc_),'{:.3f}'.format(lr_auc_),'Linear','--'],
    ['Random Forest','{:.3f}'.format(rf_acc_),'{:.3f}'.format(rf_auc_),'Parallel','Bagging'],
    ['GBM (sklearn)','{:.3f}'.format(acc_gb),'{:.3f}'.format(auc_gb),'Sequential','Boosting'],
    ['HistGBM (fast)','{:.3f}'.format(acc_hgb),'{:.3f}'.format(auc_hgb),'Sequential','LightGBM-style'],
    ['GBM WF Mean','{:.3f}'.format(np.nanmean(oos_acc)),
     '{:.3f}'.format(np.nanmean(oos_auc)),'Sequential','OOS CV'],
]
cols=['Model','Accuracy','AUC-ROC','Type','Method']
tbl=ax.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax.set_title('Model Comparison Summary',color=WHITE)

p3=os.path.join(OUT_DIR,'m39_03_comparison_dashboard.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# ================================================================
# SUMMARY
# ================================================================
print()
print('='*65)
print('  MODULE 39 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] GBM: sequential trees fit to pseudo-residuals')
print('  [2] r_im = -dL/dF(x_i)  functional gradient descent')
print('  [3] Shrinkage: F_m = F_{m-1} + eta*gamma_m*h_m')
print('  [4] Early stopping: min test deviance selects M')
print('  [5] HistGBM: binned splits => 10-100x faster')
print('  [6] Gain importance: total reduction in loss at splits')
print('  [7] GBM beats RF at same M due to lower bias')
print()
print('  GBM: Acc={:.3f} AUC={:.3f}'.format(acc_gb,auc_gb))
print('  GBM WF: Acc={:.3f} AUC={:.3f} IC={:.4f} SR={:.2f}'.format(
      np.nanmean(oos_acc),np.nanmean(oos_auc),np.nanmean(oos_ic),ls_sr))
print('  Top Gain feature: {}'.format(FACTOR_NAMES[imp_order[0]]))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)