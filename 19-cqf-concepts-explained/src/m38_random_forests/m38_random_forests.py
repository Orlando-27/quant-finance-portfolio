#!/usr/bin/env python3
# MODULE 38: RANDOM FORESTS -- CLASSIFICATION & FEATURE IMPORTANCE
# CQF Concepts Explained | Group 7: Supervised ML (3/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI
#
# THEORETICAL SCOPE
# -----------------
# Random Forests (Breiman, 2001) are an ensemble of B decorrelated
# decision trees trained via bootstrap aggregation (bagging).
#
# Each tree t is built on bootstrap sample D_t from D.
# At each split, only m = sqrt(p) features are considered (feature
# subsampling), which decorrelates the trees.
#
# Prediction (regression): f(x) = (1/B) * sum_t h_t(x)
# Prediction (classification): majority vote over {h_t(x)}
#
# Bias-variance for bagging:
#   Var[f_bag] = rho * sigma^2 + (1-rho)/B * sigma^2
#   where rho = pairwise correlation between trees.
#   Feature subsampling reduces rho => lower ensemble variance.
#
# Feature Importance (MDI -- Mean Decrease in Impurity):
#   I(j) = sum_t sum_{node in t that splits on j} p(node)*DeltaImpurity(node)
#   Normalized so sum_j I(j) = 1.
#   Gini impurity for classification: G = 1 - sum_k p_k^2
#   MSE for regression: G = mean squared error
#
# Permutation Importance (MDA -- Mean Decrease in Accuracy):
#   Permute feature j in OOS data, measure accuracy drop.
#   More reliable than MDI: not biased toward high-cardinality features.
#
# Financial application:
#   Binary classification: predict sign(fwd_return_21d) = {Up, Down}
#   Features: 12 cross-sectional factor z-scores (momentum, value, vol)
#   Evaluation: OOS accuracy, AUC-ROC, IC of predicted probability vs return
#
# SECTIONS
# --------
# 01  Universe + 12 factor features (same as M37)
# 02  Binary target: sign of 1-month forward return
# 03  Single decision tree: depth analysis, overfitting illustration
# 04  Random Forest: OOB error vs n_estimators
# 05  MDI feature importance: Gini-based ranking
# 06  MDA permutation importance: OOS accuracy drop
# 07  Partial dependence: marginal effect of top features
# 08  Rolling walk-forward backtest: OOS accuracy & AUC
# 09  Prediction-based L/S portfolio (buy high P(up))
# 10  Hyperparameter sensitivity: max_depth, min_samples
# 11  Comparison: RF vs Logistic vs single tree
# 12  Summary dashboard
#
# REFERENCES
# ----------
# Breiman, L. (2001). Random Forests. Machine Learning 45(1), 5-32.
# Breiman, L. (1996). Bagging Predictors. Machine Learning 24(2), 123-140.
# Louppe et al. (2013). Understanding Variable Importances in RF. NeurIPS.
# Lopez de Prado (2018). Advances in Financial ML. Wiley.

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              roc_curve, confusion_matrix)
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
print('  MODULE 38: RANDOM FORESTS')
print('  Classification | Feature Importance | Walk-Forward Backtest')
print('='*65)
t0=time.perf_counter()

# ================================================================
# SECTION 01: UNIVERSE + FEATURES (consistent with M36/M37)
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
# SECTION 02: PANEL + BINARY TARGET
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
             'y_ret':y_al[tk],
             'y_bin':int(y_al[tk]>0)}
        for n in FACTOR_NAMES:
            row[n]=X_row.loc[tk,n]
        rows.append(row)

panel=pd.DataFrame(rows).set_index(['date','ticker'])
X_all=panel[FACTOR_NAMES].values.astype(float)
y_ret=panel['y_ret'].values.astype(float)
y_bin=panel['y_bin'].values.astype(int)
dates_panel=panel.index.get_level_values('date')
uniq_dates=sorted(dates_panel.unique())
print('  [02] Panel: {:,} obs  binary target Up={:.1f}%  Down={:.1f}%'.format(
      len(y_bin),y_bin.mean()*100,(1-y_bin.mean())*100))

# ================================================================
# SECTION 03: SINGLE TREE DEPTH ANALYSIS
# ================================================================
TRAIN_WIN=36
train_mask=dates_panel.isin(uniq_dates[:TRAIN_WIN])
test_mask=dates_panel.isin(uniq_dates[TRAIN_WIN:TRAIN_WIN+12])
Xtr,ytr=X_all[train_mask],y_bin[train_mask]
Xte,yte=X_all[test_mask],y_bin[test_mask]
scaler=StandardScaler().fit(Xtr)
Xtr_sc=scaler.transform(Xtr)
Xte_sc=scaler.transform(Xte)

depths=list(range(1,16))
tr_acc,te_acc=[],[]
for d in depths:
    dt=DecisionTreeClassifier(max_depth=d,random_state=42).fit(Xtr_sc,ytr)
    tr_acc.append(accuracy_score(ytr,dt.predict(Xtr_sc)))
    te_acc.append(accuracy_score(yte,dt.predict(Xte_sc)))
print('  [03] Single tree depth analysis: max OOS acc={:.3f} at depth {}'.format(
      max(te_acc),depths[np.argmax(te_acc)]))

# ================================================================
# SECTION 04: RANDOM FOREST -- OOB ERROR vs N_ESTIMATORS
# ================================================================
n_trees_list=[5,10,25,50,100,200]
oob_errors=[]
for n in n_trees_list:
    rf=RandomForestClassifier(n_estimators=n,max_features='sqrt',
                              oob_score=True,random_state=42,n_jobs=-1)
    rf.fit(Xtr_sc,ytr)
    oob_errors.append(1-rf.oob_score_)
print('  [04] OOB error vs n_trees computed')

# Final RF on full training set
rf_final=RandomForestClassifier(n_estimators=200,max_features='sqrt',
                                oob_score=True,random_state=42,n_jobs=-1)
rf_final.fit(Xtr_sc,ytr)
y_prob=rf_final.predict_proba(Xte_sc)[:,1]
y_pred=rf_final.predict(Xte_sc)
acc=accuracy_score(yte,y_pred)
auc=roc_auc_score(yte,y_prob)
print('  [04] RF(200 trees): OOS Acc={:.3f}  AUC={:.3f}'.format(acc,auc))

# ================================================================
# SECTION 05: MDI FEATURE IMPORTANCE
# ================================================================
mdi_imp=rf_final.feature_importances_
mdi_std=np.std([t.feature_importances_ for t in rf_final.estimators_],axis=0)
imp_order=np.argsort(mdi_imp)[::-1]
print('  [05] MDI top feature: {}'.format(FACTOR_NAMES[imp_order[0]]))

# ================================================================
# SECTION 06: MDA PERMUTATION IMPORTANCE
# ================================================================
perm=permutation_importance(rf_final,Xte_sc,yte,
                             n_repeats=10,random_state=42,n_jobs=-1)
mda_imp=perm.importances_mean
mda_std=perm.importances_std
mda_order=np.argsort(mda_imp)[::-1]
print('  [06] MDA top feature: {}'.format(FACTOR_NAMES[mda_order[0]]))

# ================================================================
# SECTION 07: ROLLING WALK-FORWARD BACKTEST
# ================================================================
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
    yret_te=y_ret[te_m]
    sc_=StandardScaler().fit(Xtr_)
    Xtr_s=sc_.transform(Xtr_); Xte_s=sc_.transform(Xte_)
    rf_=RandomForestClassifier(n_estimators=100,max_features='sqrt',
                               random_state=42,n_jobs=-1)
    rf_.fit(Xtr_s,ytr_)
    prob_=rf_.predict_proba(Xte_s)[:,1]
    pred_=rf_.predict(Xte_s)
    if len(np.unique(yte_))<2: continue
    oos_acc.append(accuracy_score(yte_,pred_))
    oos_auc.append(roc_auc_score(yte_,prob_))
    ic_,_=spearmanr(prob_,yret_te)
    oos_ic.append(ic_)
    q80=np.percentile(prob_,80); q20=np.percentile(prob_,20)
    long_r=yret_te[prob_>=q80].mean()
    short_r=yret_te[prob_<=q20].mean()
    ls_rets.append(long_r-short_r)
    ls_dates.append(te_d)

oos_acc=np.array(oos_acc)
oos_auc=np.array(oos_auc)
oos_ic=np.array(oos_ic)
ls_series=pd.Series(ls_rets,index=ls_dates)
ls_cum=(1+ls_series.fillna(0)).cumprod()
ls_ann=ls_cum.iloc[-1]**(12/max(len(ls_series),1))-1
ls_vol=ls_series.std()*np.sqrt(12)
ls_sr=ls_ann/(ls_vol+1e-9)
print('  [07] Walk-forward: Acc={:.3f}  AUC={:.3f}  IC={:.4f}  SR={:.2f}'.format(
      np.nanmean(oos_acc),np.nanmean(oos_auc),np.nanmean(oos_ic),ls_sr))

# ================================================================
# FIGURE 1: DEPTH ANALYSIS + OOB + IMPORTANCE
# ================================================================
print()
print('  Generating figures ...')
fig=plt.figure(figsize=(16,12),facecolor=DARK)
fig.suptitle('M38 -- Random Forests: Tree Diagnostics & Feature Importance',
             color=WHITE,fontsize=14,fontweight='bold',y=0.98)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

# (a) Single tree depth: train vs test accuracy
ax=fig.add_subplot(gs[0,0])
ax.plot(depths,tr_acc,color=GREEN,lw=2,marker='o',ms=4,label='Train accuracy')
ax.plot(depths,te_acc,color=ORANGE,lw=2,marker='s',ms=4,label='Test accuracy')
ax.axvline(depths[np.argmax(te_acc)],color=RED,lw=1,ls='--',
           label='Optimal depth={}'.format(depths[np.argmax(te_acc)]))
ax.set_xlabel('Tree depth'); ax.set_ylabel('Accuracy')
ax.set_title('Single Tree: Depth vs Accuracy\n(overfitting illustration)')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# (b) OOB error vs n_estimators
ax=fig.add_subplot(gs[0,1])
ax.plot(n_trees_list,oob_errors,color=BLUE,lw=2,marker='o',ms=5)
ax.fill_between(n_trees_list,[e*0.95 for e in oob_errors],
                [e*1.05 for e in oob_errors],color=BLUE,alpha=0.15)
ax.set_xlabel('Number of trees'); ax.set_ylabel('OOB Error')
ax.set_title('Random Forest: OOB Error vs n_estimators\n(convergence of ensemble)')
ax.grid(True); wm(ax)

# (c) ROC curve
ax=fig.add_subplot(gs[0,2])
fpr,tpr,_=roc_curve(yte,y_prob)
ax.plot(fpr,tpr,color=BLUE,lw=2,label='RF AUC={:.3f}'.format(auc))
ax.plot([0,1],[0,1],color=GRAY,lw=1,ls='--',label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (initial test set)')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# (d) MDI feature importance
ax=fig.add_subplot(gs[1,:2])
x=np.arange(len(FACTOR_NAMES))
ax.bar(x-0.2,[mdi_imp[j] for j in imp_order],width=0.35,
       color=GREEN,alpha=0.8,label='MDI (Gini)')
ax.bar(x+0.2,[mda_imp[j] for j in imp_order],width=0.35,
       color=ORANGE,alpha=0.8,label='MDA (Permutation)')
ax.errorbar(x-0.2,[mdi_imp[j] for j in imp_order],
            [mdi_std[j] for j in imp_order],fmt='none',color=WHITE,capsize=3,lw=1)
ax.errorbar(x+0.2,[mda_imp[j] for j in imp_order],
            [mda_std[j] for j in imp_order],fmt='none',color=WHITE,capsize=3,lw=1)
ax.set_xticks(x)
ax.set_xticklabels([FACTOR_NAMES[j] for j in imp_order],rotation=45,ha='right',fontsize=8)
ax.set_ylabel('Importance'); ax.set_title('Feature Importance: MDI (Gini) vs MDA (Permutation)')
ax.legend(fontsize=8); ax.grid(True,axis='y'); wm(ax)

# (e) Confusion matrix
ax=fig.add_subplot(gs[1,2])
cm=confusion_matrix(yte,y_pred)
im=ax.imshow(cm,cmap='Blues',aspect='auto')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred Down','Pred Up'])
ax.set_yticklabels(['True Down','True Up'])
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                color=WHITE,fontsize=14,fontweight='bold')
ax.set_title('Confusion Matrix\nAcc={:.3f}  AUC={:.3f}'.format(acc,auc))
wm(ax)

p1=os.path.join(OUT_DIR,'m38_01_tree_diagnostics.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# ================================================================
# FIGURE 2: WALK-FORWARD OOS PERFORMANCE
# ================================================================
fig,axes=plt.subplots(2,2,figsize=(16,10),facecolor=DARK)
fig.suptitle('M38 -- Walk-Forward Out-of-Sample Performance',
             color=WHITE,fontsize=14,fontweight='bold')

# OOS accuracy rolling
ax=axes[0,0]
roll_acc=pd.Series(oos_acc).rolling(6).mean()
ax.bar(range(len(oos_acc)),oos_acc,color=BLUE,alpha=0.4,width=0.8)
ax.plot(roll_acc.values,color=ORANGE,lw=2,label='6M rolling avg')
ax.axhline(0.5,color=RED,lw=1,ls='--',label='Random=0.5')
ax.axhline(np.nanmean(oos_acc),color=GREEN,lw=1.5,ls='-.',
           label='Mean={:.3f}'.format(np.nanmean(oos_acc)))
ax.set_xlabel('Test fold'); ax.set_ylabel('Accuracy')
ax.set_title('OOS Accuracy per Fold')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# OOS AUC rolling
ax=axes[0,1]
roll_auc=pd.Series(oos_auc).rolling(6).mean()
ax.bar(range(len(oos_auc)),oos_auc,color=GREEN,alpha=0.4,width=0.8)
ax.plot(roll_auc.values,color=YELLOW,lw=2,label='6M rolling avg')
ax.axhline(0.5,color=RED,lw=1,ls='--',label='Random=0.5')
ax.axhline(np.nanmean(oos_auc),color=CYAN,lw=1.5,ls='-.',
           label='Mean={:.3f}'.format(np.nanmean(oos_auc)))
ax.set_xlabel('Test fold'); ax.set_ylabel('AUC-ROC')
ax.set_title('OOS AUC-ROC per Fold')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# IC time series
ax=axes[1,0]
roll_ic=pd.Series(oos_ic).rolling(6).mean()
ax.bar(range(len(oos_ic)),oos_ic,
       color=[GREEN if v>0 else RED for v in oos_ic],alpha=0.5,width=0.8)
ax.plot(roll_ic.values,color=ORANGE,lw=2,label='6M rolling IC')
ax.axhline(0,color=GRAY,lw=0.8)
ax.axhline(np.nanmean(oos_ic),color=BLUE,lw=1.5,ls='--',
           label='Mean IC={:.4f}'.format(np.nanmean(oos_ic)))
ax.set_xlabel('Test fold'); ax.set_ylabel('IC (Spearman)')
ax.set_title('Predicted Probability vs Forward Return IC')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

# L/S cumulative PnL
ax=axes[1,1]
ax.plot(ls_cum.index,ls_cum.values,color=PURPLE,lw=2.5,label='RF L/S PnL')
rm=ls_cum.cummax(); dd=(ls_cum/rm-1)*100
ax2b=ax.twinx()
ax2b.fill_between(ls_cum.index,dd.values,0,where=dd<0,color=RED,alpha=0.20)
ax2b.set_ylabel('Drawdown (%)',color=RED)
ax2b.tick_params(axis='y',colors=RED)
ax.axhline(1,color=GRAY,lw=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative growth')
ax.set_title('RF L/S Portfolio | Ann={:.1f}%  SR={:.2f}'.format(ls_ann*100,ls_sr))
ax.legend(fontsize=8); ax.grid(True); wm(ax)

p2=os.path.join(OUT_DIR,'m38_02_walkforward_oos.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# ================================================================
# FIGURE 3: HYPERPARAMETER SENSITIVITY + SUMMARY TABLE
# ================================================================
fig=plt.figure(figsize=(16,10),facecolor=DARK)
fig.suptitle('M38 -- Hyperparameter Sensitivity & Model Comparison',
             color=WHITE,fontsize=14,fontweight='bold')
gs3=gridspec.GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.32)

# max_depth sensitivity
ax=fig.add_subplot(gs3[0,0])
depth_grid=[2,3,4,5,6,8,10,None]
d_acc,d_auc=[],[]
for dep in depth_grid:
    rf_=RandomForestClassifier(n_estimators=100,max_depth=dep,
                               max_features='sqrt',random_state=42,n_jobs=-1)
    rf_.fit(Xtr_sc,ytr)
    p_=rf_.predict_proba(Xte_sc)[:,1]
    d_acc.append(accuracy_score(yte,rf_.predict(Xte_sc)))
    d_auc.append(roc_auc_score(yte,p_))
xlabels=['2','3','4','5','6','8','10','None']
xpos=range(len(depth_grid))
ax.bar(xpos,d_auc,color=BLUE,alpha=0.7,width=0.5,label='AUC')
ax.set_xticks(xpos); ax.set_xticklabels(xlabels)
ax.set_xlabel('max_depth'); ax.set_ylabel('AUC-ROC')
ax.set_title('Hyperparameter: max_depth Effect')
ax.legend(fontsize=8); ax.grid(True,axis='y'); wm(ax)

# min_samples_leaf sensitivity
ax=fig.add_subplot(gs3[0,1])
leaf_grid=[1,2,5,10,20,50]
l_acc,l_auc=[],[]
for ml in leaf_grid:
    rf_=RandomForestClassifier(n_estimators=100,min_samples_leaf=ml,
                               max_features='sqrt',random_state=42,n_jobs=-1)
    rf_.fit(Xtr_sc,ytr)
    p_=rf_.predict_proba(Xte_sc)[:,1]
    l_acc.append(accuracy_score(yte,rf_.predict(Xte_sc)))
    l_auc.append(roc_auc_score(yte,p_))
ax.plot(leaf_grid,l_auc,color=GREEN,lw=2,marker='o',ms=5,label='AUC')
ax.plot(leaf_grid,l_acc,color=ORANGE,lw=2,marker='s',ms=5,label='Accuracy')
ax.set_xlabel('min_samples_leaf'); ax.set_ylabel('Score')
ax.set_title('Hyperparameter: min_samples_leaf Effect')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Model comparison
ax=fig.add_subplot(gs3[1,:])
ax.set_facecolor(DARK); ax.axis('off')
lr=LogisticRegression(max_iter=1000,random_state=42).fit(Xtr_sc,ytr)
lr_acc=accuracy_score(yte,lr.predict(Xte_sc))
lr_auc=roc_auc_score(yte,lr.predict_proba(Xte_sc)[:,1])
dt_best=DecisionTreeClassifier(max_depth=depths[np.argmax(te_acc)],random_state=42).fit(Xtr_sc,ytr)
dt_acc=accuracy_score(yte,dt_best.predict(Xte_sc))
dt_auc=roc_auc_score(yte,dt_best.predict_proba(Xte_sc)[:,1])

tdata=[
    ['Logistic Regression','{:.3f}'.format(lr_acc),'{:.3f}'.format(lr_auc),'--','L2','Linear'],
    ['Decision Tree','{:.3f}'.format(dt_acc),'{:.3f}'.format(dt_auc),
     '{}'.format(depths[np.argmax(te_acc)]),'None','Single'],
    ['Random Forest (200)','{:.3f}'.format(acc),'{:.3f}'.format(auc),
     'sqrt(p)','OOB','Ensemble'],
    ['RF Walk-Fwd Mean','{:.3f}'.format(np.nanmean(oos_acc)),
     '{:.3f}'.format(np.nanmean(oos_auc)),'sqrt(p)','Rolling','OOS'],
]
cols=['Model','Accuracy','AUC-ROC','max_features','Reg.','Type']
tbl=ax.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax.set_title('Model Comparison: Logistic vs Single Tree vs Random Forest',
             color=WHITE)

p3=os.path.join(OUT_DIR,'m38_03_comparison_dashboard.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# ================================================================
# SUMMARY
# ================================================================
print()
print('='*65)
print('  MODULE 38 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] Bagging: bootstrap + average => variance reduction')
print('  [2] Feature subsampling (sqrt(p)): decorrelates trees')
print('  [3] Var[ensemble] = rho*sigma^2 + (1-rho)/B*sigma^2')
print('  [4] MDI (Gini): fast but biased toward high-cardinality')
print('  [5] MDA (permutation): OOS accuracy drop, more reliable')
print('  [6] OOB score: free internal CV, no holdout needed')
print('  [7] Walk-forward CV: no look-ahead bias in backtest')
print()
print('  OOS Accuracy: {:.3f}  AUC: {:.3f}  IC: {:.4f}'.format(
      np.nanmean(oos_acc),np.nanmean(oos_auc),np.nanmean(oos_ic)))
print('  RF L/S Sharpe: {:.3f}  Ann.Ret: {:.1f}%'.format(ls_sr,ls_ann*100))
print('  Top MDI feature: {}  ({:.3f})'.format(
      FACTOR_NAMES[imp_order[0]],mdi_imp[imp_order[0]]))
print('  Top MDA feature: {}  ({:.4f})'.format(
      FACTOR_NAMES[mda_order[0]],mda_imp[mda_order[0]]))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)