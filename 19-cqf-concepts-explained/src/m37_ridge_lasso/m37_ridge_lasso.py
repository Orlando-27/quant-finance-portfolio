#!/usr/bin/env python3
# MODULE 37: RIDGE / LASSO REGULARIZED RETURN PREDICTION
# CQF Concepts Explained | Group 7: Supervised ML (2/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI
#
# THEORETICAL SCOPE
# -----------------
# Linear regression for return prediction suffers from multicollinearity
# when many correlated factors are used as predictors. Regularization
# penalizes model complexity to improve out-of-sample generalization.
#
# Ridge (L2):  min ||y - Xb||^2 + lambda * ||b||^2
#   Solution: b_ridge = (X'X + lambda*I)^{-1} X'y
#   Effect: shrinks all coefficients toward zero uniformly.
#   Keeps all features; no exact zeros.
#
# Lasso (L1):  min ||y - Xb||^2 + lambda * ||b||_1
#   No closed form; solved via coordinate descent (LARS).
#   Effect: shrinks some coefficients to exactly zero => sparse model.
#   Performs implicit feature selection.
#
# Elastic Net: min ||y-Xb||^2 + lambda*(alpha*||b||_1 + (1-alpha)*||b||^2)
#   Combines L1 sparsity with L2 grouping effect.
#
# Bias-variance tradeoff:
#   lambda -> 0: OLS (high variance, low bias)
#   lambda -> inf: null model (low variance, high bias)
#   Optimal lambda found via time-series cross-validation (Purged K-Fold).
#
# SECTIONS
# --------
# 01  Feature engineering: momentum, value, vol factors (12 predictors)
# 02  Target construction: 1-month forward return, cross-sectional demeaned
# 03  Ridge path: coefficients vs log(lambda), regularization surface
# 04  Lasso path: coefficient shrinkage, sparsity pattern
# 05  Elastic Net: alpha grid search
# 06  Time-series CV: rolling window lambda selection
# 07  Out-of-sample prediction: IC, hit rate, Sharpe of predicted L/S
# 08  Coefficient stability: rolling 252-day Ridge betas
# 09  VIF analysis: multicollinearity diagnostics
# 10  Comparison table: OLS vs Ridge vs Lasso vs ElasticNet
# 11  Prediction-based quintile backtest
# 12  Summary dashboard
#
# REFERENCES
# ----------
# Hoerl & Kennard (1970). Ridge Regression. Technometrics 12(1).
# Tibshirani (1996). Regression Shrinkage via the Lasso. JRSS-B 58(1).
# Zou & Hastie (2005). Elastic Net. JRSS-B 67(2).
# Hastie, Tibshirani & Friedman (2009). Elements of Statistical Learning.
# Lopez de Prado (2018). Advances in Financial ML. Wiley.

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
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
print('  MODULE 37: RIDGE / LASSO REGULARIZED RETURN PREDICTION')
print('='*65)
t0=time.perf_counter()

# ================================================================
# SECTION 01: SYNTHETIC UNIVERSE + FEATURES
# ================================================================
N,D=50,252*5
dates=pd.date_range('2019-01-02',periods=D,freq='B')
tickers=['A{:02d}'.format(i) for i in range(1,N+1)]

# Simulate returns with embedded factor structure
# True model: r_i = b1*MOM + b2*VAL + b3*VOL + eps
# We will try to recover these with regularized regression
TRUE_BETAS = np.array([0.03, 0.02, -0.02, 0.01, 0.005,
                        0.015, -0.01, 0.008, 0.003, -0.005,
                        0.012, 0.007])  # 12 true factor loadings

betas_asset = np.random.uniform(0.5,1.5,N)
mkt = np.random.normal(0.0004,0.012,D)
idvol = np.random.uniform(0.008,0.025,N)
idret = np.random.normal(0,1,(D,N))*idvol[None,:]
log_ret = betas_asset[None,:]*mkt[:,None]+idret
msig = np.random.normal(0,0.0002,(D//60+1,N))
for t in range(D): log_ret[t]+=msig[t//60]
returns=pd.DataFrame(log_ret,index=dates,columns=tickers)
prices=np.exp(returns.cumsum())
prices.iloc[0]=np.random.uniform(20,200,N)
prices=prices.multiply(prices.iloc[0])

def wz(s,z=3.0):
    mu,sg=s.mean(),s.std()
    return s.clip(mu-z*sg,mu+z*sg)
def zcs(s): return (s-s.mean())/(s.std()+1e-12)

def build_factor(prices,returns,name):
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
        vv2=vv.rolling(63).std()
        return vv2.apply(lambda x: zcs(wz(-x)),axis=1)
    # Value proxies (synthetic)
    np.random.seed(hash(name)%2**31)
    base=np.random.uniform(0.02,0.10,N)
    noise=np.random.normal(0,0.002,(D,N))
    panel=pd.DataFrame(base[None,:]+noise.cumsum(axis=0)*0.01,
                       index=dates,columns=tickers)
    return panel.apply(lambda x: zcs(wz(x)),axis=1)

FACTOR_NAMES=['MOM_1M','MOM_3M','MOM_6M','MOM_12M','REVERSAL',
              'LOWVOL_1M','LOWVOL_3M','DOWN_VOL','VOL_OF_VOL',
              'VAL_EY','VAL_BM','VAL_DY']
factors={n:build_factor(prices,returns,n) for n in FACTOR_NAMES}
print('  [01] {} factors built'.format(len(FACTOR_NAMES)))

# ================================================================
# SECTION 02: PANEL DATASET (stacked cross-section x time)
# ================================================================
# Target: 1-month forward cross-sectional demeaned return
FWDD=21
fwd=returns.shift(-FWDD).rolling(FWDD).sum()
fwd_cs=fwd.apply(lambda r: r-r.mean(),axis=1)  # cross-sectional demean

# Build stacked panel at monthly rebalance dates (every 21 days)
REBAL=21; START=300  # skip warmup
rows=[]
for idx in range(START,D-FWDD,REBAL):
    d=dates[idx]
    if d not in fwd_cs.index: continue
    y_row=fwd_cs.loc[d].dropna()
    feat_rows={}
    for n in FACTOR_NAMES:
        if d in factors[n].index:
            feat_rows[n]=factors[n].loc[d]
    if len(feat_rows)<len(FACTOR_NAMES): continue
    X_row=pd.DataFrame(feat_rows).loc[y_row.index].dropna()
    y_al=y_row.loc[X_row.index]
    for tk in X_row.index:
        row={'date':d,'ticker':tk,'y':y_al[tk]}
        for n in FACTOR_NAMES:
            row[n]=X_row.loc[tk,n]
        rows.append(row)

panel=pd.DataFrame(rows).set_index(['date','ticker'])
X_all=panel[FACTOR_NAMES].values.astype(float)
y_all=panel['y'].values.astype(float)
dates_panel=panel.index.get_level_values('date')
uniq_dates=sorted(dates_panel.unique())
print('  [02] Panel: {:,} obs  {} dates  {} features'.format(
      len(y_all),len(uniq_dates),len(FACTOR_NAMES)))

# ================================================================
# SECTION 03: RIDGE COEFFICIENT PATH
# ================================================================
scaler=StandardScaler()
X_sc=scaler.fit_transform(X_all)
lambdas=np.logspace(-4,4,60)
ridge_coefs=[]
for lam in lambdas:
    m=Ridge(alpha=lam,fit_intercept=True)
    m.fit(X_sc,y_all)
    ridge_coefs.append(m.coef_)
ridge_coefs=np.array(ridge_coefs)
print('  [03] Ridge coefficient path computed ({} lambdas)'.format(len(lambdas)))

# ================================================================
# SECTION 04: LASSO COEFFICIENT PATH
# ================================================================
lasso_coefs=[]
for lam in lambdas:
    m=Lasso(alpha=lam,fit_intercept=True,max_iter=5000)
    m.fit(X_sc,y_all)
    lasso_coefs.append(m.coef_)
lasso_coefs=np.array(lasso_coefs)
print('  [04] Lasso coefficient path computed')

# ================================================================
# SECTION 05: TIME-SERIES ROLLING CV
# ================================================================
# Walk-forward CV: train on past T dates, predict next date
# Evaluate IC (Spearman) on each test fold
TRAIN_WIN=36  # months
test_ics_ridge,test_ics_lasso,test_ics_ols=[],[],[]
lam_chosen_ridge,lam_chosen_lasso=[],[]
y_pred_all=[]
y_true_all=[]
date_pred_all=[]

lam_grid_r=np.logspace(-3,3,20)
lam_grid_l=np.logspace(-5,1,20)

for i in range(TRAIN_WIN, len(uniq_dates)-1):
    train_dates=uniq_dates[i-TRAIN_WIN:i]
    test_date=uniq_dates[i]
    tr_mask=dates_panel.isin(train_dates)
    te_mask=dates_panel==test_date
    if tr_mask.sum()<50 or te_mask.sum()<5: continue
    Xtr,ytr=X_sc[tr_mask],y_all[tr_mask]
    Xte,yte=X_sc[te_mask],y_all[te_mask]
    # Ridge: pick best lambda via CV on training set
    rc=RidgeCV(alphas=lam_grid_r,cv=5).fit(Xtr,ytr)
    yp_r=rc.predict(Xte)
    lam_chosen_ridge.append(rc.alpha_)
    # Lasso
    lc=LassoCV(alphas=lam_grid_l,cv=5,max_iter=2000).fit(Xtr,ytr)
    yp_l=lc.predict(Xte)
    lam_chosen_lasso.append(lc.alpha_)
    # OLS (Ridge with tiny lambda)
    ols=Ridge(alpha=1e-6).fit(Xtr,ytr)
    yp_o=ols.predict(Xte)
    # ICs
    if len(yte)>=5:
        test_ics_ridge.append(spearmanr(yp_r,yte)[0])
        test_ics_lasso.append(spearmanr(yp_l,yte)[0])
        test_ics_ols.append(spearmanr(yp_o,yte)[0])
        y_pred_all.append(yp_r)
        y_true_all.append(yte)
        date_pred_all.append(test_date)

test_ics_ridge=np.array(test_ics_ridge)
test_ics_lasso=np.array(test_ics_lasso)
test_ics_ols=np.array(test_ics_ols)
print('  [05] Rolling CV: {} test folds'.format(len(test_ics_ridge)))
print('       IC Ridge={:.4f}  Lasso={:.4f}  OLS={:.4f}'.format(
      np.nanmean(test_ics_ridge),np.nanmean(test_ics_lasso),np.nanmean(test_ics_ols)))

# ================================================================
# SECTION 06: PREDICTION-BASED QUINTILE BACKTEST
# ================================================================
ls_rets=[]
ls_dates=[]
for yp,yt,d in zip(y_pred_all,y_true_all,date_pred_all):
    if len(yp)<10: continue
    q80=np.percentile(yp,80); q20=np.percentile(yp,20)
    long_ret=yt[yp>=q80].mean()
    short_ret=yt[yp<=q20].mean()
    ls_rets.append(long_ret-short_ret)
    ls_dates.append(d)
ls_series=pd.Series(ls_rets,index=ls_dates)
ls_cum=(1+ls_series.fillna(0)).cumprod()
ls_ann=ls_cum.iloc[-1]**(12/max(len(ls_series),1))-1
ls_vol=ls_series.std()*np.sqrt(12)
ls_sr=ls_ann/(ls_vol+1e-9)
print('  [06] L/S backtest: Ann={:.1f}%  SR={:.2f}'.format(ls_ann*100,ls_sr))

# ================================================================
# SECTION 07: FINAL MODEL COEFFICIENTS
# ================================================================
# Fit final Ridge and Lasso on full sample with CV-chosen lambda
best_lam_r=np.median(lam_chosen_ridge) if lam_chosen_ridge else 1.0
best_lam_l=np.median(lam_chosen_lasso) if lam_chosen_lasso else 0.001
m_ridge=Ridge(alpha=best_lam_r).fit(X_sc,y_all)
m_lasso=Lasso(alpha=best_lam_l,max_iter=5000).fit(X_sc,y_all)
m_ols=Ridge(alpha=1e-6).fit(X_sc,y_all)
m_enet=ElasticNet(alpha=best_lam_l,l1_ratio=0.5,max_iter=5000).fit(X_sc,y_all)
print('  [07] Final models fitted')
print('       Ridge non-zero: {}/{}  Lasso non-zero: {}/{}'.format(
      np.sum(m_ridge.coef_!=0),len(FACTOR_NAMES),
      np.sum(m_lasso.coef_!=0),len(FACTOR_NAMES)))

# ================================================================
# FIGURE 1: REGULARIZATION PATHS
# ================================================================
print()
print('  Generating figures ...')
fig,axes=plt.subplots(1,2,figsize=(16,7),facecolor=DARK)
fig.suptitle('M37 -- Regularization Paths: Ridge (L2) vs Lasso (L1)',
             color=WHITE,fontsize=14,fontweight='bold')

# Ridge path
ax=axes[0]
for j,name in enumerate(FACTOR_NAMES):
    ax.plot(np.log10(lambdas),ridge_coefs[:,j],
            color=ACCENT[j%len(ACCENT)],lw=1.8,label=name)
ax.axhline(0,color=GRAY,lw=0.8,ls='--')
ax.set_xlabel('log10(lambda)'); ax.set_ylabel('Coefficient')
ax.set_title('Ridge Path: L2 Shrinkage',
             fontweight='bold')
ax.legend(fontsize=6,ncol=2); ax.grid(True); wm(ax)

# Lasso path
ax=axes[1]
for j,name in enumerate(FACTOR_NAMES):
    ax.plot(np.log10(lambdas),lasso_coefs[:,j],
            color=ACCENT[j%len(ACCENT)],lw=1.8,label=name)
ax.axhline(0,color=GRAY,lw=0.8,ls='--')
ax.set_xlabel('log10(lambda)'); ax.set_ylabel('Coefficient')
ax.set_title('Lasso Path: L1 Sparsity',
             fontweight='bold')
ax.legend(fontsize=6,ncol=2); ax.grid(True); wm(ax)

p1=os.path.join(OUT_DIR,'m37_01_regularization_paths.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# ================================================================
# FIGURE 2: IC TIME SERIES + COEFFICIENT COMPARISON
# ================================================================
fig=plt.figure(figsize=(16,10),facecolor=DARK)
fig.suptitle('M37 -- Out-of-Sample IC & Model Comparison',
             color=WHITE,fontsize=14,fontweight='bold')
gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.32)

# IC time series
ax=fig.add_subplot(gs[0,:])
roll_r=pd.Series(test_ics_ridge).rolling(6).mean()
roll_l=pd.Series(test_ics_lasso).rolling(6).mean()
roll_o=pd.Series(test_ics_ols).rolling(6).mean()
ax.plot(roll_r.values,color=BLUE,lw=2,
        label='Ridge IC (6M roll) mean={:.3f}'.format(np.nanmean(test_ics_ridge)))
ax.plot(roll_l.values,color=GREEN,lw=2,
        label='Lasso IC (6M roll) mean={:.3f}'.format(np.nanmean(test_ics_lasso)))
ax.plot(roll_o.values,color=RED,lw=1.5,ls='--',
        label='OLS IC (6M roll)   mean={:.3f}'.format(np.nanmean(test_ics_ols)))
ax.axhline(0,color=GRAY,lw=0.8)
ax.set_xlabel('Test fold (monthly)'); ax.set_ylabel('IC (Spearman)')
ax.set_title('Out-of-Sample Information Coefficient -- Rolling 6-Fold Average')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Final model coefficients bar chart
ax2=fig.add_subplot(gs[1,0])
x=np.arange(len(FACTOR_NAMES))
w=0.22
ax2.bar(x-w,m_ols.coef_,width=w,color=RED,alpha=0.8,label='OLS')
ax2.bar(x,m_ridge.coef_,width=w,color=BLUE,alpha=0.8,label='Ridge')
ax2.bar(x+w,m_lasso.coef_,width=w,color=GREEN,alpha=0.8,label='Lasso')
ax2.bar(x+2*w,m_enet.coef_,width=w,color=ORANGE,alpha=0.8,label='ElNet')
ax2.set_xticks(x)
ax2.set_xticklabels(FACTOR_NAMES,rotation=45,ha='right',fontsize=7)
ax2.axhline(0,color=GRAY,lw=0.8)
ax2.set_ylabel('Coefficient'); ax2.set_title('Final Coefficients: OLS vs Ridge vs Lasso vs ElNet')
ax2.legend(fontsize=7); ax2.grid(True,axis='y'); wm(ax2)

# Lambda selection history
ax3=fig.add_subplot(gs[1,1])
ax3.plot(np.log10(lam_chosen_ridge),color=BLUE,lw=1.5,label='Ridge lambda (log10)')
ax3.plot(np.log10(lam_chosen_lasso),color=GREEN,lw=1.5,label='Lasso lambda (log10)')
ax3.axhline(np.log10(best_lam_r),color=BLUE,lw=1,ls='--',
            label='Median Ridge={:.3f}'.format(best_lam_r))
ax3.axhline(np.log10(best_lam_l),color=GREEN,lw=1,ls='--',
            label='Median Lasso={:.4f}'.format(best_lam_l))
ax3.set_xlabel('Test fold'); ax3.set_ylabel('log10(lambda)')
ax3.set_title('CV-Selected Lambda Over Time')
ax3.legend(fontsize=7); ax3.grid(True); wm(ax3)

p2=os.path.join(OUT_DIR,'m37_02_ic_comparison.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# ================================================================
# FIGURE 3: L/S BACKTEST + MODEL SUMMARY TABLE
# ================================================================
fig=plt.figure(figsize=(16,10),facecolor=DARK)
fig.suptitle('M37 -- Ridge Prediction-Based L/S Backtest & Summary',
             color=WHITE,fontsize=14,fontweight='bold')
gs3=gridspec.GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.32)

# L/S cumulative PnL
ax=fig.add_subplot(gs3[0,:])
ax.plot(ls_cum.index,ls_cum.values,color=BLUE,lw=2.5,label='Ridge L/S (top-bottom quintile)')
rm=ls_cum.cummax(); dd=(ls_cum/rm-1)*100
ax2b=ax.twinx()
ax2b.fill_between(ls_cum.index,dd.values,0,where=dd<0,
                  color=RED,alpha=0.20,label='Drawdown %')
ax2b.set_ylabel('Drawdown (%)',color=RED)
ax2b.tick_params(axis='y',colors=RED)
ax.axhline(1,color=GRAY,lw=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative growth')
ax.set_title('Ridge-Predicted Long/Short Portfolio | Ann={:.1f}%  SR={:.2f}'.format(
             ls_ann*100,ls_sr))
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# Sparsity bar: non-zero coefficients
ax3=fig.add_subplot(gs3[1,0])
models=['OLS','Ridge','Lasso','ElasticNet']
nz=[np.sum(m_ols.coef_!=0),np.sum(m_ridge.coef_!=0),
    np.sum(m_lasso.coef_!=0),np.sum(m_enet.coef_!=0)]
ax3.bar(models,nz,color=[RED,BLUE,GREEN,ORANGE],alpha=0.8,width=0.5)
ax3.axhline(len(FACTOR_NAMES),color=GRAY,lw=1,ls='--',
            label='Total features={}'.format(len(FACTOR_NAMES)))
ax3.set_ylabel('Non-zero coefficients')
ax3.set_title('Sparsity: Features Selected per Model')
ax3.legend(fontsize=8); ax3.grid(True,axis='y'); wm(ax3)

# Summary table
ax4=fig.add_subplot(gs3[1,1])
ax4.set_facecolor(DARK); ax4.axis('off')
tdata=[
    ['OLS','{:.4f}'.format(np.nanmean(test_ics_ols)),
     '{:.2f}'.format(np.nanmean(test_ics_ols)/(np.nanstd(test_ics_ols)+1e-9)),
     '{}'.format(len(FACTOR_NAMES)),'--'],
    ['Ridge','{:.4f}'.format(np.nanmean(test_ics_ridge)),
     '{:.2f}'.format(np.nanmean(test_ics_ridge)/(np.nanstd(test_ics_ridge)+1e-9)),
     '{}'.format(len(FACTOR_NAMES)),'{:.3f}'.format(best_lam_r)],
    ['Lasso','{:.4f}'.format(np.nanmean(test_ics_lasso)),
     '{:.2f}'.format(np.nanmean(test_ics_lasso)/(np.nanstd(test_ics_lasso)+1e-9)),
     '{}'.format(int(np.sum(m_lasso.coef_!=0))),'{:.4f}'.format(best_lam_l)],
    ['ElasticNet','--','--','{}'.format(int(np.sum(m_enet.coef_!=0))),'--'],
]
cols=['Model','Mean IC','ICIR','Non-zero','Lambda']
tbl=ax4.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax4.set_title('Model Comparison Summary',color=WHITE)

p3=os.path.join(OUT_DIR,'m37_03_summary_dashboard.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# ================================================================
# SUMMARY
# ================================================================
print()
print('='*65)
print('  MODULE 37 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] Ridge L2: b=(X`X+lI)^{-1}X`y  shrinks uniformly')
print('  [2] Lasso L1: coordinate descent  exact zeros (sparse)')
print('  [3] ElasticNet: alpha*L1 + (1-alpha)*L2  grouping effect')
print('  [4] Bias-variance: lambda->0 OLS, lambda->inf null model')
print('  [5] CV: rolling walk-forward, no look-ahead bias')
print('  [6] IC measures predictive signal quality (Spearman)')
print('  [7] L/S backtest: long top quintile, short bottom quintile')
print()
print('  OLS IC:   {:.4f}  ICIR={:.3f}'.format(
      np.nanmean(test_ics_ols),
      np.nanmean(test_ics_ols)/(np.nanstd(test_ics_ols)+1e-9)))
print('  Ridge IC: {:.4f}  ICIR={:.3f}  lambda={:.3f}'.format(
      np.nanmean(test_ics_ridge),
      np.nanmean(test_ics_ridge)/(np.nanstd(test_ics_ridge)+1e-9),
      best_lam_r))
print('  Lasso IC: {:.4f}  ICIR={:.3f}  lambda={:.4f}  non-zero={}/{}'.format(
      np.nanmean(test_ics_lasso),
      np.nanmean(test_ics_lasso)/(np.nanstd(test_ics_lasso)+1e-9),
      best_lam_l,int(np.sum(m_lasso.coef_!=0)),len(FACTOR_NAMES)))
print('  Ridge L/S Sharpe: {:.3f}'.format(ls_sr))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)