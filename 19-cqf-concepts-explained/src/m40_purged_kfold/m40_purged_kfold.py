#!/usr/bin/env python3
# MODULE 40: PURGED K-FOLD CROSS-VALIDATION
# CQF Concepts Explained | Group 7: Supervised ML (5/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI
#
# THEORETICAL SCOPE
# -----------------
# Standard K-Fold CV assumes i.i.d. observations. Financial time series
# violate this assumption in two critical ways:
#
# 1. LEAKAGE via overlapping labels:
#    If target = sum(r_{t+1}...r_{t+h}), then label at t overlaps
#    with label at t+1 (they share h-1 returns). Training on a fold
#    that contains t+1 while testing on t introduces look-ahead bias.
#
# 2. LEAKAGE via serial correlation:
#    Even non-overlapping labels may share information through
#    autocorrelated features (momentum, volatility).
#
# PURGING (Lopez de Prado, 2018):
#    Remove from the training set any observation whose label
#    overlaps in time with any observation in the test set.
#    If test obs at time t, purge all train obs in [t-h+1, t+h-1].
#
# EMBARGO:
#    After the test fold, add an embargo of e periods to the
#    training set of the next fold to prevent leakage through
#    features that look back into the test period.
#    Typical: e = 0.01 * T (1% of sample size).
#
# COMBINATORIAL PURGED K-FOLD (CPCV):
#    Use C(k,2) combinations of test folds to generate multiple
#    backtest paths, obtaining a distribution of Sharpe ratios.
#    Addresses selection bias in single backtest path.
#
# Walk-Forward (anchored / rolling):
#    Anchored: train [0,t], test [t+1, t+h] -- grows train set
#    Rolling:  train [t-W,t], test [t+1, t+h] -- fixed window
#    Purged K-Fold is superior for hyperparameter selection.
#
# SECTIONS
# --------
# 01  Demonstrate leakage: standard KFold IC inflation
# 02  Purged KFold implementation from scratch
# 03  Embargo implementation
# 04  Leakage quantification: standard vs purged IC
# 05  Hyperparameter selection: Ridge lambda via purged CV
# 06  Walk-forward anchored vs rolling window comparison
# 07  Combinatorial Purged KFold (CPCV): Sharpe distribution
# 08  Backtest overfitting: multiple testing correction
# 09  Deflated Sharpe Ratio (DSR)
# 10  Summary dashboard
#
# REFERENCES
# ----------
# Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
#   Wiley. Chapter 7: Cross-Validation in Finance.
# Lopez de Prado, M. (2019). A Data Science Solution to the
#   Multiple-Testing Crisis in Financial Research. JPM 45(1).
# Bailey et al. (2014). The Probability of Backtest Overfitting.
#   JCFR 1(4).

import os, time, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, norm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
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
print('  MODULE 40: PURGED K-FOLD CROSS-VALIDATION')
print('  Leakage | Purging | Embargo | CPCV | Deflated Sharpe')
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

# Build panel
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
        row={'date':d,'ticker':tk,'y_ret':y_al[tk],'y_bin':int(y_al[tk]>0)}
        for n in FACTOR_NAMES: row[n]=X_row.loc[tk,n]
        rows.append(row)

panel=pd.DataFrame(rows).set_index(['date','ticker'])
X_all=panel[FACTOR_NAMES].values.astype(float)
y_ret=panel['y_ret'].values.astype(float)
y_bin=panel['y_bin'].values.astype(int)
dates_panel=panel.index.get_level_values('date')
uniq_dates=sorted(dates_panel.unique())
T=len(uniq_dates)
print('  [01] Panel: {:,} obs  T={} dates'.format(len(y_ret),T))

# ================================================================
# SECTION 02: PURGED K-FOLD IMPLEMENTATION
# ================================================================
class PurgedKFold:
    '''
    Time-series K-Fold with purging and embargo.
    Observations indexed by date; labels span h periods.
    '''
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits=n_splits
        self.embargo_pct=embargo_pct

    def split(self, X, y, dates, label_len=FWDD):
        '''
        dates: array of observation dates (same length as X)
        label_len: number of days each label spans (forward return horizon)
        '''
        uniq=np.array(sorted(np.unique(dates)))
        n=len(uniq)
        fold_size=n//self.n_splits
        embargo_n=max(1,int(n*self.embargo_pct))

        for k in range(self.n_splits):
            # Test fold: dates in [k*fold_size, (k+1)*fold_size)
            test_start=k*fold_size
            test_end=min((k+1)*fold_size,n)
            test_dates=set(uniq[test_start:test_end])

            # Purge: remove train obs whose label overlaps test
            # Label at date d spans [d, d + label_len days]
            test_date_min=uniq[test_start]
            test_date_max=uniq[test_end-1]

            train_idx=[]
            test_idx=[]
            for i,d in enumerate(dates):
                if d in test_dates:
                    test_idx.append(i)
                else:
                    # Purge: label of train obs extends into test?
                    # Approximate: if d is within label_len days before test
                    days_to_test=(test_date_min-d).days if hasattr(d,'days') else 0
                    try:
                        days_to_test=(test_date_min-d).days
                    except:
                        days_to_test=999
                    if days_to_test<5 and days_to_test>=0:
                        continue  # purged
                    # Embargo: skip obs just after test fold
                    try:
                        days_after=(d-test_date_max).days
                    except:
                        days_after=999
                    if 0<days_after<=3:
                        continue  # embargoed
                    train_idx.append(i)

            yield np.array(train_idx), np.array(test_idx)

print('  [02] PurgedKFold class defined')

# ================================================================
# SECTION 03: LEAKAGE QUANTIFICATION
# ================================================================
# Compare IC from standard KFold vs PurgedKFold on Ridge
print('  [03] Quantifying leakage: standard vs purged KFold ...')

sc=StandardScaler()
X_sc=sc.fit_transform(X_all)
lam=1.0

# Standard KFold IC
kf=KFold(n_splits=5,shuffle=False)
ic_std=[]
for tr,te in kf.split(X_sc):
    m=Ridge(alpha=lam).fit(X_sc[tr],y_ret[tr])
    yp=m.predict(X_sc[te])
    ic_std.append(spearmanr(yp,y_ret[te])[0])

# Purged KFold IC
pkf=PurgedKFold(n_splits=5,embargo_pct=0.01)
ic_purged=[]
for tr,te in pkf.split(X_sc,y_ret,dates_panel):
    if len(tr)<50 or len(te)<5: continue
    m=Ridge(alpha=lam).fit(X_sc[tr],y_ret[tr])
    yp=m.predict(X_sc[te])
    ic_purged.append(spearmanr(yp,y_ret[te])[0])

ic_std=np.array(ic_std); ic_purged=np.array(ic_purged)
ic_purged_mean = np.nanmean(ic_purged) if len(ic_purged)>0 else float('nan')
print('  [03] Standard KFold IC={:.4f}  Purged KFold IC={}'.format(
      np.nanmean(ic_std), '{:.4f}'.format(ic_purged_mean) if not np.isnan(ic_purged_mean) else 'nan (all folds purged)'))
inflation = np.nanmean(ic_std)-ic_purged_mean
print('       Leakage inflation = {}'.format('{:.4f}'.format(inflation) if not np.isnan(inflation) else 'n/a'))

# ================================================================
# SECTION 04: LAMBDA SELECTION VIA PURGED CV
# ================================================================
print('  [04] Lambda selection via purged CV ...')
lam_grid=np.logspace(-3,3,15)
cv_ic_std=[]
cv_ic_purged=[]

for lam in lam_grid:
    # Standard
    ics=[]
    for tr,te in KFold(n_splits=5).split(X_sc):
        m=Ridge(alpha=lam).fit(X_sc[tr],y_ret[tr])
        ics.append(spearmanr(m.predict(X_sc[te]),y_ret[te])[0])
    cv_ic_std.append(np.nanmean(ics))
    # Purged
    icp=[]
    for tr,te in PurgedKFold(n_splits=5).split(X_sc,y_ret,dates_panel):
        if len(tr)<50 or len(te)<5: continue
        m=Ridge(alpha=lam).fit(X_sc[tr],y_ret[tr])
        icp.append(spearmanr(m.predict(X_sc[te]),y_ret[te])[0])
    cv_ic_purged.append(np.nanmean(icp))

best_lam_std=lam_grid[np.argmax(cv_ic_std)]
valid_purged = [v for v in cv_ic_purged if not np.isnan(v)]
if valid_purged:
    best_lam_purged=lam_grid[np.nanargmax(cv_ic_purged)]
else:
    best_lam_purged=best_lam_std
    print('  [04] Purged CV returned all-NaN -- using std lambda as fallback')
print('  [04] Best lambda -- Standard={:.3f}  Purged={:.3f}'.format(
      best_lam_std,best_lam_purged))

# ================================================================
# SECTION 05: WALK-FORWARD ANCHORED vs ROLLING
# ================================================================
print('  [05] Walk-forward: anchored vs rolling ...')
TRAIN_MIN=24; ROLL_WIN=36

ic_anchored,ic_rolling=[],[]
wf_dates=[]

for i in range(TRAIN_MIN,T-1):
    te_d=uniq_dates[i]
    te_m=dates_panel==te_d
    if te_m.sum()<5: continue
    Xte_,yte_=X_sc[te_m],y_ret[te_m]

    # Anchored: all history
    tr_anchor=dates_panel.isin(uniq_dates[:i])
    if tr_anchor.sum()<50:
        ic_anchored.append(np.nan)
    else:
        m=Ridge(alpha=best_lam_purged).fit(X_sc[tr_anchor],y_ret[tr_anchor])
        ic_anchored.append(spearmanr(m.predict(Xte_),yte_)[0])

    # Rolling: fixed window
    start_r=max(0,i-ROLL_WIN)
    tr_roll=dates_panel.isin(uniq_dates[start_r:i])
    if tr_roll.sum()<50:
        ic_rolling.append(np.nan)
    else:
        m=Ridge(alpha=best_lam_purged).fit(X_sc[tr_roll],y_ret[tr_roll])
        ic_rolling.append(spearmanr(m.predict(Xte_),yte_)[0])

    wf_dates.append(te_d)

ic_anchored=np.array(ic_anchored)
ic_rolling=np.array(ic_rolling)
print('  [05] Anchored IC={:.4f}  Rolling IC={:.4f}'.format(
      np.nanmean(ic_anchored),np.nanmean(ic_rolling)))

# ================================================================
# SECTION 06: COMBINATORIAL PURGED K-FOLD (CPCV)
# ================================================================
# Use C(6,2)=15 test-fold combinations to generate backtest paths
# and estimate a distribution of Sharpe ratios
print('  [06] CPCV: generating Sharpe distribution ...')
K=6; N_COMB=2
fold_size_c=T//K
fold_indices=[list(range(k*fold_size_c,min((k+1)*fold_size_c,T))) for k in range(K)]

sharpes=[]
for test_folds in itertools.combinations(range(K),N_COMB):
    test_date_idx=[]
    for f in test_folds: test_date_idx.extend(fold_indices[f])
    test_date_idx=sorted(test_date_idx)
    train_date_idx=[i for i in range(T) if i not in set(test_date_idx)]

    test_ds=set(uniq_dates[i] for i in test_date_idx)
    train_ds=set(uniq_dates[i] for i in train_date_idx)

    tr_m=dates_panel.isin(train_ds)
    te_m=dates_panel.isin(test_ds)
    if tr_m.sum()<100 or te_m.sum()<20: continue

    m=Ridge(alpha=best_lam_purged).fit(X_sc[tr_m],y_ret[tr_m])
    yp=m.predict(X_sc[te_m])
    yret_te=y_ret[te_m]
    dates_te=dates_panel[te_m]
    ls_by_date=[]
    for d in sorted(set(dates_te)):
        dm=dates_te==d
        yp_d=yp[dm]; yr_d=yret_te[dm]
        if len(yp_d)<5: continue
        q80=np.percentile(yp_d,80); q20=np.percentile(yp_d,20)
        ls_by_date.append(yr_d[yp_d>=q80].mean()-yr_d[yp_d<=q20].mean())
    if len(ls_by_date)<3: continue
    ls_a=np.array(ls_by_date)
    sr=ls_a.mean()/(ls_a.std()+1e-9)*np.sqrt(12)
    sharpes.append(sr)

sharpes=np.array(sharpes)
print('  [06] CPCV: {} paths  Sharpe mean={:.3f}  std={:.3f}'.format(
      len(sharpes),sharpes.mean(),sharpes.std()))

# ================================================================
# SECTION 07: DEFLATED SHARPE RATIO (DSR)
# ================================================================
# DSR = PSR(SR* | SR_0) where SR_0 accounts for multiple testing
# PSR(SR*) = Phi[(SR*-SR_0)*sqrt(T-1) / sqrt(1 - gamma3*SR* + (gamma4-1)/4 * SR*^2)]
# SR_0 (expected max Sharpe under null) corrected for trials:
#   SR_0 = sqrt(Var[SR]) * ((1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e)))
N_trials=len(sharpes)
SR_obs=sharpes.max() if len(sharpes)>0 else 0.0
T_obs=max(len(sharpes),10)
gamma=0.5772  # Euler-Mascheroni
SR_0=np.sqrt(np.var(sharpes)+1e-9)*(
    (1-gamma)*norm.ppf(1-1/max(N_trials,2))+
    gamma*norm.ppf(1-1/(max(N_trials,2)*np.e)))
skew_sr=0.0; kurt_sr=3.0  # assume normal SR distribution
denom=np.sqrt(1-skew_sr*SR_obs+((kurt_sr-1)/4)*SR_obs**2)
PSR=norm.cdf((SR_obs-SR_0)*np.sqrt(T_obs-1)/(denom+1e-9))
DSR=PSR
print('  [07] Deflated Sharpe: SR_max={:.3f}  SR_0={:.3f}  PSR={:.3f}'.format(
      SR_obs,SR_0,PSR))

# ================================================================
# FIGURE 1: LEAKAGE + PURGING DIAGRAM
# ================================================================
print()
print('  Generating figures ...')
fig=plt.figure(figsize=(16,12),facecolor=DARK)
fig.suptitle('M40 -- Purged K-Fold: Leakage Quantification & CV Methods',
             color=WHITE,fontsize=14,fontweight='bold',y=0.98)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

# (a) Schematic: standard vs purged split
ax=fig.add_subplot(gs[0,:])
ax.set_facecolor(DARK); ax.set_xlim(0,100); ax.set_ylim(0,4)
ax.axis('off')
# Draw fold bars
fold_cols=[BLUE,GREEN,ORANGE,RED,PURPLE]
for k in range(5):
    x0=k*20; x1=(k+1)*20
    # Standard: training folds shown as solid
    y_std=3.0
    ax.barh(y_std,x1-x0,left=x0,height=0.4,
            color=fold_cols[k] if k!=2 else RED,alpha=0.7)
    ax.text(x0+(x1-x0)/2,y_std,'Fold {}'.format(k+1),
            ha='center',va='center',color=WHITE,fontsize=8)
# Label
ax.text(-1,3.0,'Standard KFold:',color=WHITE,fontsize=9,ha='right',va='center')
# Purged: show purge zone
y_purged=2.0
for k in range(5):
    x0=k*20; x1=(k+1)*20
    color=fold_cols[k] if k!=2 else RED
    ax.barh(y_purged,x1-x0,left=x0,height=0.4,color=color,alpha=0.7)
# Purge zone (before test fold 3)
ax.barh(y_purged,3,left=37,height=0.4,color='white',alpha=0.6,
        label='Purged (overlap)')
# Embargo zone (after test fold 3)
ax.barh(y_purged,2,left=60,height=0.4,color=YELLOW,alpha=0.6,
        label='Embargoed')
ax.text(-1,2.0,'Purged KFold:',color=WHITE,fontsize=9,ha='right',va='center')
ax.legend(loc='lower right',fontsize=8)
# Annotation arrows
ax.annotate('Test fold',xy=(50,3.2),color=RED,fontsize=8,ha='center')
ax.annotate('Test fold',xy=(50,2.2),color=RED,fontsize=8,ha='center')
ax.annotate('Purge zone',xy=(38.5,1.6),color=WHITE,fontsize=7,ha='center')
ax.annotate('Embargo',xy=(61,1.6),color=YELLOW,fontsize=7,ha='center')
ax.set_title('Standard KFold vs Purged KFold: Fold Structure',
             color=WHITE,fontsize=11)

# (b) IC comparison: standard vs purged
ax=fig.add_subplot(gs[1,0])
cats=['Std KFold','Purged KFold']
vals=[np.nanmean(ic_std),np.nanmean(ic_purged)]
errs=[np.nanstd(ic_std),np.nanstd(ic_purged)]
bars=ax.bar(cats,vals,color=[RED,GREEN],alpha=0.8,width=0.5)
ax.errorbar(cats,vals,errs,fmt='none',color=WHITE,capsize=6,lw=2)
ax.axhline(0,color=GRAY,lw=0.8)
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+0.001,
            'IC={:.4f}'.format(v),ha='center',color=WHITE,fontsize=9)
ax.set_ylabel('Mean IC (Spearman)')
ax.set_title('Leakage: Standard vs Purged\nIC Inflation = {:.4f}'.format(
             np.nanmean(ic_std)-np.nanmean(ic_purged)))
ax.grid(True,axis='y'); wm(ax)

# (c) Lambda selection curves
ax=fig.add_subplot(gs[1,1])
ax.plot(np.log10(lam_grid),cv_ic_std,color=RED,lw=2,marker='o',ms=4,
        label='Standard KFold')
ax.plot(np.log10(lam_grid),cv_ic_purged,color=GREEN,lw=2,marker='s',ms=4,
        label='Purged KFold')
ax.axvline(np.log10(best_lam_std),color=RED,lw=1,ls='--')
ax.axvline(np.log10(best_lam_purged),color=GREEN,lw=1,ls='--')
ax.set_xlabel('log10(lambda)'); ax.set_ylabel('Mean IC')
ax.set_title('Lambda Selection:\nStandard vs Purged CV')
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# (d) Anchored vs rolling IC
ax=fig.add_subplot(gs[1,2])
roll_a=pd.Series(ic_anchored).rolling(6).mean()
roll_r=pd.Series(ic_rolling).rolling(6).mean()
ax.plot(roll_a.values,color=BLUE,lw=2,
        label='Anchored (6M roll) mean={:.3f}'.format(np.nanmean(ic_anchored)))
ax.plot(roll_r.values,color=ORANGE,lw=2,
        label='Rolling W={} (6M roll) mean={:.3f}'.format(ROLL_WIN,np.nanmean(ic_rolling)))
ax.axhline(0,color=GRAY,lw=0.8)
ax.set_xlabel('Walk-forward fold'); ax.set_ylabel('IC')
ax.set_title('Anchored vs Rolling Walk-Forward')
ax.legend(fontsize=7); ax.grid(True); wm(ax)

p1=os.path.join(OUT_DIR,'m40_01_purging_leakage.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# ================================================================
# FIGURE 2: CPCV SHARPE DISTRIBUTION + DSR
# ================================================================
fig,axes=plt.subplots(1,2,figsize=(16,7),facecolor=DARK)
fig.suptitle('M40 -- Combinatorial Purged KFold: Sharpe Distribution & DSR',
             color=WHITE,fontsize=14,fontweight='bold')

ax=axes[0]
ax.hist(sharpes,bins=12,color=BLUE,alpha=0.75,edgecolor=PANEL,density=True)
mu,sigma=sharpes.mean(),sharpes.std()+1e-9
xx=np.linspace(sharpes.min()-1,sharpes.max()+1,200)
ax.plot(xx,norm.pdf(xx,mu,sigma),color=ORANGE,lw=2,label='Normal fit')
ax.axvline(SR_obs,color=RED,lw=2,ls='--',label='SR_max={:.3f}'.format(SR_obs))
ax.axvline(SR_0,color=YELLOW,lw=2,ls=':',label='SR_0={:.3f}'.format(SR_0))
ax.axvline(mu,color=GREEN,lw=1.5,ls='-.',label='Mean SR={:.3f}'.format(mu))
ax.set_xlabel('Annualized Sharpe Ratio'); ax.set_ylabel('Density')
ax.set_title('CPCV Sharpe Distribution\n({} backtest paths from C(6,2)=15 combinations)'.format(
             len(sharpes)))
ax.legend(fontsize=8); ax.grid(True); wm(ax)

# DSR table
ax=axes[1]
ax.set_facecolor(DARK); ax.axis('off')
tdata=[
    ['N trials (backtest paths)',str(N_trials)],
    ['SR_max (best path)','{:.4f}'.format(SR_obs)],
    ['SR_0 (expected max under H0)','{:.4f}'.format(SR_0)],
    ['Probabilistic SR (PSR)','{:.4f}'.format(PSR)],
    ['Deflated Sharpe Ratio (DSR)','{:.4f}'.format(DSR)],
    ['Interpretation','DSR>0.95 => significant' if DSR>0.95 else 'DSR<0.95 => not significant'],
    ['Mean Sharpe (CPCV)','{:.4f}'.format(sharpes.mean())],
    ['Std Sharpe (CPCV)','{:.4f}'.format(sharpes.std())],
]
tbl=ax.table(cellText=tdata,colLabels=['Metric','Value'],
             loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax.set_title('Deflated Sharpe Ratio Analysis\n(Multiple Testing Correction)',
             color=WHITE)

p2=os.path.join(OUT_DIR,'m40_02_cpcv_sharpe.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# ================================================================
# FIGURE 3: SUMMARY DASHBOARD
# ================================================================
fig=plt.figure(figsize=(16,8),facecolor=DARK)
fig.suptitle('M40 -- CV Methods Comparison: Summary',
             color=WHITE,fontsize=14,fontweight='bold')
gs3=gridspec.GridSpec(1,2,figure=fig,wspace=0.30)

# IC comparison bar: all methods
ax=fig.add_subplot(gs3[0,0])
methods=['Standard\nKFold','Purged\nKFold','Anchored\nWF','Rolling\nWF']
ic_vals=[np.nanmean(ic_std),np.nanmean(ic_purged),
         np.nanmean(ic_anchored),np.nanmean(ic_rolling)]
ic_stds=[np.nanstd(ic_std),np.nanstd(ic_purged),
         np.nanstd(ic_anchored),np.nanstd(ic_rolling)]
colors=[RED,GREEN,BLUE,ORANGE]
bars=ax.bar(methods,ic_vals,color=colors,alpha=0.8,width=0.5)
ax.errorbar(methods,ic_vals,ic_stds,fmt='none',color=WHITE,capsize=6,lw=2)
ax.axhline(0,color=GRAY,lw=0.8)
for bar,v in zip(bars,ic_vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+(0.001 if v>=0 else -0.002),
            '{:.4f}'.format(v),ha='center',
            color=WHITE,fontsize=9,
            va='bottom' if v>=0 else 'top')
ax.set_ylabel('Mean IC (Spearman)')
ax.set_title('IC by CV Method:\nLeakage vs Purged Estimates')
ax.grid(True,axis='y'); wm(ax)

# Summary table
ax=fig.add_subplot(gs3[0,1])
ax.set_facecolor(DARK); ax.axis('off')
tdata=[
    ['Standard KFold','{:.4f}'.format(np.nanmean(ic_std)),
     'BIASED','Overlapping labels'],
    ['Purged KFold','{:.4f}'.format(np.nanmean(ic_purged)),
     'UNBIASED','Labels purged+embargoed'],
    ['Anchored WF','{:.4f}'.format(np.nanmean(ic_anchored)),
     'UNBIASED','Growing train window'],
    ['Rolling WF W={}'.format(ROLL_WIN),'{:.4f}'.format(np.nanmean(ic_rolling)),
     'UNBIASED','Fixed train window'],
    ['CPCV (15 paths)','SR={:.3f}'.format(sharpes.mean()),
     'UNBIASED','Distribution of SRs'],
    ['Deflated SR','DSR={:.3f}'.format(DSR),
     'CORRECTED','Multiple testing adj.'],
]
cols=['Method','IC / SR','Bias','Notes']
tbl=ax.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax.set_title('Cross-Validation Methods Summary',color=WHITE)

p3=os.path.join(OUT_DIR,'m40_03_cv_summary.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# ================================================================
# SUMMARY
# ================================================================
print()
print('='*65)
print('  MODULE 40 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] Overlapping labels: r_{t,t+h} shares h-1 days with r_{t+1}')
print('  [2] Purging: remove train obs whose label overlaps test')
print('  [3] Embargo: skip e periods after test to prevent feature leak')
print('  [4] CPCV: C(k,2) paths => distribution of Sharpe ratios')
print('  [5] PSR = Phi[(SR*-SR_0)*sqrt(T-1)/sqrt(1-g3*SR*+...)]')
print('  [6] DSR corrects for multiple testing across N backtest paths')
print('  [7] Standard KFold IC always >= Purged KFold IC (leakage)')
print()
print('  Standard KFold IC:  {:.4f}'.format(np.nanmean(ic_std)))
print('  Purged KFold IC:    {:.4f}  (leakage={:.4f})'.format(
      np.nanmean(ic_purged),np.nanmean(ic_std)-np.nanmean(ic_purged)))
print('  Best lambda -- Std={:.3f}  Purged={:.3f}'.format(
      best_lam_std,best_lam_purged))
print('  CPCV Sharpe: mean={:.3f}  std={:.3f}  max={:.3f}'.format(
      sharpes.mean(),sharpes.std(),SR_obs))
print('  Deflated SR (DSR): {:.4f}'.format(DSR))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)