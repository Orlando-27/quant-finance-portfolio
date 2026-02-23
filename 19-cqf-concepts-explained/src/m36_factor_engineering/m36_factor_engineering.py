#!/usr/bin/env python3
# MODULE 36: FACTOR ENGINEERING -- MOMENTUM, VALUE & VOLATILITY
# CQF Concepts Explained | Group 7: Supervised ML (1/5)
# Author: Jose Orlando Bobadilla Fuentes | CQF | MSc AI

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import spearmanr
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

def watermark(ax, text='Jose O. Bobadilla | CQF', alpha=0.10):
    ax.text(0.5,0.5,text,transform=ax.transAxes,fontsize=14,
            color=WHITE,alpha=alpha,ha='center',va='center',
            rotation=30,fontweight='bold')

def winsorize(s, z=3.0):
    mu,sig=s.mean(),s.std()
    return s.clip(mu-z*sig, mu+z*sig)

def zscore_cs(s):
    return (s-s.mean())/(s.std()+1e-12)

def build_momentum(prices, lb, skip=21):
    ret = prices.shift(skip)/prices.shift(lb+skip)-1
    return ret.apply(lambda r: zscore_cs(winsorize(r)), axis=1)

def compute_ic(fac, fwd, freq=21):
    ic_vals,ic_dates=[],[]
    for d in fac.index[::freq]:
        if d not in fwd.index: continue
        f=fac.loc[d].dropna(); r=fwd.loc[d].dropna()
        common=f.index.intersection(r.index)
        if len(common)<10: continue
        rho,_=spearmanr(f[common].values,r[common].values)
        ic_vals.append(rho); ic_dates.append(d)
    return pd.Series(ic_vals,index=ic_dates)

def quintile_bt(fac,ret,n_q=5,rebal=21):
    rdates=fac.index[::rebal]
    q_rets={i:[] for i in range(1,n_q+1)}; q_dates=[]
    for j,d in enumerate(rdates[:-1]):
        dn=rdates[j+1]; f=fac.loc[d].dropna()
        if len(f)<n_q*3: continue
        bounds=np.percentile(f,np.linspace(0,100,n_q+1))
        labels=pd.cut(f,bins=bounds,labels=False,include_lowest=True)
        mask=(ret.index>d)&(ret.index<=dn)
        pr=ret.loc[mask]
        if pr.empty: continue
        for q in range(n_q):
            assets=f.index[labels==q]
            q_rets[q+1].append(pr[assets].mean().mean() if len(assets)>0 else np.nan)
        q_dates.append(dn)
    df=pd.DataFrame(q_rets,index=q_dates)
    df['LS']=df[n_q]-df[1]
    return df

# --- SECTION 01: UNIVERSE ---
print('='*65)
print('  MODULE 36: FACTOR ENGINEERING')
print('  Momentum | Value | Volatility | IC | Quintile PnL')
print('='*65)
t0=time.perf_counter()
N,D=50,252*5
dates=pd.date_range('2019-01-02',periods=D,freq='B')
tickers=['A{:02d}'.format(i) for i in range(1,N+1)]
betas=np.random.uniform(0.5,1.5,N)
mkt=np.random.normal(0.0004,0.012,D)
idvol=np.random.uniform(0.008,0.025,N)
idret=np.random.normal(0,1,(D,N))*idvol[None,:]
log_ret=betas[None,:]*mkt[:,None]+idret
msig=np.random.normal(0,0.0002,(D//60+1,N))
for t in range(D): log_ret[t]+=msig[t//60]
returns=pd.DataFrame(log_ret,index=dates,columns=tickers)
prices=np.exp(returns.cumsum())
prices.iloc[0]=np.random.uniform(20,200,N)
prices=prices.multiply(prices.iloc[0])
print('  [01] Universe: {} assets x {} days'.format(N,D))

# --- SECTION 02: MOMENTUM ---
factors={}
for name,lb in [('MOM_1M',21),('MOM_3M',63),('MOM_6M',126),('MOM_12M',252)]:
    factors[name]=build_momentum(prices,lb,21)
rev=prices/prices.shift(21)-1
factors['REVERSAL']=rev.apply(lambda r: zscore_cs(winsorize(-r)),axis=1)
print('  [02] Momentum factors: MOM_1M/3M/6M/12M + REVERSAL')

# --- SECTION 03: VALUE ---
fund=pd.DataFrame({
    'ey':np.random.uniform(0.02,0.10,N),
    'bm':np.random.uniform(0.20,2.50,N),
    'dy':np.random.uniform(0.00,0.06,N),
},index=tickers)
for col in fund.columns:
    base=fund[col].values
    noise=np.random.normal(0,0.05*base.std(),(D,N))
    panel=pd.DataFrame(base[None,:]+noise.cumsum(axis=0)*0.01,
                       index=dates,columns=tickers)
    panel=panel.clip(lower=fund[col].min()*0.5)
    factors['VAL_'+col.upper()]=panel.apply(lambda r: zscore_cs(winsorize(r)),axis=1)
print('  [03] Value factors: VAL_EY, VAL_BM, VAL_DY')

# --- SECTION 04: VOLATILITY ---
rv21=returns.rolling(21).std()*np.sqrt(252)
rv63=returns.rolling(63).std()*np.sqrt(252)
def dsemi(ret,w):
    def _s(x):
        neg=x[x<0]
        return neg.std()*np.sqrt(252) if len(neg)>2 else np.nan
    return ret.rolling(w).apply(_s,raw=False)
dvol=dsemi(returns,63)
vvol=rv21.rolling(63).std()
factors['LOWVOL_1M']=rv21.apply(lambda r: zscore_cs(winsorize(-r)),axis=1)
factors['LOWVOL_3M']=rv63.apply(lambda r: zscore_cs(winsorize(-r)),axis=1)
factors['DOWN_VOL']=dvol.apply(lambda r: zscore_cs(winsorize(-r)),axis=1)
factors['VOL_OF_VOL']=vvol.apply(lambda r: zscore_cs(winsorize(-r)),axis=1)
print('  [04] Volatility factors: LOWVOL_1M/3M, DOWN_VOL, VOL_OF_VOL')

# --- SECTION 05: IC ---
FWDD=21
fwd_ret=returns.shift(-FWDD).rolling(FWDD).sum()
print('  [05] Computing IC ...')
ic_results={}
for name in factors:
    ic_s=compute_ic(factors[name],fwd_ret,freq=21)
    if len(ic_s)==0: continue
    ic_results[name]={
        'ic':ic_s,'mean':ic_s.mean(),'std':ic_s.std(),
        'ir':ic_s.mean()/(ic_s.std()+1e-9),
        't':ic_s.mean()/(ic_s.std()/np.sqrt(len(ic_s))+1e-9),
        'pp':(ic_s>0).mean(),
    }

# --- SECTION 06: IC DECAY ---
horizons=[5,10,21,42,63,126]
decay_res={}
for name in ['MOM_12M','VAL_BM','LOWVOL_3M']:
    if name not in factors: continue
    icd={}
    for h in horizons:
        fv=returns.shift(-h).rolling(h).sum()
        icd[h]=compute_ic(factors[name],fv,freq=21).mean()
    decay_res[name]=pd.Series(icd)
print('  [06] IC decay computed')

# --- SECTION 07: COMPOSITE ---
valid={k:v for k,v in ic_results.items() if abs(v['ir'])>0.05}
tot=sum(abs(v['ir']) for v in valid.values())+1e-9
wts={k:v['ir']/tot for k,v in valid.items()}
first=list(valid.keys())[0]
comp=pd.DataFrame(0.0,index=factors[first].index,columns=tickers)
for name,w in wts.items():
    comp=comp.add(factors[name].fillna(0)*w,fill_value=0)
comp=comp.apply(lambda r: zscore_cs(winsorize(r)),axis=1)
ic_comp=compute_ic(comp,fwd_ret,freq=21)
ir_comp=ic_comp.mean()/(ic_comp.std()+1e-9)
print('  [07] Composite IC={:.4f}  IR={:.3f}'.format(ic_comp.mean(),ir_comp))

# --- SECTION 08: QUINTILE BACKTESTS ---
print('  [08] Quintile backtests ...')
key_facs=['MOM_12M','MOM_6M','MOM_1M','REVERSAL','VAL_BM','LOWVOL_3M']
bt_res={n:quintile_bt(factors[n],returns) for n in key_facs if n in factors}
comp_bt=quintile_bt(comp,returns)
names_s=sorted(ic_results,key=lambda k:abs(ic_results[k]['mean']),reverse=True)

# =================================================================
# FIGURE 1: FACTOR DISTRIBUTIONS & IC SUMMARY
# =================================================================
print()
print('  Generating figures ...')
fig=plt.figure(figsize=(16,12),facecolor=DARK)
fig.suptitle('M36 -- Factor Engineering: Distributions & IC Analysis',
             color=WHITE,fontsize=14,fontweight='bold',y=0.98)
gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.35)

ax1=fig.add_subplot(gs[0,0])
for i,name in enumerate(['MOM_12M','MOM_6M','VAL_BM']):
    if name not in factors: continue
    row=factors[name].iloc[-1].dropna()
    ax1.hist(row,bins=20,alpha=0.55,color=ACCENT[i],label=name,density=True)
xx=np.linspace(-3,3,200)
ax1.plot(xx,stats.norm.pdf(xx),color=WHITE,lw=1.5,ls='--',label='N(0,1)')
ax1.set_xlabel('Z-score'); ax1.set_ylabel('Density')
ax1.set_title('Factor Cross-Sectional Distributions')
ax1.legend(fontsize=7); ax1.grid(True); watermark(ax1)

ax2=fig.add_subplot(gs[0,1:])
ic_means={k:ic_results[k]['mean'] for k in names_s}
ic_ir={k:ic_results[k]['ir'] for k in names_s}
x_pos=np.arange(len(names_s))
bc=[GREEN if ic_means[n]>0 else RED for n in names_s]
ax2.bar(x_pos,[ic_means[n] for n in names_s],color=bc,alpha=0.75,width=0.6)
ax2.axhline(0,color=GRAY,lw=0.8)
for i,n in enumerate(names_s):
    v=ic_means[n]
    ax2.text(i,v+np.sign(v)*0.003,'IR={:.2f}'.format(ic_ir[n]),
             color=WHITE,fontsize=6,ha='center',
             va='bottom' if v>=0 else 'top')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(names_s,rotation=45,ha='right',fontsize=7)
ax2.set_ylabel('Mean IC (Spearman)')
ax2.set_title('Mean IC by Factor (1M forward return)')
ax2.grid(True,axis='y'); watermark(ax2)

ax3=fig.add_subplot(gs[1,:2])
if 'MOM_12M' in ic_results:
    ic_s=ic_results['MOM_12M']['ic']
    roll=ic_s.rolling(6).mean()
    ax3.bar(ic_s.index,ic_s.values,color=BLUE,alpha=0.4,width=20)
    ax3.plot(roll.index,roll.values,color=ORANGE,lw=2,label='6M rolling avg')
    ax3.axhline(0,color=GRAY,lw=0.8)
    ax3.axhline(ic_s.mean(),color=GREEN,lw=1.5,ls='--',
                label='Mean IC={:.3f}'.format(ic_s.mean()))
    ax3.set_xlabel('Date'); ax3.set_ylabel('IC')
    ax3.set_title('Rolling IC -- MOM_12M  IR={:.3f}'.format(ic_results['MOM_12M']['ir']))
    ax3.legend(fontsize=7); ax3.grid(True); watermark(ax3)

ax4=fig.add_subplot(gs[1,2])
for i,(name,ds) in enumerate(decay_res.items()):
    ax4.plot(ds.index,ds.values,marker='o',color=ACCENT[i],lw=2,ms=5,label=name)
ax4.axhline(0,color=GRAY,lw=0.8)
ax4.set_xlabel('Forward horizon (days)'); ax4.set_ylabel('IC')
ax4.set_title('IC Decay by Horizon')
ax4.legend(fontsize=7); ax4.grid(True); watermark(ax4)

ax5=fig.add_subplot(gs[2,:])
ax5.set_facecolor(DARK); ax5.axis('off')
tdata=[]
cols=['Factor','Mean IC','Std IC','ICIR','t-stat','%pos']
for name in names_s:
    r=ic_results[name]
    tdata.append([name,'{:.4f}'.format(r['mean']),'{:.4f}'.format(r['std']),
                  '{:.3f}'.format(r['ir']),'{:.2f}'.format(r['t']),
                  '{:.0f}%'.format(r['pp']*100)])
tbl=ax5.table(cellText=tdata,colLabels=cols,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(PANEL if row%2==0 else DARK)
    cell.set_edgecolor(GRAY); cell.set_text_props(color=WHITE)
    if row==0: cell.set_facecolor('#1F6FEB')
ax5.set_title('IC Summary Table -- All Factors',color=WHITE)

p1=os.path.join(OUT_DIR,'m36_01_factor_ic.png')
plt.savefig(p1,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p1))

# =================================================================
# FIGURE 2: QUINTILE BACKTESTS
# =================================================================
fig,axes=plt.subplots(2,3,figsize=(16,10),facecolor=DARK)
fig.suptitle('M36 -- Quintile Portfolio Backtests (Long-Short Spreads)',
             color=WHITE,fontsize=14,fontweight='bold')
for ax,(name,bt) in zip(axes.flat,bt_res.items()):
    ls=(1+bt['LS'].fillna(0)).cumprod()
    ar=ls.iloc[-1]**(252/max(len(bt),1))-1
    av=bt['LS'].std()*np.sqrt(252/21)
    sr=ar/(av+1e-9)
    ax.plot(ls.index,(ls-1)*100,color=BLUE,lw=2,label='L/S PnL %')
    rm=ls.cummax(); dd=(ls/rm-1)*100
    ax.fill_between(ls.index,dd,0,where=dd<0,color=RED,alpha=0.25,label='DD')
    ax.axhline(0,color=GRAY,lw=0.6)
    if 5 in bt.columns and 1 in bt.columns:
        ax.plot((1+bt[5].fillna(0)).cumprod().index,
                ((1+bt[5].fillna(0)).cumprod()-1)*100,
                color=GREEN,lw=1.2,ls='--',alpha=0.7,label='Q5')
        ax.plot((1+bt[1].fillna(0)).cumprod().index,
                ((1+bt[1].fillna(0)).cumprod()-1)*100,
                color=RED,lw=1.2,ls='--',alpha=0.7,label='Q1')
    ax.set_title('{} | AR={:.1f}% SR={:.2f}'.format(name,ar*100,sr))
    ax.set_xlabel('Date'); ax.set_ylabel('Cum PnL (%)')
    ax.legend(fontsize=6); ax.grid(True); watermark(ax)
for ax in axes.flat[len(bt_res):]: ax.set_visible(False)
p2=os.path.join(OUT_DIR,'m36_02_quintile_backtests.png')
plt.savefig(p2,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p2))

# =================================================================
# FIGURE 3: COMPOSITE DASHBOARD
# =================================================================
fig=plt.figure(figsize=(16,10),facecolor=DARK)
fig.suptitle('M36 -- Composite Alpha | IC-Weighted Factor Combination',
             color=WHITE,fontsize=14,fontweight='bold')
gs3=gridspec.GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.30)

axc=fig.add_subplot(gs3[0,:])
icr=ic_comp.rolling(6).mean()
axc.bar(ic_comp.index,ic_comp.values,color=PURPLE,alpha=0.4,width=20,label='Monthly IC')
axc.plot(icr.index,icr.values,color=YELLOW,lw=2,label='6M rolling avg')
axc.axhline(0,color=GRAY,lw=0.8)
axc.axhline(ic_comp.mean(),color=GREEN,lw=1.5,ls='--',
            label='Mean IC={:.3f}'.format(ic_comp.mean()))
axc.set_title('Composite Alpha IC | ICIR={:.3f}'.format(ir_comp))
axc.set_xlabel('Date'); axc.set_ylabel('IC')
axc.legend(fontsize=8); axc.grid(True); watermark(axc)

axq=fig.add_subplot(gs3[1,0])
if 'MOM_12M' in bt_res:
    bt=bt_res['MOM_12M']
    qann={'Q{}'.format(q):bt[q].mean()*(252/21)*100
          for q in [1,2,3,4,5] if q in bt.columns}
    axq.bar(list(qann.keys()),list(qann.values()),
            color=[RED,ORANGE,GRAY,CYAN,GREEN],alpha=0.8,width=0.6)
    axq.axhline(0,color=GRAY,lw=0.8)
    axq.set_xlabel('Quintile'); axq.set_ylabel('Ann. Return (%)')
    axq.set_title('MOM_12M -- Annual Return per Quintile')
    axq.grid(True,axis='y'); watermark(axq)

axcq=fig.add_subplot(gs3[1,1])
lsc=(1+comp_bt['LS'].fillna(0)).cumprod()
anc=lsc.iloc[-1]**(252/max(len(comp_bt),1))-1
vc=comp_bt['LS'].std()*np.sqrt(252/21)
src=anc/(vc+1e-9)
axcq.plot(lsc.index,(lsc-1)*100,color=PURPLE,lw=2.5,label='Composite L/S')
rm=lsc.cummax(); ddc=(lsc/rm-1)*100
axcq.fill_between(lsc.index,ddc,0,where=ddc<0,color=RED,alpha=0.25)
axcq.axhline(0,color=GRAY,lw=0.6)
axcq.set_title('Composite L/S | AR={:.1f}% SR={:.2f}'.format(anc*100,src))
axcq.set_xlabel('Date'); axcq.set_ylabel('Cum PnL (%)')
axcq.legend(fontsize=8); axcq.grid(True); watermark(axcq)

p3=os.path.join(OUT_DIR,'m36_03_composite_dashboard.png')
plt.savefig(p3,dpi=150,bbox_inches='tight',facecolor=DARK); plt.close()
print('    [OK] {}'.format(p3))

# =================================================================
# SUMMARY
# =================================================================
best=max(ic_results,key=lambda k:abs(ic_results[k]['ir']))
print()
print('='*65)
print('  MODULE 36 COMPLETE -- 3 figures saved')
print('  Key Concepts:')
print('  [1] Momentum: total return t-lb to t-skip (skip=21d)')
print('  [2] Value: earnings yield, B/M, div yield (slow-moving)')
print('  [3] Low-vol anomaly: negative z-score of realized vol')
print('  [4] Winsorize +/-3sigma + cross-sectional z-score')
print('  [5] IC = Spearman corr(signal_t, fwd_return_{t+h})')
print('  [6] ICIR = mean(IC)/std(IC)  Grinold-Kahn quality')
print('  [7] Composite = IC-weighted factor z-scores')
print('  [8] Quintile L/S = Q5 minus Q1, monthly rebalanced')
print()
print('  Best factor: {}  ICIR={:.4f}'.format(best,ic_results[best]['ir']))
print('  Composite IC={:.4f}  ICIR={:.4f}'.format(ic_comp.mean(),ir_comp))
print('  Composite Sharpe: {:.3f}'.format(src))
print('  Elapsed: {:.1f}s'.format(time.perf_counter()-t0))
print('='*65)