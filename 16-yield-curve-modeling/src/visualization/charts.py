"""Yield Curve Visualization Engine — dark theme + watermark."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
warnings.filterwarnings("ignore")

BG_COLOR    = "#0D1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR  = "#C9D1D9"
GRID_COLOR  = "#21262D"
ACCENT      = ["#00D4FF","#FF6B35","#7FFF7F","#FFD700","#DA70D6","#00CED1","#FF4500","#32CD32","#FF8C00","#9370DB"]
WATERMARK   = "Jose Orlando Bobadilla Fuentes, CQF | MSc AI"
DPI         = 150

def _dark(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    ax_list = np.array(axes).flatten() if hasattr(axes,"__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR); ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for sp in ax.spines.values(): sp.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, ls="--", alpha=0.6)

def _wm(fig):
    fig.text(0.5,0.5,WATERMARK,fontsize=9,color="white",alpha=0.07,
             ha="center",va="center",rotation=30,transform=fig.transFigure,zorder=0,fontweight="bold")

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor=BG_COLOR,edgecolor="none")
    plt.close(fig)
    print(f"  [+] Saved: {path}")

def plot_ns_nss_fit(tenors,yields_obs,ns_fit,nss_fit,tau_fine,ns_fine,nss_fine,spline_fine,ns_params,nss_params,title="",save_path=None):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,10),gridspec_kw={"height_ratios":[3,1]})
    _dark(fig,[ax1,ax2]); _wm(fig)
    fig.suptitle(f"Nelson-Siegel / NSS Fit — {title}",color=TEXT_COLOR,fontsize=13,fontweight="bold")
    ax1.scatter(tenors,yields_obs*100,color=ACCENT[3],s=60,zorder=5,label="Observed",edgecolors=PANEL_COLOR,linewidths=0.5)
    ax1.plot(tau_fine,ns_fine*100,color=ACCENT[0],lw=2.0,label=f"NS  RMSE={ns_params.rmse*1e4:.1f}bps  R²={ns_params.r2:.5f}")
    ax1.plot(tau_fine,nss_fine*100,color=ACCENT[1],lw=2.0,ls="--",label=f"NSS RMSE={nss_params.rmse*1e4:.1f}bps  R²={nss_params.r2:.5f}")
    ax1.plot(tau_fine,spline_fine*100,color=ACCENT[2],lw=1.5,ls=":",label="Cubic spline",alpha=0.8)
    ax1.set_ylabel("Yield (%)",color=TEXT_COLOR,fontsize=10)
    ax1.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=9)
    p=ns_params
    ax1.text(0.02,0.04,f"β₀={p.beta0*100:.2f}%  β₁={p.beta1*100:.2f}%  β₂={p.beta2*100:.2f}%  λ={p.lam:.3f}",transform=ax1.transAxes,color=ACCENT[0],fontsize=8)
    ax2.bar(tenors-0.08,(yields_obs-ns_fit)*1e4,width=0.15,color=ACCENT[0],alpha=0.8,label="NS")
    ax2.bar(tenors+0.08,(yields_obs-nss_fit)*1e4,width=0.15,color=ACCENT[1],alpha=0.8,label="NSS")
    ax2.axhline(0,color=TEXT_COLOR,lw=0.8,alpha=0.5)
    ax2.set_xlabel("Maturity (years)",color=TEXT_COLOR,fontsize=10)
    ax2.set_ylabel("Residual (bps)",color=TEXT_COLOR,fontsize=10)
    ax2.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=9)
    if save_path: _save(fig,save_path)
    return fig,(ax1,ax2)

def plot_factor_dynamics(factors_df,title="",save_path=None):
    cols=factors_df.columns.tolist()
    labels=["β₀ — Level","β₁ — Slope","β₂ — Curvature"]
    fig,axes=plt.subplots(3,1,figsize=(16,12),sharex=True)
    _dark(fig,axes); _wm(fig)
    fig.suptitle(f"NS Factor Dynamics — {title}",color=TEXT_COLOR,fontsize=13,fontweight="bold",y=0.99)
    for ax,col,label,color in zip(axes,cols,labels,ACCENT[:3]):
        s=factors_df[col].dropna()*100; rm=s.rolling(12,min_periods=3).mean()
        ax.plot(s.index,s.values,color=color,lw=0.7,alpha=0.8)
        ax.fill_between(s.index,s.mean(),s.values,color=color,alpha=0.15)
        ax.plot(rm.index,rm.values,color="white",lw=1.8,ls="--",alpha=0.7,label="12m MA")
        ax.axhline(s.mean(),color=color,lw=1.0,ls=":",alpha=0.5,label=f"Mean={s.mean():.2f}%")
        q75,q25=s.quantile(0.75),s.quantile(0.25)
        ax.fill_between(s.index,q75,s.values,where=s.values>q75,color=ACCENT[3],alpha=0.20)
        ax.fill_between(s.index,s.values,q25,where=s.values<q25,color=ACCENT[1],alpha=0.20)
        ax.set_ylabel(f"{label} (%)",color=TEXT_COLOR,fontsize=9)
        ax.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=8)
    axes[-1].set_xlabel("Date",color=TEXT_COLOR,fontsize=10)
    if save_path: _save(fig,save_path)
    return fig,axes

def plot_pca_analysis(pca_model,factor_scores,yields_panel,save_path=None):
    fig=plt.figure(figsize=(18,12))
    gs=GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.30)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])
    _dark(fig,[ax1,ax2,ax3,ax4]); _wm(fig)
    fig.suptitle("Yield Curve PCA — Level / Slope / Curvature",color=TEXT_COLOR,fontsize=13,fontweight="bold")
    loadings=pca_model.loadings_; tenors=np.array(loadings.index,dtype=float); bw=0.25
    for i,(col,color,lbl) in enumerate(zip(loadings.columns,ACCENT[:3],["PC1 Level","PC2 Slope","PC3 Curvature"])):
        ax1.bar(np.arange(len(tenors))+(i-1)*bw,loadings[col].values,width=bw,color=color,alpha=0.85,label=lbl)
    ax1.set_xticks(np.arange(len(tenors)))
    ax1.set_xticklabels([f"{t:.1f}y" for t in tenors],rotation=45,ha="right",color=TEXT_COLOR,fontsize=8)
    ax1.axhline(0,color=TEXT_COLOR,lw=0.7,alpha=0.5); ax1.set_title("PCA Loadings",color=TEXT_COLOR,fontsize=10)
    ax1.set_ylabel("Loading",color=TEXT_COLOR,fontsize=9); ax1.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=8)
    ev=pca_model.explained_var_; cum_ev=np.cumsum(ev)*100; pcs=[f"PC{i+1}" for i in range(len(ev))]
    ax2.bar(pcs,ev*100,color=ACCENT[0],alpha=0.8,label="Individual")
    ax2r=ax2.twinx(); ax2r.plot(pcs,cum_ev,color=ACCENT[3],lw=2.0,marker="o",ms=6,label="Cumulative")
    ax2r.set_ylabel("Cumulative (%)",color=ACCENT[3],fontsize=9); ax2r.tick_params(colors=ACCENT[3]); ax2r.set_facecolor(PANEL_COLOR); ax2r.set_ylim(0,105)
    ax2.set_title("Scree Plot",color=TEXT_COLOR,fontsize=10); ax2.set_ylabel("Var. Explained (%)",color=TEXT_COLOR,fontsize=9)
    h1,l1=ax2.get_legend_handles_labels(); h2,l2=ax2r.get_legend_handles_labels()
    ax2.legend(h1+h2,l1+l2,facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=8)
    for col,color in zip(factor_scores.columns,ACCENT[:3]):
        ax3.plot(factor_scores[col].index,factor_scores[col].values,color=color,lw=0.8,label=col,alpha=0.85)
    ax3.axhline(0,color=TEXT_COLOR,lw=0.6,alpha=0.4); ax3.set_title("Factor Scores",color=TEXT_COLOR,fontsize=10)
    ax3.set_ylabel("Score",color=TEXT_COLOR,fontsize=9); ax3.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=8)
    recon=pca_model.reconstruct(yields_panel); common=yields_panel.index.intersection(recon.index)
    rmse_t=np.sqrt(((yields_panel.loc[common]-recon.loc[common])**2).mean())*1e4
    ax4.bar(np.arange(len(tenors)),rmse_t.values,color=ACCENT[1],alpha=0.8)
    ax4.set_xticks(np.arange(len(tenors))); ax4.set_xticklabels([f"{t:.1f}y" for t in tenors],rotation=45,ha="right",color=TEXT_COLOR,fontsize=8)
    avg=rmse_t.mean(); ax4.axhline(avg,color=ACCENT[3],lw=1.5,ls="--",label=f"Mean={avg:.2f}bps")
    ax4.set_title("Reconstruction RMSE by Tenor (bps)",color=TEXT_COLOR,fontsize=10); ax4.set_ylabel("RMSE (bps)",color=TEXT_COLOR,fontsize=9)
    ax4.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=8)
    if save_path: _save(fig,save_path)
    return fig,[ax1,ax2,ax3,ax4]

def plot_var_forecast(tenors,historical_last,forecast_curves,history_panel,title="",save_path=None):
    fig=plt.figure(figsize=(18,9)); ax1=fig.add_subplot(121); ax2=fig.add_subplot(122,projection="3d")
    _dark(fig,[ax1]); _wm(fig); fig.patch.set_facecolor(BG_COLOR); ax2.set_facecolor(PANEL_COLOR)
    fig.suptitle(f"Yield Curve Forecast — {title}",color=TEXT_COLOR,fontsize=13,fontweight="bold")
    ax1.plot(tenors,historical_last*100,color="white",lw=2.5,label="Current",zorder=5)
    h=min(len(forecast_curves),12); idx=sorted(set([0,min(2,h-1),min(5,h-1),h-1]))
    lbls=["1m","3m","6m",f"{h}m"]
    for ri,color,lbl in zip(idx,ACCENT[:4],lbls):
        fc=forecast_curves.iloc[ri].values
        ax1.plot(tenors,fc*100,color=color,lw=1.8,ls="--",label=f"{lbl} ahead",alpha=0.9)
        ax1.fill_between(tenors,historical_last*100,fc*100,color=color,alpha=0.06)
    ax1.set_xlabel("Maturity (years)",color=TEXT_COLOR,fontsize=10); ax1.set_ylabel("Yield (%)",color=TEXT_COLOR,fontsize=10)
    ax1.set_title("Diebold-Li VAR Forecast",color=TEXT_COLOR,fontsize=10); ax1.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=9)
    n_t=len(tenors); h_panel=history_panel.dropna().iloc[-12:,: n_t]
    fc_vals=forecast_curves.iloc[:,:n_t]; combined=pd.concat([h_panel,fc_vals]); n_d=len(combined)
    X=np.tile(tenors,(n_d,1)); Y=np.tile(np.arange(n_d),(n_t,1)).T; Z=combined.values[:,:n_t]
    ax2.plot_surface(X,Y,Z*100,cmap="plasma",alpha=0.85,linewidth=0,antialiased=True)
    ax2.set_xlabel("Maturity",color=TEXT_COLOR,fontsize=8,labelpad=6); ax2.set_ylabel("Time",color=TEXT_COLOR,fontsize=8,labelpad=6)
    ax2.set_zlabel("Yield (%)",color=TEXT_COLOR,fontsize=8,labelpad=6); ax2.set_title("Curve Surface",color=TEXT_COLOR,fontsize=9,pad=8)
    ax2.tick_params(colors=TEXT_COLOR,labelsize=7); ax2.xaxis.pane.fill=False; ax2.yaxis.pane.fill=False; ax2.zaxis.pane.fill=False
    if save_path: _save(fig,save_path)
    return fig,(ax1,ax2)

def plot_bootstrap_bands(tenors,yields_obs,tau_grid,boot_result,ns_fine,title="",save_path=None):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,8))
    _dark(fig,[ax1,ax2]); _wm(fig)
    fig.suptitle(f"Bootstrap Confidence Bands — {title}",color=TEXT_COLOR,fontsize=13,fontweight="bold")
    all_c=boot_result["all_curves"]
    ax1.fill_between(tau_grid,boot_result["lower"]*100,boot_result["upper"]*100,color=ACCENT[0],alpha=0.20,label="95% CI")
    ax1.fill_between(tau_grid,np.percentile(all_c,25,axis=0)*100,np.percentile(all_c,75,axis=0)*100,color=ACCENT[0],alpha=0.30,label="IQR")
    ax1.plot(tau_grid,boot_result["mean_curve"]*100,color=ACCENT[0],lw=2.0,label="Bootstrap mean")
    ax1.plot(tau_grid,ns_fine*100,color="white",lw=1.5,ls="--",label="NS estimate",alpha=0.8)
    ax1.scatter(tenors,yields_obs*100,color=ACCENT[3],s=50,zorder=5,label="Observed",edgecolors=PANEL_COLOR)
    ax1.set_xlabel("Maturity (years)",color=TEXT_COLOR,fontsize=10); ax1.set_ylabel("Yield (%)",color=TEXT_COLOR,fontsize=10)
    ax1.set_title("NS Fit with Bootstrap Uncertainty",color=TEXT_COLOR,fontsize=10); ax1.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=9)
    b0=all_c[:,-1]; b1=all_c[:,0]-all_c[:,-1]; b2=all_c[:,len(tau_grid)//2]-0.5*(all_c[:,0]+all_c[:,-1])
    for vals,color,lbl in [(b0*100,ACCENT[0],"β₀ Level (%)"),(b1*100,ACCENT[1],"β₁ Slope (%)"),(b2*100,ACCENT[2],"β₂ Curvature (%)")]:
        ax2.hist(vals,bins=40,color=color,alpha=0.55,label=lbl,edgecolor=PANEL_COLOR,linewidth=0.3)
        ax2.axvline(vals.mean(),color=color,lw=2.0,ls="--",alpha=0.9)
    ax2.set_xlabel("Value (%)",color=TEXT_COLOR,fontsize=10); ax2.set_ylabel("Frequency",color=TEXT_COLOR,fontsize=10)
    ax2.set_title("Bootstrap Factor Distributions",color=TEXT_COLOR,fontsize=10); ax2.legend(facecolor=PANEL_COLOR,labelcolor=TEXT_COLOR,fontsize=9)
    if save_path: _save(fig,save_path)
    return fig,(ax1,ax2)
