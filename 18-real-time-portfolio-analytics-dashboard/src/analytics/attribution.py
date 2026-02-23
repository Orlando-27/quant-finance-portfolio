# =============================================================================
# src/analytics/attribution.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Brinson-Hood-Beebower performance attribution
# =============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def brinson_attribution(
    port_weights: np.ndarray,
    bench_weights: np.ndarray,
    port_returns: np.ndarray,
    bench_returns: np.ndarray,
    asset_names: List[str],
) -> pd.DataFrame:
    """
    BHB Attribution:
      Allocation Effect  = (wp - wb) * (rb - R_b)
      Selection Effect   = wb * (rp - rb)
      Interaction Effect = (wp - wb) * (rp - rb)
      Active Return      = Allocation + Selection + Interaction
    """
    R_b = float(np.dot(bench_weights, bench_returns))
    alloc   = (port_weights - bench_weights) * (bench_returns - R_b)
    select  = bench_weights * (port_returns - bench_returns)
    interact= (port_weights - bench_weights) * (port_returns - bench_returns)
    active  = alloc + select + interact

    return pd.DataFrame({
        "Asset":       asset_names,
        "Port Wt":     np.round(port_weights * 100, 2),
        "Bench Wt":    np.round(bench_weights * 100, 2),
        "Port Ret":    np.round(port_returns * 100, 3),
        "Bench Ret":   np.round(bench_returns * 100, 3),
        "Allocation":  np.round(alloc * 100, 4),
        "Selection":   np.round(select * 100, 4),
        "Interaction": np.round(interact * 100, 4),
        "Active Ret":  np.round(active * 100, 4),
    })

def summary_attribution(df: pd.DataFrame) -> Dict:
    """Aggregate BHB effects across all assets."""
    return {
        "Total Allocation":  round(df["Allocation"].sum(), 4),
        "Total Selection":   round(df["Selection"].sum(), 4),
        "Total Interaction": round(df["Interaction"].sum(), 4),
        "Total Active Ret":  round(df["Active Ret"].sum(), 4),
    }
