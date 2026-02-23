"""
Cointegration Testing: Engle-Granger & Johansen
================================================

Implements the two principal cointegration testing methodologies used
in pairs trading and statistical arbitrage applications.

The Engle-Granger procedure is a two-step residual-based test suitable
for bivariate systems. The Johansen procedure is a system-based approach
using reduced-rank regression that handles multivariate cointegration
and avoids the normalization sensitivity of Engle-Granger.

References:
    Engle & Granger (1987), Johansen (1991), MacKinnon (1991),
    Hamilton (1994) Chapter 19
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class EngleGranger:
    """
    Engle-Granger two-step cointegration test.

    Step 1: OLS regression of Y on X to obtain residuals.
    Step 2: ADF test on residuals with MacKinnon critical values.

    The test is asymmetric: results may differ depending on which
    variable is the dependent variable. For robustness, the class
    tests both orderings and reports the stronger result.

    Parameters
    ----------
    significance : float
        Significance level for the ADF test (default 0.05).
    max_lags : int or None
        Maximum lag order for ADF. If None, uses automatic selection
        via the AIC criterion.
    """

    def __init__(self, significance: float = 0.05,
                 max_lags: Optional[int] = None):
        self.significance = significance
        self.max_lags = max_lags
        self.results = None

    def test(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Perform the Engle-Granger cointegration test.

        Tests both orderings (Y~X and X~Y) and reports the result
        with the strongest evidence of cointegration (lowest ADF p-value).

        Parameters
        ----------
        y : pd.Series
            First price series (typically log prices).
        x : pd.Series
            Second price series.

        Returns
        -------
        dict
            Keys: cointegrated (bool), adf_stat, adf_pvalue, critical_values,
            hedge_ratio (beta), intercept (alpha), residuals (spread),
            ordering (which variable was dependent in best result).
        """
        # Align series
        df = pd.DataFrame({"y": y, "x": x}).dropna()
        if len(df) < 30:
            return self._empty_result("Insufficient data")

        # Test ordering 1: Y = alpha + beta * X + eps
        res1 = self._test_one_direction(df["y"].values, df["x"].values,
                                         "Y ~ X")

        # Test ordering 2: X = alpha + beta * Y + eps
        res2 = self._test_one_direction(df["x"].values, df["y"].values,
                                         "X ~ Y")

        # Select the ordering with stronger cointegration evidence
        best = res1 if res1["adf_pvalue"] < res2["adf_pvalue"] else res2

        # Also store the statsmodels coint() result for comparison
        coint_stat, coint_pval, coint_crit = coint(df["y"], df["x"],
                                                     maxlag=self.max_lags)
        best["coint_stat"] = coint_stat
        best["coint_pvalue"] = coint_pval
        best["coint_critical"] = dict(zip(["1%", "5%", "10%"], coint_crit))

        # Final cointegration decision
        best["cointegrated"] = best["adf_pvalue"] < self.significance

        # Store residuals as pd.Series with proper index
        best["residuals"] = pd.Series(
            best["residuals_raw"], index=df.index, name="spread"
        )

        self.results = best
        return best

    def _test_one_direction(self, y: np.ndarray, x: np.ndarray,
                             label: str) -> Dict:
        """Run OLS + ADF for one ordering."""
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        resid = ols.resid

        adf_result = adfuller(resid, maxlag=self.max_lags,
                               autolag="AIC" if self.max_lags is None else None)

        return {
            "ordering": label,
            "hedge_ratio": ols.params[1],
            "intercept": ols.params[0],
            "r_squared": ols.rsquared,
            "adf_stat": adf_result[0],
            "adf_pvalue": adf_result[1],
            "adf_lags": adf_result[2],
            "adf_nobs": adf_result[3],
            "critical_values": adf_result[4],
            "residuals_raw": resid,
        }

    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure."""
        return {
            "cointegrated": False, "reason": reason,
            "adf_stat": np.nan, "adf_pvalue": 1.0,
            "hedge_ratio": np.nan, "intercept": np.nan,
            "residuals": pd.Series(dtype=float),
        }

    def get_summary(self) -> str:
        """Return formatted test summary."""
        if self.results is None:
            return "Run test() first."
        r = self.results
        sig = "***" if r["adf_pvalue"] < 0.01 else \
              "**" if r["adf_pvalue"] < 0.05 else \
              "*" if r["adf_pvalue"] < 0.10 else ""
        lines = [
            "=" * 60,
            "ENGLE-GRANGER COINTEGRATION TEST",
            "=" * 60,
            f"Ordering: {r.get('ordering', 'N/A')}",
            f"Hedge ratio (beta): {r['hedge_ratio']:.6f}",
            f"Intercept (alpha):  {r['intercept']:.6f}",
            f"R-squared:          {r.get('r_squared', np.nan):.4f}",
            "-" * 60,
            f"ADF statistic:      {r['adf_stat']:.4f} {sig}",
            f"ADF p-value:        {r['adf_pvalue']:.6f}",
            f"ADF lags used:      {r.get('adf_lags', 'N/A')}",
            f"Cointegrated:       {r['cointegrated']}",
            "-" * 60,
            "Critical values:",
        ]
        for k, v in r.get("critical_values", {}).items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class JohansenTest:
    """
    Johansen multivariate cointegration test.

    Tests for the number of cointegrating relationships in a system
    of I(1) variables using reduced-rank regression on a VECM.

    The trace statistic tests H0: r <= r0 against H1: r > r0.
    The max-eigenvalue statistic tests H0: r = r0 against H1: r = r0 + 1.

    Parameters
    ----------
    det_order : int
        Deterministic term specification:
        -1 = no constant, 0 = restricted constant, 1 = unrestricted constant.
    k_ar_diff : int
        Number of lagged difference terms in the VECM (default 1).
    """

    def __init__(self, det_order: int = 0, k_ar_diff: int = 1):
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff
        self.results = None

    def test(self, data: pd.DataFrame) -> Dict:
        """
        Perform the Johansen cointegration test.

        Parameters
        ----------
        data : pd.DataFrame
            (T x N) matrix of price series (typically log prices).

        Returns
        -------
        dict
            Keys: n_coint_trace, n_coint_eigen (number of cointegrating
            vectors at 5%), trace_stats, eigen_stats, critical_values_trace,
            critical_values_eigen, eigenvectors (cointegrating vectors),
            eigenvalues.
        """
        X = data.dropna().values
        if X.shape[0] < 50 or X.shape[1] < 2:
            return {"n_coint_trace": 0, "n_coint_eigen": 0,
                    "error": "Insufficient data"}

        try:
            joh = coint_johansen(X, det_order=self.det_order,
                                  k_ar_diff=self.k_ar_diff)
        except Exception as e:
            return {"n_coint_trace": 0, "n_coint_eigen": 0,
                    "error": str(e)}

        n = X.shape[1]
        # Count cointegrating relations at 5% (column index 1 = 5%)
        n_trace = sum(joh.lr1[i] > joh.cvt[i, 1] for i in range(n))
        n_eigen = sum(joh.lr2[i] > joh.cvm[i, 1] for i in range(n))

        result = {
            "n_coint_trace": n_trace,
            "n_coint_eigen": n_eigen,
            "trace_stats": joh.lr1,
            "eigen_stats": joh.lr2,
            "critical_values_trace_5pct": joh.cvt[:, 1],
            "critical_values_eigen_5pct": joh.cvm[:, 1],
            "eigenvectors": joh.evec,
            "eigenvalues": joh.eig,
            "column_names": list(data.columns),
        }

        self.results = result
        return result

    def get_cointegrating_vector(self, idx: int = 0) -> Optional[np.ndarray]:
        """Return the idx-th cointegrating vector (normalized)."""
        if self.results is None or "eigenvectors" not in self.results:
            return None
        evec = self.results["eigenvectors"][:, idx]
        return evec / evec[0]  # Normalize first element to 1

    def get_summary(self) -> str:
        """Return formatted Johansen test summary."""
        if self.results is None:
            return "Run test() first."
        r = self.results
        n = len(r["trace_stats"])
        lines = [
            "=" * 60,
            "JOHANSEN COINTEGRATION TEST",
            "=" * 60,
            f"Variables: {r.get('column_names', 'N/A')}",
            f"Cointegrating relations (trace, 5%): {r['n_coint_trace']}",
            f"Cointegrating relations (eigen, 5%): {r['n_coint_eigen']}",
            "-" * 60,
            f"{'H0: r<=':<10} {'Trace Stat':<14} {'5% CV':<14} {'Reject?':<10}",
            "-" * 60,
        ]
        for i in range(n):
            reject = "YES" if r["trace_stats"][i] > r["critical_values_trace_5pct"][i] else "no"
            lines.append(
                f"{i:<10} {r['trace_stats'][i]:<14.4f} "
                f"{r['critical_values_trace_5pct'][i]:<14.4f} {reject:<10}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
