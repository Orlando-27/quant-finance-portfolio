"""
Factor Construction & Fama-French Replication
=============================================

Implements systematic factor portfolio construction from firm characteristics
and replicates the Fama-French three-factor and five-factor models using
either real data or a realistic synthetic cross-section for demonstration.

Methodology:
    For each characteristic c (book-to-market, profitability, investment,
    momentum) at rebalancing date t:

    1. Rank the universe of N stocks by characteristic c_it
    2. Form quantile-sorted portfolios (quintiles or 2x3 independent sorts)
    3. Compute factor return as the long-short spread:
       F_t = R_top(c)_t - R_bottom(c)_t

    This yields a zero-cost portfolio whose return isolates the premium
    associated with characteristic c, controlling for other dimensions
    via double-sorting or orthogonalization.

    The 2x3 independent sort (Fama-French methodology):
    - Sort independently on Size (median breakpoint) and B/M (30/70 percentiles)
    - Six portfolios: SH, SM, SL, BH, BM, BL
    - SMB = (SH + SM + SL)/3 - (BH + BM + BL)/3
    - HML = (SH + BH)/2 - (SL + BL)/2

References:
    Fama & French (1993), Carhart (1997), Fama & French (2015),
    Asness, Moskowitz & Pedersen (2013)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats as sp_stats


class FactorConstructor:
    """
    Constructs long-short factor portfolios from firm characteristics.

    The constructor supports single-sort (quintile spreads) and double-sort
    (2x3 Fama-French methodology) approaches. Value-weighting within each
    portfolio is optional.

    Parameters
    ----------
    n_quantiles : int
        Number of quantile bins for single-sort (default 5 = quintiles).
    holding_period : int
        Rebalancing frequency in periods (default 1 = every period).
    value_weight : bool
        If True, value-weight returns within each sorted portfolio using
        market capitalization. If False, equal-weight.
    """

    def __init__(self, n_quantiles: int = 5, holding_period: int = 1,
                 value_weight: bool = True):
        self.n_quantiles = n_quantiles
        self.holding_period = holding_period
        self.value_weight = value_weight
        self.factor_returns = None
        self.quantile_portfolios = None

    def single_sort(self, returns: pd.DataFrame,
                    characteristic: pd.DataFrame,
                    market_cap: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Construct a long-short factor from single characteristic sort.

        At each date t, stocks are ranked by characteristic value into
        n_quantiles groups. The factor return is the spread between the
        top and bottom quantile portfolios.

        Parameters
        ----------
        returns : pd.DataFrame
            (T x N) panel of asset returns, index = dates, columns = stock IDs.
        characteristic : pd.DataFrame
            (T x N) panel of characteristic values aligned with returns.
        market_cap : pd.DataFrame, optional
            (T x N) panel of market capitalizations for value-weighting.

        Returns
        -------
        pd.Series
            Time series of long-short factor returns.
        """
        dates = returns.index
        factor_ret = pd.Series(index=dates, dtype=float, name="factor_return")
        quantile_data = {}

        for t in dates:
            ret_t = returns.loc[t].dropna()
            char_t = characteristic.loc[t].reindex(ret_t.index).dropna()
            common = ret_t.index.intersection(char_t.index)
            if len(common) < self.n_quantiles * 2:
                factor_ret.loc[t] = np.nan
                continue

            ret_t = ret_t.loc[common]
            char_t = char_t.loc[common]

            # Assign quantile labels (1 = lowest, n_quantiles = highest)
            labels = pd.qcut(char_t, self.n_quantiles, labels=False,
                             duplicates="drop") + 1
            q_max = labels.max()
            q_min = labels.min()

            if self.value_weight and market_cap is not None:
                cap_t = market_cap.loc[t].reindex(common).fillna(0)
                # Value-weighted return for top and bottom quantiles
                top_mask = labels == q_max
                bot_mask = labels == q_min
                w_top = cap_t[top_mask] / cap_t[top_mask].sum()
                w_bot = cap_t[bot_mask] / cap_t[bot_mask].sum()
                r_top = (ret_t[top_mask] * w_top).sum()
                r_bot = (ret_t[bot_mask] * w_bot).sum()
            else:
                r_top = ret_t[labels == q_max].mean()
                r_bot = ret_t[labels == q_min].mean()

            factor_ret.loc[t] = r_top - r_bot
            quantile_data[t] = labels

        self.factor_returns = factor_ret.dropna()
        self.quantile_portfolios = quantile_data
        return self.factor_returns

    def double_sort_2x3(self, returns: pd.DataFrame,
                        char_primary: pd.DataFrame,
                        char_secondary: pd.DataFrame,
                        market_cap: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Fama-French 2x3 independent double sort.

        Sorts independently on char_primary (median breakpoint: Small/Big)
        and char_secondary (30th/70th percentiles: Low/Medium/High).
        Produces six portfolios and constructs two factors.

        Parameters
        ----------
        returns : pd.DataFrame
            (T x N) asset returns.
        char_primary : pd.DataFrame
            (T x N) primary sort characteristic (typically market cap for size).
        char_secondary : pd.DataFrame
            (T x N) secondary sort characteristic (e.g., book-to-market).
        market_cap : pd.DataFrame
            (T x N) market capitalizations.

        Returns
        -------
        dict
            Keys 'primary_factor' (e.g., SMB) and 'secondary_factor' (e.g., HML),
            each a pd.Series of factor returns.
        """
        dates = returns.index
        primary_factor = pd.Series(index=dates, dtype=float, name="primary")
        secondary_factor = pd.Series(index=dates, dtype=float, name="secondary")

        for t in dates:
            ret_t = returns.loc[t].dropna()
            cp = char_primary.loc[t].reindex(ret_t.index).dropna()
            cs = char_secondary.loc[t].reindex(ret_t.index).dropna()
            cap_t = market_cap.loc[t].reindex(ret_t.index).fillna(0)
            common = ret_t.index.intersection(cp.index).intersection(cs.index)
            if len(common) < 12:
                primary_factor.loc[t] = np.nan
                secondary_factor.loc[t] = np.nan
                continue

            ret_t, cp, cs, cap_t = (x.loc[common] for x in
                                     [ret_t, cp, cs, cap_t])

            # Independent breakpoints
            med_p = cp.median()
            p30, p70 = cs.quantile(0.3), cs.quantile(0.7)

            # 2x3 classification
            is_small = cp <= med_p
            is_big = cp > med_p
            is_low = cs <= p30
            is_mid = (cs > p30) & (cs <= p70)
            is_high = cs > p70

            def vw_ret(mask):
                if mask.sum() == 0:
                    return 0.0
                w = cap_t[mask] / cap_t[mask].sum()
                return (ret_t[mask] * w).sum()

            sh = vw_ret(is_small & is_high)
            sm = vw_ret(is_small & is_mid)
            sl = vw_ret(is_small & is_low)
            bh = vw_ret(is_big & is_high)
            bm = vw_ret(is_big & is_mid)
            bl = vw_ret(is_big & is_low)

            # SMB = avg(small) - avg(big)
            primary_factor.loc[t] = (sh + sm + sl) / 3 - (bh + bm + bl) / 3
            # HML = avg(high) - avg(low)
            secondary_factor.loc[t] = (sh + bh) / 2 - (sl + bl) / 2

        return {
            "primary_factor": primary_factor.dropna(),
            "secondary_factor": secondary_factor.dropna(),
        }


class FamaFrenchReplicator:
    """
    Generates a realistic synthetic cross-section and replicates Fama-French
    factors for demonstration and testing purposes.

    The synthetic universe embeds known factor structures:
    - Size effect: small-cap stocks earn higher expected returns
    - Value effect: high book-to-market stocks earn higher expected returns
    - Momentum: past winners continue outperforming
    - Profitability: high-ROE firms earn higher returns
    - Investment: conservative (low asset growth) firms earn higher returns

    Parameters
    ----------
    n_stocks : int
        Number of stocks in the synthetic universe.
    n_periods : int
        Number of monthly return observations.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_stocks: int = 500, n_periods: int = 240,
                 seed: int = 42):
        self.n_stocks = n_stocks
        self.n_periods = n_periods
        self.seed = seed
        self.returns = None
        self.characteristics = None
        self.market_cap = None
        self.factor_returns = None

    def generate_universe(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame],
                                          pd.DataFrame]:
        """
        Generate synthetic stock universe with embedded factor structure.

        The data generating process follows a linear factor model:
            R_it = alpha_i + sum_k beta_ik * F_kt + eps_it

        where factor loadings beta_ik correlate with observable characteristics
        (size, B/M, momentum, profitability, investment), ensuring that
        characteristic-sorted portfolios recover the underlying factor premia.

        Returns
        -------
        returns : pd.DataFrame
            (T x N) monthly returns.
        characteristics : dict
            Keys = characteristic names, values = (T x N) DataFrames.
        market_cap : pd.DataFrame
            (T x N) market capitalizations.
        """
        rng = np.random.RandomState(self.seed)
        N, T = self.n_stocks, self.n_periods
        dates = pd.date_range("2004-01-31", periods=T, freq="M")
        stocks = [f"S{i:04d}" for i in range(N)]

        # -- Latent factor returns (monthly) --
        # MKT, SMB, HML, UMD, RMW, CMA with realistic moments
        factor_mu = np.array([0.006, 0.002, 0.003, 0.005, 0.003, 0.002])
        factor_vol = np.array([0.045, 0.030, 0.028, 0.040, 0.022, 0.018])
        # Factor correlation matrix
        n_f = len(factor_mu)
        corr = np.eye(n_f)
        corr[0, 1] = corr[1, 0] = 0.2
        corr[0, 3] = corr[3, 0] = -0.1
        corr[2, 3] = corr[3, 2] = -0.15
        corr[2, 4] = corr[4, 2] = 0.3
        corr[4, 5] = corr[5, 4] = 0.25
        cov = np.outer(factor_vol, factor_vol) * corr
        factor_rets = rng.multivariate_normal(factor_mu, cov, size=T)
        factor_names = ["MKT", "SMB", "HML", "UMD", "RMW", "CMA"]
        F = pd.DataFrame(factor_rets, index=dates, columns=factor_names)

        # -- Firm characteristics (static + slow-varying) --
        log_size = rng.normal(7.0, 1.5, N)        # log market cap
        bm_ratio = rng.lognormal(-0.3, 0.8, N)    # book-to-market
        profitability = rng.normal(0.12, 0.06, N)  # ROE
        investment = rng.normal(0.08, 0.05, N)     # asset growth

        # Factor loadings correlate with characteristics
        betas = np.zeros((N, n_f))
        betas[:, 0] = rng.uniform(0.6, 1.4, N)                       # MKT beta
        betas[:, 1] = -0.3 * (log_size - log_size.mean()) / log_size.std() \
                       + rng.normal(0, 0.15, N)                       # SMB
        betas[:, 2] = 0.4 * (bm_ratio - bm_ratio.mean()) / bm_ratio.std() \
                       + rng.normal(0, 0.15, N)                       # HML
        betas[:, 3] = rng.normal(0, 0.3, N)                          # UMD
        betas[:, 4] = 0.3 * (profitability - profitability.mean()) \
                       / profitability.std() + rng.normal(0, 0.1, N)  # RMW
        betas[:, 5] = -0.3 * (investment - investment.mean()) \
                       / investment.std() + rng.normal(0, 0.1, N)     # CMA

        # Idiosyncratic vol scales inversely with size
        idio_vol = 0.08 * np.exp(-0.15 * (log_size - log_size.mean()))

        # -- Generate returns --
        R = np.zeros((T, N))
        for t in range(T):
            systematic = betas @ factor_rets[t]
            idio = rng.normal(0, idio_vol)
            R[t] = systematic + idio

        returns = pd.DataFrame(R, index=dates, columns=stocks)

        # -- Build time-varying characteristics --
        # Size evolves with cumulative returns
        cum_ret = (1 + returns).cumprod()
        init_cap = np.exp(log_size)
        market_cap = cum_ret.multiply(init_cap, axis=1)

        # Book-to-market: slow mean-reversion + noise
        bm_panel = pd.DataFrame(index=dates, columns=stocks, dtype=float)
        bm_current = bm_ratio.copy()
        for t_idx, t in enumerate(dates):
            bm_current = bm_current * (1 + rng.normal(0, 0.02, N))
            bm_panel.loc[t] = bm_current

        # Momentum: trailing 12-1 month return
        momentum = returns.rolling(11).sum().shift(1)

        # Profitability and investment: slow-varying
        prof_panel = pd.DataFrame(index=dates, columns=stocks, dtype=float)
        inv_panel = pd.DataFrame(index=dates, columns=stocks, dtype=float)
        prof_curr = profitability.copy()
        inv_curr = investment.copy()
        for t_idx, t in enumerate(dates):
            prof_curr += rng.normal(0, 0.005, N)
            inv_curr += rng.normal(0, 0.003, N)
            prof_panel.loc[t] = prof_curr
            inv_panel.loc[t] = inv_curr

        characteristics = {
            "book_to_market": bm_panel.astype(float),
            "momentum": momentum,
            "profitability": prof_panel.astype(float),
            "investment": inv_panel.astype(float),
        }

        self.returns = returns
        self.characteristics = characteristics
        self.market_cap = market_cap
        self.factor_returns = F
        return returns, characteristics, market_cap

    def replicate_factors(self) -> pd.DataFrame:
        """
        Replicate Fama-French factors from the synthetic cross-section
        using the 2x3 double-sort methodology.

        Returns
        -------
        pd.DataFrame
            Columns: MKT, SMB_rep, HML_rep, UMD_rep, RMW_rep, CMA_rep
        """
        if self.returns is None:
            self.generate_universe()

        fc = FactorConstructor(value_weight=True)
        ret = self.returns
        cap = self.market_cap

        # Market factor: equal-weighted market return minus risk-free proxy
        mkt = ret.mean(axis=1) - 0.002 / 12  # ~2% annual Rf

        # SMB and HML via 2x3 sort on size and B/M
        size_val = fc.double_sort_2x3(
            ret, cap, self.characteristics["book_to_market"], cap
        )
        smb = size_val["primary_factor"]
        hml = size_val["secondary_factor"]

        # Momentum via single sort (quintile spread)
        fc_mom = FactorConstructor(n_quantiles=5, value_weight=True)
        umd = fc_mom.single_sort(ret, self.characteristics["momentum"], cap)
        umd.name = "UMD_rep"

        # RMW via 2x3 sort on size and profitability
        size_prof = fc.double_sort_2x3(
            ret, cap, self.characteristics["profitability"], cap
        )
        rmw = size_prof["secondary_factor"]

        # CMA via 2x3 sort on size and investment (inverted: low investment = conservative)
        inv_neg = -self.characteristics["investment"]
        size_inv = fc.double_sort_2x3(ret, cap, inv_neg, cap)
        cma = size_inv["secondary_factor"]

        replicated = pd.DataFrame({
            "MKT": mkt, "SMB_rep": smb, "HML_rep": hml,
            "UMD_rep": umd, "RMW_rep": rmw, "CMA_rep": cma,
        }).dropna()

        return replicated

    def compute_factor_statistics(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive factor return statistics.

        Includes annualized return, volatility, Sharpe ratio, skewness,
        kurtosis, maximum drawdown, and t-statistic for mean != 0.

        Parameters
        ----------
        factors : pd.DataFrame
            Factor return time series (monthly frequency assumed).

        Returns
        -------
        pd.DataFrame
            Statistics table with factors as columns.
        """
        ann_ret = factors.mean() * 12
        ann_vol = factors.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol
        skew = factors.skew()
        kurt = factors.kurtosis()
        t_stat = factors.mean() / (factors.std() / np.sqrt(len(factors)))
        cum = (1 + factors).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

        stats_df = pd.DataFrame({
            "Ann. Return": ann_ret, "Ann. Volatility": ann_vol,
            "Sharpe Ratio": sharpe, "t-stat (H0: mu=0)": t_stat,
            "Skewness": skew, "Excess Kurtosis": kurt,
            "Max Drawdown": max_dd,
        })
        return stats_df
