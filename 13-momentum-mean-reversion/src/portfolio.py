"""
Portfolio Construction & Risk Management
==========================================

Combines momentum and mean-reversion signals using regime-dependent
weights to construct a dynamically-blended multi-asset portfolio.

Key features:
    - Regime-adaptive signal blending (momentum alpha from RegimeDetector)
    - EWMA volatility scaling for portfolio-level vol targeting
    - Maximum position limits per asset and per asset class
    - Drawdown-based risk overlay: reduces exposure when DD exceeds threshold
    - Turnover constraints to control transaction costs

Author: Jose Orlando Bobadilla Fuentes, CQF, MSc AI
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class PortfolioConstructor:
    """
    Constructs a dynamically-blended momentum/mean-reversion portfolio.

    Parameters
    ----------
    vol_target : float
        Annualized portfolio volatility target (default 0.10).
    max_position : float
        Maximum absolute weight per asset (default 0.20).
    max_class_weight : float
        Maximum total absolute weight per asset class (default 0.50).
    max_drawdown_threshold : float
        Drawdown level that triggers exposure reduction (default 0.10).
    drawdown_scale_factor : float
        Factor by which to reduce exposure when DD > threshold (default 0.5).
    turnover_penalty : float
        Penalty for turnover in signal smoothing (default 0.10).
    rebalance_freq : int
        Rebalancing frequency in trading days (default 21 ~ monthly).
    """

    def __init__(
        self,
        vol_target: float = 0.10,
        max_position: float = 0.20,
        max_class_weight: float = 0.50,
        max_drawdown_threshold: float = 0.10,
        drawdown_scale_factor: float = 0.50,
        turnover_penalty: float = 0.10,
        rebalance_freq: int = 21,
    ):
        self.vol_target = vol_target
        self.max_position = max_position
        self.max_class_weight = max_class_weight
        self.max_dd_threshold = max_drawdown_threshold
        self.dd_scale = drawdown_scale_factor
        self.turnover_penalty = turnover_penalty
        self.rebalance_freq = rebalance_freq

    def blend_signals(
        self,
        mom_signal: pd.DataFrame,
        mr_signal: pd.DataFrame,
        regime_alpha: pd.Series,
    ) -> pd.DataFrame:
        """
        Blend momentum and mean-reversion signals using regime alpha.

        composite = alpha * MOM + (1 - alpha) * MR

        Parameters
        ----------
        mom_signal : pd.DataFrame
            Momentum signal (dates x assets).
        mr_signal : pd.DataFrame
            Mean-reversion signal (dates x assets).
        regime_alpha : pd.Series
            Regime-dependent blending weight for momentum (0 to 1).

        Returns
        -------
        pd.DataFrame
            Blended composite signal.
        """
        # Broadcast regime_alpha across assets
        alpha_2d = regime_alpha.values[:, np.newaxis] * np.ones((1, mom_signal.shape[1]))
        alpha_df = pd.DataFrame(alpha_2d, index=mom_signal.index, columns=mom_signal.columns)

        composite = alpha_df * mom_signal + (1.0 - alpha_df) * mr_signal
        return composite

    def apply_vol_scaling(
        self,
        raw_weights: pd.DataFrame,
        returns: pd.DataFrame,
        vol_lookback: int = 60,
    ) -> pd.DataFrame:
        """
        Scale portfolio weights to target a specific volatility level.

        The portfolio is scaled by: vol_target / realized_vol_portfolio.

        Parameters
        ----------
        raw_weights : pd.DataFrame
            Unscaled portfolio weights.
        returns : pd.DataFrame
            Daily returns for vol estimation.
        vol_lookback : int
            EWMA span for portfolio vol estimation.

        Returns
        -------
        pd.DataFrame
            Volatility-scaled weights.
        """
        # Estimate portfolio volatility from recent returns and weights
        port_ret = (returns * raw_weights.shift(1)).sum(axis=1)
        port_vol = port_ret.ewm(span=vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)

        scale = self.vol_target / port_vol.replace(0, np.nan)
        scale = scale.clip(0.1, 3.0)  # Cap extreme scaling

        scaled = raw_weights.multiply(scale, axis=0)
        return scaled

    def apply_position_limits(
        self,
        weights: pd.DataFrame,
        asset_class_map: Optional[Dict[str, list]] = None,
    ) -> pd.DataFrame:
        """
        Apply per-asset and per-class position limits.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights.
        asset_class_map : dict, optional
            Mapping from asset class to list of tickers.

        Returns
        -------
        pd.DataFrame
            Constrained weights.
        """
        # Per-asset limit
        constrained = weights.clip(-self.max_position, self.max_position)

        # Per-class limit
        if asset_class_map is not None:
            for ac, tickers in asset_class_map.items():
                cols = [c for c in tickers if c in constrained.columns]
                if not cols:
                    continue
                class_sum = constrained[cols].abs().sum(axis=1)
                excess = (class_sum / self.max_class_weight).clip(lower=1.0)
                for c in cols:
                    constrained[c] = constrained[c] / excess

        return constrained

    def apply_drawdown_control(
        self,
        weights: pd.DataFrame,
        cumulative_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Reduce portfolio exposure when drawdown exceeds threshold.

        When the strategy's drawdown from peak exceeds max_dd_threshold,
        all weights are scaled down by dd_scale_factor.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights.
        cumulative_returns : pd.Series
            Cumulative strategy returns (for drawdown calculation).

        Returns
        -------
        pd.DataFrame
            Drawdown-adjusted weights.
        """
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max.replace(0, 1.0)

        # Scale factor: 1.0 when DD < threshold, dd_scale when DD >= threshold
        dd_flag = (drawdown < -self.max_dd_threshold).astype(float)
        scale = 1.0 - dd_flag * (1.0 - self.dd_scale)

        adjusted = weights.multiply(scale, axis=0)
        return adjusted

    def construct_portfolio(
        self,
        mom_signal: pd.DataFrame,
        mr_signal: pd.DataFrame,
        regime_alpha: pd.Series,
        returns: pd.DataFrame,
        asset_class_map: Optional[Dict[str, list]] = None,
    ) -> pd.DataFrame:
        """
        Full portfolio construction pipeline.

        Steps:
            1. Blend momentum and mean-reversion signals
            2. Apply rebalancing frequency (hold between rebalance dates)
            3. Scale by volatility target
            4. Apply position limits
            5. Apply drawdown control overlay

        Parameters
        ----------
        mom_signal : pd.DataFrame
            Momentum composite signal.
        mr_signal : pd.DataFrame
            Mean-reversion composite signal.
        regime_alpha : pd.Series
            Regime blending parameter.
        returns : pd.DataFrame
            Daily returns.
        asset_class_map : dict, optional
            Asset class groupings for class-level limits.

        Returns
        -------
        pd.DataFrame
            Final portfolio weights (dates x assets).
        """
        # Step 1: Blend signals
        raw_signal = self.blend_signals(mom_signal, mr_signal, regime_alpha)

        # Step 2: Rebalancing -- only update weights at rebalance dates
        rebal_mask = np.arange(len(raw_signal)) % self.rebalance_freq == 0
        weights = raw_signal.copy()
        for i in range(len(weights)):
            if not rebal_mask[i] and i > 0:
                weights.iloc[i] = weights.iloc[i - 1]

        # Step 3: Normalize raw weights to sum of abs = 1 (fully invested)
        abs_sum = weights.abs().sum(axis=1).replace(0, 1.0)
        weights = weights.div(abs_sum, axis=0)

        # Step 4: Vol scaling
        weights = self.apply_vol_scaling(weights, returns)

        # Step 5: Position limits
        weights = self.apply_position_limits(weights, asset_class_map)

        # Step 6: Drawdown control (using portfolio returns so far)
        port_ret = (returns * weights.shift(1)).sum(axis=1)
        cum_ret = (1.0 + port_ret).cumprod()
        weights = self.apply_drawdown_control(weights, cum_ret)

        return weights
