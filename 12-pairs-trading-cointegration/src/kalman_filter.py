"""
Kalman Filter for Adaptive Hedge Ratio Estimation
==================================================

Implements a linear Kalman filter that treats the hedge ratio (beta)
as a latent state variable, allowing it to evolve over time. This
captures structural shifts in the pair relationship that a static
OLS regression would miss.

State-space formulation:
    State equation:       beta_t = beta_{t-1} + w_t,   w_t ~ N(0, Q)
    Observation equation: y_t = x_t * beta_t + v_t,    v_t ~ N(0, R)

where y_t = P_A,t (dependent price), x_t = [1, P_B,t] (regressors),
and beta_t = [alpha_t, beta_t] (time-varying intercept and hedge ratio).

The Kalman filter recursion:
    Predict: beta_{t|t-1} = beta_{t-1|t-1}
             P_{t|t-1} = P_{t-1|t-1} + Q
    Update:  K_t = P_{t|t-1} * x_t' * (x_t * P_{t|t-1} * x_t' + R)^{-1}
             beta_{t|t} = beta_{t|t-1} + K_t * (y_t - x_t * beta_{t|t-1})
             P_{t|t} = (I - K_t * x_t) * P_{t|t-1}

References:
    Kalman (1960), Montana et al. (2009), Elliott et al. (2005)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class KalmanHedgeRatio:
    """
    Kalman filter for time-varying hedge ratio estimation.

    Parameters
    ----------
    delta : float
        State transition covariance scaling. Controls how quickly the
        hedge ratio can change. Larger delta = more responsive but
        noisier estimates (default 1e-4).
    observation_noise : float
        Observation noise variance R. Controls the filter's trust in
        observations vs. predictions (default 1e-3).
    """

    def __init__(self, delta: float = 1e-4,
                 observation_noise: float = 1e-3):
        self.delta = delta
        self.R = observation_noise
        self.betas = None
        self.spreads = None
        self.kalman_gains = None

    def filter(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Run the Kalman filter to estimate time-varying hedge ratios.

        Parameters
        ----------
        y : pd.Series
            Dependent price series (stock A).
        x : pd.Series
            Independent price series (stock B).

        Returns
        -------
        dict
            betas: (T x 2) array [alpha_t, beta_t],
            spreads: pd.Series of Kalman-filtered residuals,
            forecast_errors: pd.Series,
            kalman_gains: (T x 2) array.
        """
        # Align series
        df = pd.DataFrame({"y": y, "x": x}).dropna()
        T = len(df)
        y_vals = df["y"].values
        x_vals = df["x"].values

        # State dimension: [alpha, beta]
        n_state = 2

        # Initialize state
        beta = np.zeros(n_state)  # [alpha_0, beta_0]
        P = np.eye(n_state) * 1.0  # Initial uncertainty

        # Transition covariance
        Q = np.eye(n_state) * self.delta
        R = self.R

        # Storage
        betas = np.zeros((T, n_state))
        errors = np.zeros(T)
        gains = np.zeros((T, n_state))
        spreads = np.zeros(T)

        for t in range(T):
            # Observation vector: x_t = [1, P_B,t]
            x_t = np.array([1.0, x_vals[t]])

            # Predict
            beta_pred = beta.copy()
            P_pred = P + Q

            # Innovation (forecast error)
            y_hat = x_t @ beta_pred
            e_t = y_vals[t] - y_hat
            errors[t] = e_t

            # Innovation covariance
            S = x_t @ P_pred @ x_t + R

            # Kalman gain
            K = P_pred @ x_t / S
            gains[t] = K

            # Update
            beta = beta_pred + K * e_t
            P = P_pred - np.outer(K, x_t) @ P_pred

            betas[t] = beta
            spreads[t] = e_t

        self.betas = pd.DataFrame(
            betas, index=df.index, columns=["alpha", "hedge_ratio"]
        )
        self.spreads = pd.Series(spreads, index=df.index, name="kalman_spread")
        self.kalman_gains = pd.DataFrame(
            gains, index=df.index, columns=["K_alpha", "K_beta"]
        )

        return {
            "betas": self.betas,
            "spreads": self.spreads,
            "forecast_errors": pd.Series(errors, index=df.index),
            "kalman_gains": self.kalman_gains,
            "final_hedge_ratio": betas[-1, 1],
            "final_alpha": betas[-1, 0],
        }

    def get_adaptive_spread(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """
        Compute the spread using the Kalman-filtered hedge ratio.

        S_t = y_t - alpha_t - beta_t * x_t

        This spread uses the filtered (time-varying) hedge ratio,
        producing a more stationary residual than static OLS.

        Parameters
        ----------
        y : pd.Series
            Dependent price series.
        x : pd.Series
            Independent price series.

        Returns
        -------
        pd.Series
            Adaptive (Kalman-filtered) spread.
        """
        if self.betas is None:
            self.filter(y, x)

        df = pd.DataFrame({"y": y, "x": x}).dropna()
        common = df.index.intersection(self.betas.index)
        alpha = self.betas.loc[common, "alpha"]
        beta = self.betas.loc[common, "hedge_ratio"]

        spread = df.loc[common, "y"] - alpha - beta * df.loc[common, "x"]
        spread.name = "kalman_adaptive_spread"
        return spread
