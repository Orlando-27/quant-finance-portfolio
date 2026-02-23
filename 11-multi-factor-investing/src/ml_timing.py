"""
Machine Learning Factor Timing & Regime Detection
==================================================

Extends traditional (static) factor investing with dynamic allocation
conditioned on macro-economic signals and market regimes.

Components:
    1. Feature Engineering: Constructs predictive features from factor
       returns (momentum, volatility, autocorrelation) and macro indicators
       (yield curve slope, credit spread, equity volatility).

    2. Factor Return Prediction: Gradient Boosting (XGBoost) and Random
       Forest trained to predict next-period factor returns. Walk-forward
       cross-validation prevents lookahead bias.

    3. Regime Detection: Hidden Markov Model (HMM) with Gaussian emissions
       identifies latent market states (e.g., risk-on, risk-off, transition)
       from factor return dynamics. Each state has distinct mean and
       covariance, enabling state-dependent factor allocation.

    4. Signal Combination: Ensemble of ML predictions and regime probabilities
       to generate dynamic factor weight recommendations.

References:
    Gu, Kelly & Xiu (2020), Ang & Timmermann (2012),
    Hamilton (1989), Asness et al. (2017)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Conditional import for HMM (may not be available in all environments)
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class FeatureEngineering:
    """
    Constructs predictive features for factor timing models.

    Features capture time-series properties of factor returns (momentum,
    volatility clustering, mean-reversion) and cross-factor dynamics
    (correlation regime shifts, relative strength).

    Parameters
    ----------
    lookback_windows : list of int
        Windows for rolling statistics (default [3, 6, 12] months).
    """

    def __init__(self, lookback_windows: Optional[List[int]] = None):
        self.windows = lookback_windows or [3, 6, 12]

    def build_features(self, factor_returns: pd.DataFrame,
                       macro_indicators: Optional[pd.DataFrame] = None
                       ) -> pd.DataFrame:
        """
        Generate feature matrix from factor returns and optional macro data.

        Features per factor f and window w:
            - Rolling mean (momentum signal): mean(F_f, w)
            - Rolling volatility (risk regime): std(F_f, w)
            - Rolling Sharpe: mean/std ratio
            - Rolling skewness (tail risk signal)
            - Autocorrelation at lag 1 (mean-reversion vs persistence)
            - Z-score of current return relative to rolling distribution

        Cross-factor features:
            - Rolling correlation between factor pairs
            - Factor dispersion (cross-sectional std of factor returns)

        Parameters
        ----------
        factor_returns : pd.DataFrame
            (T x K) factor return series.
        macro_indicators : pd.DataFrame, optional
            (T x M) macro-economic indicators.

        Returns
        -------
        pd.DataFrame
            (T x P) feature matrix (rows with NaN from rolling windows dropped).
        """
        features = {}
        fnames = factor_returns.columns

        for w in self.windows:
            for f in fnames:
                s = factor_returns[f]
                prefix = f"{f}_w{w}"
                features[f"{prefix}_mom"] = s.rolling(w).mean()
                features[f"{prefix}_vol"] = s.rolling(w).std()
                r_mean = s.rolling(w).mean()
                r_std = s.rolling(w).std().replace(0, np.nan)
                features[f"{prefix}_sharpe"] = r_mean / r_std
                features[f"{prefix}_skew"] = s.rolling(w).skew()
                features[f"{prefix}_autocorr"] = s.rolling(w).apply(
                    lambda x: pd.Series(x).autocorr(lag=1), raw=False
                )
                features[f"{prefix}_zscore"] = (s - r_mean) / r_std

            # Cross-factor features
            roll_corr = factor_returns.rolling(w).corr()
            for i, f1 in enumerate(fnames):
                for f2 in fnames[i + 1:]:
                    key = f"corr_{f1}_{f2}_w{w}"
                    corr_series = pd.Series(index=factor_returns.index, dtype=float)
                    for t in factor_returns.index[w - 1:]:
                        try:
                            corr_series[t] = roll_corr.loc[(t, f1), f2]
                        except (KeyError, TypeError):
                            pass
                    features[key] = corr_series

            # Factor dispersion
            features[f"dispersion_w{w}"] = factor_returns.rolling(w).std().mean(axis=1)

        feat_df = pd.DataFrame(features, index=factor_returns.index)

        # Macro features (passthrough with lags)
        if macro_indicators is not None:
            for col in macro_indicators.columns:
                feat_df[f"macro_{col}"] = macro_indicators[col]
                feat_df[f"macro_{col}_lag1"] = macro_indicators[col].shift(1)
                feat_df[f"macro_{col}_chg"] = macro_indicators[col].diff()

        return feat_df.dropna()


class FactorTimingML:
    """
    Machine learning models for predicting factor returns.

    Implements walk-forward (expanding window) cross-validation to ensure
    strict temporal ordering and prevent lookahead bias.

    Parameters
    ----------
    model_type : str
        'rf' for Random Forest, 'xgb' for XGBoost, 'ensemble' for both.
    n_splits : int
        Number of folds for TimeSeriesSplit cross-validation.
    """

    def __init__(self, model_type: str = "ensemble", n_splits: int = 5):
        self.model_type = model_type
        self.n_splits = n_splits
        self.models = {}
        self.feature_importance = {}
        self.cv_results = {}
        self.scaler = StandardScaler()

    def _get_models(self) -> Dict:
        """Initialize model instances with calibrated hyperparameters."""
        models = {}
        if self.model_type in ("rf", "ensemble"):
            models["RandomForest"] = RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                max_features="sqrt", n_jobs=-1, random_state=42
            )
        if self.model_type in ("xgb", "ensemble"):
            models["XGBoost"] = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
        return models

    def walk_forward_cv(self, features: pd.DataFrame,
                        target: pd.Series) -> Dict:
        """
        Walk-forward cross-validation for factor return prediction.

        Uses expanding training window with fixed test periods.
        No future information leaks into training at any point.

        Parameters
        ----------
        features : pd.DataFrame
            (T x P) feature matrix.
        target : pd.Series
            (T,) next-period factor return to predict.

        Returns
        -------
        dict
            Keys: model names. Values: dict with 'predictions', 'actuals',
            'rmse', 'r2', 'ic' (information coefficient = rank correlation).
        """
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        models = self._get_models()
        results = {}

        for name, model in models.items():
            all_preds = []
            all_actuals = []

            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                X_te_s = scaler.transform(X_test)

                model.fit(X_tr_s, y_train)
                preds = model.predict(X_te_s)

                all_preds.extend(preds)
                all_actuals.extend(y_test.values)

            preds_arr = np.array(all_preds)
            actuals_arr = np.array(all_actuals)

            rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr))
            r2 = r2_score(actuals_arr, preds_arr)
            # Information coefficient: Spearman rank correlation
            from scipy.stats import spearmanr
            ic, ic_pval = spearmanr(preds_arr, actuals_arr)

            results[name] = {
                "predictions": preds_arr,
                "actuals": actuals_arr,
                "rmse": rmse,
                "r2": r2,
                "ic": ic,
                "ic_pval": ic_pval,
            }

        self.cv_results = results
        return results

    def fit_predict(self, features: pd.DataFrame,
                    target: pd.Series,
                    train_end: int) -> Tuple[Dict, pd.DataFrame]:
        """
        Fit models on training data and predict full series.

        Parameters
        ----------
        features : pd.DataFrame
            Full feature matrix.
        target : pd.Series
            Full target series.
        train_end : int
            Index position marking end of training period.

        Returns
        -------
        predictions : dict
            Model name -> prediction array for test period.
        importance : pd.DataFrame
            Feature importance from each model.
        """
        common = features.index.intersection(target.index)
        X, y = features.loc[common], target.loc[common]

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test = X.iloc[train_end:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        models = self._get_models()
        predictions = {}
        importance_dict = {}

        for name, model in models.items():
            model.fit(X_tr_s, y_train)
            predictions[name] = pd.Series(
                model.predict(X_te_s), index=X_test.index, name=name
            )
            if hasattr(model, "feature_importances_"):
                importance_dict[name] = pd.Series(
                    model.feature_importances_, index=features.columns
                ).sort_values(ascending=False)

            self.models[name] = model

        self.feature_importance = importance_dict
        return predictions, pd.DataFrame(importance_dict)


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Identifies latent states from factor return dynamics. Each state has
    a distinct multivariate Gaussian distribution (mean vector and
    covariance matrix), capturing regime-dependent factor behavior.

    Common interpretation:
        State 0: Low volatility, positive mean (risk-on)
        State 1: High volatility, negative mean (risk-off)
        State 2: Transition / mixed (if n_states=3)

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 2: risk-on / risk-off).
    n_iter : int
        Maximum EM iterations for HMM fitting.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self.states = None
        self.state_means = None
        self.state_covars = None
        self.transition_matrix = None

    def fit_predict(self, factor_returns: pd.DataFrame) -> pd.Series:
        """
        Fit HMM and decode most likely state sequence.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            (T x K) factor returns.

        Returns
        -------
        pd.Series
            Decoded state labels aligned with factor_returns index.
        """
        if not HMM_AVAILABLE:
            # Fallback: simple volatility regime (above/below median)
            vol = factor_returns.std(axis=1).rolling(6).mean()
            med_vol = vol.median()
            states = (vol > med_vol).astype(int)
            self.states = states
            return states

        X = factor_returns.values
        model = GaussianHMM(
            n_components=self.n_states, covariance_type="full",
            n_iter=self.n_iter, random_state=42
        )
        model.fit(X)

        states = model.predict(X)

        # Sort states by mean return (state 0 = best environment)
        mean_rets = [factor_returns.iloc[states == s].mean().mean()
                     for s in range(self.n_states)]
        state_order = np.argsort(mean_rets)[::-1]
        remap = {old: new for new, old in enumerate(state_order)}
        states = np.array([remap[s] for s in states])

        self.model = model
        self.states = pd.Series(states, index=factor_returns.index, name="regime")
        self.state_means = {
            remap[s]: model.means_[s] for s in range(self.n_states)
        }
        self.state_covars = {
            remap[s]: model.covars_[s] for s in range(self.n_states)
        }
        self.transition_matrix = model.transmat_[np.ix_(state_order, state_order)]

        return self.states

    def get_state_statistics(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics for each regime state.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            (T x K) factor returns.

        Returns
        -------
        pd.DataFrame
            MultiIndex (state, statistic) x factor summary.
        """
        if self.states is None:
            self.fit_predict(factor_returns)

        records = []
        for s in range(self.n_states):
            mask = self.states == s
            sub = factor_returns.loc[mask.values]
            n_obs = len(sub)
            pct = n_obs / len(factor_returns) * 100
            ann_ret = sub.mean() * 12
            ann_vol = sub.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol.replace(0, np.nan)

            for stat_name, stat_vals in [("Ann Return", ann_ret),
                                          ("Ann Vol", ann_vol),
                                          ("Sharpe", sharpe)]:
                for f in factor_returns.columns:
                    records.append({
                        "State": s, "Statistic": stat_name,
                        "Factor": f, "Value": stat_vals[f],
                        "Observations": n_obs, "Pct Time": pct,
                    })

        return pd.DataFrame(records)
