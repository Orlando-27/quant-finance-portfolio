"""
PCA Factor Analysis of Yield Curves
=====================================
Decomposes yield changes into level, slope, and curvature factors
following Litterman & Scheinkman (1991).

References:
    Litterman, R. & Scheinkman, J. (1991). JFI 1(1), 54-61.
    Joslin, S., Singleton, K.J. & Zhu, H. (2011). RFS 24(3), 926-970.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class YieldCurvePCA:
    """
    PCA decomposition of yield curve changes.

    Attributes
    ----------
    n_components : int
        Number of principal components to retain (default 3).
    scale : bool
        Whether to standardise yields before PCA. Standardising is
        appropriate when tenors have very different variances.
    """

    FACTOR_LABELS = {0: "Level (PC1)", 1: "Slope (PC2)", 2: "Curvature (PC3)"}

    def __init__(self, n_components: int = 3, scale: bool = True):
        self.n_components = n_components
        self.scale        = scale
        self._scaler      = StandardScaler() if scale else None
        self._pca         = PCA(n_components=n_components)
        self.fitted_      = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, yields_panel: pd.DataFrame) -> "YieldCurvePCA":
        """
        Fit PCA on yield levels (or changes if differentiated externally).

        Parameters
        ----------
        yields_panel : pd.DataFrame  (dates x tenors).

        Returns
        -------
        self (fitted)
        """
        X = yields_panel.dropna().values.astype(float)
        if self.scale:
            X = self._scaler.fit_transform(X)
        self._pca.fit(X)
        self.fitted_         = True
        self.tenors_         = yields_panel.columns.astype(float).tolist()
        self.explained_var_  = self._pca.explained_variance_ratio_
        self.loadings_       = pd.DataFrame(
            self._pca.components_.T,
            index   = self.tenors_,
            columns = [f"PC{i+1}" for i in range(self.n_components)],
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, yields_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Project yield panel onto principal components (factor scores).

        Returns
        -------
        pd.DataFrame  (dates x n_components) factor time series.
        """
        if not self.fitted_:
            raise RuntimeError("Fit PCA first.")
        clean = yields_panel.dropna()
        X     = clean.values.astype(float)
        if self.scale:
            X = self._scaler.transform(X)
        scores = self._pca.transform(X)
        cols   = [self.FACTOR_LABELS.get(i, f"PC{i+1}")
                  for i in range(self.n_components)]
        return pd.DataFrame(scores, index=clean.index, columns=cols)

    # ------------------------------------------------------------------
    # Fit + transform shortcut
    # ------------------------------------------------------------------
    def fit_transform(self, yields_panel: pd.DataFrame) -> pd.DataFrame:
        return self.fit(yields_panel).transform(yields_panel)

    # ------------------------------------------------------------------
    # Variance explained
    # ------------------------------------------------------------------
    def variance_explained(self) -> pd.Series:
        """Return cumulative variance explained by each component."""
        if not self.fitted_:
            raise RuntimeError("Fit PCA first.")
        labels = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.Series(
            np.cumsum(self.explained_var_),
            index=labels,
            name="cum_var_explained",
        )

    # ------------------------------------------------------------------
    # Reconstruct yields from k components
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        yields_panel: pd.DataFrame,
        n_components : int | None = None,
    ) -> pd.DataFrame:
        """
        Reconstruct yields using only the first k components.

        Useful for assessing how much of the yield curve is explained
        by level, slope, and curvature alone.
        """
        if not self.fitted_:
            raise RuntimeError("Fit PCA first.")
        k = n_components or self.n_components
        clean = yields_panel.dropna()
        X     = clean.values.astype(float)
        if self.scale:
            X = self._scaler.transform(X)

        # Project onto first k components and back
        scores = self._pca.transform(X)[:, :k]
        W      = self._pca.components_[:k]
        X_rec  = scores @ W + self._pca.mean_

        if self.scale:
            X_rec = self._scaler.inverse_transform(X_rec)

        return pd.DataFrame(X_rec, index=clean.index,
                            columns=yields_panel.columns)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Print-ready summary of PCA results."""
        if not self.fitted_:
            raise RuntimeError("Fit PCA first.")
        rows = []
        cum  = 0.0
        for i, ev in enumerate(self.explained_var_):
            cum += ev
            rows.append({
                "Component"         : f"PC{i+1}",
                "Label"             : self.FACTOR_LABELS.get(i, f"PC{i+1}"),
                "Eigenvalue"        : self._pca.explained_variance_[i],
                "Var Explained (%)" : ev * 100,
                "Cum Var (%)"       : cum * 100,
            })
        return pd.DataFrame(rows)
