"""Surrogate models for the Stage 2 observables.

- TreeSurrogate: one LightGBM model per observable (baseline).
- MLPEnsemble: a deep ensemble of small MLPs predicting all observables at once (standardised targets);
  the spread across members is the surrogate's own uncertainty.
- SpectrumSurrogate: binned electron/photon spectra via PCA coefficients predicted by an MLP ensemble.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TREE = {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 10,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, "n_jobs": 1}


class TreeSurrogate:
    def __init__(self, quick=False, seed=0):
        self.params = dict(TREE, random_state=seed)
        if quick:
            self.params["n_estimators"] = 150

    def fit(self, X, Y):
        self.cols = list(Y.columns)
        self.models = {c: lgb.LGBMRegressor(**self.params).fit(X, Y[c]) for c in self.cols}
        return self

    def predict(self, X):
        return np.column_stack([self.models[c].predict(X) for c in self.cols])


class MLPEnsemble:
    def __init__(self, n_members=5, hidden=(128, 128), quick=False, seed=0):
        self.n_members = 2 if quick else n_members
        self.hidden = (32, 32) if quick else hidden
        self.max_iter = 300 if quick else 3000
        self.seed = seed

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.xs, self.ys = StandardScaler().fit(X), StandardScaler().fit(Y)
        Xs, Ys = self.xs.transform(X), self.ys.transform(Y)
        self.members = [MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-4, learning_rate_init=1e-3,
                                     batch_size=64, max_iter=self.max_iter, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=40, random_state=self.seed + m)
                        .fit(Xs, Ys) for m in range(self.n_members)]
        return self

    def predict_members(self, X):
        Xs = self.xs.transform(np.asarray(X, float))
        return np.stack([self.ys.inverse_transform(m.predict(Xs).reshape(len(Xs), -1)) for m in self.members])

    def predict(self, X):
        P = self.predict_members(X)
        return P.mean(0), P.std(0, ddof=1) if len(P) > 1 else np.zeros_like(P[0])


class SpectrumSurrogate:
    """Spectra (rows = runs, cols = bins) -> PCA on sqrt(spectrum) -> MLP ensemble on the coefficients."""

    def __init__(self, n_pc=12, quick=False, seed=0):
        self.n_pc, self.quick, self.seed = n_pc, quick, seed

    def fit(self, X, S):
        T = np.sqrt(np.clip(S, 0, None))
        self.pca = PCA(self.n_pc, svd_solver="full").fit(T)
        self.net = MLPEnsemble(quick=self.quick, seed=self.seed).fit(X, self.pca.transform(T))
        return self

    def predict(self, X):
        C, _ = self.net.predict(X)
        return np.clip(self.pca.inverse_transform(C), 0, None) ** 2
