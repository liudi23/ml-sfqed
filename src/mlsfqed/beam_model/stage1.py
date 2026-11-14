"""Stage 1: predict the electron spectrum dW/dE from shot diagnostics, with uncertainty.

Design
- Inputs: laser energy, 6 far-field features, PCA of the top-view scattering profile and of the density
  profile (fitted on training shots only). This mirrors the paper's 18 VAE-compressed inputs.
- Target: dW/dE [J/GeV] on 200 bins, compressed to its leading PCA components.
- Models: bagged ensembles (each member sees a random 50% of the training shots, as in the paper)
  of LightGBM and of small MLPs, plus ridge and train-mean baselines.
- Uncertainty: ensemble spread (epistemic), calibrated with out-of-bag residuals on the training set
  (jackknife+-after-bootstrap style), so intervals cover aleatoric shot-to-shot noise too.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from mlsfqed.data.gemini import physical_spectra


@dataclass
class Featuriser:
    n_iz: int = 5
    n_ne: int = 5

    def fit(self, d, idx):
        self.p_iz = PCA(self.n_iz).fit(d["I_z_data"][idx])
        self.p_ne = PCA(self.n_ne).fit(d["n_e_z_data"][idx])
        self.sc = StandardScaler().fit(self._raw(d)[idx])
        return self

    def _raw(self, d):
        return np.column_stack([d["laser_E"], d["f40_data"],
                                self.p_iz.transform(d["I_z_data"]), self.p_ne.transform(d["n_e_z_data"])])

    def transform(self, d, idx):
        return self.sc.transform(self._raw(d)[idx])

    def names(self):
        return (["laser_E", "f40_I0", "f40_x", "f40_y", "f40_sx", "f40_sy", "f40_theta"]
                + [f"scatter_pc{i+1}" for i in range(self.n_iz)] + [f"density_pc{i+1}" for i in range(self.n_ne)])


class BaggedSpectrumModel:
    """Ensemble over random subsamples (default 50%); each member predicts the target PCA coefficients."""

    def __init__(self, kind: str, n_members: int = 20, n_pc: int = 10, seed: int = 0, frac: float = 0.5):
        self.kind, self.n_members, self.n_pc, self.seed, self.frac = kind, n_members, n_pc, seed, frac

    def _member(self, s):
        if self.kind == "lgbm":
            return [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      reg_lambda=1.0, random_state=s, n_jobs=1, verbose=-1) for _ in range(self.n_pc)]
        if self.kind == "mlp":
            return MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-2, learning_rate_init=3e-3,
                                max_iter=3000, early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=50, random_state=s)
        raise ValueError(self.kind)

    def fit(self, X, Y):
        self.pca = PCA(self.n_pc).fit(Y)
        C = self.pca.transform(Y)
        self.csc = StandardScaler().fit(C)
        Cs = self.csc.transform(C)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.members, self.inbag = [], np.zeros((self.n_members, n), bool)
        for m in range(self.n_members):
            ib = rng.choice(n, int(n * self.frac), replace=False)
            self.inbag[m, ib] = True
            mod = self._member(self.seed * 1000 + m)
            if isinstance(mod, list):
                for k, r in enumerate(mod):
                    r.fit(X[ib], Cs[ib, k])
            else:
                mod.fit(X[ib], Cs[ib])
            self.members.append(mod)
        self.Y_train, self.X_train = Y, X
        return self

    def _predict_member(self, mod, X):
        Cs = np.column_stack([r.predict(X) for r in mod]) if isinstance(mod, list) else mod.predict(X)
        return self.pca.inverse_transform(self.csc.inverse_transform(Cs))

    def predict_members(self, X):
        """(n_members, n_shots, n_bins)"""
        return np.stack([self._predict_member(m, X) for m in self.members])

    def oob_predictions(self):
        """Out-of-bag mean and spread for each training shot (members that did not see it)."""
        P = self.predict_members(self.X_train)
        oob = ~self.inbag[:, :, None]
        cnt = oob.sum(0)
        mean = (P * oob).sum(0) / cnt
        sd = np.sqrt(((P - mean) ** 2 * oob).sum(0) / np.maximum(cnt - 1, 1))
        return mean, sd


def beam_energy(Y, E_MeV):
    return np.trapezoid(Y, E_MeV / 1e3, axis=-1)


def interval_from_oob(model, E_MeV, alpha=0.1):
    """Calibrated scale factor q so that mean +/- q*sd covers (1-alpha) of OOB beam energies."""
    P = model.predict_members(model.X_train)
    oob = ~model.inbag
    W = beam_energy(P, E_MeV)
    Wm = np.array([W[oob[:, i], i].mean() for i in range(W.shape[1])])
    Ws = np.array([W[oob[:, i], i].std(ddof=1) for i in range(W.shape[1])])
    Wt = beam_energy(model.Y_train, E_MeV)
    resid = np.abs(Wt - Wm)
    # normalised score; floor the spread so near-zero ensemble disagreement doesn't explode the score
    floor = np.median(Ws)
    score = resid / np.maximum(Ws, floor)
    n = len(score)
    q = np.quantile(score, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return q, floor


def ridge_predict(Xtr, Ytr, Xte, n_pc=10):
    pca = PCA(n_pc).fit(Ytr)
    r = Ridge(alpha=1.0).fit(Xtr, pca.transform(Ytr))
    return pca.inverse_transform(r.predict(Xte))


def crps_ensemble(samples, y):
    """CRPS for an ensemble: E|X-y| - 0.5 E|X-X'|, per shot. samples: (M, N)."""
    t1 = np.abs(samples - y[None]).mean(0)
    t2 = np.abs(samples[:, None, :] - samples[None, :, :]).mean((0, 1))
    return t1 - 0.5 * t2


def lgbm_beam_energy_shap(X, w, names, seed=0):
    """Single LightGBM on total beam energy; mean |SHAP| per feature (LightGBM's built-in TreeSHAP)."""
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1).fit(X, w)
    contrib = m.predict(X, pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(0)
    order = np.argsort(imp)[::-1]
    return [(names[i], float(imp[i])) for i in order]


__all__ = [
    "BaggedSpectrumModel",
    "Featuriser",
    "beam_energy",
    "crps_ensemble",
    "interval_from_oob",
    "lgbm_beam_energy_shap",
    "physical_spectra",
    "ridge_predict",
]
