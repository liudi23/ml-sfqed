"""Two-part beam model and beam sampler.

dW/dE = W * s(E), with
  W     total beam energy [J]  - bagged LightGBM on log W, from downsampled raw diagnostic profiles
  s(E)  spectral shape, unit integral over E in GeV - bagged LightGBM on 10 shape-PCA coefficients

Each part is calibrated with out-of-bag (OOB) residuals on the training shots, so samples include
shot-to-shot noise that the diagnostics cannot explain, not only ensemble disagreement.

The sampler draws, for a shot's diagnostics x:
  log W = mu(x) + r * s(x)     r = OOB normalised residual of a random training shot (non-Gaussian tails kept)
  shape = member-mean coefficients + that same shot's OOB coefficient residual, clipped >= 0, renormalised
Energy and shape residuals are drawn as a pair from the same training shot, which keeps their
correlation (energy and shape errors are anti-correlated, r ~ -0.3 to -0.5).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler

from mlsfqed.beam_model.stage1 import BaggedSpectrumModel, beam_energy

E_CHARGE = 1.602176634e-19


class RawFeaturiser:
    """Laser energy, 6 far-field features, scattering profile every `iz_step` points, density every `ne_step`."""

    def __init__(self, iz_step: int = 5, ne_step: int = 10):
        self.iz_step, self.ne_step = iz_step, ne_step

    def _raw(self, d, idx):
        return np.column_stack([d["laser_E"][idx], d["f40_data"][idx],
                                d["I_z_data"][idx][:, :: self.iz_step], d["n_e_z_data"][idx][:, :: self.ne_step]])

    def fit(self, d, idx):
        self.sc = StandardScaler().fit(self._raw(d, idx))
        self.n_iz = d["I_z_data"][:, :: self.iz_step].shape[1]
        self.n_ne = d["n_e_z_data"][:, :: self.ne_step].shape[1]
        self.z_iz = d["z_mm"][:: self.iz_step]
        self.z_ne = d["lmi_z_mm"][:: self.ne_step]
        return self

    def transform(self, d, idx):
        return self.sc.transform(self._raw(d, idx))

    def names(self):
        return (["laser_E", "f40_I0", "f40_x", "f40_y", "f40_sx", "f40_sy", "f40_theta"]
                + [f"scatter_z{z:.1f}mm" for z in self.z_iz] + [f"density_z{z:.1f}mm" for z in self.z_ne])


# n_jobs=1: on ~600 shots multi-threading is far slower, and LightGBM 4.x defaults to all cores
LGBM_W = {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 8, "min_child_samples": 10,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5, "reg_lambda": 1.0,
          "verbose": -1, "n_jobs": 1}


class EnergyEnsemble:
    """Bagged LightGBM on one scalar target (log beam energy, peak parameters), members on `frac` of shots."""

    def __init__(self, n_members=20, frac=0.8, seed=0):
        self.n_members, self.frac, self.seed = n_members, frac, seed

    def fit(self, X, logW):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.inbag = np.zeros((self.n_members, n), bool)
        self.members = []
        for m in range(self.n_members):
            ib = rng.choice(n, int(n * self.frac), replace=False)
            self.inbag[m, ib] = True
            self.members.append(lgb.LGBMRegressor(random_state=self.seed * 1000 + m, **LGBM_W).fit(X[ib], logW[ib]))
        P = self.predict_members(X)
        oob = ~self.inbag
        has = oob.sum(0) > 1                              # shots left out by >= 2 members
        self.oob_mu = np.array([P[oob[:, i], i].mean() if has[i] else np.nan for i in range(n)])
        self.oob_sd = np.array([P[oob[:, i], i].std(ddof=1) if oob[:, i].sum() > 1 else np.nan for i in range(n)])
        self.sd_floor = np.nanmedian(self.oob_sd)
        sd = np.maximum(np.nan_to_num(self.oob_sd, nan=self.sd_floor), self.sd_floor)
        self.has_oob = has
        self.oob_sd_used = sd
        self.scores_all = (logW - self.oob_mu) / sd       # signed, normalised OOB residuals (nan where no OOB)
        self.scores = self.scores_all[has]
        return self

    def predict_members(self, X):
        return np.stack([m.predict(X) for m in self.members])

    def predict(self, X):
        P = self.predict_members(X)
        return P.mean(0), np.maximum(P.std(0, ddof=1), self.sd_floor)

    def interval(self, X, level=0.9):
        mu, sd = self.predict(X)
        lo, hi = np.quantile(self.scores, [(1 - level) / 2, (1 + level) / 2])
        return np.exp(mu), np.exp(mu + lo * sd), np.exp(mu + hi * sd)


class GPBlendEnsemble(EnergyEnsemble):
    """Bagged LightGBM averaged with a Gaussian process (RBF kernel with one length scale per feature).

    Trees give step-like predictions and the GP smooth ones, so their errors differ and the average beats
    either: peak position error 35.7 -> 32.7 MeV on held-out run 13. The member spread (sd) still comes from
    the trees. For training shots the GP part of the out-of-bag mean is an out-of-fold prediction, so the
    residuals the sampler draws from stay honest."""

    def __init__(self, n_members=20, frac=0.8, seed=0, n_folds=5):
        super().__init__(n_members=n_members, frac=frac, seed=seed)
        self.n_folds = n_folds

    def _gp(self, d):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        return GaussianProcessRegressor(ConstantKernel() * RBF(np.full(d, 3.0)) + WhiteKernel(),
                                        normalize_y=True, random_state=self.seed)

    def fit(self, X, y):
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        super().fit(X, y)
        folds = np.random.default_rng(self.seed + 7).permutation(len(X)) % self.n_folds
        gp_oof = np.empty(len(X))
        with warnings.catch_warnings():
            # a length scale at its upper bound means that feature carries no information: expected, harmless
            warnings.simplefilter("ignore", ConvergenceWarning)
            for f in range(self.n_folds):
                m = folds == f
                gp_oof[m] = self._gp(X.shape[1]).fit(X[~m], y[~m]).predict(X[m])
            self.gp = self._gp(X.shape[1]).fit(X, y)
        self.oob_mu = 0.5 * (self.oob_mu + gp_oof)
        self.scores_all = (y - self.oob_mu) / self.oob_sd_used
        self.scores = self.scores_all[self.has_oob]
        return self

    def predict(self, X):
        mu, sd = super().predict(X)
        return 0.5 * (mu + self.gp.predict(X)), sd


class MarginalEnsemble:
    """A target the diagnostics cannot predict (e.g. peak width, held-out R^2 < 0): its prediction is the
    training mean, and samples draw from its training distribution. Same interface as EnergyEnsemble."""

    def __init__(self, n_members=20, frac=0.8, seed=0):
        self.seed = seed

    def fit(self, X, y):
        n = len(y)
        self.mu, self.sd = float(np.mean(y)), float(np.std(y, ddof=1))
        self.sd_floor = self.sd
        # leave-one-out mean, so a training shot's residual is out of sample
        self.oob_mu = (y.sum() - y) / (n - 1)
        self.oob_sd_used = np.full(n, self.sd)
        self.has_oob = np.ones(n, bool)
        self.scores_all = (y - self.oob_mu) / self.sd
        self.scores = self.scores_all
        return self

    def predict(self, X):
        n = len(X)
        return np.full(n, self.mu), np.full(n, self.sd)


# ---------------------------------------------------------------- per-bin band calibration
def _cqr_scores(samples, Y, level=0.9):
    """Per-bin nonconformity of Y against a sampled band (scaled conformalised quantile regression)."""
    a = (1 - level) / 2
    med = np.median(samples, axis=1)
    lo, hi = np.quantile(samples, [a, 1 - a], axis=1)
    eps = 1e-3 * np.abs(Y).max()
    return np.maximum((lo - Y) / (med - lo + eps), (Y - hi) / (hi - med + eps))


def calibration_factors(samples, Y, level=0.9, smooth=9):
    """Per-bin factor k_b: stretching sample deviations from the median by (1 + q_b) gives `level` coverage
    on the calibration shots. A running median over `smooth` bins removes bin-to-bin noise."""
    e = _cqr_scores(samples, Y, level)
    n = len(e)
    q = np.quantile(e, min(1.0, np.ceil((n + 1) * level) / n), axis=0)
    pad = smooth // 2
    qp = np.pad(q, pad, mode="edge")
    q = np.array([np.median(qp[i:i + smooth]) for i in range(len(q))])
    return 1.0 + q


def apply_calibration(samples, k):
    med = np.median(samples, axis=1, keepdims=True)
    return np.clip(med + k * (samples - med), 0, None)


class _CalibratedMixin:
    """Adds OOB-based per-bin calibration. Subclasses provide _oob_train_samples(n, rng)."""

    def calibrate(self, level=0.9, n_samples=100, seed=1):
        rng = np.random.default_rng(seed)
        samp, Yc = self._oob_train_samples(n_samples, rng)
        self.k_bins = calibration_factors(samp, Yc, level)
        return self

    def sample_calibrated(self, X, n_samples=100, seed=0):
        return apply_calibration(self.sample(X, n_samples, seed), self.k_bins)


def _draw_other(rng, n_pool, i, size):
    j = rng.integers(n_pool - 1, size=size)
    return j + (j >= i)                                   # never the shot's own residual


class TwoPartBeamModel(_CalibratedMixin):
    def __init__(self, E_MeV, n_members=20, seed=0, n_pc=10):
        self.E = np.asarray(E_MeV)
        self.energy = EnergyEnsemble(n_members=n_members, seed=seed)
        self.shape = BaggedSpectrumModel("lgbm", n_members=n_members, n_pc=n_pc, seed=seed, frac=0.8)

    def fit(self, X, Y):
        W = beam_energy(Y, self.E)
        self.energy.fit(X, np.log(W))
        S = Y / W[:, None]
        self.shape.fit(X, S)
        # OOB shape residuals in (scaled) PCA-coefficient space, for the sampler
        Pc = np.stack([self._member_coeffs(m, X) for m in self.shape.members])
        oob = ~self.shape.inbag
        has = oob.sum(0) > 1
        true_c = self.shape.csc.transform(self.shape.pca.transform(S))
        resid = np.full_like(true_c, np.nan)
        for i in np.flatnonzero(has):
            resid[i] = true_c[i] - Pc[oob[:, i], i].mean(0)
        # keep energy and shape residuals paired by training shot, so their correlation is preserved
        both = has & self.energy.has_oob
        self.joint_scores = self.energy.scores_all[both]
        self.shape_resid = resid[both]
        mean_c = np.full_like(true_c, np.nan)
        for i in np.flatnonzero(has):
            mean_c[i] = Pc[oob[:, i], i].mean(0)
        self._both, self._oob_shape_mean, self.Y_train = both, mean_c, Y
        self._W_train = W
        return self

    def _member_coeffs(self, mod, X):
        return np.column_stack([r.predict(X) for r in mod]) if isinstance(mod, list) else mod.predict(X)

    def _shape_from_coeffs(self, Cs):
        lead = Cs.shape[:-1]
        Cs = Cs.reshape(-1, Cs.shape[-1])
        S = self.shape.pca.inverse_transform(self.shape.csc.inverse_transform(Cs)).reshape(*lead, -1)
        S = np.clip(S, 0, None)
        return S / beam_energy(S, self.E)[..., None]

    def predict(self, X):
        """Point prediction: dW/dE [J/GeV], beam energy W [J]."""
        mu, _ = self.energy.predict(X)
        Cs = np.mean([self._member_coeffs(m, X) for m in self.shape.members], 0)
        W = np.exp(mu)
        return self._shape_from_coeffs(Cs) * W[:, None], W

    def sample(self, X, n_samples=100, seed=0):
        """Draw spectra for each shot: array (n_shots, n_samples, n_bins) in J/GeV."""
        rng = np.random.default_rng(seed)
        mu, sd = self.energy.predict(X)
        j = rng.integers(len(self.joint_scores), size=(len(X), n_samples))   # one training shot per draw
        W = np.exp(mu[:, None] + self.joint_scores[j] * sd[:, None])
        Cm = np.mean([self._member_coeffs(m, X) for m in self.shape.members], 0)
        res = self.shape_resid[j]
        S = self._shape_from_coeffs(Cm[:, None, :] + res)
        return S * W[..., None]


    def _oob_train_samples(self, n_samples, rng):
        """Samples for each training shot from OOB centres + other shots' paired residuals."""
        E = self.E
        both = self._both
        idx = np.flatnonzero(both)
        out = np.empty((len(idx), n_samples, len(E)))
        for r, i in enumerate(idx):
            j = _draw_other(rng, len(idx), r, n_samples)
            logW = self.energy.oob_mu[i] + self.joint_scores[j] * self.energy.oob_sd_used[i]
            C = self._oob_shape_mean[i][None, :] + self.shape_resid[j]
            out[r] = self._shape_from_coeffs(C) * np.exp(logW)[:, None]
        return out, self.Y_train[idx]


def to_electron_spectrum(Y, E_MeV):
    """dW/dE [J/GeV] -> dN/dE [electrons/MeV] and charge [pC] in the spectrometer window."""
    EJ = np.asarray(E_MeV) * 1e6 * E_CHARGE
    dNdE_perGeV = Y / EJ
    dNdE = dNdE_perGeV / 1e3
    Q_pC = np.trapezoid(dNdE_perGeV, np.asarray(E_MeV) / 1e3, axis=-1) * E_CHARGE * 1e12
    return dNdE, Q_pC


# ---------------------------------------------------------------- peak-aware model
class PeakBeamModel(_CalibratedMixin):
    """dW/dE = W * (continuum(E) + peak(A, position, sigma)), renormalised.

    The peak is a Gaussian (shape="gauss", position = centre) or the physics-based square-root edge
    (shape="sqrt_edge", position = edge energy E_max; adopted, see peaks.sqrt_edge).
    W, peak height (log A), position and width (log sigma) each come from a bagged LightGBM ensemble;
    the continuum (spectrum minus fitted peak) from 10 PCA coefficients. All ensembles share the same
    bags, so out-of-bag residuals of every part are paired by training shot and drawn together.
    """

    PEAK_PARAMS = ("log_height", "position", "log_width")

    def __init__(self, E_MeV, n_members=20, seed=0, n_pc=10, gp_blend=(), shape="gauss", marginal=()):
        """gp_blend: peak parameters (names from PEAK_PARAMS) whose LightGBM ensemble is averaged with a
        Gaussian process (GPBlendEnsemble). shape: "gauss" or "sqrt_edge" (peaks.sqrt_edge; position = edge
        E_max). marginal: peak parameters not predicted from the diagnostics but drawn from their training
        distribution (MarginalEnsemble). Defaults = the original model, as used for the Stage 2 beam pool."""
        self.E = np.asarray(E_MeV)
        self.seed, self.n_members, self.gp_blend = seed, n_members, tuple(gp_blend)
        self.shape, self.marginal = shape, tuple(marginal)
        self.energy = EnergyEnsemble(n_members=n_members, seed=seed)
        def make(name):
            cls = MarginalEnsemble if name in self.marginal else GPBlendEnsemble if name in self.gp_blend else EnergyEnsemble
            return cls(n_members=n_members, seed=seed)
        self.peak = [make(name) for name in self.PEAK_PARAMS]
        self.cont = BaggedSpectrumModel("lgbm", n_members=n_members, n_pc=n_pc, seed=seed, frac=0.8)

    def fit(self, X, Y):
        from mlsfqed.beam_model.peaks import fit_peaks
        E = self.E
        W = beam_energy(Y, E)
        S = Y / W[:, None]
        P = fit_peaks(S, E, shape=self.shape)
        ok = ~np.isnan(P[:, 0])
        X, Y, S, W, P = X[ok], Y[ok], S[ok], W[ok], P[ok]
        self.n_dropped = int((~ok).sum())
        T = np.column_stack([np.log(P[:, 0]), P[:, 1], np.log(P[:, 2])])
        self.energy.fit(X, np.log(W))
        for k in range(3):
            self.peak[k].fit(X, T[:, k])
        C = S - self._profile(P[:, 0], P[:, 1], P[:, 2])
        self.cont.fit(X, C)
        Pc = np.stack([np.column_stack([r.predict(X) for r in m]) for m in self.cont.members])
        oob = ~self.cont.inbag
        has = oob.sum(0) > 1
        true_c = self.cont.csc.transform(self.cont.pca.transform(C))
        mean_c = np.full_like(true_c, np.nan)
        for i in np.flatnonzero(has):
            mean_c[i] = Pc[oob[:, i], i].mean(0)
        both = has & self.energy.has_oob & np.all([p.has_oob for p in self.peak], 0)
        self._both, self._oob_cont_mean, self.Y_train = both, mean_c, Y
        self.res_W = self.energy.scores_all[both]
        self.res_P = np.column_stack([p.scores_all[both] for p in self.peak])
        self.res_C = (true_c - mean_c)[both]
        self.peak_fits = P
        return self

    def _profile(self, A, pos, sig):
        from mlsfqed.beam_model.peaks import gaussian, sqrt_edge
        return (sqrt_edge if getattr(self, "shape", "gauss") == "sqrt_edge" else gaussian)(self.E, A, pos, sig)

    def _assemble(self, logW, T, Cs):
        lead = Cs.shape[:-1]
        C = self.cont.pca.inverse_transform(self.cont.csc.inverse_transform(Cs.reshape(-1, Cs.shape[-1])))
        C = C.reshape(*lead, -1)
        S = C + self._profile(np.exp(T[..., 0]), T[..., 1], np.exp(T[..., 2]))
        S = np.clip(S, 0, None)
        S = S / beam_energy(S, self.E)[..., None]
        return S * np.exp(logW)[..., None]

    def sample(self, X, n_samples=100, seed=0):
        rng = np.random.default_rng(seed)
        muW, sdW = self.energy.predict(X)
        mp = [p.predict(X) for p in self.peak]
        Cm = np.mean([np.column_stack([r.predict(X) for r in m]) for m in self.cont.members], 0)
        j = rng.integers(len(self.res_W), size=(len(X), n_samples))
        logW = muW[:, None] + self.res_W[j] * sdW[:, None]
        T = np.stack([mp[k][0][:, None] + self.res_P[j, k] * mp[k][1][:, None] for k in range(3)], -1)
        return self._assemble(logW, T, Cm[:, None, :] + self.res_C[j])

    def predict(self, X, n_samples=200, seed=0):
        """Mean of the predictive distribution (smooths the peak over its position uncertainty)."""
        Y = self.sample(X, n_samples, seed).mean(1)
        return Y, beam_energy(Y, self.E)

    def predict_point(self, X):
        """Spectrum at the predicted parameters (beam energy, peak height/position/width, continuum).

        Unlike the mean or per-bin median of samples, which average over where the peak may sit and
        flatten it (~80% of the measured height on held-out shots), this keeps the peak sharp (~97%).
        Its MSE is slightly higher, because a sharp peak in the wrong place costs twice."""
        muW, _ = self.energy.predict(X)
        T = np.column_stack([p.predict(X)[0] for p in self.peak])
        Cm = np.mean([np.column_stack([r.predict(X) for r in m]) for m in self.cont.members], 0)
        return self._assemble(muW, T, Cm)

    def _oob_train_samples(self, n_samples, rng):
        idx = np.flatnonzero(self._both)
        out = np.empty((len(idx), n_samples, len(self.E)))
        for r, i in enumerate(idx):
            j = _draw_other(rng, len(idx), r, n_samples)
            logW = self.energy.oob_mu[i] + self.res_W[j] * self.energy.oob_sd_used[i]
            T = np.stack([self.peak[k].oob_mu[i] + self.res_P[j, k] * self.peak[k].oob_sd_used[i]
                          for k in range(3)], -1)
            out[r] = self._assemble(logW, T, self._oob_cont_mean[i][None, :] + self.res_C[j])
        return out, self.Y_train[idx]
