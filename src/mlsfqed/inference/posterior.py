"""Surrogate-based Bayesian inference of collision parameters for one shot.

Likelihood (electron observables only; definitions match Los et al.):
  observed: mean energy above 305 MeV and P70 of the measured post-collision spectrum (P70 after 10 MeV
  smoothing, as in the simulation). For each of the K predicted incoming beams k,
    predicted mean = <E>_in,k - loss(theta, beam_k),   predicted P70 = P70(theta, beam_k),
  Gaussian errors with variance = surrogate variance (calibrated) + measurement variance.
  The beam uncertainty is marginalised by averaging the likelihood over the K predictions.
Priors: uniform over the surrogate's training range, except the longitudinal offset with dz_prior="focus":
density proportional to |dz|^(-1/2) on +-250 um - the distribution the scan was drawn from (half within
+-62 um), expressing that collisions are timed near focus. Parameters not inferred are fixed (FIXED).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from mlsfqed.surrogate.data import assemble_inputs, describe_beams

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "data" / "processed" / "models" / "surrogate.pkl"
PRIOR = {"a0": (5.0, 20.0), "dz_um": (-250.0, 250.0), "length_um": (1.0, 10.0), "r_offset_um": (0.0, 5.0)}
# Gemini 2021 (Los et al.): waist ~2.47 um, 45 fs, source size 0.68 um; divergence a few mrad
FIXED = {"waist_um": 2.5, "fwhm_fs": 45.0, "radius_um": 0.7, "div_mrad": 2.0, "phi_offset": 0.0, "r_offset_um": 0.0}
TARGETS = ("loss_window_MeV", "e_P70")
P70_REL_SD = 0.05      # measurement + definition systematics (energy scale ~2%, binning/smoothing 2-4%)


def load_model(path: Path = MODEL) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


class ShotPosterior:
    """log-posterior for one shot. free: names of inferred parameters (subset of PRIOR)."""

    def __init__(self, model, beams_dWdE, obs_mean, obs_p70, meas_sd_mean, free=("a0", "dz_um", "length_um"),
                 fixed=None, beam_rmse=None, dz_prior="uniform", use=(0, 1)):
        """beam_rmse: optional (mean energy, P70) realistic prediction errors of the incoming beam. If given,
        the spread of the supplied beam predictions is widened per observable to that error by adding
        the missing variance (max(0, rmse^2 - spread^2)); None uses the predictions as they are."""
        self.m = model
        self.free = list(free)
        self.fixed = dict(FIXED, **(fixed or {}))
        for k in self.free:
            self.fixed.pop(k, None)
        bd = describe_beams(beams_dWdE, model["pool_E_MeV"], model["pool_pca"], model["pool_Q_median_pC"])
        self.beams = {c: bd[c].values for c in bd.columns}
        self.K = len(bd)
        self.obs = np.array([obs_mean, obs_p70])
        self.meas_var = np.array([meas_sd_mean ** 2, (P70_REL_SD * obs_p70) ** 2])
        self.beam_extra_var = np.zeros(2)
        if beam_rmse is not None:
            from mlsfqed.inference.los2024 import spectrum_observables
            E = model["pool_E_MeV"]
            dN = np.atleast_2d(beams_dWdE) / E
            spread = np.array([[spectrum_observables(E, n)[k] for k in ("mean_window_MeV", "P70")] for n in dN]).std(0)
            self.beam_extra_var = np.maximum(0.0, np.asarray(beam_rmse) ** 2 - spread ** 2)
        self.meas_var = self.meas_var + self.beam_extra_var
        self.ti = [model["targets"].index(t) for t in TARGETS]
        f = model["conformal_factor90"]
        self.fac = np.array([f[t] for t in TARGETS]) / 1.645
        self.noise = np.array([model["mc_noise"].get(t, 0.0) for t in TARGETS])
        self.lo = np.array([PRIOR[k][0] for k in self.free])
        self.hi = np.array([PRIOR[k][1] for k in self.free])
        self.dz_prior = dz_prior
        self.use = tuple(use)
        self.i_dz = self.free.index("dz_um") if "dz_um" in self.free else None

    def inputs(self, theta):
        """theta (W, n_free) -> surrogate inputs for every (walker, beam) pair, walker-major."""
        W = len(theta)
        c = {k: np.repeat(theta[:, i], self.K) for i, k in enumerate(self.free)}
        c.update({k: np.full(W * self.K, v) for k, v in self.fixed.items()})
        b = {k: np.tile(v, W) for k, v in self.beams.items()}
        return assemble_inputs(c, b)

    def predict(self, theta):
        """Predicted (mean energy, P70) and variances, shape (W, K, 2)."""
        X = self.inputs(theta)
        P = self.m["mlp"].predict_members(X)                 # (members, W*K, targets)
        mu, sd = P.mean(0)[:, self.ti], P.std(0, ddof=1)[:, self.ti]
        var = (self.fac * np.sqrt(sd ** 2 + self.noise ** 2)) ** 2
        mean_out = np.tile(self.beams["beam_mean_window_MeV"], len(theta)) - mu[:, 0]
        pred = np.column_stack([mean_out, mu[:, 1]]).reshape(len(theta), self.K, 2)
        return pred, var.reshape(len(theta), self.K, 2)

    def loglike(self, theta, use=(0, 1)):
        """log-likelihood (beam-marginalised) for theta inside the prior box. use: observables to include
        (0 = mean energy, 1 = P70)."""
        pred, var = self.predict(np.atleast_2d(theta))
        v = var + self.meas_var
        terms = -0.5 * (((self.obs - pred) ** 2) / v + np.log(2 * np.pi * v))            # (W, K, 2)
        ll = terms[..., list(use)].sum(-1)
        return logsumexp(ll, axis=1) - np.log(self.K)

    def __call__(self, theta):
        theta = np.atleast_2d(theta)
        lp = np.full(len(theta), -np.inf)
        ok = np.all((theta >= self.lo) & (theta <= self.hi), axis=1)
        if ok.any():
            lp[ok] = self.loglike(theta[ok], self.use)
            if self.dz_prior == "focus" and self.i_dz is not None:
                lp[ok] += -0.5 * np.log(np.abs(theta[ok, self.i_dz]) + 1e-3)
        return lp

    def sample_prior(self, n, rng):
        """Draws from the (normalised) prior: uniform box, dz ~ 250 sign(v) v^2 under the focus prior."""
        th = self.lo + rng.random((n, len(self.free))) * (self.hi - self.lo)
        if self.dz_prior == "focus" and self.i_dz is not None:
            v = 2 * rng.random(n) - 1
            th[:, self.i_dz] = np.sign(v) * v ** 2 * PRIOR["dz_um"][1]
        return th

    def log_evidence(self, n=40000, seed=0, batch=4000, use=None):
        """log Z = log E_prior[L] by Monte Carlo over the prior; returns (logZ, standard error of logZ).
        Adequate here: 3-4 parameters and a likelihood that is broad relative to the prior."""
        rng = np.random.default_rng(seed)
        use = self.use if use is None else use
        ll = np.concatenate([self.loglike(self.sample_prior(batch, rng), use) for _ in range(n // batch)])
        m = ll.max()
        w = np.exp(ll - m)
        logZ = m + np.log(w.mean())
        se = float(w.std(ddof=1) / np.sqrt(len(w)) / w.mean())                 # delta method on log
        return float(logZ), se, float(w.sum() ** 2 / (w ** 2).sum())        # + effective sample size

    def a0_eff(self, theta):
        """Geometric effective a0 (the surrogate's a0_eff_est) for posterior samples."""
        X = self.inputs(np.atleast_2d(theta))
        return X["a0_eff_est"].values.reshape(len(theta), self.K)[:, 0]


def run_mcmc(post, nwalkers=32, nsteps=3000, burn=1000, seed=0):
    import emcee
    rng = np.random.default_rng(seed)
    p0 = post.lo + rng.random((nwalkers, len(post.free))) * (post.hi - post.lo)
    s = emcee.EnsembleSampler(nwalkers, len(post.free), post, vectorize=True)
    s.random_state = np.random.RandomState(seed).get_state()      # emcee 3 seeds through random_state
    s.run_mcmc(p0, nsteps, progress=False)
    chain = s.get_chain(discard=burn, thin=5, flat=True)
    tau = float(np.max(s.get_autocorr_time(discard=burn, quiet=True)))   # quiet: warns, not raises, if short
    return chain, {"acceptance": float(np.mean(s.acceptance_fraction)), "autocorr_max_steps": tau,
                   "n_samples": len(chain)}
