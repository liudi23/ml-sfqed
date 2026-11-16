"""Full-spectrum likelihood for one collision shot, built on the transfer-kernel surrogate.

Data: the outgoing electron spectrum above 310 MeV in 20 MeV bins, normalised to unit sum (the shape; the
charge is not used). Prediction for collision parameters theta and incoming spectrum q: y(theta, q) =
bin(K(theta) q), normalised. The likelihood is Gaussian with a full covariance:

    y_obs ~ N( mean_s y(theta, q_s),  C_beam(theta) + C_energy + C_surr + C_floor )

* C_beam: the spread of y over S incoming-spectrum samples q_s. Each sample is the predicted incoming spectrum
  plus one shape residual measured on held-out Gemini shots (scripts/beam_shape_residuals.py), so the
  incoming-shape error and its correlations across energy are propagated exactly through the (linear) kernel.
* C_energy: the spectrometer's energy-calibration uncertainty, sigma_E = 32.45e-6 E^2 MeV (Los et al.), as a
  fully correlated shift of the measured spectrum.
* C_surr: the surrogate's error against Ptarmigan, per bin, measured on held-out runs.
* C_floor: a small independent term per bin for everything else (fraction of the bin content, plus a constant).
Free parameters and priors follow the summary-statistic posterior (mlsfqed.inference.posterior).
"""

from __future__ import annotations

import numpy as np
import torch

from mlsfqed.inference.posterior import FIXED
from mlsfqed.inference.posterior import PRIOR as _PRIOR

# the summary-statistic priors, plus the geometric nuisances the full spectrum is sensitive to (scan ranges)
PRIOR = {**_PRIOR, "radius_um": (0.5, 2.0), "div_mrad": (1.0, 4.0), "waist_um": (2.0, 3.0)}
from mlsfqed.surrogate.kernel import COLLISION

EDGES = np.arange(310.0, 1010.1, 20.0)          # analysis bins


def rebin(E_fine, y_fine, edges=EDGES):
    """Integrate a density sampled on E_fine into the analysis bins (trapezoid on a refined grid)."""
    Ef = np.linspace(edges[0], edges[-1], 2001)
    yf = np.interp(Ef, E_fine, y_fine, left=0.0, right=0.0)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (yf[1:] + yf[:-1]) * np.diff(Ef))])
    return np.diff(np.interp(edges, Ef, c))


def energy_shift_error(E, dNdE, edges=EDGES):
    """Half the difference between the binned spectrum with the energy axis shifted up and down by sigma_E."""
    s = 32.45e-6 * E ** 2
    up = rebin(E + s, dNdE, edges)
    dn = rebin(E - s, dNdE, edges)
    up, dn = up / up.sum(), dn / dn.sum()
    return 0.5 * (up - dn)


class FullSpectrumPosterior:
    def __init__(self, kernel, E_in, incoming, obs_E, obs_dNdE, residuals=None, n_samples=100, surr_sd=None,
                 free=("a0", "dz_um", "length_um"), fixed=None, dz_prior="focus", floor=(0.03, 2e-4), seed=0,
                 beam_scale_sd=None):
        """kernel: TransferKernel; E_in: its incoming grid; incoming: (K, m) predicted incoming dN/dE on E_in;
        residuals: (R, m) unit-area shape residuals (None = use the K predictions as given); obs: measured
        outgoing dN/dE on obs_E. surr_sd: per-bin surrogate error (normalised units).
        beam_scale_sd: if given, the incoming energy scale is a free parameter 'beam_scale' (E -> E(1 + s)) with
        a Gaussian prior of this width, for the error in the predicted beam energy (the spectral edge)."""
        self.k = kernel
        self.free = list(free) + (["beam_scale"] if beam_scale_sd else [])
        self.scale_sd = beam_scale_sd
        self.fixed = dict(FIXED, **(fixed or {}))
        for f in self.free:
            self.fixed.pop(f, None)
        prior = dict(PRIOR, beam_scale=(-3 * (beam_scale_sd or 0), 3 * (beam_scale_sd or 0)))
        self.lo = np.array([prior[f][0] for f in self.free])
        self.hi = np.array([prior[f][1] for f in self.free])
        self.dz_prior = dz_prior
        self.i_dz = self.free.index("dz_um") if "dz_um" in self.free else None
        rng = np.random.default_rng(seed)
        inc = np.atleast_2d(np.asarray(incoming, float))
        inc = inc / np.trapezoid(inc, E_in, axis=1)[:, None]
        if residuals is not None:
            base = np.median(inc, 0)
            r = residuals[rng.choice(len(residuals), n_samples, replace=len(residuals) < n_samples)]
            inc = np.clip(base[None, :] + r, 0.0, None)
        dE = np.gradient(E_in)[None, :]
        q = inc * dE                                              # electrons per grid point
        self.q = torch.tensor(q / q.sum(1, keepdims=True), dtype=torch.float32)
        if beam_scale_sd:
            # incoming samples rescaled in energy on a grid of scale factors, interpolated per theta
            self.sgrid = np.linspace(-3 * beam_scale_sd, 3 * beam_scale_sd, 61)
            qs = []
            for sc in self.sgrid:
                sh = np.array([np.interp(E_in / (1 + sc), E_in, x, left=0.0, right=0.0) for x in inc]) * dE
                qs.append(sh / sh.sum(1, keepdims=True))
            self.qgrid = torch.tensor(np.array(qs), dtype=torch.float32)          # (G, S, m)
        # 10 MeV output bins -> analysis bins
        e10 = kernel.edges
        c10 = 0.5 * (e10[1:] + e10[:-1])
        self.A = torch.tensor(((c10[None, :] >= EDGES[:-1, None]) & (c10[None, :] < EDGES[1:, None])).astype(np.float32))
        y = rebin(obs_E, obs_dNdE)
        self.y = torch.tensor(y / y.sum(), dtype=torch.float32)
        d = energy_shift_error(obs_E, obs_dNdE)
        B = len(y)
        cov = np.outer(d, d) + np.diag((floor[0] * y / y.sum()) ** 2 + floor[1] ** 2)
        if surr_sd is not None:
            cov += np.diag(np.asarray(surr_sd) ** 2)
        self.C0 = torch.tensor(cov, dtype=torch.float32)
        self.B = B

    def theta_full(self, theta):
        theta = np.atleast_2d(theta)
        cols = {f: theta[:, i] for i, f in enumerate(self.free)}
        return np.column_stack([cols[c] if c in cols else np.full(len(theta), self.fixed[c]) for c in COLLISION])

    def _q(self, theta):
        """Incoming samples for each theta: (S, m), or (n, S, m) when the beam energy scale is free."""
        if not self.scale_sd:
            return self.q
        sc = np.atleast_2d(theta)[:, self.free.index("beam_scale")]
        x = np.clip((sc - self.sgrid[0]) / (self.sgrid[1] - self.sgrid[0]), 0, len(self.sgrid) - 1.000001)
        i = np.floor(x).astype(int)
        w = torch.tensor(x - i, dtype=torch.float32)[:, None, None]
        return (1 - w) * self.qgrid[i] + w * self.qgrid[i + 1]

    def predict(self, theta):
        """-> mean (n, B) and beam covariance (n, B, B) of the normalised binned outgoing spectrum."""
        with torch.no_grad():
            K = self.k.kernel(self.theta_full(theta))                       # (n, m, 150)
            q = self._q(theta)
            eq = "sm,nmb->nsb" if q.dim() == 2 else "nsm,nmb->nsb"
            out = torch.einsum(eq, q, K) @ self.A.T                          # (n, S, B)
            out = out / out.sum(-1, keepdim=True).clamp_min(1e-12)
            mu = out.mean(1)
            dev = out - mu[:, None, :]
            C = dev.transpose(1, 2) @ dev / (out.shape[1] - 1)
        return mu, C

    def loglike(self, theta):
        mu, C = self.predict(theta)
        C = C + self.C0[None]
        L = torch.linalg.cholesky(C.double())
        r = (self.y[None, :] - mu).double()[..., None]
        z = torch.cholesky_solve(r, L)
        quad = (r * z).sum((1, 2))
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        return (-0.5 * (quad + logdet + self.B * np.log(2 * np.pi))).numpy()

    def __call__(self, theta):
        theta = np.atleast_2d(theta)
        lp = np.full(len(theta), -np.inf)
        ok = np.all((theta >= self.lo) & (theta <= self.hi), axis=1)
        if ok.any():
            lp[ok] = self.loglike(theta[ok])
            if self.dz_prior == "focus" and self.i_dz is not None:
                lp[ok] += -0.5 * np.log(np.abs(theta[ok, self.i_dz]) + 1e-3)
            if self.scale_sd:
                lp[ok] += -0.5 * (theta[ok, self.free.index("beam_scale")] / self.scale_sd) ** 2
        return lp

    def sample_prior(self, n, rng):
        th = self.lo + rng.random((n, len(self.free))) * (self.hi - self.lo)
        if self.dz_prior == "focus" and self.i_dz is not None:
            v = 2 * rng.random(n) - 1
            th[:, self.i_dz] = np.sign(v) * v ** 2 * PRIOR["dz_um"][1]
        if self.scale_sd:
            th[:, self.free.index("beam_scale")] = np.clip(rng.normal(0, self.scale_sd, n), self.lo[-1], self.hi[-1])
        return th

    def log_evidence(self, n=20000, seed=0, batch=500):
        """log Z = log E_prior[L] by Monte Carlo over the prior (the focus prior is sampled directly)."""
        rng = np.random.default_rng(seed)
        th = self.sample_prior(n, rng)
        ll = np.concatenate([self.loglike(th[i:i + batch]) for i in range(0, n, batch)])
        m = ll.max()
        w = np.exp(ll - m)
        logZ = m + np.log(w.mean())
        se = float(np.std(w) / np.sqrt(n) / w.mean())
        return float(logZ), se, float(w.sum() ** 2 / (w ** 2).sum())

    def a0_eff(self, theta):
        t = self.theta_full(theta)
        a0, w, r, dz, rad = t[:, 0], t[:, 1], t[:, 3], t[:, 5], t[:, 6]
        zR = np.pi * w ** 2 / 0.8
        return a0 / np.sqrt(1 + (dz / 2 / zR) ** 2) * np.sqrt(np.exp(-r ** 2 / (w ** 2 / 2 + rad ** 2)))
