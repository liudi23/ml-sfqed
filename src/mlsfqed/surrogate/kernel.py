"""Transfer-kernel surrogate of Ptarmigan's outgoing electron spectrum.

Radiation reaction acts on each electron independently, so the outgoing spectrum is linear in the incoming one:

    N_out(E) = sum_j N_in(E0_j) K(E | E0_j, theta)

K is the distribution of an electron's final energy given its initial energy E0 and the collision parameters
theta, averaged over where in the beam it sits. It is written as a mixture of M Gaussians in the fractional
energy loss f = 1 - E/E0, whose weights, means and widths come from a small MLP of (E0, theta). Integrating
each component over an output bin gives the probability that the electron lands there, so a whole spectrum
is one matrix product, for any incoming spectrum. Trained on Ptarmigan runs with the multinomial likelihood
of their outgoing histograms; one kernel per radiation-reaction model.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

COLLISION = ["a0", "waist_um", "fwhm_fs", "r_offset_um", "phi_offset", "dz_um", "radius_um", "length_um", "div_mrad"]
HBAR_OMEGA_EV, ME_MEV = 1.5498, 0.51099895


def features(theta: np.ndarray, E0: np.ndarray) -> np.ndarray:
    """theta (n, 9) in COLLISION order, E0 (m,) in MeV -> (n, m, F) kernel-network inputs."""
    a0, w, fs, r, phi, dz, rad, L, div = theta.T
    zR = np.pi * w ** 2 / 0.8
    overlap = np.exp(-r ** 2 / (w ** 2 / 2 + rad ** 2))
    a_eff = a0 / np.sqrt(1 + (dz / 2 / zR) ** 2) * np.sqrt(overlap)
    n, m = len(theta), len(E0)
    per_run = np.column_stack([a0 / 20, w / 3, fs / 50, r / 5, np.cos(phi), dz / 250, np.abs(dz) / zR / 4,
                               rad / 2, L / 10, div / 4, a_eff / 20, r / w])
    e = E0 / 1000.0
    chi = 2 * (E0[None, :] / ME_MEV) * a_eff[:, None] * HBAR_OMEGA_EV * 1e-6 / ME_MEV
    X = np.concatenate([np.repeat(per_run[:, None, :], m, 1), np.broadcast_to(e[None, :, None], (n, m, 1)),
                        (chi * 10)[..., None], (chi * a_eff[:, None] / 20)[..., None]], axis=2)
    return X.astype(np.float32)


class KernelNet(nn.Module):
    def __init__(self, n_in=15, hidden=96, n_comp=3):
        super().__init__()
        self.M = n_comp
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * n_comp))

    def forward(self, X):
        o = self.net(X)
        logw, mu, ls = o[..., :self.M], o[..., self.M:2 * self.M], o[..., 2 * self.M:]
        w = torch.softmax(logw, -1)
        mu = torch.sigmoid(mu) * 0.9                      # mean fractional loss in [0, 0.9]
        sig = nn.functional.softplus(ls) * 0.05 + 2e-3    # width in fractional loss
        return w, mu, sig


def _ncdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def bin_probs(w, mu, sig, E0, edges):
    """P(final energy in bin b | E0): (..., m, B). f = 1 - E/E0, so E in [lo, hi) <=> f in (1-hi/E0, 1-lo/E0]."""
    E0 = E0[:, None]
    f_hi = 1 - edges[None, :-1] / E0          # (m, B) upper f for each bin (from its lower energy edge)
    f_lo = 1 - edges[None, 1:] / E0
    z_hi = (f_hi[..., None] - mu[..., None, :]) / sig[..., None, :]
    z_lo = (f_lo[..., None] - mu[..., None, :]) / sig[..., None, :]
    p = ((_ncdf(z_hi) - _ncdf(z_lo)) * w[..., None, :]).sum(-1)       # (..., m, B)
    return p


class TransferKernel:
    """Wraps the network: fit on runs, predict outgoing spectra for (theta, incoming spectrum) pairs."""

    def __init__(self, E0, edges, n_comp=3, hidden=96, seed=0):
        torch.manual_seed(seed)
        self.E0 = np.asarray(E0, float)
        self.edges = np.asarray(edges, float)
        self.net = KernelNet(n_comp=n_comp, hidden=hidden)
        self._E0 = torch.tensor(self.E0, dtype=torch.float32)
        self._edges = torch.tensor(self.edges, dtype=torch.float32)

    def kernel(self, theta):
        """theta (n, 9) -> K (n, m, B) as a torch tensor."""
        X = torch.from_numpy(features(np.atleast_2d(theta), self.E0))
        w, mu, sig = self.net(X)
        return bin_probs(w, mu, sig, self._E0, self._edges)

    def predict(self, theta, N_in):
        """theta (n, 9), N_in (n, m) electrons per incoming grid point (any scale) -> (n, B) fractions."""
        with torch.no_grad():
            K = self.kernel(theta)
            q = torch.from_numpy(np.asarray(N_in, np.float32))
            q = q / q.sum(1, keepdim=True)
            return torch.einsum("nm,nmb->nb", q, K).numpy()

    def fit(self, theta, N_in, N_out, n_eff=5000.0, epochs=40, batch=64, lr=2e-3, val=None, log=print):
        """Multinomial NLL of the outgoing histograms (normalised counts x n_eff macro-electrons)."""
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        q_all = torch.from_numpy((N_in / N_in.sum(1, keepdims=True)).astype(np.float32))
        y_all = torch.from_numpy((N_out / N_out.sum(1, keepdims=True)).astype(np.float32))
        n = len(theta)
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            self.net.train()
            idx = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, batch):
                b = idx[i:i + batch]
                K = self.kernel(theta[b])
                p = torch.einsum("nm,nmb->nb", q_all[b], K).clamp_min(1e-9)
                p = p / p.sum(1, keepdim=True)
                loss = -(y_all[b] * torch.log(p)).sum(1).mean() * n_eff
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            sched.step()
            msg = f"epoch {ep + 1:3d}  train NLL {tot / n:.2f}"
            if val is not None:
                msg += f"  val NLL {self.nll(*val, n_eff=n_eff):.2f}"
            log(msg)
        return self

    def nll(self, theta, N_in, N_out, n_eff=5000.0):
        p = np.clip(self.predict(theta, N_in), 1e-9, None)
        y = N_out / N_out.sum(1, keepdims=True)
        return float(-(y * np.log(p)).sum(1).mean() * n_eff)
