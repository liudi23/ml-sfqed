"""Fit the high-energy peak of each LWFA spectrum.

Every Gemini spectrum has a peak near the top of its energy range (810-990 MeV, sigma 22-41 MeV, about
11% of the beam energy). A Gaussian plus a local linear background is fitted in a window around the
highest-energy prominent maximum above 650 MeV; the fit succeeds for all 645 shots, with local R^2 > 0.9
in 99.7% of them.

The measured peaks are asymmetric (sharp high-energy edge, slow fall toward low energy). `sqrt_edge` is the
phase-space-rotation shape: near dephasing the final energy is stationary in the injection phase, so electrons
pile up below the maximum energy E_max as dN/dE ~ (E_max - E)^(-1/2); smeared by a Gaussian of width sigma
(spectrometer resolution + bunch spread). It has the same three parameters as the Gaussian and fits the 645
peaks with median local R^2 0.993 vs 0.959 (scripts/peak_shape_study.py).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks
from scipy.special import pbdv


def _edge_profile(z):
    """exp(-z^2/4) D_{-1/2}(z) = (1/sqrt(pi)) * integral_0^inf t^(-1/2) exp(-(z+t)^2/2) dt: the convolution of
    u^(-1/2) (u > 0) with a unit Gaussian, at z = (E - E_max)/sigma. Far below the edge (z < -12) D_{-1/2}
    overflows, so the asymptotic series sqrt(2/|z|) (1 + 3/(8 z^2) + 105/(128 z^4)) is used (rel. error < 1e-5)."""
    z = np.asarray(z, float)
    far = z < -12.0
    zc = np.where(far, -12.0, z)
    exact = np.exp(-0.25 * zc ** 2) * pbdv(-0.5, zc)[0]
    za = np.abs(np.where(far, z, -12.0))
    return np.where(far, np.sqrt(2.0 / za) * (1 + 3.0 / (8 * za ** 2) + 105.0 / (128 * za ** 4)), exact)


_Z_PEAK = minimize_scalar(lambda z: -_edge_profile(z), bounds=(-3, 1), method="bounded").x
_EDGE_MAX = float(_edge_profile(_Z_PEAK))


def sqrt_edge(E, A, Emax, sig):
    """Square-root edge, normalised so that A is the peak height (the maximum lies at E_max + z_peak*sigma)."""
    E = np.asarray(E)
    A, Emax, sig = (np.asarray(v)[..., None] for v in (A, Emax, sig))
    return A * _edge_profile((E - Emax) / sig) / _EDGE_MAX


def _local_edge(E, A, Emax, sig, b0, b1):
    return A * _edge_profile((E - Emax) / sig) / _EDGE_MAX + b0 + b1 * (E - Emax)


def gaussian(E, A, mu, sig):
    E = np.asarray(E)
    A, mu, sig = (np.asarray(v)[..., None] for v in (A, mu, sig))
    return A * np.exp(-0.5 * ((E - mu) / sig) ** 2)


def _local(E, A, mu, sig, b0, b1):
    return A * np.exp(-0.5 * ((E - mu) / sig) ** 2) + b0 + b1 * (E - mu)


def fit_peaks(S, E, min_energy=650.0, shape="gauss"):
    """Return (n, 4) array: peak height A, position [MeV], width sigma [MeV], local R^2.
    shape "gauss": position = centre mu; shape "sqrt_edge": position = edge E_max."""
    E = np.asarray(E)
    model = _local if shape == "gauss" else _local_edge
    out = np.full((len(S), 4), np.nan)
    for n, s in enumerate(S):
        pk, _ = find_peaks(s, prominence=0.05 * s.max())
        pk = pk[E[pk] > min_energy]
        if len(pk) == 0:
            continue
        i = pk[-1]
        m = (E > E[i] - 180) & (E < E[i] + 150)
        if shape == "gauss":
            p0, lo, hi = [s[i] - s[m].min(), E[i], 25, s[m].min(), 0], [0, E[i] - 60, 4, -1, -1e-2], [10, E[i] + 60, 120, 10, 1e-2]
        else:
            p0, lo, hi = ([s[i] - s[m].min(), E[i] + 15, 12, s[m].min(), 0], [0, E[i] - 40, 2, -1, -1e-2],
                          [10, E[i] + 100, 80, 10, 1e-2])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(model, E[m], s[m], p0=p0, bounds=(lo, hi), maxfev=5000)
        except RuntimeError:
            continue
        r2 = 1 - np.sum((s[m] - model(E[m], *p)) ** 2) / np.sum((s[m] - s[m].mean()) ** 2)
        out[n] = [p[0], p[1], p[2], r2]
    return out
