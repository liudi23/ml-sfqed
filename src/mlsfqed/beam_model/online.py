"""Online adaptation during a run, using only shots measured before the one being predicted.

Adaptive conformal calibration (Gibbs & Candes 2021, quantile-tracking form): the sampled 90% band of
each shot is stretched around its median by (1 + q). After each shot is measured, q moves up if the
measurement fell outside and down otherwise:  q <- q + gamma * (miss - alpha).  Long-run miss rate
then tracks alpha even when the error spread drifts. q is tracked per energy bin ("per_bin") or as
one number shared by all bins ("global").
"""

from __future__ import annotations

import numpy as np


def cqr_scores(samples, Y, level=0.9):
    """Per-bin overshoot of Y beyond the sampled band, in units of the band half-width on that side.
    Y is covered by the band stretched by (1 + q) iff score <= q."""
    a = (1 - level) / 2
    med = np.median(samples, axis=1)
    lo, hi = np.quantile(samples, [a, 1 - a], axis=1)
    eps = 1e-3 * np.abs(Y).max()
    return np.maximum((lo - Y) / (med - lo + eps), (Y - hi) / (hi - med + eps))


def track(scores, alpha=0.1, gamma=0.02, mode="per_bin", q0=0.0):
    """Run quantile tracking over shots in time order. scores: (n_shots, n_bins).
    Returns q used for each shot (before seeing it) and the miss indicators."""
    n, b = scores.shape
    q = np.full(b, q0) if mode == "per_bin" else np.full(1, q0)
    qs, miss = np.empty((n, b)), np.empty((n, b), bool)
    for t in range(n):
        qt = q if mode == "per_bin" else np.full(b, q[0])
        qs[t] = qt
        miss[t] = scores[t] > qt
        if mode == "per_bin":
            q = q + gamma * (miss[t] - alpha)
        else:
            q = q + gamma * (miss[t].mean() - alpha)
    return qs, miss


def stretched_band(samples, q, level=0.9):
    a = (1 - level) / 2
    med = np.median(samples, axis=1)
    lo, hi = np.quantile(samples, [a, 1 - a], axis=1)
    f = np.clip(1 + q, 0, None)
    return med - f * (med - lo), med + f * (hi - med)
