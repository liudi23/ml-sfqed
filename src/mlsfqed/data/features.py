"""Extra per-shot features for the beam model.

physics_features(d)
    Scalars motivated by laser-wakefield physics, computed from the raw profiles:
    density plateau level, integral, up/down-ramp positions and plateau length (interferometry);
    scattering total, centroid, peak position and cut-off where it falls below half its maximum
    (a proxy for where the laser depletes or loses guiding); laser energy per unit density.

recent_shot_features(d, targets, k=5)
    For each shot, the measured targets (e.g. log beam energy, peak position, peak height) of the
    previous shot and the mean of the previous k shots, in time order (run, then shot number). Only
    strictly earlier shots are used. NaN where no history exists (LightGBM handles NaN natively).
    Deployment assumption: spectra of recent reference shots (no collision) are available, as in this
    campaign, where such shots are interleaved with collision shots.
"""

from __future__ import annotations

import numpy as np


def _crossings(z, y, level):
    above = y >= level
    if not above.any():
        return np.nan, np.nan
    idx = np.flatnonzero(above)
    return z[idx[0]], z[idx[-1]]


def _smooth(y, w=5):
    k = np.ones(w) / w
    return np.convolve(np.pad(y, w // 2, mode="edge"), k, mode="valid")


def physics_features(d) -> tuple[np.ndarray, list[str]]:
    zn, N = d["lmi_z_mm"], d["n_e_z_data"]
    zi, I = d["z_mm"], d["I_z_data"]
    rows = []
    plate = (zn >= 5) & (zn <= 21)
    for n, s in zip(N, I):
        ns = _smooth(n, 9)
        plat = ns[plate].mean()
        up, down = _crossings(zn, ns, 0.5 * plat)
        integ = np.trapezoid(np.clip(ns, 0, None), zn)
        ss = _smooth(s, 5)
        tot = np.trapezoid(ss, zi)
        cen = np.trapezoid(ss * zi, zi) / tot
        zmax = zi[np.argmax(ss)]
        _, cut = _crossings(zi, ss, 0.5 * ss.max())
        late = np.trapezoid(ss[zi > 10], zi[zi > 10]) / tot
        rows.append([plat, np.percentile(ns[plate], 95), integ, up, down, down - up,
                     tot, cen, zmax, cut, late])
    F = np.array(rows)
    names = ["ne_plateau", "ne_p95", "ne_integral", "ne_upramp_z", "ne_downramp_z", "ne_length",
             "scatter_total", "scatter_centroid_z", "scatter_peak_z", "scatter_cutoff_z", "scatter_late_frac"]
    lasE = d["laser_E"][:, None]
    F = np.column_stack([F, lasE[:, 0] / np.clip(F[:, 0], 0.1, None), 1.0 / np.clip(F[:, 0], 0.1, None)])
    names += ["laserE_per_ne", "inv_ne_plateau"]
    return F, names


def recent_shot_features(d, targets: dict, k: int = 5) -> tuple[np.ndarray, list[str]]:
    """targets: name -> per-shot array. Assumes d is ordered by (run, shot), as load_all() returns."""
    order = np.lexsort((d["shot"], d["run"]))
    assert np.all(order == np.arange(len(order))), "data must be ordered by (run, shot)"
    cols, names = [], []
    for name, v in targets.items():
        lag1 = np.full(len(v), np.nan)
        mean_k = np.full(len(v), np.nan)
        lag1[1:] = v[:-1]
        for i in range(1, len(v)):
            mean_k[i] = np.nanmean(v[max(0, i - k):i])
        cols += [lag1, mean_k]
        names += [f"prev_{name}", f"prev{k}_mean_{name}"]
    return np.column_stack(cols), names
