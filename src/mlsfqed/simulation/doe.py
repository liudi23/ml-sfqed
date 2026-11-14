"""Design of experiments for the Stage 2 Ptarmigan scan.

Beam pool: the 645 measured Gemini spectra plus two calibrated draws per shot from the Stage 1 peak model
(trained on all shots), so beams cover both the real population and its shot-to-shot uncertainty.

Collision parameters are drawn with a scrambled Sobol sequence over the ranges below. Ranges follow the
Gemini 2021 setup (Los et al.): nominal peak a0 ~ 21 (capped at 20 by the LMA rate tables), 45 fs,
~2.5 um focus, inferred effective a0 7-13;
offsets up to ~10 Rayleigh ranges (z_R ~ 25 um at 2.5 um waist), denser near focus.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

RANGES = {                       # name: (low, high)
    "a0": (5.0, 20.0),           # LMA tables stop near a local a ~ 20
    "waist_um": (2.0, 3.0),
    "fwhm_fs": (35.0, 55.0),
    "r_offset_um": (0.0, 5.0),   # transverse offset magnitude
    "phi_offset": (0.0, np.pi / 2),  # its angle from the polarisation axis (x); symmetric quadrant
    "dz_um": (-250.0, 250.0),    # longitudinal offset; sampled densely near focus (see sobol_design)
    "radius_um": (0.5, 2.0),     # rms beam radius at the collision
    "length_um": (1.0, 10.0),    # rms bunch length (3-33 fs)
    "div_mrad": (1.0, 4.0),      # rms divergence
}


def sobol_design(n: int, n_beams: int, seed: int = 0) -> dict:
    """n should be a power of two. Returns dict of arrays incl. beam index and Cartesian offsets."""
    names = list(RANGES)
    s = qmc.Sobol(d=len(names) + 1, scramble=True, seed=seed).random(n)
    out = {k: lo + s[:, i] * (hi - lo) for i, (k, (lo, hi)) in enumerate(RANGES.items())}
    # longitudinal offset: dz = 250 um * sign(v) * v^2 with v uniform in [-1, 1], so half of the runs fall
    # within +-62 um (~2.5 Rayleigh ranges) while the full +-250 um range is still covered
    v = 2 * s[:, names.index("dz_um")] - 1
    out["dz_um"] = np.sign(v) * v ** 2 * RANGES["dz_um"][1]
    out["beam_index"] = np.minimum((s[:, -1] * n_beams).astype(int), n_beams - 1)
    out["dx_um"] = out["r_offset_um"] * np.cos(out["phi_offset"])
    out["dy_um"] = out["r_offset_um"] * np.sin(out["phi_offset"])
    out["run_id"] = np.arange(n)
    return out


def beam_pool(seed: int = 0, draws_per_shot: int = 2):
    """Measured spectra + calibrated model draws, as dN/dE [electrons/MeV] and charge [pC]."""
    from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser, to_electron_spectrum
    from mlsfqed.data.gemini import load_all, physical_spectra

    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    allm = np.ones(len(Y), bool)
    rf = RawFeaturiser().fit(d, allm)
    X = rf.transform(d, allm)
    m = PeakBeamModel(E, seed=seed).fit(X, Y).calibrate()
    S = m.sample_calibrated(X, draws_per_shot, seed)                    # (n_shots, draws, bins)
    Ys = np.concatenate([Y[:, None, :], S], axis=1).reshape(-1, Y.shape[1])
    source = np.tile(np.array(["measured"] + ["model"] * draws_per_shot), len(Y))
    shot = np.repeat(np.arange(len(Y)), draws_per_shot + 1)
    dNdE, Q = to_electron_spectrum(Ys, E)
    return {"E_MeV": E, "dNdE_per_MeV": dNdE, "Q_pC": Q, "dWdE_J_per_GeV": Ys,
            "source": source, "shot_index": shot, "run": d["run"][shot], "shot": d["shot"][shot]}
