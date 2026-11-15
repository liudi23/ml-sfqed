"""Stage 3 data: scan results -> surrogate inputs and targets.

Inputs (all known before a collision is simulated):
  collision: a0, waist, FWHM, transverse offset (magnitude, angle), longitudinal offset, beam radius,
             bunch length, divergence
  beam:      charge, beam energy, mean energy, and the first shape-PCA coefficients of the beam's
             energy-weighted spectrum (PCA fitted on the whole beam pool)
  derived:   a0_eff, a crude geometric estimate of the field the beam sees (Rayleigh-range and
             transverse-overlap factors), and chi_est = 2 gamma a0_eff hbar omega / m c^2. Deterministic
             functions of the inputs, given to help the tree models.

Targets: electron energy loss (all and in the >= 305 MeV window), energy spread, P70, photon yield and
energy per electron (log), critical energy, rms angles; plus binned electron and photon spectra.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "data" / "processed" / "stage2"
HBAR_OMEGA_EV = 1.5498     # 0.8 um
ME_EV = 0.51099895e6

COLLISION = ["a0", "waist_um", "fwhm_fs", "r_offset_um", "phi_offset", "dz_um",
             "radius_um", "length_um", "div_mrad"]
BEAM_COLS = ["beam_Q_pC", "beam_W_J", "beam_mean_MeV", "beam_mean_window_MeV"] + [f"beam_pc{k}" for k in range(1, 7)]
DERIVED = ["a0_eff_est", "dz_over_zR", "r_over_w", "chi_est"]
INPUTS = COLLISION + DERIVED + BEAM_COLS


def assemble_inputs(c, b) -> pd.DataFrame:
    """Surrogate inputs from collision parameters c and beam descriptors b (dict-likes of equal-length or
    broadcastable arrays). The single place where derived features are computed - used for training and
    inference alike."""
    c = {k: np.asarray(c[k], float) for k in COLLISION}
    b = {k: np.asarray(b[k], float) for k in BEAM_COLS}
    zR = np.pi * c["waist_um"] ** 2 / 0.8
    overlap = np.exp(-c["r_offset_um"] ** 2 / (c["waist_um"] ** 2 / 2 + c["radius_um"] ** 2))
    a_eff = c["a0"] / np.sqrt(1 + (c["dz_um"] / 2 / zR) ** 2) * np.sqrt(overlap)
    gamma = b["beam_mean_MeV"] * 1e6 / ME_EV
    cols = {**c, "a0_eff_est": a_eff, "dz_over_zR": c["dz_um"] / zR, "r_over_w": c["r_offset_um"] / c["waist_um"],
            "chi_est": 2 * gamma * a_eff * HBAR_OMEGA_EV / ME_EV, **b}
    n = max(np.size(v) for v in cols.values())
    return pd.DataFrame({k: np.broadcast_to(cols[k], n) for k in INPUTS})


TARGETS = {                  # name: how it is built from the scan columns
    "loss_MeV": "beam_mean_MeV - e_mean_MeV",
    "loss_window_MeV": "beam_mean_window_MeV - e_mean_window_MeV",
    "e_std_MeV": "e_std_MeV",
    "e_P70": "e_P70",
    "log_g_number_per_e": "log10(g_number_per_e)",
    "log_g_energy_per_e": "log10(g_total_energy_MeV_per_e)",
    "g_crit_MeV": "g_crit_MeV",
    "g_theta_x_rms_mrad": "g_theta_x_rms_mrad",
    "g_theta_y_rms_mrad": "g_theta_y_rms_mrad",
}
# Monte Carlo noise at 5,000 electrons, measured in the pilot (std over 4 seeds; typical value)
NOISE = {"loss_MeV": 1.3, "loss_window_MeV": 1.5, "e_P70": 0.08, "log_g_number_per_e": 0.005,
         "log_g_energy_per_e": 0.01, "g_crit_MeV": 0.5, "g_theta_x_rms_mrad": 0.05, "g_theta_y_rms_mrad": 0.05}


def load_runs(stage2: Path = STAGE2) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read finished runs straight from the part files (works while a scan is still running).
    Returns (table, electron spectra Ne, photon spectra Ng), rows aligned and sorted by run_id."""
    recs = [json.loads(line) for f in sorted((stage2 / "results").glob("part-*.jsonl"))
            for line in f.read_text().splitlines() if line.strip()]
    recs = [r for r in recs if r.get("ok")]
    recs = list({r["run_id"]: r for r in recs}.values())          # de-duplicate reruns
    recs.sort(key=lambda r: r["run_id"])
    Ne = np.array([r.pop("Ne") for r in recs], dtype=float)
    Ng = np.array([r.pop("Ng") for r in recs], dtype=float)
    res = pd.DataFrame(recs)
    des = pd.read_parquet(stage2 / "design.parquet")
    df = des.merge(res, on="run_id", how="inner").sort_values("run_id").reset_index(drop=True)
    return df, Ne, Ng


def pool_shape_pca(stage2: Path = STAGE2, n_pc: int = 6):
    """PCA of the beam pool's normalised energy-weighted spectra (unit area over E in GeV)."""
    pool = dict(np.load(stage2 / "beam_pool.npz", allow_pickle=False))
    E, Y = pool["E_MeV"], pool["dWdE_J_per_GeV"]
    shape = Y / np.trapezoid(Y, E / 1e3, axis=1)[:, None]
    return E, PCA(n_pc, svd_solver="full").fit(shape)


def describe_beams(dWdE, E, pca, Q_pC):
    """Beam descriptors for spectra not in the pool (e.g. Los et al. predictions), on the pool grid E.
    dWdE: (n, len(E)) any scale; Q_pC: charge to assign (per-electron observables don't depend on it)."""
    dWdE = np.atleast_2d(dWdE)
    shape = dWdE / np.trapezoid(dWdE, E / 1e3, axis=1)[:, None]
    dNdE = dWdE / E[None, :]
    mean = (dNdE * E).sum(1) / dNdE.sum(1)
    win = E >= 305.0
    out = pd.DataFrame({"beam_Q_pC": np.broadcast_to(Q_pC, len(dWdE)).astype(float),
                        "beam_W_J": np.broadcast_to(Q_pC, len(dWdE)) * mean * 1e-6,   # (Q/e) * <E> in joules
                        "beam_mean_MeV": mean,
                        "beam_mean_window_MeV": (dNdE[:, win] * E[win]).sum(1) / dNdE[:, win].sum(1)})
    pcs = pca.transform(shape)
    for k in range(pcs.shape[1]):
        out[f"beam_pc{k + 1}"] = pcs[:, k]
    return out


def beam_descriptors(stage2: Path = STAGE2, n_pc: int = 6) -> pd.DataFrame:
    """One row per beam in the pool: charge, energy, mean energy, mean in window, shape PCA coefficients."""
    pool = dict(np.load(stage2 / "beam_pool.npz", allow_pickle=False))
    E, N = pool["E_MeV"], pool["dNdE_per_MeV"]
    Y = pool["dWdE_J_per_GeV"]
    W = np.trapezoid(Y, E / 1e3, axis=1)
    shape = Y / W[:, None]
    pcs = pool_shape_pca(stage2, n_pc)[1].transform(shape)
    out = pd.DataFrame({"beam_Q_pC": pool["Q_pC"], "beam_W_J": W,
                        "beam_mean_MeV": (N * E).sum(1) / N.sum(1)})
    win = E >= 305.0
    out["beam_mean_window_MeV"] = (N[:, win] * E[win]).sum(1) / N[:, win].sum(1)
    for k in range(n_pc):
        out[f"beam_pc{k + 1}"] = pcs[:, k]
    out["beam_source"] = pool["source"]
    out["beam_shot_index"] = pool["shot_index"]
    return out


def build(stage2: Path = STAGE2):
    """Returns X (DataFrame of inputs), Y (DataFrame of targets), groups (beam shot index), spectra."""
    df, Ne, Ng = load_runs(stage2)
    bd = beam_descriptors(stage2)
    b = bd.iloc[df.beam_index.values].reset_index(drop=True)
    df = pd.concat([df, b.drop(columns=[c for c in b.columns if c in df.columns])], axis=1)
    X = assemble_inputs(df, df)
    Y = pd.DataFrame({
        "loss_MeV": df.beam_mean_MeV - df.e_mean_MeV,
        "loss_window_MeV": df.beam_mean_window_MeV - df.e_mean_window_MeV,
        "e_std_MeV": df.e_std_MeV,
        "e_P70": df.e_P70,
        "log_g_number_per_e": np.log10(df.g_number_per_e),
        "log_g_energy_per_e": np.log10(df.g_total_energy_MeV_per_e.clip(lower=1e-6)),
        "g_crit_MeV": df.g_crit_MeV,
        "g_theta_x_rms_mrad": df.g_theta_x_rms_mrad,
        "g_theta_y_rms_mrad": df.g_theta_y_rms_mrad,
    })
    ok = Y.notna().all(1).values
    meta = df[["run_id", "beam_index", "beam_shot_index", "beam_source", "wall_s"]]
    # spectra per electron: electrons per 10 MeV bin, photons per log bin
    nreal = (df.beam_Q_pC * 1e-12 / 1.602176634e-19).values[:, None]
    return (X[ok].reset_index(drop=True), Y[ok].reset_index(drop=True), meta[ok].reset_index(drop=True),
            Ne[ok] / nreal[ok], Ng[ok] / nreal[ok])
