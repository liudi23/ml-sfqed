"""The 10 collision shots analysed by Los et al. (2024/2025), prepared for surrogate-based inference.

Data (scripts/fetch_los2024.sh -> data/raw/los2024/, from github.com/ELos385/RadiationReaction):
  mcmc_input_spectra.h5   per shot: measured post-collision electron spectrum (dN/dE per MeV on a
                          gamma axis 597-2446, i.e. 305-1250 MeV), photon spectrum (log10), and the
                          predicted pre-collision spectrum - 100 network predictions, each stored as
                          30 Gaussians in gamma (mean, sigma, height per prediction).
  Model_evidences.csv     log-evidences per shot for the classical, quantum-continuous and
                          quantum-stochastic models (their Bayes factors are ratios of these).

Reconstruction follows their calc_null_spec: prediction j is sum_i N(gamma; mu_i, sigma_i) weighted by
h_ij sigma_i / sum_i h_ij sigma_i, i.e. a unit-area dN/dgamma (shape only; no charge).

Checks (15 Nov 2026): our P70 matches their function on all 10 measured spectra to ~0.1%; the rebuilt
predictions reproduce their stored ones for shot 1 in distribution (mean energy 572.1 +- 6.9 vs
571.9 +- 7.0 MeV; the 100 predictions are stored in a different order). Their energy axis uses
m_e = 0.511 MeV (vs 0.51099895 here; 2e-6 difference). All public spectra are normalised to 1000 pC,
so absolute charge is not available - every observable used here is per electron. Shot labels in their
results file count from 1 ("shot 1" = index 0 here).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mlsfqed.simulation.ptarmigan import critical_energy, p70

ROOT = Path(__file__).resolve().parents[3]
LOS = ROOT / "data" / "raw" / "los2024"
ME_MEV = 0.51099895
WINDOW_MEV = 305.0


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / np.sqrt(2 * np.pi * sig ** 2)


def predicted_pre_spectra(mu, sig, h, gamma_ax):
    """mu, sig: (30,), h: (30, 100) -> (100, n_gamma) unit-area dN/dgamma (Los et al. calc_null_spec)."""
    G = _gauss(gamma_ax[None, :], mu[:, None], sig[:, None])          # (30, n_gamma)
    w = h * sig[:, None]                                              # (30, 100)
    w = w / w.sum(0, keepdims=True)
    return (w.T @ G)                                                  # (100, n_gamma)


def load(los_dir: Path = LOS) -> dict:
    with h5py.File(los_dir / "mcmc_input_spectra.h5", "r") as f:
        d = {k: f[k][()] for k in f}
    gamma = d["espec_gamma_ax"]
    E = gamma * ME_MEV                                                # MeV
    shots = []
    for s in range(d["pre_coll_espec_gamma_mean"].shape[0]):
        pre = predicted_pre_spectra(d["pre_coll_espec_gamma_mean"][s], d["pre_coll_espec_gamma_sigma"][s],
                                    d["pre_coll_espec_heights"][s], gamma) / ME_MEV   # unit-area dN/dE [1/MeV]
        post = d["post_coll_espec_dN_dE_per_MeV_measured"][s]
        shots.append({"shot": int(d["shot_number"][s]), "pre_dNdE": pre, "post_dNdE": post,
                      # photon axis is in units of m_e c^2; the stored spectrum is E dN/dE normalised to its
                      # maximum (their calc_gamma_spec), so dN/dE is recovered by dividing by E
                      "photon_E_MeV": d["photon_gamma_ax"] * 0.511,
                      "photon_EdNdE": 10 ** d["log10_photon_spec"][s],
                      "photon_spec_log10_std": d["std_log10_photon_spec"][s]})
    ev = pd.read_csv(los_dir / "Model_evidences.csv", index_col=0)
    return {"E_MeV": E, "shots": shots, "log_evidence": ev, "raw": d}


def spectrum_observables(E, dNdE, smooth_mev=0.0):
    """Mean energy (all, and >= 305 MeV), P70 (Los et al. definition), charge proxy (integral)."""
    n = np.trapezoid(dNdE, E)
    win = E >= WINDOW_MEV
    return {"mean_MeV": float(np.trapezoid(dNdE * E, E) / n),
            "mean_window_MeV": float(np.trapezoid(dNdE[win] * E[win], E[win]) / np.trapezoid(dNdE[win], E[win])),
            "P70": p70(E[win], dNdE[win], smooth_mev=smooth_mev),
            "integral": float(n)}


def shot_table(data: dict) -> pd.DataFrame:
    """One row per shot: measured post-collision observables, predicted pre-collision observables
    (median and spread over the 100 predictions), photon critical energy fitted to their spectrum,
    and their log-evidences."""
    E = data["E_MeV"]
    rows = []
    for s in data["shots"]:
        post = spectrum_observables(E, s["post_dNdE"])
        pre = [spectrum_observables(E, p) for p in s["pre_dNdE"]]
        pre_mean = np.array([p["mean_window_MeV"] for p in pre])
        pre_p70 = np.array([p["P70"] for p in pre])
        Eg = s["photon_E_MeV"]
        g = s["photon_EdNdE"] / Eg                                   # dN/dE, arbitrary scale
        row = {"shot": s["shot"],
               "post_mean_window_MeV": post["mean_window_MeV"], "post_P70": post["P70"],
               "pre_mean_window_MeV_median": float(np.median(pre_mean)), "pre_mean_window_MeV_std": float(pre_mean.std()),
               "pre_P70_median": float(np.median(pre_p70)),
               "loss_window_MeV_median": float(np.median(pre_mean) - post["mean_window_MeV"]),
               "photon_crit_MeV_fit": critical_energy(Eg, g)}
        for m in data["log_evidence"].columns:
            row[f"logZ_{m}"] = float(data["log_evidence"].loc[s["shot"], m])
        rows.append(row)
    return pd.DataFrame(rows)


def beams_on_pool_grid(data: dict, E_pool: np.ndarray) -> list[np.ndarray]:
    """Each shot's 100 predicted pre-collision spectra as dW/dE (arbitrary scale) on the beam-pool grid
    (305-1400 MeV), zero above the 1250 MeV end of their axis. Shape (100, n_pool) per shot."""
    E = data["E_MeV"]
    out = []
    for s in data["shots"]:
        dNdE = np.array([np.interp(E_pool, E, p, left=0.0, right=0.0) for p in s["pre_dNdE"]])
        out.append(dNdE * E_pool[None, :])
    return out
