"""Loader for the Streeter et al. (2023) Gemini LWFA dataset (Zenodo 10.5281/zenodo.7510352, CC BY 4.0).

Each pickle is a dict of NumPy arrays (one row per shot):
  spectra      (N, 200)  dW/dE on E_MeV (305-1400 MeV), normalised to unit integral over E in GeV;
                         the physical spectrum [J/GeV] is spectra * spectra_J (see physical_spectra)
  spectra_pC   (N,)      total charge [pC];  spectra_J (N,) total beam energy [J]
  laser_E      (N,)      laser energy [J] before the compressor (~60% throughput, so ~6.6 J on target at 11 J)
  f40_data     (N, 6)    far-field focal-spot features, standardised with f40_mean / f40_std
  I_z_data     (N, 100)  top-view laser scattering profile on z_mm (-5..20 mm)
  n_e_z_data   (N, 310)  interferometry density profile on lmi_z_mm (0..26 mm)
  gamma_sig    (N,)      C_gamma, the gamma-ray signal from collision with the counter-propagating laser
                         (99.99th-percentile CsI pixel). Only 'null' shots with C_gamma <= 100 are included.
  index_list   list of (run, shot) pairs, e.g. ("20210620/run10", 254)

The files contain only NumPy objects, but pickle can run code: load only these published files.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
FILES = {"train": "train_data_20220609.pkl", "test": "test_data_20220609.pkl"}


def load_split(split: str, raw_dir: Path = RAW) -> dict:
    """Load the authors' 'train' or 'test' split as a dict of arrays."""
    with open(Path(raw_dir) / FILES[split], "rb") as f:
        return pickle.load(f)


def load_all(raw_dir: Path = RAW) -> dict:
    """Concatenate both splits, ordered by (run, shot), with a 'split' label per shot."""
    parts = [load_split(s, raw_dir) for s in ("train", "test")]
    per_shot = [k for k, v in parts[0].items()
                if isinstance(v, np.ndarray) and v.shape[:1] == (len(parts[0]["index_list"]),)]
    out = {k: np.concatenate([p[k] for p in parts]) for k in per_shot}
    idx = parts[0]["index_list"] + parts[1]["index_list"]
    out["run"] = np.array([r for r, _ in idx])
    out["shot"] = np.array([int(s) for _, s in idx])
    out["split"] = np.array(["train"] * len(parts[0]["index_list"]) + ["test"] * len(parts[1]["index_list"]))
    order = np.lexsort((out["shot"], out["run"]))
    out = {k: v[order] for k, v in out.items()}
    for k in ("E_MeV", "z_mm", "lmi_z_mm", "f40_mean", "f40_std"):
        out[k] = parts[0][k]
    return out


def shot_table(d: dict) -> pd.DataFrame:
    """One row per shot with the scalar quantities."""
    return pd.DataFrame({k: d[k] for k in
                         ("run", "shot", "split", "laser_E", "spectra_pC", "spectra_J", "gamma_sig")})


def physical_spectra(d: dict) -> np.ndarray:
    """dW/dE in J/GeV: the quantity the paper's MSE is computed on."""
    return d["spectra"] * d["spectra_J"][:, None]


def time_ordered_split(d: dict, test_runs=("20210620/run13",)) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks holding out whole later runs, to avoid leakage from drift within a run."""
    test = np.isin(d["run"], test_runs)
    return ~test, test
