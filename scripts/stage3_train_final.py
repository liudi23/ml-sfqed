"""Train and save the final surrogate used for inference.

MLP ensemble trained on all scan runs except a calibration set (15% of the measured shots, whole shots);
per-observable split-conformal factors from the calibration set; saved with everything inference needs.
  python scripts/stage3_train_final.py            (~2-3 min on a laptop)
  python scripts/stage3_train_final.py --quick    (small model, smoke test)
Writes data/processed/models/surrogate.pkl (or surrogate_quick.pkl).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stage3_surrogate import conformal_factors, interval_scale

from mlsfqed.surrogate.data import INPUTS, NOISE, build, pool_shape_pca
from mlsfqed.surrogate.models import MLPEnsemble

MODELS = ROOT / "data" / "processed" / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    X, Y, meta, _, _ = build()
    assert list(X.columns) == INPUTS
    rng = np.random.default_rng(1)
    shots = meta.beam_shot_index.unique()
    cal = meta.beam_shot_index.isin(rng.choice(shots, int(0.15 * len(shots)), replace=False)).values
    t0 = time.time()
    mlp = MLPEnsemble(quick=a.quick).fit(X[~cal], Y[~cal])
    cols = list(Y.columns)
    Pc, Sc = mlp.predict(X[cal])
    factors = conformal_factors(Pc, Y[cal].values, interval_scale(Pc, Y[cal].values, cols, Sc))
    cov = np.mean(np.abs(Pc - Y[cal].values) <= factors * interval_scale(Pc, Y[cal].values, cols, Sc), axis=0)
    E_pool, pca = pool_shape_pca()
    pool = np.load(ROOT / "data" / "processed" / "stage2" / "beam_pool.npz")
    model = {"mlp": mlp, "inputs": INPUTS, "targets": cols, "conformal_factor90": dict(zip(cols, factors.tolist())),
             "mc_noise": NOISE, "pool_E_MeV": E_pool, "pool_pca": pca, "pool_Q_median_pC": float(np.median(pool["Q_pC"])),
             "input_min": X.min().to_dict(), "input_max": X.max().to_dict(),
             "n_train": int((~cal).sum()), "n_calibration": int(cal.sum()), "quick": a.quick,
             "calibration_run_ids": meta.run_id[cal].tolist()}
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / f"surrogate{'_quick' if a.quick else ''}.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"trained on {model['n_train']} runs, calibrated on {model['n_calibration']} ({time.time() - t0:.0f} s) -> {out}")
    print("calibration-set coverage after conformal scaling:", dict(zip(cols, np.round(cov, 2))))


if __name__ == "__main__":
    main()
