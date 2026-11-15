"""Train and save the final surrogate used for inference.

MLP ensemble trained on all scan runs except a calibration set (15% of the measured shots, whole shots);
per-observable split-conformal factors from the calibration set; saved with everything inference needs.
  python scripts/stage3_train_final.py            (~2-3 min on a laptop)
  python scripts/stage3_train_final.py --quick    (small model, smoke test)
  python scripts/stage3_train_final.py --model classical --max-runs 4096   (one surrogate per RR model)
Writes data/processed/models/surrogate.pkl (stochastic, all runs) or surrogate_<model>_<N>[_quick].pkl.
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
    ap.add_argument("--model", choices=["stochastic", "classical", "continuous"], default="stochastic")
    ap.add_argument("--max-runs", type=int)
    a = ap.parse_args()
    stage2 = ROOT / "data" / "processed" / ("stage2" if a.model == "stochastic" else f"stage2_{a.model}")
    X, Y, meta, _, _ = build(stage2, a.max_runs)
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
             "calibration_run_ids": meta.run_id[cal].tolist(), "rr_model": a.model, "max_runs": a.max_runs}
    MODELS.mkdir(parents=True, exist_ok=True)
    if a.model == "stochastic" and a.max_runs is None:
        out = MODELS / f"surrogate{'_quick' if a.quick else ''}.pkl"
    else:
        out = MODELS / f"surrogate_{a.model}_{a.max_runs or 'all'}{'_quick' if a.quick else ''}.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"trained on {model['n_train']} runs, calibrated on {model['n_calibration']} ({time.time() - t0:.0f} s) -> {out}")
    print("calibration-set coverage after conformal scaling:", dict(zip(cols, np.round(cov, 2))))


if __name__ == "__main__":
    main()
