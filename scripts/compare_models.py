"""Radiation-reaction model comparison with the surrogates: classical vs quantum-continuous vs
quantum-stochastic, as in Los et al. (2024/2025).

Needs one surrogate per model trained on the same collisions:
  python scripts/stage3_train_final.py --model stochastic --max-runs 4096
  python scripts/stage3_train_final.py --model classical  --max-runs 4096
  python scripts/stage3_train_final.py --model continuous --max-runs 4096
Then:
  python scripts/compare_models.py recovery --n 30   # simulated shots: does the evidence pick the true model?
  python scripts/compare_models.py shots             # the 10 public collision shots
Evidence log Z = log E_prior[L] by Monte Carlo over the prior (near-focus dz prior). Variants: observables
(mean energy + P70, or mean energy only) and incoming-beam uncertainty (published / calibrated).
Writes results/model_comparison/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import h5py
from infer_los2024 import SETUPS, beam_rmse

from mlsfqed.inference.los2024 import LOS, beams_on_pool_grid, load, spectrum_observables
from mlsfqed.inference.posterior import MODEL, ShotPosterior, load_model
from mlsfqed.surrogate.data import COLLISION, build

OUT = ROOT / "results" / "model_comparison"
RR = ("classical", "continuous", "stochastic")
LOS_NAMES = {"classical": "classical", "continuous": "quantum-continuous", "stochastic": "quantum-stochastic"}
USES = {"mean+P70": (0, 1), "mean": (0,)}


def surrogate_path(model, max_runs, quick):
    return MODEL.with_name(f"surrogate_{model}_{max_runs}{'_quick' if quick else ''}.pkl")


def available(max_runs, quick):
    return [m for m in RR if surrogate_path(m, max_runs, quick).exists()]


# ---------------------------------------------------------------- recovery
def _rec_task(args):
    row, models, max_runs, quick, n_prior = args
    out = {"true_model": row["true_model"], "run_id": row["run_id"], "loss": row["loss"]}
    fixed = {k: row[k] for k in COLLISION if k not in SETUPS["offset"]}
    for mdl in models:
        m = load_model(surrogate_path(mdl, max_runs, quick))
        post = ShotPosterior(m, row["beam"], row["obs_mean"], row["obs_p70"], 10.0, free=SETUPS["offset"],
                             fixed=fixed, dz_prior="focus")
        for name, use in USES.items():
            out[f"logZ_{mdl}_{name}"] = post.log_evidence(n=n_prior, seed=row["run_id"], use=use)[0]
    return out


def recovery(n, max_runs, quick, workers, n_prior):
    models = available(max_runs, quick)
    pool = np.load(ROOT / "data" / "processed" / "stage2" / "beam_pool.npz")
    cal_ids = set(load_model(surrogate_path(models[0], max_runs, quick))["calibration_run_ids"])
    rows, rng = [], np.random.default_rng(0)
    for true in models:
        d = ROOT / "data" / "processed" / ("stage2" if true == "stochastic" else f"stage2_{true}")
        X, Y, meta, _, _ = build(d, max_runs)
        idx = np.flatnonzero(meta.run_id.isin(cal_ids).values & (Y.loss_window_MeV.values > 5.0))
        for i in rng.choice(idx, min(n, len(idx)), replace=False):
            r = {k: float(X[k].iloc[i]) for k in X.columns}
            r.update(true_model=true, run_id=int(meta.run_id.iloc[i]), loss=float(Y.loss_window_MeV.iloc[i]),
                     beam=pool["dWdE_J_per_GeV"][[int(meta.beam_index.iloc[i])]],
                     obs_mean=float(X.beam_mean_window_MeV.iloc[i] - Y.loss_window_MeV.iloc[i]),
                     obs_p70=float(Y.e_P70.iloc[i]))
            rows.append(r)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = pd.DataFrame(list(ex.map(_rec_task, [(r, models, max_runs, quick, n_prior) for r in rows])))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if quick else ""
    res.to_csv(OUT / f"recovery{tag}.csv", index=False)
    print(f"model-comparison recovery: {len(res)} simulated shots (loss > 5 MeV) the surrogates never saw, "
          f"{time.time() - t0:.0f} s")
    for name in USES:
        Z = res[[f"logZ_{m}_{name}" for m in models]].values
        best = np.array(models)[Z.argmax(1)]
        print(f"\nobservables: {name}   (rows: true model; columns: fraction of shots where each model has the highest evidence)")
        print(f"{'':12s}" + "".join(f"{m:>12s}" for m in models) + f"{'log BF true vs best other (median)':>36s}")
        for true in models:
            sel = (res.true_model == true).values
            frac = [np.mean(best[sel] == m) for m in models]
            j = models.index(true)
            others = np.delete(Z[sel], j, axis=1).max(1)
            print(f"{true:12s}" + "".join(f"{f:12.2f}" for f in frac) + f"{np.median(Z[sel, j] - others):36.2f}")


# ---------------------------------------------------------------- real shots
def _shot_task(args):
    k, mdl, unc, setup, max_runs, quick, K, n_prior = args
    m = load_model(surrogate_path(mdl, max_runs, quick))
    data = load()
    E = data["E_MeV"]
    beams = beams_on_pool_grid(data, m["pool_E_MeV"])[k]
    beams = beams[np.random.default_rng(100 + k).choice(len(beams), min(K, len(beams)), replace=False)]
    s = data["shots"][k]
    obs = spectrum_observables(E, s["post_dNdE"], smooth_mev=10.0)
    with h5py.File(LOS / "Bayesian_inference_data.h5", "r") as f:
        dE = f["shot 1/Electrons/Energy_uncertainty[MeV]"][()]
    meas_sd = float(np.trapezoid(s["post_dNdE"] * dE, E) / np.trapezoid(s["post_dNdE"], E))
    post = ShotPosterior(m, beams, obs["mean_window_MeV"], obs["P70"], meas_sd, free=SETUPS[setup],
                         beam_rmse=beam_rmse() if unc == "calibrated" else None, dz_prior="focus")
    out = {"shot": s["shot"], "model": mdl, "beam_unc": unc, "setup": setup}
    for name, use in USES.items():
        logZ, se, ess = post.log_evidence(n=n_prior, seed=k, use=use)
        out.update({f"logZ_{name}": logZ, f"se_{name}": se, f"ess_{name}": ess})
    return out


def shots(max_runs, quick, workers, K, n_prior):
    models = available(max_runs, quick)
    tasks = [(k, mdl, unc, setup, max_runs, quick, K, n_prior)
             for k in range(10) for mdl in models for unc in ("published", "calibrated") for setup in SETUPS]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = pd.DataFrame(list(ex.map(_shot_task, tasks)))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if quick else ""
    res.to_csv(OUT / f"shots{tag}.csv", index=False)
    los = pd.read_csv(LOS / "Model_evidences.csv", index_col=0)
    print(f"{len(res)} evidence estimates in {time.time() - t0:.0f} s; max MC standard error "
          f"{res[[c for c in res if c.startswith('se_')]].max().max():.3f} in log Z")
    ref = "classical" if "classical" in models else models[0]
    for setup in SETUPS:
        for unc in ("published", "calibrated"):
            for name in USES:
                sub = res[(res.setup == setup) & (res.beam_unc == unc)].pivot(index="shot", columns="model", values=f"logZ_{name}")
                print(f"\n[{setup}, beam {unc}, {name}]  log Bayes factor vs {ref} per shot (shots 0-9) and summed:")
                for mdl in models:
                    if mdl == ref:
                        continue
                    d = sub[mdl] - sub[ref]
                    print(f"  {mdl:10s}: " + " ".join(f"{v:+5.2f}" for v in d.values) + f"   sum {d.sum():+6.2f}")
    print(f"\nLos et al. (their log-evidences, PSIS-LOO) vs {LOS_NAMES[ref]}:")
    for mdl in models:
        if mdl == ref:
            continue
        d = los[LOS_NAMES[mdl]] - los[LOS_NAMES[ref]]
        print(f"  {mdl:10s}: " + " ".join(f"{v:+5.2f}" for v in d.values) + f"   sum {d.sum():+6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["recovery", "shots"])
    ap.add_argument("--n", type=int, default=30, help="recovery: simulated shots per true model")
    ap.add_argument("--max-runs", default="4096")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--beams", type=int, default=50)
    ap.add_argument("--prior-samples", type=int, default=40000)
    a = ap.parse_args()
    mr = int(a.max_runs) if a.max_runs != "all" else None
    if a.what == "recovery":
        recovery(a.n, mr, a.quick, a.workers, a.prior_samples)
    else:
        shots(a.max_runs, a.quick, a.workers, a.beams, a.prior_samples)


if __name__ == "__main__":
    main()
