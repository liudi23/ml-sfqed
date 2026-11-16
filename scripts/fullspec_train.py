"""Train the transfer-kernel spectrum surrogate (one per radiation-reaction model) for the full-spectrum likelihood.

  pip install -e ".[fullspec]"
  python scripts/fullspec_train.py                         # all three models, ~10-20 min on a laptop
  python scripts/fullspec_train.py --model classical --epochs 5    # quick check

Data: the Stage 2 Ptarmigan scans (incoming spectrum from the beam pool, outgoing 10 MeV histogram). Split by
incoming beam as in Stage 3 (20% of the measured shots held out). Writes models/kernel_<model>.pt, and
results/fullspec/kernel_<model>.json: test accuracy (outgoing mean energy and P70 recomputed from the predicted
spectrum, against Ptarmigan) and the per-bin surrogate error used in the likelihood.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlsfqed.inference.fullspec import EDGES
from mlsfqed.simulation.ptarmigan import p70
from mlsfqed.surrogate.kernel import COLLISION, TransferKernel

MODELS = ("stochastic", "continuous", "classical")
N_POOL_SHOTS = 645


def load(model, max_runs=None):
    d = ROOT / "data" / "processed" / ("stage2" if model == "stochastic" else f"stage2_{model}")
    r = pd.read_parquet(d / "runs.parquet")
    s = np.load(d / "spectra.npz")
    order = np.argsort(s["run_id"])
    sid, Ne = s["run_id"][order], s["Ne"][order]
    r = r[r.ok].sort_values("run_id")
    keep = np.isin(sid, r.run_id.values)
    sid, Ne = sid[keep], Ne[keep]
    r = r.set_index("run_id").loc[sid].reset_index()
    if max_runs:
        m = (r.run_id < max_runs).values
        r, Ne = r[m].reset_index(drop=True), Ne[m]
    pool = np.load(ROOT / "data" / "processed" / "stage2" / "beam_pool.npz")
    Nin = pool["dNdE_per_MeV"][r.beam_index.values]
    return r, r[COLLISION].values.astype(float), Nin, Ne, pool["E_MeV"], s["E_edges"], pool["shot_index"][r.beam_index.values]


def beam_split(shot, seed=0, frac=0.2):
    """Hold out 20% of the measured pool shots (and every beam drawn from them); the same for every scan."""
    held = np.random.default_rng(seed).choice(np.arange(N_POOL_SHOTS), int(frac * N_POOL_SHOTS), replace=False)
    te = np.isin(shot, held)
    return ~te, te


def evaluate(k, r, th, Nin, Ne, edges, te):
    P = k.predict(th[te], Nin[te])
    c = 0.5 * (edges[1:] + edges[:-1])
    w = c >= 305
    pm = (P[:, w] * c[w]).sum(1) / P[:, w].sum(1)
    pp = np.array([p70(c[w], p[w], 10.0) for p in P])
    tm, tp = r.e_mean_window_MeV.values[te], r.e_P70.values[te]
    r2 = lambda a, b: float(1 - np.mean((a - b) ** 2) / np.var(b))
    A = ((c[None, :] >= EDGES[:-1, None]) & (c[None, :] < EDGES[1:, None])).astype(float)
    yp, yt = P @ A.T, Ne[te] @ A.T
    yp, yt = yp / yp.sum(1, keepdims=True), yt / yt.sum(1, keepdims=True)
    res = yp - yt
    # The residual includes Ptarmigan's own macro-particle noise in the labels, which is about as large; its RMS
    # is used as the surrogate error, an upper bound.
    mc = yt * (1 - yt) / 5000
    surr = np.sqrt((res ** 2).mean(0))
    return {"n_test": int(te.sum()), "mean_window_rmse_MeV": float(np.sqrt(np.mean((pm - tm) ** 2))), "mean_window_R2": r2(pm, tm),
            "P70_rmse": float(np.sqrt(np.mean((pp - tp) ** 2))), "P70_R2": r2(pp, tp), "test_nll": k.nll(th[te], Nin[te], Ne[te]),
            "edges": EDGES.tolist(), "surr_sd": surr.round(6).tolist(), "mc_sd": np.sqrt(mc.mean(0)).round(6).tolist(),
            "mean_bin": yt.mean(0).round(6).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS + ("all",), default="all")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    out = ROOT / "results" / "fullspec"
    out.mkdir(parents=True, exist_ok=True)
    for model in (MODELS if a.model == "all" else (a.model,)):
        r, th, Nin, Ne, E0, edges, shot = load(model, None if model == "stochastic" else 4096)
        tr, te = beam_split(shot)
        k = TransferKernel(E0, edges)
        t0 = time.time()
        def log(msg, model=model, t0=t0):
            print(f"{model}: {msg} ({time.time() - t0:.0f} s)", flush=True)

        k.fit(th[tr], Nin[tr], Ne[tr], epochs=a.epochs, log=log)
        torch.save(k.net.state_dict(), ROOT / "models" / f"kernel_{model}.pt")
        res = evaluate(k, r, th, Nin, Ne, edges, te)
        res.update(model=model, n_train=int(tr.sum()), epochs=a.epochs, fit_s=round(time.time() - t0))
        (out / f"kernel_{model}.json").write_text(json.dumps(res, indent=1))
        print(f"{model}: mean energy RMSE {res['mean_window_rmse_MeV']:.2f} MeV (R2 {res['mean_window_R2']:.3f}), "
              f"P70 RMSE {res['P70_rmse']:.3f} (R2 {res['P70_R2']:.3f})")


if __name__ == "__main__":
    main()
