"""Stage 3: train and evaluate surrogates of the Ptarmigan scan.

  python scripts/stage3_surrogate.py                 # all splits, full models
  python scripts/stage3_surrogate.py --quick         # small models, for a smoke test
  python scripts/stage3_surrogate.py --split beam    # one split: random | beam | a0

Splits:
  random  20% of runs held out at random
  beam    20% of the measured shots held out, with every beam drawn from them (new beams)
  a0      train on a0 <= 16.5, test on a0 > 16.5 (extrapolation to the strongest fields)
Metrics per observable: R^2, RMSE, RMSE relative to the Monte Carlo noise floor, 90% coverage of the
MLP ensemble's interval, and the speed-up over Ptarmigan. Writes results/stage3/metrics_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlsfqed.surrogate.data import NOISE, build
from mlsfqed.surrogate.models import MLPEnsemble, SpectrumSurrogate, TreeSurrogate

OUT = ROOT / "results" / "stage3"


def split_masks(name, X, meta, seed=0):
    rng = np.random.default_rng(seed)
    if name == "random":
        test = rng.random(len(X)) < 0.2
    elif name == "beam":
        shots = meta.beam_shot_index.unique()
        held = rng.choice(shots, int(0.2 * len(shots)), replace=False)
        test = meta.beam_shot_index.isin(held).values
    elif name == "a0":
        test = (X.a0 > 16.5).values
    else:
        raise ValueError(name)
    return ~test, test


def scores(P, Yte, cols, sd=None):
    out = {}
    for j, c in enumerate(cols):
        y, p = Yte[:, j], P[:, j]
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        r = {"R2": float(1 - np.mean((p - y) ** 2) / np.var(y)), "RMSE": rmse}
        if c in NOISE:
            r["RMSE_over_MC_noise"] = rmse / NOISE[c]
        if sd is not None:
            # interval = ensemble spread combined with the Monte Carlo noise of the target itself
            s = np.sqrt(sd[:, j] ** 2 + NOISE.get(c, 0.0) ** 2)
            r["coverage90"] = float(np.mean(np.abs(p - y) <= 1.645 * s))
        out[c] = r
    return out


def run_split(name, X, Y, meta, Ne, Ng, quick):
    tr, te = split_masks(name, X, meta)
    cols = list(Y.columns)
    res = {"split": name, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    t0 = time.time()
    tree = TreeSurrogate(quick=quick).fit(X[tr], Y[tr])
    res["tree_fit_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    P = tree.predict(X[te])
    res["tree_predict_s_per_sample"] = (time.time() - t0) / te.sum()
    res["tree"] = scores(P, Y[te].values, cols)
    t0 = time.time()
    mlp = MLPEnsemble(quick=quick).fit(X[tr], Y[tr])
    res["mlp_fit_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    Pm, Sm = mlp.predict(X[te])
    res["mlp_predict_s_per_sample"] = (time.time() - t0) / te.sum()
    res["mlp"] = scores(Pm, Y[te].values, cols, Sm)
    # spectra: electron spectrum per electron (150 bins); score mean energy implied by it and shape error
    spec = SpectrumSurrogate(quick=quick).fit(X[tr], Ne[tr])
    Sp = spec.predict(X[te])
    Ec = np.arange(5, 1500, 10.0)
    true_mean = (Ne[te] * Ec).sum(1) / Ne[te].sum(1)
    pred_mean = (Sp * Ec).sum(1) / Sp.sum(1)
    res["electron_spectrum"] = {"rel_L1": float(np.mean(np.abs(Sp - Ne[te]).sum(1) / Ne[te].sum(1))),
                                "mean_energy_RMSE_MeV": float(np.sqrt(np.mean((pred_mean - true_mean) ** 2)))}
    res["ptarmigan_mean_wall_s"] = float(meta.wall_s[te].mean())
    res["speedup_tree"] = res["ptarmigan_mean_wall_s"] / res["tree_predict_s_per_sample"]
    res["speedup_mlp"] = res["ptarmigan_mean_wall_s"] / res["mlp_predict_s_per_sample"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "beam", "a0", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    X, Y, meta, Ne, Ng = build()
    print(f"{len(X)} runs, {X.shape[1]} inputs, {Y.shape[1]} observables")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (["random", "beam", "a0"] if a.split == "all" else [a.split]):
        r = run_split(name, X, Y, meta, Ne, Ng, a.quick)
        tag = "_quick" if a.quick else ""
        (OUT / f"metrics_{name}{tag}.json").write_text(json.dumps(r, indent=2))
        print(f"\n== split {name}: train {r['n_train']}, test {r['n_test']} | speed-up tree {r['speedup_tree']:.0f}x, "
              f"MLP {r['speedup_mlp']:.0f}x (Ptarmigan {r['ptarmigan_mean_wall_s']:.1f} s/run)")
        print(f"{'observable':22s}{'tree R2':>9s}{'MLP R2':>9s}{'MLP RMSE/noise':>16s}{'MLP cov90':>11s}")
        for c in Y.columns:
            t, m = r["tree"][c], r["mlp"][c]
            print(f"{c:22s}{t['R2']:9.3f}{m['R2']:9.3f}{m.get('RMSE_over_MC_noise', float('nan')):16.2f}{m['coverage90']:11.2f}")
        es = r["electron_spectrum"]
        print(f"electron spectrum: relative L1 {es['rel_L1']:.3f}, implied mean-energy RMSE {es['mean_energy_RMSE_MeV']:.2f} MeV")


if __name__ == "__main__":
    main()
