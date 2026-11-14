"""Feature study: do recent-shot history or physics-based features improve the beam model?

Part 1: leave-one-run-out CV on training runs (05-11), then the held-out run13, for log beam energy,
        peak position and log peak height with feature sets raw / +physics / +recent shots / compact.
Part 2: are shot-to-shot prediction errors correlated, and does correcting by recent errors help?
Part 3: full peak model on run13, current features vs raw+physics+recent.

Result (14 Nov 2026): no meaningful gain; errors are nearly independent shot to shot, so the remaining
spread is not drift the history could correct. The default model keeps the raw features.

Usage: python scripts/feature_study.py [--part 1|2|3]          (all parts by default, ~4 min)
Saved output: python scripts/feature_study.py | tee results/stage1/feature_study.txt
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_shape_models import evaluate

from mlsfqed.beam_model.peaks import fit_peaks
from mlsfqed.beam_model.stage1 import beam_energy
from mlsfqed.beam_model.twopart import EnergyEnsemble, PeakBeamModel, RawFeaturiser
from mlsfqed.data.features import physics_features, recent_shot_features
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split

warnings.filterwarnings("ignore")
FEATURE_SETS = ["raw", "raw+lag", "raw+phys", "raw+phys+lag", "compact", "compact+lag"]
RUN13 = "20210620/run13"


def r2(p, y):
    return 1 - np.mean((p - y) ** 2) / np.var(y)


def setup():
    d = load_all()
    E = d["E_MeV"]
    Y = physical_spectra(d)
    W = beam_energy(Y, E)
    P = fit_peaks(Y / W[:, None], E)
    T = {"logW": np.log(W), "peak_pos": P[:, 1], "log_height": np.log(P[:, 0])}
    PH, _ = physics_features(d)
    LAG, _ = recent_shot_features(d, T, k=5)
    return d, E, Y, W, T, PH, LAG


def features(d, fs, tr, PH, LAG):
    R = RawFeaturiser().fit(d, tr).transform(d, np.ones(len(d["run"]), bool))
    base = np.column_stack([d["laser_E"], d["f40_data"]])
    parts = {"raw": [R], "raw+lag": [R, LAG], "raw+phys": [R, PH], "raw+phys+lag": [R, PH, LAG],
             "compact": [base, PH], "compact+lag": [base, PH, LAG]}[fs]
    return np.column_stack(parts)


def part1(d, W, T, PH, LAG):
    t0 = time.time()
    trall, te = time_ordered_split(d)
    runs = [r for r in sorted(set(d["run"][trall])) if (d["run"] == r).sum() > 10]
    print("Leave-one-run-out CV on training runs (pooled R2; higher is better)")
    print(f"{'features':14s}" + "".join(f"{k:>12s}" for k in T))
    for fs in FEATURE_SETS:
        preds = {k: np.full(len(W), np.nan) for k in T}
        for r in runs:
            tst = trall & (d["run"] == r)
            trn = trall & ~tst
            X = features(d, fs, trn, PH, LAG)
            for k, y in T.items():
                preds[k][tst] = EnergyEnsemble(n_members=5, seed=0).fit(X[trn], y[trn]).predict(X[tst])[0]
        m = trall & ~np.isnan(preds["logW"])
        print(f"{fs:14s}" + "".join(f"{r2(preds[k][m], y[m]):12.3f}" for k, y in T.items()))
    print("held-out run13 (R2; slope pred-vs-true)")
    for fs in FEATURE_SETS:
        X = features(d, fs, trall, PH, LAG)
        out = []
        for y in T.values():
            p = EnergyEnsemble(n_members=5, seed=0).fit(X[trall], y[trall]).predict(X[te])[0]
            out.append(f"{r2(p, y[te]):6.3f} /{np.polyfit(y[te], p, 1)[0]:5.2f}")
        print(f"{fs:14s}" + "".join(f"{o:>14s}" for o in out))
    print("seconds", round(time.time() - t0))


def part2(d, W, T):
    trall, te = time_ordered_split(d)
    runs = [r for r in sorted(set(d["run"][trall])) if (d["run"] == r).sum() > 10] + [RUN13]
    pred = {k: np.full(len(W), np.nan) for k in T}
    for r in runs:  # leave-one-run-out for training runs; run13 predicted from all training runs
        tst = d["run"] == r
        trn = trall & ~tst
        X = RawFeaturiser().fit(d, trn).transform(d, np.ones(len(W), bool))
        for k, y in T.items():
            pred[k][tst] = EnergyEnsemble(n_members=10, seed=0).fit(X[trn], y[trn]).predict(X[tst])[0]
    print("residual lag-1 autocorrelation within run (0 = shot-to-shot errors independent):")
    for k, y in T.items():
        res = y - pred[k]
        ac = []
        for r in runs:
            x = res[np.flatnonzero(d["run"] == r)]
            ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
        print(f"  {k:10s}", " ".join(f"{a:+.2f}" for a in ac), "(last = run13)")
    print("online bias correction: prediction + a * mean of the previous k residuals in the same run")
    cvm = trall & np.isin(d["run"], runs)
    for k, y in T.items():
        res = y - pred[k]
        for kk in (5, 20):
            corr = np.zeros(len(W))
            for r in runs:
                m = np.flatnonzero(d["run"] == r)
                for n, i in enumerate(m):
                    corr[i] = np.mean(res[m[max(0, n - kk):n]]) if n > 0 else 0.0
            # choose a on training runs only, then apply to run13
            a_grid = np.linspace(0, 1, 11)
            a = a_grid[np.argmax([r2(pred[k][cvm] + g * corr[cvm], y[cvm]) for g in a_grid])]
            print(f"  {k:10s} k={kk:2d}: best a on training runs {a:.1f} | "
                  f"CV R2 {r2(pred[k][cvm], y[cvm]):.3f} -> {r2(pred[k][cvm] + a * corr[cvm], y[cvm]):.3f} | "
                  f"run13 R2 {r2(pred[k][te], y[te]):.3f} -> {r2(pred[k][te] + a * corr[te], y[te]):.3f}")


def part3(d, E, Y, W, PH, LAG):
    tr, te = time_ordered_split(d)
    R = RawFeaturiser().fit(d, tr).transform(d, np.ones(len(W), bool))
    for name, X in (("current (raw)", R), ("raw+phys+lag", np.column_stack([R, PH, LAG]))):
        m = PeakBeamModel(E, seed=0).fit(X[tr], Y[tr]).calibrate()
        r = evaluate(m.sample_calibrated(X[te], 200, 0), Y[te], E)
        print(f"{name:14s} MSE {r['MSE']:.4f}  W err {r['beam_energy_rel_rms']:.3f}  CRPS {r['CRPS_per_bin']:.4f}  "
              f"cov all {r['coverage90_all_bins']:.3f} on-beam {r['coverage90_on_beam']:.3f}  "
              f"peak ht {r['peak_height_samples_mean']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    part = ap.parse_args().part
    d, E, Y, W, T, PH, LAG = setup()
    if part in (None, 1):
        part1(d, W, T, PH, LAG)
    if part in (None, 2):
        part2(d, W, T)
    if part in (None, 3):
        part3(d, E, Y, W, PH, LAG)


if __name__ == "__main__":
    main()
