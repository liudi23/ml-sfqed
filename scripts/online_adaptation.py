"""Use earlier shots of the same run while predicting it: adaptive calibration and rolling retraining.

Every prediction for a shot uses only data measured before it (training runs + earlier shots of the
current run). Deployment assumption: reference shots (no collision) are measured during the run.

  --stage calib-tune   train runs 05-10, run11 online: choose step size gamma and mode
  --stage calib-test   train runs 05-12, run13 online with the chosen setting
  --stage retrain --block N   rolling retraining on run13, block N of 20 shots (see below)
  --stage retrain-report
Results go to results/stage1/online_*.json; the work files to results/stage1/_online/ (gitignored).
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
sys.path.insert(0, str(ROOT / "scripts"))

from compare_shape_models import crps_bins, evaluate

from mlsfqed.beam_model.online import cqr_scores, stretched_band, track
from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser
from mlsfqed.data.gemini import load_all, physical_spectra

OUT = ROOT / "results" / "stage1"
WORK = OUT / "_online"
RUNS = ["run05", "run06", "run08", "run09", "run10", "run11", "run12", "run13"]
GAMMAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
MODES = ["per_bin", "global"]
N_SAMPLES = 200
BLOCK = 20


def run_mask(d, names):
    return np.isin(d["run"], [f"20210620/{r}" for r in names])


def base_samples(d, E, Y, train_runs, test_run, seed=0, members=20):
    tr, te = run_mask(d, train_runs), run_mask(d, [test_run])
    rf = RawFeaturiser().fit(d, tr)
    m = PeakBeamModel(E, n_members=members, seed=seed).fit(rf.transform(d, tr), Y[tr]).calibrate()
    return m.sample_calibrated(rf.transform(d, te), N_SAMPLES, seed), Y[te]


def coverage_report(samples, Y, q):
    lo, hi = stretched_band(samples, q)
    inside = (Y >= lo) & (Y <= hi)
    on = Y > 0.2 * Y.max(1, keepdims=True)
    half = len(Y) // 2
    width = np.mean((hi - lo) / (Y.max(1, keepdims=True)))
    return {"coverage_all": float(inside.mean()), "coverage_on_beam": float(inside[on].mean()),
            "coverage_first_half": float(inside[:half].mean()), "coverage_second_half": float(inside[half:].mean()),
            "mean_band_width_rel": float(width)}


def calib(stage):
    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    if stage == "calib-tune":
        train, test = RUNS[:5], "run11"
    else:
        train, test = RUNS[:7], "run13"
    t0 = time.time()
    S, Yt = base_samples(d, E, Y, train, test)
    sc = cqr_scores(S, Yt)
    res = {"train_runs": train, "test_run": test, "n_test": len(Yt),
           "fixed (training calibration)": coverage_report(S, Yt, np.zeros_like(sc))}
    settings = [(g, m) for g in GAMMAS for m in MODES]
    if stage == "calib-test":
        chosen = json.loads((OUT / "online_calib_tune.json").read_text())["chosen"]
        settings = [(chosen["gamma"], chosen["mode"])]
    for g, m in settings:
        q, _ = track(sc, gamma=g, mode=m)
        res[f"adaptive gamma={g} {m}"] = coverage_report(S, Yt, q)
    if stage == "calib-tune":
        cands = {k: v for k, v in res.items() if k.startswith("adaptive")}
        # closest to 90% on beam; ties broken by narrower band
        best = min(cands, key=lambda k: (round(abs(cands[k]["coverage_on_beam"] - 0.9), 3),
                                         cands[k]["mean_band_width_rel"]))
        g, m = best.split()[1].split("=")[1], best.split()[2]
        res["chosen"] = {"gamma": float(g), "mode": m, "key": best}
    res["seconds"] = round(time.time() - t0, 1)
    name = "online_calib_tune.json" if stage == "calib-tune" else "online_calib_test.json"
    (OUT / name).write_text(json.dumps(res, indent=2))
    print(f"train {train} -> online on {test} ({len(Yt)} shots)")
    print(f"{'setting':34s}{'all bins':>10s}{'on beam':>10s}{'1st half':>10s}{'2nd half':>10s}{'width':>8s}")
    for k, v in res.items():
        if isinstance(v, dict) and "coverage_all" in v:
            print(f"{k:34s}{v['coverage_all']:10.3f}{v['coverage_on_beam']:10.3f}"
                  f"{v['coverage_first_half']:10.3f}{v['coverage_second_half']:10.3f}{v['mean_band_width_rel']:8.3f}")
    if "chosen" in res:
        print("chosen on run11:", res["chosen"])


def retrain_block(block, members):
    """Block b covers run13 shots [b*20, (b+1)*20); trained on runs 05-12 + run13 shots before the block."""
    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    tr0 = run_mask(d, RUNS[:7])
    idx13 = np.flatnonzero(run_mask(d, ["run13"]))
    blk = idx13[block * BLOCK:(block + 1) * BLOCK]
    if len(blk) == 0:
        print("no shots in block", block)
        return
    tr = tr0.copy()
    tr[idx13[:block * BLOCK]] = True
    te = np.zeros(len(Y), bool)
    te[blk] = True
    rf = RawFeaturiser().fit(d, tr)
    t0 = time.time()
    m = PeakBeamModel(E, n_members=members, seed=0).fit(rf.transform(d, tr), Y[tr]).calibrate()
    S = m.sample_calibrated(rf.transform(d, te), N_SAMPLES, 0)
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / f"retrain_block{block}.npy", S.astype(np.float32))
    print(f"block {block}: {len(blk)} shots, trained on {tr.sum()} shots ({block * BLOCK} from run13), "
          f"{time.time() - t0:.0f} s")


def retrain_report(members):
    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    idx13 = np.flatnonzero(run_mask(d, ["run13"]))
    n_blocks = int(np.ceil(len(idx13) / BLOCK))
    S = np.concatenate([np.load(WORK / f"retrain_block{b}.npy") for b in range(n_blocks)])
    S0 = np.load(WORK / "retrain_block0.npy")  # block 0 has no run13 data: it IS the fixed model on those shots
    # fixed model on all run13 shots = the block-0 model; refit it once for the remaining shots
    fixed = WORK / "fixed_all.npy"
    if not fixed.exists():
        tr0 = run_mask(d, RUNS[:7])
        rf = RawFeaturiser().fit(d, tr0)
        m = PeakBeamModel(E, n_members=members, seed=0).fit(rf.transform(d, tr0), Y[tr0]).calibrate()
        X13 = rf.transform(d, run_mask(d, ["run13"]))
        # sample block by block, as the rolling models do, so both use identical random draws
        F = np.concatenate([m.sample_calibrated(X13[b * BLOCK:(b + 1) * BLOCK], N_SAMPLES, 0) for b in range(n_blocks)])
        np.save(fixed, F.astype(np.float32))
    F = np.load(fixed)
    assert np.allclose(F[:len(S0)], S0, atol=1e-5), "block 0 should equal the fixed model"
    Yt = Y[idx13]
    later = slice(BLOCK, None)                                     # shots where retraining had run13 data
    res = {"note": "shots 20-113 of run13; models with 0 vs 20-100 earlier run13 shots in training"}
    for name, Z in (("fixed", F), ("rolling retrain", S)):
        r = evaluate(Z[later], Yt[later], E)
        r["CRPS_per_bin"] = crps_bins(Z[later], Yt[later])
        res[name] = r
    (OUT / "online_retrain.json").write_text(json.dumps(res, indent=2))
    keys = ["MSE", "beam_energy_rel_rms", "CRPS_per_bin", "coverage90_all_bins", "coverage90_on_beam",
            "coverage90_800-1000 (peak)", "peak_height_samples_mean", "peak_height_measured_mean"]
    print(f"{'metric':30s}{'fixed':>12s}{'rolling':>12s}")
    for k in keys:
        print(f"{k:30s}{res['fixed'][k]:12.4f}{res['rolling retrain'][k]:12.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calib-tune", "calib-test", "retrain", "retrain-report"])
    ap.add_argument("--block", type=int)
    ap.add_argument("--members", type=int, default=20)
    a = ap.parse_args()
    if a.stage.startswith("calib"):
        calib(a.stage)
    elif a.stage == "retrain":
        retrain_block(a.block, a.members)
    else:
        retrain_report(a.members)


if __name__ == "__main__":
    main()
