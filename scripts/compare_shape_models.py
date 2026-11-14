"""Compare spectral shape models and per-bin band calibration on held-out shots.

Variants: pca (current two-part model), pca+cal, peak (explicit peak position/height/width), peak+cal.
Metrics: point MSE (mean of the predictive distribution), beam-energy error, 90%-band coverage per
energy region, mean per-bin CRPS, and how sharp the sampled peaks are compared with measured ones.
Usage: python scripts/compare_shape_models.py --split time   (or paper)
Writes results/stage1/shape_models_<split>.json
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

from mlsfqed.beam_model.peaks import fit_peaks  # noqa: E402
from mlsfqed.beam_model.stage1 import beam_energy  # noqa: E402
from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser, TwoPartBeamModel  # noqa: E402
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split  # noqa: E402

OUT = ROOT / "results" / "stage1"
REGIONS = {"305-600": (305, 600), "600-800": (600, 800), "800-1000 (peak)": (800, 1000), "1000-1400": (1000, 1401)}


def crps_bins(samples, Y):
    """Mean over bins and shots of the sample CRPS; samples (n, m, bins)."""
    t1 = np.abs(samples - Y[:, None]).mean(1)
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    w = (2 * np.arange(1, m + 1) - m - 1)[None, :, None]
    t2 = (w * s).sum(1) / m ** 2                       # = 0.5 E|X-X'|
    return float((t1 - t2).mean())


def peak_height(Y, E):
    """Height of the normalised spectrum's highest point in 700-1100 MeV."""
    S = Y / beam_energy(Y, E)[..., None]
    m = (E >= 700) & (E <= 1100)
    return S[..., m].max(-1)


def evaluate(samples, Y, E):
    lo, hi = np.percentile(samples, [5, 95], axis=1)
    inside = (Y >= lo) & (Y <= hi)
    point = samples.mean(1)
    W, Wp = beam_energy(Y, E), beam_energy(point, E)
    out = {"MSE": float(np.mean((point - Y) ** 2)),
           "beam_energy_rel_rms": float(np.sqrt(np.mean((Wp - W) ** 2 / W ** 2))),
           "coverage90_all_bins": float(inside.mean()),
           "coverage90_on_beam": float(inside[Y > 0.2 * Y.max(1, keepdims=True)].mean()),
           "CRPS_per_bin": crps_bins(samples, Y)}
    for name, (a, b) in REGIONS.items():
        m = (E >= a) & (E < b)
        out[f"coverage90_{name}"] = float(inside[:, m].mean())
    ph_meas, ph_samp = peak_height(Y, E), peak_height(samples, E)
    out["peak_height_measured_mean"] = float(ph_meas.mean())
    out["peak_height_samples_mean"] = float(ph_samp.mean())
    out["peak_height_point_pred_mean"] = float(peak_height(point, E).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["paper", "time"], default="time")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = load_all()
    E = d["E_MeV"]
    Y = physical_spectra(d)
    tr, te = (d["split"] == "train", d["split"] == "test") if a.split == "paper" else time_ordered_split(d)
    rf = RawFeaturiser().fit(d, tr)
    Xtr, Xte = rf.transform(d, tr), rf.transform(d, te)
    res = {"split": a.split, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    for name, cls in (("pca", TwoPartBeamModel), ("peak", PeakBeamModel)):
        t0 = time.time()
        m = cls(E, seed=a.seed).fit(Xtr, Y[tr]).calibrate()
        raw = m.sample(Xte, a.samples, a.seed)
        cal = m.sample_calibrated(Xte, a.samples, a.seed)
        res[name] = evaluate(raw, Y[te], E)
        res[name + "+cal"] = evaluate(cal, Y[te], E)
        res[name + "_k_bins"] = {"min": float(m.k_bins.min()), "median": float(np.median(m.k_bins)),
                                 "max": float(m.k_bins.max())}
        res[name + "_seconds"] = round(time.time() - t0, 1)
        np.save(OUT / f"_samples_{name}_{a.split}.npy", cal[:, :50].astype(np.float32))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"shape_models_{a.split}.json").write_text(json.dumps(res, indent=2))
    keys = ["MSE", "beam_energy_rel_rms", "CRPS_per_bin", "coverage90_all_bins", "coverage90_on_beam",
            *[f"coverage90_{r}" for r in REGIONS], "peak_height_samples_mean", "peak_height_point_pred_mean"]
    print(f"split={a.split}  measured mean peak height {res['pca']['peak_height_measured_mean']:.2f}")
    print(f"{'metric':34s}" + "".join(f"{v:>10s}" for v in ("pca", "pca+cal", "peak", "peak+cal")))
    for k in keys:
        print(f"{k:34s}" + "".join(f"{res[v][k]:10.4f}" for v in ("pca", "pca+cal", "peak", "peak+cal")))
    print("k_bins", {n: res[n + "_k_bins"] for n in ("pca", "peak")}, "seconds", res["pca_seconds"], res["peak_seconds"])


if __name__ == "__main__":
    main()
