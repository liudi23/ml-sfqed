"""Ptarmigan sweep of laser strength a0 for the three radiation-reaction models, for the README animation.

  python scripts/a0_sweep.py                 # ~10 min on 4 cores, ~3 min on 12

One measured Gemini beam (the one with median P70), one geometry (head-on at focus: waist 2.5 um, 45 fs,
beam radius 0.7 um, length 5 um, divergence 2 mrad), a0 = 2-9 in steps of 1 with 20,000 macro-electrons and
a0 = 10-16 in steps of 2 with 6,000 (run time grows ~a0^2). Same seed for every run.
Writes results/diagnostics/a0_sweep_spectra.json: energy-weighted spectra, normalised above 305 MeV and lightly
smoothed (Gaussian, one 10 MeV bin), plus P70 and mean energy above 305 MeV per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlsfqed.inference.los2024 import spectrum_observables
from mlsfqed.simulation.ptarmigan import E_BINS_E, run_one

POOL = ROOT / "data" / "processed" / "stage2" / "beam_pool.npz"
OUT = ROOT / "results" / "diagnostics" / "a0_sweep_spectra.json"
MODELS = ("classical", "continuous", "stochastic")
GRID = [(a, 20000) for a in range(2, 10)] + [(a, 6000) for a in (10, 12, 14, 16)]
EC = 0.5 * (E_BINS_E[1:] + E_BINS_E[:-1])
KEEP = (EC >= 300) & (EC <= 900)


def median_beam():
    pool = np.load(POOL)
    E, N, Q = pool["E_MeV"], pool["dNdE_per_MeV"], pool["Q_pC"]
    meas = np.flatnonzero(pool["source"] == "measured")
    p70 = np.array([spectrum_observables(E, N[i], 10.0)["P70"] for i in meas])
    i = int(meas[np.argsort(p70)[len(p70) // 2]])
    return i, E, N[i], float(Q[i])


def task(args):
    a0, n_el, mdl = args
    _, E, N, Q = median_beam()
    p = {"a0": a0, "waist_um": 2.5, "fwhm_fs": 45.0, "radius_um": 0.7, "length_um": 5.0, "div_mrad": 2.0,
         "dx_um": 0.0, "dy_um": 0.0, "dz_um": 0.0, "charge_pC": Q}
    o, Ne, _ = run_one(p, N, E, n_el, seed=7, model=mdl)
    if not o["ok"]:
        raise RuntimeError(f"a0={a0} {mdl}: {o['error']}")
    return {"a0": a0, "model": mdl, "Ne": Ne.tolist(), "P70": o["e_P70"], "mean_window": o["e_mean_window_MeV"]}


def norm(E, dNdE):
    w = dNdE * E
    win = E >= 305
    return w / np.trapezoid(w[win], E[win] / 1e3)


def smooth(y, s=1.0):
    k = np.arange(-4, 5)
    g = np.exp(-0.5 * (k / s) ** 2)
    return np.convolve(np.pad(y, 4, mode="edge"), g / g.sum(), mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    bi, E, N, _ = median_beam()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(task, [(a0, n, m) for a0, n in GRID for m in MODELS]))
    out = {"beam_index": bi, "E": EC[KEEP].round(0).tolist(),
           "incoming": np.interp(EC[KEEP], E, norm(E, N)).round(3).tolist(),
           "incoming_P70": round(spectrum_observables(E, N, 10.0)["P70"], 3),
           "incoming_mean": round(spectrum_observables(E, N)["mean_window_MeV"], 1), "frames": []}
    for a0, _ in GRID:
        fr = {"a0": a0}
        for r in (r for r in res if r["a0"] == a0):
            y = smooth(norm(EC, np.array(r["Ne"]) / np.diff(E_BINS_E)))
            fr[r["model"]] = {"y": y[KEEP].round(3).tolist(), "P70": round(r["P70"], 3), "mean": round(r["mean_window"], 1)}
        out["frames"].append(fr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
