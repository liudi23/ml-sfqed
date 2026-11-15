"""Beam-chirp sensitivity: does an energy-position correlation in the electron bunch change the observables?

The scan (and surrogate) assume no chirp. Los et al. (HPLSE 2024) found chirp negligible for their
observables; Magnusson et al. (PRAB 2023) found a large effect on the energy spread. Here: take N collisions
from the stochastic scan with a clear interaction (energy loss > 10 MeV), rerun them with Ptarmigan's
energy_chirp = -0.5, -0.2, +0.2, +0.5 (same seeds; positive = head of the bunch more energetic), and compare
each observable with the no-chirp run, against the surrogate error and the measurement uncertainty.
  python scripts/chirp_check.py --n 64        (~3-5 min on a 12-core laptop)
Writes results/chirp/chirp_check.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlsfqed.simulation.ptarmigan import run_one

OUT = ROOT / "results" / "chirp"
CHIRPS = (-0.5, -0.2, 0.2, 0.5)
OBS = {"loss_MeV": "energy loss", "e_std_MeV": "energy spread", "e_P70": "P70",
       "g_crit_MeV": "photon critical energy", "log_g_number_per_e": "photon yield (log10)"}
# measurement sd: spectrometer energy uncertainty (9-14 MeV for the Los et al. shots) and 5% on P70
MEASUREMENT_SD = {"loss_MeV": 11.0, "e_P70": 0.09}


def surrogate_rmse():
    """Surrogate RMSE on new beams, from the latest Stage 3 evaluation."""
    m = json.loads((ROOT / "results" / "stage3" / "metrics_beam.json").read_text())["mlp"]
    return {k: m[k]["RMSE"] for k in ("loss_MeV", "e_P70", "g_crit_MeV", "log_g_number_per_e", "e_std_MeV")}


def _job(args):
    row, spec, E, q, chirp = args
    p = {k: row[k] for k in ("a0", "waist_um", "fwhm_fs", "dx_um", "dy_um", "dz_um", "radius_um", "length_um", "div_mrad")}
    p.update(charge_pC=float(q), chirp=chirp)
    o, _, _ = run_one(p, spec, E, 5000, seed=int(row["run_id"]) + 1)
    return int(row["run_id"]), chirp, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    base = ROOT / "data" / "processed" / "stage2"
    runs = pd.read_parquet(base / "runs.parquet")
    runs["loss_MeV"] = runs.beam_mean_MeV - runs.e_mean_MeV
    runs["log_g_number_per_e"] = np.log10(runs.g_number_per_e)
    pick = runs[runs.loss_MeV > 10].sample(a.n, random_state=0)
    pool = np.load(base / "beam_pool.npz")
    E, spec, Q = pool["E_MeV"], pool["dNdE_per_MeV"], pool["Q_pC"]
    jobs = [(r, spec[int(r["beam_index"])], E, Q[int(r["beam_index"])], c) for r in pick.to_dict("records") for c in CHIRPS]
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_job, jobs))
    ref = pick.set_index("run_id")
    rows = []
    for rid, c, o in out:
        if not o["ok"]:
            continue
        r = {"run_id": rid, "chirp": c, "loss_ref": ref.loc[rid, "loss_MeV"]}
        o["loss_MeV"] = ref.loc[rid, "beam_mean_MeV"] - o["e_mean_MeV"]
        o["log_g_number_per_e"] = np.log10(o["g_number_per_e"])
        for k in OBS:
            r[f"d_{k}"] = o[k] - ref.loc[rid, k]
        rows.append(r)
    df = pd.DataFrame(rows)
    rmse = surrogate_rmse()
    res = {"n_collisions": a.n, "chirps": CHIRPS, "surrogate_rmse": rmse, "median_loss_MeV": float(pick.loss_MeV.median()), "effects": {}}
    print(f"{a.n} collisions with energy loss > 10 MeV (median {pick.loss_MeV.median():.0f} MeV); change vs no chirp")
    print(f"{'observable':24s}" + "".join(f"{'chirp ' + format(c, '+.1f'):>18s}" for c in CHIRPS) + f"{'surrogate RMSE':>16s}{'meas. sd':>10s}")
    for k, lab in OBS.items():
        cells, eff = [], {}
        for c in CHIRPS:
            d = df.loc[df.chirp == c, f"d_{k}"]
            eff[str(c)] = {"mean": float(d.mean()), "rms": float(np.sqrt((d ** 2).mean()))}
            cells.append(f"{d.mean():+8.3g} (rms {np.sqrt((d ** 2).mean()):.2g})")
        res["effects"][k] = eff
        print(f"{lab:24s}" + "".join(f"{x:>18s}" for x in cells) + f"{rmse.get(k, float('nan')):16.3g}{MEASUREMENT_SD.get(k, float('nan')):10.3g}")
    print("(rms includes Monte Carlo noise: compare with the seed-to-seed noise ~1.3 MeV loss, ~4% P70)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chirp_check.json").write_text(json.dumps(res, indent=2))
    df.to_csv(OUT / "chirp_runs.csv", index=False)


if __name__ == "__main__":
    main()
