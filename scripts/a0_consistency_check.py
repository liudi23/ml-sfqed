"""Why is our effective a0 lower than Los et al.'s? Direct Ptarmigan check, no surrogate.

  python scripts/a0_consistency_check.py            # ~1 min on 4 cores

For each of the 10 public shots: the median predicted incoming beam, a head-on collision at focus
(dz = 0, transverse offset 0, waist 2.5 um, 45 fs, source size 0.68 um, length 5 um), quantum-stochastic
radiation reaction, a0 = 3, 5, 7, 9, 12. Records the outgoing mean energy above 305 MeV and the photon
critical energy, and finds the a0 at focus that reproduces the measured mean energy.
Result (15 Nov 2026): shots 3, 4, 6 (34-43 MeV implied loss) need a0 = 7.2, 8.2, 8.1, against Los et al.'s
effective <a0~> 6.7, 9.4, 7.2 (their shot n = our shot n-1); shot 0 (77 MeV) needs ~12.5 (theirs 7.8); shots whose
predicted incoming mean is within ~20 MeV of the measured mean (about the incoming-beam error, 18 MeV) give a0 3.5-5.5
or are undetermined. Los et al. also fit the gamma-ray signal, which constrains a0 independently. The public photon
spectra are not directly usable here: a critical energy fitted to them is 45-60 MeV on every shot vs 13-35 MeV
simulated, i.e. the detector response must be forward-modelled first. Writes results/inference/a0_consistency_check.json.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT + "/src")
from mlsfqed.inference.los2024 import beams_on_pool_grid, load, spectrum_observables
from mlsfqed.simulation.ptarmigan import run_one

pool = np.load(ROOT + "/data/processed/stage2/beam_pool.npz")
Ep = pool["E_MeV"]
Q = float(np.median(pool["Q_pC"]))
data = load()
E = data["E_MeV"]
B = beams_on_pool_grid(data, Ep)


def task(t):
    k, a0, w = t
    b = B[k]
    means = np.array([spectrum_observables(Ep, x / Ep)["mean_window_MeV"] for x in b])
    bm = b[np.argsort(means)[50]]
    p = {
        "a0": a0,
        "waist_um": w,
        "fwhm_fs": 45.0,
        "radius_um": 0.68,
        "length_um": 5.0,
        "div_mrad": 2.0,
        "dx_um": 0.0,
        "dy_um": 0.0,
        "dz_um": 0.0,
        "charge_pC": Q,
    }
    o, _, _ = run_one(p, bm / Ep, Ep, 4000, seed=11, model="stochastic")
    return {
        "shot": k,
        "a0": a0,
        "waist": w,
        "mean": o["e_mean_window_MeV"],
        "g_crit": o["g_crit_MeV"],
        "inc_mean": float(np.median(means)),
        "obs_mean": spectrum_observables(E, data["shots"][k]["post_dNdE"])["mean_window_MeV"],
    }


if __name__ == "__main__":
    shots = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(10))
    tasks = [(k, a0, w) for k in shots for w in (2.5,) for a0 in (3.0, 5.0, 7.0, 9.0, 12.0)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(task, tasks))
    with open(Path(ROOT) / "results" / "inference" / "a0_consistency_check.json", "w") as f:
        json.dump(res, f, indent=1)
    for k in shots:
        rk = sorted((r for r in res if r["shot"] == k), key=lambda r: r["a0"])
        a0s = np.array([r["a0"] for r in rk])
        sim = np.array([r["mean"] for r in rk])
        obs, inc = rk[0]["obs_mean"], rk[0]["inc_mean"]
        # sim falls with a0; interpolate the a0 that gives the observed mean (nan if out of range)
        a0_match = float(np.interp(-obs, -sim, a0s, left=np.nan, right=np.nan))
        print(
            f"shot {k}: incoming {inc:.1f}  observed {obs:.1f}  loss {inc - obs:5.1f} MeV  a0 at focus {a0_match:.1f}"
        )
