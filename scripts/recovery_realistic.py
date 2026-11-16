"""Fair recovery test of the summary-statistic model comparison: realistic incoming-beam error.

  python scripts/recovery_realistic.py            # ~30 min on 2 cores, a few minutes on 12

The earlier recovery test (compare_models.py recovery) fitted each simulated shot with its true incoming beam.
Here, as for a real collision shot, the incoming spectrum is only predicted: the true one plus a shape residual
measured on held-out Gemini shots (scripts/beam_shape_residuals.py). The likelihood uses 25 further residual
samples around that prediction and the measured beam error; laser waist, duration, beam radius and divergence
stay at nominal values. 15 held-out Ptarmigan runs per true model (energy loss >= 5 MeV), the three 4,096-run
surrogates in models/. Writes results/model_comparison/recovery_realistic.json; scripts/make_figures.py turns it
into the expected 10-shot evidence for each true model (fair_recovery.png).
"""

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
MODELS = ("stochastic", "continuous", "classical")


def task(args):
    import pickle

    from mlsfqed.inference.posterior import ShotPosterior
    warnings.filterwarnings("ignore")
    true, run_id, obs_mean, obs_p70, beams, rmse = args
    z = {}
    for m in MODELS:
        with open(ROOT / "models" / f"surrogate_{m}_4096.pkl", "rb") as f:
            sur = pickle.load(f)
        post = ShotPosterior(sur, beams, obs_mean, obs_p70, 10.0, free=("a0", "dz_um", "length_um", "r_offset_um"),
                             beam_rmse=rmse, dz_prior="focus")
        z[m] = post.log_evidence(n=6000, seed=1)[0]
    return {"true": true, "run_id": run_id, "bf_s": z["stochastic"] - z["classical"], "bf_c": z["continuous"] - z["classical"],
            "sel": max(MODELS, key=lambda m: z[m])}


def main():
    from fullspec_train import beam_split, load
    R = np.load(ROOT / "results" / "inference" / "beam_shape_residuals.npz")["residuals"].astype(float)
    err = json.loads((ROOT / "results" / "inference" / "beam_prediction_error.json").read_text())
    rmse = (err["mean_window_MeV"]["rmse"], err["P70"]["rmse"])
    rng = np.random.default_rng(0)
    tasks = []
    for true in MODELS:
        r, _, Nin, _, E_in, _, shot = load(true, 4096)
        _, te = beam_split(shot)
        loss = (Nin * E_in).sum(1) / Nin.sum(1) - r.e_mean_window_MeV.values
        for i in rng.choice(np.flatnonzero(te & (loss >= 5.0)), 15, replace=False):
            inc = Nin[i] / np.trapezoid(Nin[i], E_in)
            pred = np.clip(inc + R[rng.integers(len(R))], 0, None)
            beams = np.clip(pred[None, :] + R[rng.choice(len(R), 25)], 0, None) * E_in[None, :]
            tasks.append((true, int(r.run_id[i]), float(r.e_mean_window_MeV.iloc[i]), float(r.e_P70.iloc[i]), beams, rmse))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(task, tasks))
    out = ROOT / "results" / "model_comparison" / "recovery_realistic.json"
    out.write_text(json.dumps({"setup": __doc__.split("\n\n")[1].replace("\n", " "), "runs": rows}, indent=1))
    for t in MODELS:
        b = np.array([x["bf_s"] for x in rows if x["true"] == t])
        print(f"true {t}: stochastic chosen {np.mean([x['sel'] == 'stochastic' for x in rows if x['true'] == t]):.0%}, "
              f"median log BF stochastic vs classical {np.median(b):+.2f}")


if __name__ == "__main__":
    main()
