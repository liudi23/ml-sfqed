"""Full-spectrum inference: recovery tests on held-out Ptarmigan runs, and the 10 public collision shots.

  python scripts/fullspec_infer.py recovery --n 20       # 20 held-out runs per true model (~20 min on 12 cores)
  python scripts/fullspec_infer.py shots                 # the 10 Los et al. shots, both beam uncertainties (~15 min)

Needs scripts/fullspec_train.py (kernels) and scripts/beam_shape_residuals.py (measured incoming-shape errors).
Recovery: the 'measured' spectrum is a held-out Ptarmigan run; the 'predicted' incoming spectrum is the true one
plus a measured shape residual, so the test sees a realistic incoming-beam error; laser waist, duration, beam
radius and divergence stay at nominal values, not the run's own. Real shots: the incoming spectrum is either the
100 published predictions as given ('published') or their median plus measured residuals ('measured'). With the
measured beam error the incoming energy scale is also free (prior width --beam-scale-sd, default 4%, the measured
error of the predicted spectral edge), because the full spectrum is most sensitive to the edge position.
Writes results/fullspec/recovery.json and results/fullspec/shots.json.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODELS = ("stochastic", "continuous", "classical")
OUT = ROOT / "results" / "fullspec"


def setup():
    import torch

    from mlsfqed.surrogate.kernel import TransferKernel
    torch.set_num_threads(1)
    pool = np.load(ROOT / "data" / "processed" / "stage2" / "beam_pool.npz")
    s = np.load(ROOT / "data" / "processed" / "stage2" / "spectra.npz")
    K, sd = {}, {}
    for m in MODELS:
        k = TransferKernel(pool["E_MeV"], s["E_edges"])
        k.net.load_state_dict(torch.load(ROOT / "models" / f"kernel_{m}.pt"))
        k.net.eval()
        K[m] = k
        sd[m] = np.array(json.loads((OUT / f"kernel_{m}.json").read_text())["surr_sd"])
    R = np.load(ROOT / "results" / "inference" / "beam_shape_residuals.npz")["residuals"].astype(float)
    return K, sd, R, pool["E_MeV"]


def a0_eff_true(t):
    zR = np.pi * t[1] ** 2 / 0.8
    return t[0] / np.sqrt(1 + (t[5] / 2 / zR) ** 2) * np.sqrt(np.exp(-t[3] ** 2 / (t[1] ** 2 / 2 + t[6] ** 2)))


def _recovery_task(args):
    true, run_id, theta, obs_E, obs, pred, n_ev, steps, scale_sd = args
    from mlsfqed.inference.fullspec import FullSpectrumPosterior
    from mlsfqed.inference.posterior import run_mcmc
    K, sd, R, E_in = setup()
    t0 = time.time()
    row = {"true_model": true, "run_id": run_id, "true_a0": float(theta[0]), "true_a0_eff": float(a0_eff_true(theta))}
    for m in MODELS:
        post = FullSpectrumPosterior(K[m], E_in, pred[None, :], obs_E, obs, residuals=R, n_samples=100, surr_sd=sd[m],
                                     free=("a0", "dz_um", "length_um", "r_offset_um"), seed=run_id, beam_scale_sd=scale_sd)
        row[f"logZ_{m}"] = post.log_evidence(n=n_ev, seed=1)[0]
        if m == true:
            ch, _ = run_mcmc(post, nwalkers=32, nsteps=steps, burn=steps // 3, seed=0)
            for name, x in (("a0", ch[:, 0]), ("a0_eff", post.a0_eff(ch))):
                lo, med, hi = np.percentile(x, [5, 50, 95])
                row[name] = {"p5": float(lo), "p50": float(med), "p95": float(hi)}
    row["wall_s"] = time.time() - t0
    return row


def recovery(n, n_ev, steps, workers, scale_sd):
    from fullspec_train import beam_split, load
    _, _, R, E_in = setup()
    rng = np.random.default_rng(0)
    tasks = []
    for true in MODELS:
        r, th, Nin, Ne, _, edges, shot = load(true, 4096)
        _, te = beam_split(shot)
        loss = (Nin * E_in).sum(1) / Nin.sum(1) - r.e_mean_window_MeV.values
        c = 0.5 * (edges[1:] + edges[:-1])
        for i in rng.choice(np.flatnonzero(te & (loss >= 5.0)), n, replace=False):
            inc = Nin[i] / np.trapezoid(Nin[i], E_in)
            pred = np.clip(inc + R[rng.integers(len(R))], 0, None)
            tasks.append((true, int(r.run_id[i]), th[i], c, Ne[i] / 10.0, pred, n_ev, steps, scale_sd))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_recovery_task, tasks))
    for row in rows:
        row["selected"] = max(MODELS, key=lambda m, row=row: row[f"logZ_{m}"])
    summ = {"n_per_model": n, "beam_scale_sd": scale_sd, "identified": {m: float(np.mean([x["selected"] == m for x in rows if x["true_model"] == m])) for m in MODELS}}
    t = [x for x in rows]
    summ["a0_eff_coverage90"] = float(np.mean([x["a0_eff"]["p5"] <= x["true_a0_eff"] <= x["a0_eff"]["p95"] for x in t]))
    summ["a0_coverage90"] = float(np.mean([x["a0"]["p5"] <= x["true_a0"] <= x["a0"]["p95"] for x in t]))
    summ["confusion"] = {tm: {m: int(sum(x["selected"] == m for x in rows if x["true_model"] == tm)) for m in MODELS} for tm in MODELS}
    (OUT / "recovery.json").write_text(json.dumps({"summary": summ, "runs": rows}, indent=1))
    print(json.dumps(summ, indent=1))


def _shot_task(args):
    k, unc, n_ev, steps, scale_sd = args
    from mlsfqed.inference.fullspec import FullSpectrumPosterior
    from mlsfqed.inference.los2024 import beams_on_pool_grid, load
    from mlsfqed.inference.posterior import run_mcmc
    K, sd, R, E_in = setup()
    data = load()
    inc = beams_on_pool_grid(data, E_in)[k] / E_in[None, :]
    obs = data["shots"][k]["post_dNdE"]
    t0 = time.time()
    row = {"shot": k, "beam_unc": unc}
    for m in MODELS:
        post = FullSpectrumPosterior(K[m], E_in, inc, data["E_MeV"], obs, residuals=R if unc == "measured" else None,
                                     n_samples=100, surr_sd=sd[m], seed=k,
                                     beam_scale_sd=scale_sd if unc == "measured" else None)
        row[f"logZ_{m}"] = post.log_evidence(n=n_ev, seed=1)[0]
        ch, _ = run_mcmc(post, nwalkers=32, nsteps=steps, burn=steps // 3, seed=0)
        ae = post.a0_eff(ch)
        row[f"a0_eff_{m}"] = dict(zip(("p5", "p16", "p50", "p84", "p95"), np.percentile(ae, [5, 16, 50, 84, 95]).round(3).tolist()))
        mu, C = post.predict(np.median(ch, 0)[None, :])
        row[f"fit_{m}"] = {"pred": mu[0].numpy().round(5).tolist(),
                           "sd": np.sqrt(np.diag(C[0].numpy()) + np.diag(post.C0.numpy())).round(5).tolist()}
        row["obs"] = post.y.numpy().round(5).tolist()
    row["wall_s"] = time.time() - t0
    return row


def shots(n_ev, steps, workers, scale_sd, which=range(10)):
    from mlsfqed.inference.fullspec import EDGES
    tasks = [(k, unc, n_ev, steps, scale_sd) for unc in ("published", "measured") for k in which]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_shot_task, tasks))
    summ = {}
    for unc in ("published", "measured"):
        rr = [x for x in rows if x["beam_unc"] == unc]
        summ[unc] = {m: round(sum(x[f"logZ_{m}"] - x["logZ_classical"] for x in rr), 2) for m in ("stochastic", "continuous")}
    tag = "" if scale_sd == 0.04 else f"_scale{scale_sd or 0:g}"
    (OUT / f"shots{tag}{'' if len(rows) == 20 else '_partial'}.json").write_text(json.dumps(
        {"edges": EDGES.tolist(), "beam_scale_sd": scale_sd, "summary_logBF_vs_classical": summ, "shots": rows}, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["recovery", "shots"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-evidence", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--beam-scale-sd", type=float, default=0.04,
                    help="prior width of the incoming energy scale (measured beam error); 0 to fix it")
    ap.add_argument("--shots", default="0,1,2,3,4,5,6,7,8,9", help="comma-separated shot numbers")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "recovery":
        recovery(a.n, a.n_evidence, a.steps, a.workers, a.beam_scale_sd or None)
    else:
        shots(a.n_evidence, a.steps, a.workers, a.beam_scale_sd or None, [int(x) for x in a.shots.split(",")])


if __name__ == "__main__":
    main()
