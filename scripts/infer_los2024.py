"""Surrogate-based inference of collision parameters for Los et al.'s 10 public shots.

  python scripts/infer_los2024.py recovery --n 24        # simulated shots with known answers (do first)
  python scripts/infer_los2024.py shots                  # the 10 real shots, two set-ups
  add --quick to use the small smoke-test surrogate and short chains; --workers N for parallel tasks.

Incoming-beam uncertainty (--beam-unc):
  published   the spread of Los et al.'s 100 network predictions, as used in their analysis
  calibrated  that spread widened to the realistic prediction error measured on a held-out run
              (scripts/beam_prediction_error.py; their spread is ~3x too narrow)
  both        run both (default)
Set-ups for the real shots:
  los     a0, longitudinal offset, bunch length free; transverse offset fixed at 0 (as Los et al.)
  offset  the same plus the transverse offset free (what their cost did not allow)
Needs data/processed/models/surrogate.pkl (scripts/stage3_train_final.py) and data/raw/los2024
(scripts/fetch_los2024.sh). Writes results/inference/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")           # one thread per process: tasks run in parallel instead

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import h5py
from run_stage1 import check_fits

from mlsfqed.inference.los2024 import LOS, beams_on_pool_grid, load, spectrum_observables
from mlsfqed.inference.posterior import MODEL, ShotPosterior, load_model, run_mcmc
from mlsfqed.surrogate.data import COLLISION, build

OUT = ROOT / "results" / "inference"
LOS_A0_EFF_RANGE = (7.0, 13.0)     # Los et al. (Nat. Commun.): inferred effective a0 across the 10 shots
LOS_CPU_HOURS = 19200.0
SETUPS = {"los": ("a0", "dz_um", "length_um"), "offset": ("a0", "dz_um", "length_um", "r_offset_um")}


def model_path(quick):
    return MODEL.with_name("surrogate_quick.pkl") if quick else MODEL


def mcmc_kw(quick):
    return {"nwalkers": 24, "nsteps": 1500, "burn": 500} if quick else {"nwalkers": 32, "nsteps": 8000, "burn": 3000}


def q(x, ps=(5, 16, 50, 84, 95)):
    return dict(zip([f"p{p}" for p in ps], np.percentile(x, ps).round(4).tolist()))


# ---------------------------------------------------------------- recovery test
def _recovery_task(args):
    run_row, quick, seed, dz_prior = args
    m = load_model(model_path(quick))
    pool = np.load(ROOT / "data" / "processed" / "stage2" / "beam_pool.npz")
    beam = pool["dWdE_J_per_GeV"][[int(run_row["beam_index"])]]
    fixed = {k: run_row[k] for k in COLLISION if k not in SETUPS["offset"]}
    post = ShotPosterior(m, beam, run_row["obs_mean"], run_row["obs_p70"], meas_sd_mean=10.0,
                         free=SETUPS["offset"], fixed=fixed, dz_prior=dz_prior)
    t0 = time.time()
    chain, diag = run_mcmc(post, seed=seed, **mcmc_kw(quick))
    a_eff = post.a0_eff(chain)
    out = {"run_id": int(run_row["run_id"]), "true_a0": run_row["a0"], "true_a0_eff": run_row["a0_eff_est"],
           "true_loss_MeV": run_row["loss"], "wall_s": time.time() - t0, **diag,
           "a0": q(chain[:, 0]), "a0_eff": q(a_eff)}
    for name, true, s in (("a0", run_row["a0"], chain[:, 0]), ("a0_eff", run_row["a0_eff_est"], a_eff)):
        lo, hi = np.percentile(s, [5, 95])
        out[f"{name}_in90"] = bool(lo <= true <= hi)
    return out


def recovery(n, quick, workers, dz_prior):
    m = load_model(model_path(quick))
    X, Y, meta, _, _ = build()
    cal = meta.run_id.isin(m["calibration_run_ids"]).values          # runs the surrogate never trained on
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(cal & (Y.loss_window_MeV.values > 2.0)), n, replace=False)
    rows = []
    for i in pick:
        r = {k: float(X[k].iloc[i]) for k in X.columns}
        r.update(run_id=int(meta.run_id.iloc[i]), beam_index=int(meta.beam_index.iloc[i]),
                 loss=float(Y.loss_window_MeV.iloc[i]),
                 obs_mean=float(X.beam_mean_window_MeV.iloc[i] - Y.loss_window_MeV.iloc[i]),
                 obs_p70=float(Y.e_P70.iloc[i]))
        rows.append(r)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_recovery_task, [(r, quick, k, dz_prior) for k, r in enumerate(rows)]))
    bias = float(np.median([r["a0_eff"]["p50"] - r["true_a0_eff"] for r in res]))
    summ = {"n": n, "dz_prior": dz_prior, "a0_eff_median_bias": bias, "a0_coverage90": float(np.mean([r["a0_in90"] for r in res])),
            "a0_eff_coverage90": float(np.mean([r["a0_eff_in90"] for r in res])),
            "median_wall_s_per_shot": float(np.median([r["wall_s"] for r in res])),
            "total_wall_s": time.time() - t0, "quick": quick}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"recovery_{dz_prior}{'_quick' if quick else ''}.json").write_text(json.dumps({"summary": summ, "runs": res}, indent=2))
    print(f"recovery test on {n} simulated shots the surrogate never saw (4 parameters free, 2 observables), "
          f"dz prior: {dz_prior}:")
    print(f"  true value inside 90% interval: a0 {summ['a0_coverage90']:.2f}, effective a0 {summ['a0_eff_coverage90']:.2f}")
    print(f"  median wall time per shot: {summ['median_wall_s_per_shot']:.0f} s (one core)")
    w = np.array([r["a0_eff"]["p95"] - r["a0_eff"]["p5"] for r in res])
    print(f"  effective-a0 90% interval width: median {np.median(w):.1f}; median bias {bias:+.2f}")


# ---------------------------------------------------------------- real shots
def beam_rmse():
    f = OUT / "beam_prediction_error.json"
    if not f.exists():
        sys.exit("run scripts/beam_prediction_error.py first")
    r = json.loads(f.read_text())
    return (r["mean_window_MeV"]["rmse"], r["P70"]["rmse"])


def _shot_task(args):
    k, setup, unc, quick, K, seed, dz_prior = args
    m = load_model(model_path(quick))
    data = load()
    E = data["E_MeV"]
    beams = beams_on_pool_grid(data, m["pool_E_MeV"])[k]
    beams = beams[np.random.default_rng(seed).choice(len(beams), min(K, len(beams)), replace=False)]
    s = data["shots"][k]
    obs = spectrum_observables(E, s["post_dNdE"], smooth_mev=10.0)
    with h5py.File(LOS / "Bayesian_inference_data.h5", "r") as f:
        dE = f["shot 1/Electrons/Energy_uncertainty[MeV]"][()]
    meas_sd = float(np.trapezoid(s["post_dNdE"] * dE, E) / np.trapezoid(s["post_dNdE"], E))
    post = ShotPosterior(m, beams, obs["mean_window_MeV"], obs["P70"], meas_sd, free=SETUPS[setup],
                         beam_rmse=beam_rmse() if unc == "calibrated" else None, dz_prior=dz_prior)
    t0 = time.time()
    chain, diag = run_mcmc(post, seed=seed, **mcmc_kw(quick))
    wall = time.time() - t0
    a_eff = post.a0_eff(chain)
    rng = np.random.default_rng(1)
    sub = chain[rng.choice(len(chain), 200)]
    pred, var = post.predict(sub)
    # posterior predictive draws including every variance term (surrogate, measurement, beam widening);
    # PIT = fraction of draws below the observation (0.5 = centred; 5-95% expected for 90% of fits)
    draws = pred + rng.standard_normal(pred.shape) * np.sqrt(var + post.meas_var)
    pit = (draws < post.obs).reshape(-1, 2).mean(0)
    res = {"shot": s["shot"], "setup": setup, "beam_unc": unc, "beam_extra_sd": np.sqrt(post.beam_extra_var).tolist(),
           "wall_s": wall, "K_beams": len(beams), "meas_sd_mean_MeV": meas_sd,
           "obs_mean_window_MeV": obs["mean_window_MeV"], "obs_P70": obs["P70"],
           "pred_mean_window_MeV": q(pred[..., 0].ravel()), "pred_P70": q(pred[..., 1].ravel()),
           "a0_eff": q(a_eff), "pit_mean_window": float(pit[0]), "pit_P70": float(pit[1]), **diag}
    for i, name in enumerate(post.free):
        res[name] = q(chain[:, i])
    tag = ("_quick" if quick else "") + ("_dzuniform" if dz_prior == "uniform" else "")
    np.save(OUT / "chains" / f"shot{s['shot']}_{setup}_{unc}{tag}.npy", chain.astype(np.float32))
    return res


def shots(quick, workers, K, unc, dz_prior):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(exist_ok=True)
    uncs = ["published", "calibrated"] if unc == "both" else [unc]
    tasks = [(k, setup, u, quick, K, 100 + k, dz_prior) for u in uncs for setup in SETUPS for k in range(10)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_shot_task, tasks))
    total = time.time() - t0
    tag = ("_quick" if quick else "") + ("_dzuniform" if dz_prior == "uniform" else "")
    summ = {"total_wall_s": total, "cpu_s": float(sum(r["wall_s"] for r in res)), "workers": workers,
            "los_cpu_hours_per_inference": LOS_CPU_HOURS, "quick": quick, "dz_prior": dz_prior}
    summ["speedup_per_inference"] = LOS_CPU_HOURS * 3600 / (summ["cpu_s"] / len(res))
    (OUT / f"los2024_posteriors{tag}.json").write_text(json.dumps({"summary": summ, "results": res}, indent=2))
    print(f"{'shot':>4s} {'beam unc.':>10s} {'setup':>7s} {'eff. a0 (16-84%)':>18s} {'a0 (16-84%)':>16s} {'r_offset um':>12s} "
          f"{'obs/pred <E>':>16s} {'obs/pred P70':>14s} {'accept':>7s} {'wall':>6s}")
    for r in res:
        ro = f"{r['r_offset_um']['p50']:.1f}" if "r_offset_um" in r else "0 (fixed)"
        print(f"{r['shot']:4d} {r['beam_unc']:>10s} {r['setup']:>7s} {r['a0_eff']['p50']:6.1f} ({r['a0_eff']['p16']:4.1f}-{r['a0_eff']['p84']:4.1f})"
              f" {r['a0']['p50']:6.1f} ({r['a0']['p16']:4.1f}-{r['a0']['p84']:4.1f}) {ro:>12s}"
              f" {r['obs_mean_window_MeV']:6.0f}/{r['pred_mean_window_MeV']['p50']:6.0f}"
              f" {r['obs_P70']:6.2f}/{r['pred_P70']['p50']:5.2f} {r['acceptance']:7.2f} {r['wall_s']:5.0f}s")
    print(f"\nall {len(res)} inferences: {total / 60:.1f} min wall on {workers} workers; "
          f"{summ['cpu_s'] / len(res):.0f} CPU-s per inference vs Los et al. ~{LOS_CPU_HOURS:,.0f} CPU-hours "
          f"(~{summ['speedup_per_inference']:,.0f}x)")
    for u in uncs:
        rr = [r for r in res if r["beam_unc"] == u]
        for key, lab in (("pit_mean_window", "mean energy"), ("pit_P70", "P70")):
            pits = np.array([r[key] for r in rr])
            print(f"posterior predictive ({u:10s}) {lab:11s}: PIT median {np.median(pits):.2f} (0.5 = centred), "
                  f"outside 5-95% in {np.mean((pits < 0.05) | (pits > 0.95)):.0%} of fits (~10% expected)")
    plot(res, tag)


def plot(res, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    uncs = [u for u in ("published", "calibrated") if any(r["beam_unc"] == u for r in res)]
    fig, axs = plt.subplots(1, len(uncs), figsize=(6.0 * len(uncs), 3.4), sharey=True, squeeze=False)
    for ax, u in zip(axs[0], uncs):
        _panel(ax, [r for r in res if r["beam_unc"] == u], f"beam uncertainty: {u}")
    axs[0][0].set_ylabel("effective a0 (median, 68%)")
    axs[0][0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Surrogate-based inference on the 10 public shots", fontsize=10)
    fig.tight_layout()
    check_fits(fig, "los2024_a0_eff.png")
    fig.savefig(OUT / f"los2024_a0_eff{tag}.png", dpi=150)


def _panel(ax, res, title):
    ax.axhspan(*LOS_A0_EFF_RANGE, color="#e1e0d9", alpha=0.6, lw=0, label="Los et al. inferred range (all shots)")
    for setup, col, dx in (("los", "#2a78d6", -0.15), ("offset", "#eb6834", 0.15)):
        rr = sorted([r for r in res if r["setup"] == setup], key=lambda r: r["shot"])
        x = np.array([r["shot"] for r in rr]) + dx
        med = np.array([r["a0_eff"]["p50"] for r in rr])
        lo = med - np.array([r["a0_eff"]["p16"] for r in rr])
        hi = np.array([r["a0_eff"]["p84"] for r in rr]) - med
        lab = "transverse offset fixed at 0" if setup == "los" else "transverse offset free"
        ax.errorbar(x, med, yerr=[lo, hi], fmt="o", color=col, ms=5, elinewidth=1.5, capsize=0, label=lab)
    ax.set(xlabel="shot (Los et al. index)", xticks=range(10))
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["recovery", "shots"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--beams", type=int, default=50, help="predicted incoming beams used per shot (of 100)")
    ap.add_argument("--beam-unc", choices=["published", "calibrated", "both"], default="both")
    ap.add_argument("--dz-prior", choices=["focus", "uniform"], default="focus",
                    help="longitudinal-offset prior: concentrated near focus (as the scan) or uniform")
    a = ap.parse_args()
    if a.what == "recovery":
        recovery(a.n, a.quick, a.workers, a.dz_prior)
    else:
        shots(a.quick, a.workers, a.beams, a.beam_unc, a.dz_prior)


if __name__ == "__main__":
    main()
