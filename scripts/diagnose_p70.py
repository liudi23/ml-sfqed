"""Why do the real shots' P70 values sit where no radiation-reaction model predicts them?

  python scripts/diagnose_p70.py                # all checks, ~3 min on 12 cores
  python scripts/diagnose_p70.py --no-sim       # skip the Ptarmigan runs (part C)

A  Definitions (data only)
   A1 observed P70 vs extra smoothing; the spectrometer's quoted energy uncertainty
   A2 does the simulation pipeline (10 MeV histogram + smoothing) give the same P70 as the data pipeline
      (fine grid + smoothing) for the same spectrum? (a) pool beams rebinned; (b) near-zero-loss scan runs
   A3 are Los et al.'s predicted incoming beams like the beams the surrogates were trained on?
      (incoming P70 and shape-PCA distance vs the 1,935-beam pool)
B  Posterior predictive (surrogates): fit mean energy ONLY, per shot and model, then predict P70.
   If the observed P70 is outside what a model predicts after matching the energy loss, P70 is what
   separates the models - for the right or the wrong reason.
C  Ptarmigan check (direct simulation): each shot x model at the part-B posterior median with the shot's
   median predicted beam. Full outgoing spectra for comparison with the measured one, P70 from Ptarmigan
   (is the surrogate right on these out-of-pool beams?), and P70 after smearing with the quoted
   spectrometer energy uncertainty.
Writes results/diagnostics/p70/ (summary.json, spectra.png).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from compare_models import RR, surrogate_path
from infer_los2024 import SETUPS

from mlsfqed.inference.los2024 import LOS, beams_on_pool_grid, load, spectrum_observables
from mlsfqed.inference.posterior import ShotPosterior, load_model, run_mcmc
from mlsfqed.simulation.ptarmigan import E_BINS_E, p70, run_one
from mlsfqed.surrogate.data import build

OUT = ROOT / "results" / "diagnostics" / "p70"
STAGE2 = ROOT / "data" / "processed" / "stage2"
EC = 0.5 * (E_BINS_E[1:] + E_BINS_E[:-1])
WIN = EC >= 305.0
K_BEAMS = 50


def resolution():
    """Quoted energy uncertainty of the electron spectrometer (MeV) vs energy (MeV)."""
    with h5py.File(LOS / "Bayesian_inference_data.h5", "r") as f:
        return f["shot 1/Electrons/Energy[MeV]"][()], f["shot 1/Electrons/Energy_uncertainty[MeV]"][()]


def rebin(E, dNdE, edges=E_BINS_E):
    """Number per histogram bin from a density on a fine grid (exact for piecewise-linear dN/dE)."""
    cum = np.concatenate([[0], np.cumsum(0.5 * (dNdE[1:] + dNdE[:-1]) * np.diff(E))])
    return np.diff(np.interp(edges, E, cum, left=0.0, right=cum[-1]))


def smear(Ne, sigma_at):
    """Convolve a 10 MeV histogram with an energy-dependent Gaussian (sigma from the spectrometer)."""
    s = np.maximum(sigma_at(EC), 1e-6)
    G = np.exp(-0.5 * ((EC[:, None] - EC[None, :]) / s[None, :]) ** 2)
    return (G / G.sum(0, keepdims=True)) @ Ne


def sim_obs(Ne):
    """Observables exactly as the scan computes them from a 10 MeV histogram."""
    dens = Ne / np.diff(E_BINS_E)
    return {"mean_window_MeV": float(np.sum(Ne[WIN] * EC[WIN]) / Ne[WIN].sum()),
            "P70": p70(EC[WIN], dens[WIN])}


# ---------------------------------------------------------------- A: definitions
def part_a():
    data = load()
    E = data["E_MeV"]
    Er, dEr = resolution()
    out = {"A1": {"observed_P70_vs_smoothing_MeV": {}, "quoted_energy_uncertainty_MeV":
                  {str(e): float(np.interp(e, Er, dEr)) for e in (400, 500, 600, 800)}}}
    for k, s in enumerate(data["shots"]):
        out["A1"]["observed_P70_vs_smoothing_MeV"][k] = {sm: spectrum_observables(E, s["post_dNdE"], sm)["P70"]
                                                       for sm in (0, 5, 10, 15, 20)}
    # A2a: same spectrum through both pipelines
    pool = dict(np.load(STAGE2 / "beam_pool.npz"))
    Ep, N = pool["E_MeV"], pool["dNdE_per_MeV"]
    meas = np.flatnonzero(pool["source"] == "measured")
    r = []
    for i in meas:
        fine = spectrum_observables(Ep, N[i], smooth_mev=10.0)["P70"]
        hist = sim_obs(rebin(Ep, N[i]))["P70"]
        if np.isfinite(fine) and np.isfinite(hist) and fine > 0:
            r.append(hist / fine)
    r = np.array(r)
    out["A2_same_spectrum_hist_over_fine"] = {"n": len(r), "median": float(np.median(r)),
                                              "p5": float(np.percentile(r, 5)), "p95": float(np.percentile(r, 95))}
    # A2b: scan runs that barely interacted: simulated P70 vs P70 of the incoming beam (data pipeline)
    _, Y, meta, _, _ = build(STAGE2)
    low = np.flatnonzero(Y.loss_window_MeV.abs().values < 1.0)
    rr = []
    for i in low:
        b = int(meta.beam_index.iloc[i])
        rr.append(Y.e_P70.iloc[i] / spectrum_observables(Ep, N[b], smooth_mev=10.0)["P70"])
    rr = np.array(rr)
    out["A2_low_loss_runs_sim_over_beam"] = {"n": len(rr), "median": float(np.median(rr)) if len(rr) else None,
                                             "p5": float(np.percentile(rr, 5)) if len(rr) else None,
                                             "p95": float(np.percentile(rr, 95)) if len(rr) else None}
    # A3: incoming beams vs the training pool
    pool_p70 = np.array([spectrum_observables(Ep, n, smooth_mev=10.0)["P70"] for n in N])
    m = load_model(surrogate_path("stochastic", 4096, False))
    pca = m["pool_pca"]
    W = pool["dWdE_J_per_GeV"]
    Zp = pca.transform(W / np.trapezoid(W, Ep / 1e3, axis=1)[:, None])
    mu, Ci = Zp.mean(0), np.linalg.inv(np.cov(Zp.T))
    d2p = np.einsum("ij,jk,ik->i", Zp - mu, Ci, Zp - mu)
    beams = beams_on_pool_grid(data, m["pool_E_MeV"])
    a3 = []
    for k, (s, b) in enumerate(zip(data["shots"], beams)):
        inc_p70 = np.array([spectrum_observables(E, p, smooth_mev=10.0)["P70"] for p in s["pre_dNdE"]])
        Z = pca.transform(b / np.trapezoid(b, Ep / 1e3, axis=1)[:, None])
        d2 = np.einsum("ij,jk,ik->i", Z - mu, Ci, Z - mu)
        rec = pca.inverse_transform(Z)
        shp = b / np.trapezoid(b, Ep / 1e3, axis=1)[:, None]
        a3.append({"shot": k, "incoming_P70_median": float(np.median(inc_p70)),
                   "incoming_P70_pool_percentile": float((pool_p70 < np.median(inc_p70)).mean() * 100),
                   "observed_P70": spectrum_observables(E, s["post_dNdE"], 10.0)["P70"],
                   "shape_mahalanobis_pool_percentile": float((d2p < np.median(d2)).mean() * 100),
                   "pca_residual_rel": float(np.median(np.linalg.norm(shp - rec, axis=1) / np.linalg.norm(shp, axis=1)))})
    out["A3"] = a3
    out["A3_pool_P70"] = {"p5": float(np.nanpercentile(pool_p70, 5)), "p50": float(np.nanpercentile(pool_p70, 50)),
                          "p95": float(np.nanpercentile(pool_p70, 95))}
    rec_pool = pca.inverse_transform(Zp)
    shp_pool = W / np.trapezoid(W, Ep / 1e3, axis=1)[:, None]
    out["A3_pool_pca_residual_rel_median"] = float(np.median(np.linalg.norm(shp_pool - rec_pool, axis=1)
                                                             / np.linalg.norm(shp_pool, axis=1)))
    return out


# ---------------------------------------------------------------- B + C: per shot and model
def _task(args):
    k, mdl, do_sim, n_el = args
    m = load_model(surrogate_path(mdl, 4096, False))
    data = load()
    E = data["E_MeV"]
    s = data["shots"][k]
    allb = beams_on_pool_grid(data, m["pool_E_MeV"])[k]
    beams = allb[np.random.default_rng(100 + k).choice(len(allb), K_BEAMS, replace=False)]
    obs = spectrum_observables(E, s["post_dNdE"], smooth_mev=10.0)
    Er, dEr = resolution()
    meas_sd = float(np.trapezoid(s["post_dNdE"] * dEr, E) / np.trapezoid(s["post_dNdE"], E))
    out = {"shot": k, "model": mdl, "obs_mean_window_MeV": obs["mean_window_MeV"], "obs_P70": obs["P70"]}
    rng = np.random.default_rng(1)
    for name, use in (("mean_only", (0,)), ("mean_P70", (0, 1))):
        post = ShotPosterior(m, beams, obs["mean_window_MeV"], obs["P70"], meas_sd, free=SETUPS["los"],
                             dz_prior="focus", use=use)
        chain, _ = run_mcmc(post, nsteps=3000, burn=1000, seed=k)
        sub = chain[rng.choice(len(chain), 200)]
        pred, var = post.predict(sub)
        draws = pred + rng.standard_normal(pred.shape) * np.sqrt(var + post.meas_var)
        out[name] = {"theta_median": dict(zip(post.free, np.median(chain, 0).round(3).tolist())),
                     "pred_P70_p5_p50_p95": np.percentile(draws[..., 1], [5, 50, 95]).round(4).tolist(),
                     "pred_mean_p50": float(np.median(draws[..., 0])),
                     "pit_P70": float((draws[..., 1] < obs["P70"]).mean()),
                     "pit_mean": float((draws[..., 0] < obs["mean_window_MeV"]).mean())}
    if do_sim:
        Ep = m["pool_E_MeV"]
        means = np.array([spectrum_observables(Ep, b / Ep)["mean_window_MeV"] for b in allb])
        b = allb[np.argsort(means)[len(means) // 2]]                 # median predicted beam
        th = out["mean_only"]["theta_median"]
        p = {"a0": th["a0"], "dz_um": th["dz_um"], "length_um": th["length_um"], "waist_um": 2.5, "fwhm_fs": 45.0,
             "radius_um": 0.7, "div_mrad": 2.0, "dx_um": 0.0, "dy_um": 0.0, "charge_pC": float(m["pool_Q_median_pC"])}
        o, Ne, _ = run_one(p, b / Ep, Ep, n_el, seed=1000 + k, model=mdl)
        if o["ok"]:
            def sig(e):
                return np.interp(e, Er, dEr)

            post1 = ShotPosterior(m, b[None, :], obs["mean_window_MeV"], obs["P70"], meas_sd, free=SETUPS["los"])
            sp, _ = post1.predict(np.array([[th["a0"], th["dz_um"], th["length_um"]]]))
            out["ptarmigan"] = {"P70": sim_obs(Ne)["P70"], "P70_smeared": sim_obs(smear(Ne, sig))["P70"],
                                "mean_window_MeV": sim_obs(Ne)["mean_window_MeV"],
                                "surrogate_P70_same_point": float(sp[0, 0, 1]),
                                "surrogate_mean_same_point": float(sp[0, 0, 0]),
                                "incoming_P70": spectrum_observables(Ep, b / Ep, 10.0)["P70"],
                                "Ne": Ne.tolist(), "beam_dWdE": b.tolist(), "wall_s": o["wall_s"]}
        else:
            out["ptarmigan"] = {"error": o.get("error")}
    return out


def figure(res, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_stage1 import check_fits
    E = data["E_MeV"]
    cols = {"classical": "#d95f02", "continuous": "#7570b3", "stochastic": "#1b9e77"}
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
    m = load_model(surrogate_path("stochastic", 4096, False))
    Ep = m["pool_E_MeV"]
    for k, ax in enumerate(axs.ravel()):
        s = data["shots"][k]
        win = E >= 305
        w = s["post_dNdE"] * E
        ax.plot(E, w / np.trapezoid(w[win], E[win] / 1e3), color="k", lw=1.6, label="measured")
        drawn_beam = False
        for r in res:
            if r["shot"] != k or "Ne" not in r.get("ptarmigan", {}):
                continue
            Ne = np.array(r["ptarmigan"]["Ne"])
            wv = Ne / 10.0 * EC
            ax.plot(EC[WIN], wv[WIN] / np.trapezoid(wv[WIN], EC[WIN] / 1e3), color=cols[r["model"]], lw=1.2,
                    label=r["model"])
            if not drawn_beam:
                b = np.array(r["ptarmigan"]["beam_dWdE"])
                ax.plot(Ep, b / np.trapezoid(b[Ep >= 305], Ep[Ep >= 305] / 1e3), color="0.6", ls="--", lw=1,
                        label="incoming (median prediction)")
                drawn_beam = True
        ax.set_xlim(300, 1000)
        ax.set_title(f"shot {k}", fontsize=10)
        if k % 5 == 0:
            ax.set_ylabel("E dN/dE, normalised [1/GeV]")
        if k >= 5:
            ax.set_xlabel("electron energy [MeV]")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Measured outgoing spectra vs Ptarmigan at each model's mean-energy-only posterior median", fontsize=11)
    fig.tight_layout()
    check_fits(fig, "spectra.png")
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--electrons", type=int, default=5000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A = part_a()
    tasks = [(k, mdl, not a.no_sim, a.electrons) for k in range(10) for mdl in RR]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_task, tasks))
    data = load()
    slim = [{k: ({kk: vv for kk, vv in v.items() if kk not in ("Ne", "beam_dWdE")} if isinstance(v, dict) else v)
             for k, v in r.items()} for r in res]
    (OUT / "summary.json").write_text(json.dumps({"A": A, "per_shot_model": slim}, indent=1, default=float))
    if not a.no_sim:
        np.savez(OUT / "ptarmigan_spectra.npz", **{f"s{r['shot']}_{r['model']}": np.array(r["ptarmigan"]["Ne"])
                                                  for r in res if "Ne" in r.get("ptarmigan", {})})
        try:
            figure(res, data, OUT / "spectra.png")
        except (RuntimeError, ValueError, OSError) as e:             # a plotting error must not lose the run
            print(f"figure failed ({e}); results are saved")

    # ---------------- report
    print(f"P70 diagnostics ({time.time() - t0:.0f} s)\n")
    a1 = A["A1"]
    print("A1 observed P70 with 0/5/10/15/20 MeV extra smoothing (shot: values):")
    for k, v in a1["observed_P70_vs_smoothing_MeV"].items():
        print(f"   {k}: " + " ".join(f"{x:.3f}" for x in v.values()))
    print("   quoted spectrometer energy uncertainty [MeV] at 400/500/600/800 MeV: "
          + " ".join(f"{v:.1f}" for v in a1["quoted_energy_uncertainty_MeV"].values()))
    x = A["A2_same_spectrum_hist_over_fine"]
    print(f"\nA2 same spectrum, simulation pipeline / data pipeline P70: median {x['median']:.3f} "
          f"(5-95%: {x['p5']:.3f}-{x['p95']:.3f}, n={x['n']})")
    x = A["A2_low_loss_runs_sim_over_beam"]
    if x["n"]:
        print(f"   scan runs with |loss| < 1 MeV, simulated P70 / incoming-beam P70: median {x['median']:.3f} "
              f"(5-95%: {x['p5']:.3f}-{x['p95']:.3f}, n={x['n']})")
    print(f"\nA3 incoming beams vs training pool (pool P70 5/50/95%: {A['A3_pool_P70']['p5']:.2f} / "
          f"{A['A3_pool_P70']['p50']:.2f} / {A['A3_pool_P70']['p95']:.2f}; pool PCA residual "
          f"{A['A3_pool_pca_residual_rel_median']:.3f})")
    print("   shot  incoming P70 (pool pct)  observed P70   shape distance pool pct   PCA residual")
    for r in A["A3"]:
        print(f"   {r['shot']:4d}  {r['incoming_P70_median']:6.2f} ({r['incoming_P70_pool_percentile']:5.1f})"
              f"       {r['observed_P70']:6.2f}        {r['shape_mahalanobis_pool_percentile']:6.1f}"
              f"            {r['pca_residual_rel']:.3f}")
    print("\nB fit mean energy only, then predict P70 (posterior predictive 5/50/95%, PIT of the observation):")
    print("   shot  obs P70 | " + " | ".join(f"{m:^26s}" for m in RR))
    for k in range(10):
        row = [r for r in res if r["shot"] == k]
        cells = []
        for mdl in RR:
            r = next(x for x in row if x["model"] == mdl)["mean_only"]
            p5, p50, p95 = r["pred_P70_p5_p50_p95"]
            cells.append(f"{p50:.2f} [{p5:.2f},{p95:.2f}] PIT {r['pit_P70']:.2f}")
        print(f"   {k:4d}  {row[0]['obs_P70']:6.2f}  | " + " | ".join(f"{c:26s}" for c in cells))
    print("   (same with mean+P70 fitted - PIT of P70):")
    for mdl in RR:
        print(f"   {mdl:10s} " + " ".join(f"{next(x for x in res if x['shot'] == k and x['model'] == mdl)['mean_P70']['pit_P70']:.2f}"
                                       for k in range(10)))
    if not a.no_sim:
        print("\nC Ptarmigan at the mean-only posterior median, median incoming beam:")
        print("   shot  model        obs P70  sim P70  smeared  surrogate | obs mean  sim mean  surrogate | incoming P70")
        for r in res:
            p = r.get("ptarmigan", {})
            if "P70" not in p:
                print(f"   {r['shot']:4d}  {r['model']:10s}  failed: {p.get('error')}")
                continue
            print(f"   {r['shot']:4d}  {r['model']:10s}  {r['obs_P70']:6.2f}   {p['P70']:6.2f}   {p['P70_smeared']:6.2f}"
                  f"   {p['surrogate_P70_same_point']:6.2f}    | {r['obs_mean_window_MeV']:7.1f}  {p['mean_window_MeV']:7.1f}"
                  f"   {p['surrogate_mean_same_point']:7.1f}   | {p['incoming_P70']:.2f}")
        print(f"\nfigure: {OUT / 'spectra.png'}")
    print(f"summary: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
