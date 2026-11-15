"""Uncertainty propagation: real shot-to-shot beam uncertainty -> distributions of collision observables.

For every shot of run 13 (never seen by the Stage 1 model), draw 100 calibrated incoming beams from the
Stage 1 peak model given that shot's diagnostics, and push them through the surrogate at a fixed nominal
collision. Then check the surrogate's distributions against Ptarmigan run directly on a subset of the
same beams (--validate shots x beams).
  python scripts/propagate_uncertainty.py --validate-shots 4 --validate-beams 12     (~5 min on 12 cores)
Writes results/propagation/propagation.json and propagation.png.
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

from run_stage1 import check_fits

from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser, to_electron_spectrum
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split
from mlsfqed.inference.posterior import load_model
from mlsfqed.simulation.ptarmigan import run_one
from mlsfqed.surrogate.data import assemble_inputs, describe_beams

OUT = ROOT / "results" / "propagation"
NOMINAL = {"a0": 12.0, "waist_um": 2.5, "fwhm_fs": 45.0, "r_offset_um": 0.0, "phi_offset": 0.0, "dz_um": 0.0,
           "radius_um": 0.7, "length_um": 5.5, "div_mrad": 2.0}
SHOW = {"loss_window_MeV": "energy loss >=305 MeV [MeV]", "log_g_number_per_e": "photons per electron (log10)",
        "g_crit_MeV": "photon critical energy [MeV]", "e_P70": "P70"}


def _ptarmigan(args):
    spec, E, q, seed, n_e = args
    p = {k: NOMINAL[k] for k in ("a0", "waist_um", "fwhm_fs", "dz_um", "radius_um", "length_um", "div_mrad")}
    p.update(dx_um=0.0, dy_um=0.0, charge_pC=float(q))
    o, _, _ = run_one(p, spec, E, n_e, seed)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--validate-shots", type=int, default=4)
    ap.add_argument("--validate-beams", type=int, default=12)
    ap.add_argument("--validate-electrons", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    m = load_model()
    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    tr, te = time_ordered_split(d)
    rf = RawFeaturiser().fit(d, tr)
    beam_model = PeakBeamModel(E, seed=0).fit(rf.transform(d, tr), Y[tr]).calibrate()
    S = beam_model.sample_calibrated(rf.transform(d, te), a.samples, 0)          # (shots, samples, bins)
    n_shots = S.shape[0]
    flat = S.reshape(-1, S.shape[-1])
    dNdE, Q = to_electron_spectrum(flat, E)
    bd = describe_beams(flat, m["pool_E_MeV"], m["pool_pca"], Q)
    t0 = time.time()
    X = assemble_inputs({k: np.full(len(bd), v) for k, v in NOMINAL.items()}, {c: bd[c].values for c in bd.columns})
    P, _ = m["mlp"].predict(X)
    t_sur = time.time() - t0
    ti = {t: m["targets"].index(t) for t in SHOW}
    pred = {t: P[:, ti[t]].reshape(n_shots, a.samples) for t in SHOW}
    # same quantities with the measured beam of each shot (what a deterministic analysis would use)
    _, Q_meas = to_electron_spectrum(Y[te], E)
    bdm = describe_beams(Y[te], m["pool_E_MeV"], m["pool_pca"], Q_meas)
    Pm, _ = m["mlp"].predict(assemble_inputs({k: np.full(len(bdm), v) for k, v in NOMINAL.items()},
                                             {c: bdm[c].values for c in bdm.columns}))
    # Ptarmigan on a subset of the same beams
    rng = np.random.default_rng(0)
    vs = rng.choice(n_shots, a.validate_shots, replace=False)
    jobs, idx = [], []
    for s in vs:
        for b in range(a.validate_beams):
            k = s * a.samples + b
            jobs.append((dNdE[k], E, Q[k], 1000 + k, a.validate_electrons))
            idx.append((s, b))
    with ProcessPoolExecutor(a.workers) as ex:
        sims = list(ex.map(_ptarmigan, jobs))
    cpu_pt = float(np.sum([o["wall_s"] for o in sims]))
    sim_vals = {"loss_window_MeV": [], "log_g_number_per_e": [], "g_crit_MeV": [], "e_P70": []}
    for (s, b), o in zip(idx, sims):
        k = s * a.samples + b
        bm = bd.beam_mean_window_MeV.values[k]
        sim_vals["loss_window_MeV"].append(bm - o["e_mean_window_MeV"])
        sim_vals["log_g_number_per_e"].append(np.log10(o["g_number_per_e"]))
        sim_vals["g_crit_MeV"].append(o["g_crit_MeV"])
        sim_vals["e_P70"].append(o["e_P70"])
    res = {"nominal_collision": NOMINAL, "n_shots": n_shots, "samples_per_shot": a.samples,
           "validation_electrons": a.validate_electrons,
           "surrogate_seconds_total": t_sur, "n_surrogate_evaluations": len(bd),
           "ptarmigan_cpu_s_per_run_5000e": cpu_pt / len(sims) * 5000 / a.validate_electrons,
           "ptarmigan_cpu_hours_equivalent": cpu_pt / len(sims) * 5000 / a.validate_electrons * len(bd) / 3600,
           "validation": {}}
    for t in SHOW:
        sur = np.array([pred[t][s, b] for s, b in idx])
        sim = np.array(sim_vals[t])
        # per validated shot: spread across the beam samples, surrogate vs Ptarmigan
        per = [{"shot": int(s), "sur_mean": float(pred[t][s, :a.validate_beams].mean()), "sim_mean": float(np.mean(sim[[i for i, (ss, _) in enumerate(idx) if ss == s]])),
                "sur_sd": float(pred[t][s, :a.validate_beams].std()), "sim_sd": float(np.std(sim[[i for i, (ss, _) in enumerate(idx) if ss == s]]))}
               for s in vs]
        res["validation"][t] = {"rmse_paired": float(np.sqrt(np.mean((sur - sim) ** 2))), "corr_paired": float(np.corrcoef(sur, sim)[0, 1]),
                                "per_shot": per}
        res[f"{t}_population"] = {"p5": float(np.percentile(pred[t], 5)), "p50": float(np.median(pred[t])),
                                  "p95": float(np.percentile(pred[t], 95)),
                                  "median_per_shot_sd": float(np.median(pred[t].std(1))),
                                  "spread_using_measured_beams_sd": float(Pm[:, ti[t]].std())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "propagation.json").write_text(json.dumps(res, indent=2))
    print(f"{n_shots} shots x {a.samples} beams = {len(bd)} surrogate evaluations in {t_sur:.2f} s; "
          f"Ptarmigan would need ~{res['ptarmigan_cpu_hours_equivalent']:.0f} CPU-hours "
          f"({res['ptarmigan_cpu_s_per_run_5000e']:.1f} s per 5,000-electron run)")
    print(f"{'observable':22s}{'per-shot sd':>12s}{'paired RMSE vs Ptarmigan':>26s}{'corr':>7s}   per validated shot: mean sur/sim, sd sur/sim")
    for t in SHOW:
        v = res["validation"][t]
        ps = "  ".join(f"{p['sur_mean']:.3g}/{p['sim_mean']:.3g}, {p['sur_sd']:.2g}/{p['sim_sd']:.2g}" for p in v["per_shot"])
        print(f"{t:22s}{res[t + '_population']['median_per_shot_sd']:12.3g}{v['rmse_paired']:26.3g}{v['corr_paired']:7.2f}   {ps}")
    plot(pred, Pm, ti, idx, sim_vals, vs, a)


def plot(pred, Pm, ti, idx, sim_vals, vs, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, blue, orange = "#52514e", "#2a78d6", "#eb6834"
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    t = "loss_window_MeV"
    order = np.argsort(np.median(pred[t], 1))
    pick = order[np.linspace(0, len(order) - 1, 12).astype(int)]
    pick = np.unique(np.concatenate([pick, vs]))
    pick = pick[np.argsort(np.median(pred[t][pick], 1))]
    ax = axs[0]
    for j, s in enumerate(pick):
        lo, med, hi = np.percentile(pred[t][s], [5, 50, 95])
        ax.plot([j, j], [lo, hi], color=blue, lw=2, solid_capstyle="round", label="surrogate, 5-95%" if j == 0 else None)
        ax.plot(j, med, "o", color=blue, ms=4)
        if s in vs:
            v = [sim_vals[t][i] for i, (ss, _) in enumerate(idx) if ss == s]
            ax.plot(np.full(len(v), j + 0.18), v, "o", color=orange, ms=3, alpha=0.7,
                    label=f"Ptarmigan, same beams ({a.validate_electrons} e-)" if s == vs[0] else None)
    ax.set(xticks=[], xlabel="run-13 shots (sorted)", ylabel="energy loss >=305 MeV [MeV]")
    ax.set_title("Per-shot predictive distributions", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axs[1]
    t = "loss_window_MeV"
    bins = np.linspace(min(pred[t].min(), Pm[:, ti[t]].min()), max(pred[t].max(), Pm[:, ti[t]].max()), 40)
    ax.hist(pred[t].ravel(), bins=bins, density=True, color=blue, alpha=0.35, label="sampled beams (calibrated)")
    ax.hist(Pm[:, ti[t]], bins=bins, density=True, histtype="step", color=ink2, lw=1.5, label="measured beams")
    ax.set(xlabel="energy loss >=305 MeV [MeV]", ylabel="density")
    ax.set_title("Run 13 population at the nominal collision", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Propagating beam uncertainty through the surrogate (a0 = 12, aligned)", fontsize=10)
    fig.tight_layout()
    check_fits(fig, "propagation.png")
    fig.savefig(OUT / "propagation.png", dpi=150)


if __name__ == "__main__":
    main()
