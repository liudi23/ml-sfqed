"""Stage 1: Gemini LWFA diagnostics -> electron spectrum, with calibrated uncertainty.

Runs on two splits:
  paper : the authors' fixed 570/75 split (random within runs) - for comparison with Streeter et al.
  time  : hold out the last run (run13, 114 shots), train on runs 05-12 - leakage-free, the honest number

Writes results/stage1/{metrics.json, metrics.md, *.png}.
Usage: python scripts/run_stage1.py [--members 20] [--seed 0]
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

from mlsfqed.beam_model.stage1 import (  # noqa: E402
    BaggedSpectrumModel, Featuriser, beam_energy, crps_ensemble, interval_from_oob,
    lgbm_beam_energy_shap, ridge_predict,
)
from mlsfqed.beam_model.twopart import RawFeaturiser, TwoPartBeamModel  # noqa: E402
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split  # noqa: E402

PAPER = {"ensemble_MSE": 0.057, "single_model_MSE": 0.073, "naive_MSE": 0.106,
         "beam_energy_rel_rms": 0.124, "beam_energy_rel_spread": 0.30, "autoencoder_MSE": 0.011}
OUT = ROOT / "results" / "stage1"


def evaluate(name, d, tr, te, args):
    E = d["E_MeV"]
    Y = physical_spectra(d)
    feat = Featuriser().fit(d, tr)
    Xtr, Xte = feat.transform(d, tr), feat.transform(d, te)
    Ytr, Yte = Y[tr], Y[te]
    Wte = beam_energy(Yte, E)
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
           "test_beam_energy_rel_spread": float(Wte.std() / Wte.mean())}

    def score(pred):
        Wp = beam_energy(pred, E)
        return {"MSE": float(np.mean((pred - Yte) ** 2)),
                "beam_energy_rel_rms": float(np.sqrt(np.mean((Wp - Wte) ** 2 / Wte ** 2)))}

    res["naive_mean"] = score(np.repeat(Ytr.mean(0, keepdims=True), len(Yte), 0))
    res["ridge_pca"] = score(ridge_predict(Xtr, Ytr, Xte))
    preds = {}
    for kind in ("lgbm", "mlp"):
        t0 = time.time()
        m = BaggedSpectrumModel(kind, n_members=args.members, seed=args.seed).fit(Xtr, Ytr)
        P = m.predict_members(Xte)
        mean = P.mean(0)
        s = score(mean)
        s["single_member_MSE_mean"] = float(np.mean([np.mean((p - Yte) ** 2) for p in P]))
        # beam-energy uncertainty: raw ensemble vs OOB-calibrated
        Wm_all = beam_energy(P, E)
        Wm, Ws = Wm_all.mean(0), Wm_all.std(0, ddof=1)
        q, floor = interval_from_oob(m, E, alpha=0.1)
        sd_cal = np.maximum(Ws, floor) * q / 1.645
        s["raw_90pc_coverage"] = float(np.mean(np.abs(Wte - Wm) <= 1.645 * Ws))
        s["calibrated_90pc_coverage"] = float(np.mean(np.abs(Wte - Wm) <= 1.645 * sd_cal))
        s["calibrated_mean_interval_halfwidth_rel"] = float(np.mean(1.645 * sd_cal / Wm))
        s["CRPS_beam_energy_raw_J"] = float(crps_ensemble(Wm_all, Wte).mean())
        z = (Wte - Wm) / sd_cal
        from math import pi, sqrt
        from scipy.stats import norm
        s["CRPS_beam_energy_calibrated_J"] = float(np.mean(sd_cal * (z * (2 * norm.cdf(z) - 1)
                                                                    + 2 * norm.pdf(z) - 1 / sqrt(pi))))
        s["fit_seconds"] = round(time.time() - t0, 1)
        res[f"{kind}_ensemble"] = s
        preds[kind] = (P, Wm, sd_cal)
    # two-part model: W (raw-profile LightGBM on log W) x shape
    t0 = time.time()
    rf = RawFeaturiser().fit(d, tr)
    Rtr, Rte = rf.transform(d, tr), rf.transform(d, te)
    tp = TwoPartBeamModel(E, n_members=args.members, seed=args.seed).fit(Rtr, Ytr)
    Yp, Wp = tp.predict(Rte)
    s = score(Yp)
    _, lo, hi = tp.energy.interval(Rte, 0.9)
    s["calibrated_90pc_coverage"] = float(np.mean((Wte >= lo) & (Wte <= hi)))
    s["calibrated_mean_interval_halfwidth_rel"] = float(np.mean((hi - lo) / 2 / Wp))
    samp = tp.sample(Rte, n_samples=200, seed=args.seed)
    s["CRPS_beam_energy_sampler_J"] = float(crps_ensemble(beam_energy(samp, E).T, Wte).mean())
    s["fit_seconds"] = round(time.time() - t0, 1)
    res["two_part"] = s
    preds["two_part"] = (samp.transpose(1, 0, 2), Wp, (hi - lo) / 2 / 1.645)
    shap = lgbm_beam_energy_shap(np.vstack([Xtr]), beam_energy(Ytr, E), feat.names(), seed=args.seed)
    res["beam_energy_shap_top"] = shap[:8]
    return res, preds, Yte, Wte


def plots(name, d, te, preds, Yte, Wte):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": surf,
                         "axes.facecolor": surf, "font.size": 9, "lines.linewidth": 2})
    E = d["E_MeV"]
    P, Wm, sd = preds["two_part"]
    # 1) predicted vs measured beam energy with calibrated 90% intervals
    fig, ax = plt.subplots(figsize=(4.2, 4))
    lo, hi = min(Wte.min(), Wm.min()) * 0.9, max(Wte.max(), Wm.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=ink2, lw=1, ls="--", label="perfect prediction")
    ax.errorbar(Wte, Wm, yerr=1.645 * sd, fmt="o", ms=4, color=blue, ecolor=blue, elinewidth=0.8,
                alpha=0.8, mec=surf, mew=0.8, label="two-part model, 90% interval")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="measured beam energy [J]", ylabel="predicted beam energy [J]",
           title=f"Beam energy, {name} split")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / f"beam_energy_{name}.png", dpi=150); plt.close(fig)
    # 2) example spectra across the beam-energy range
    order = np.argsort(Wte); pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axs = plt.subplots(2, 3, figsize=(8, 4.2), sharex=True, sharey=True)
    for ax, i in zip(axs.flat, pick):
        lo_b, hi_b = np.percentile(P[:, i], [5, 95], axis=0)
        ax.fill_between(E, lo_b, hi_b, color=blue, alpha=0.2, lw=0, label="sampler 5–95%")
        ax.plot(E, np.median(P[:, i], 0), color=blue, label="sampler median")
        ax.plot(E, Yte[i], color=ink, lw=1.2, label="measured")
        ax.set_xlim(300, 1200)
    for ax in axs[1]: ax.set_xlabel("E [MeV]")
    for ax in axs[:, 0]: ax.set_ylabel("dW/dE [J/GeV]")
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Test spectra, {name} split (sorted by beam energy)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"spectra_{name}.png", dpi=150); plt.close(fig)


def plot_shap(shap):
    import matplotlib.pyplot as plt
    names = [n for n, _ in shap][::-1]; vals = [v for _, v in shap][::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.barh(names, vals, color="#2a78d6", height=0.6)
    ax.set_xlabel("mean |SHAP| on beam energy [J]"); ax.grid(axis="y", visible=False)
    fig.suptitle("What drives beam energy", fontsize=10, x=0.02, ha="left")
    ax.set_title("time split, training shots", fontsize=8, color="#52514e", loc="left")
    fig.tight_layout()
    fig.canvas.draw()  # check every text element fits inside the figure
    fb = fig.bbox
    for t in [fig._suptitle, ax.title, ax.xaxis.label, *ax.get_yticklabels()]:
        bb = t.get_window_extent()
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y1 > fb.y1 + 1:
            raise RuntimeError(f"text {t.get_text()!r} is clipped in shap_beam_energy.png")
    fig.savefig(OUT / "shap_beam_energy.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_all()
    splits = {"paper": (d["split"] == "train", d["split"] == "test"), "time": time_ordered_split(d)}
    results = {"paper_reported": PAPER, "members": args.members, "seed": args.seed}
    for name, (tr, te) in splits.items():
        res, preds, Yte, Wte = evaluate(name, d, tr, te, args)
        results[name] = res
        plots(name, d, te, preds, Yte, Wte)
        if name == "time":
            plot_shap(res["beam_energy_shap_top"])
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    rows = ["| split | model | MSE | beam-energy rel. RMS | 90% coverage (calibrated) |", "|---|---|---|---|---|"]
    for name in splits:
        r = results[name]
        for m in ("naive_mean", "ridge_pca", "lgbm_ensemble", "mlp_ensemble", "two_part"):
            cov = f"{r[m]['calibrated_90pc_coverage']:.2f}" if "calibrated_90pc_coverage" in r[m] else "–"
            rows.append(f"| {name} | {m} | {r[m]['MSE']:.4f} | {r[m]['beam_energy_rel_rms']:.3f} | {cov} |")
    rows.append(f"| paper | Streeter et al. VNN ensemble | {PAPER['ensemble_MSE']} | {PAPER['beam_energy_rel_rms']} | – |")
    (OUT / "metrics.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
