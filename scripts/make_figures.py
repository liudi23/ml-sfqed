"""Figures for the README and report, built from the result files.

  python scripts/make_figures.py

Writes docs/figures/:
  rr_sweep.gif         animated outgoing spectra vs laser strength for three radiation-reaction models
                       (needs results/diagnostics/a0_sweep_spectra.json from scripts/a0_sweep.py)
  p70_shots.png        observed P70 on the 10 collision shots vs each model's prediction after fitting energy loss
  bayes_factors.png    summed log Bayes factors vs classical under four uncertainty set-ups, and Los et al.
  a0_eff.png           effective a0 per shot with published vs measured incoming-beam uncertainty
  recovery_a0.png      parameter recovery on 48 simulated shots: true vs inferred a0 and effective a0
  model_recovery.png   model comparison on 90 simulated shots: which model wins, per true model
  surrogate_accuracy.png  surrogate vs Ptarmigan for energy loss and P70, unseen beams and a0 > 16.5, with LightGBM
  pipeline_checks.png  one predicted-vs-actual panel per stage (needs results/stage3/predictions_beam.json from
                       `stage3_surrogate.py --predictions` and per-shot arrays from beam_prediction_error.py)

  python scripts/make_figures.py recovery_a0 model_recovery   # only the named figures
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_stage1 import check_fits

R = ROOT / "results"
OUT = ROOT / "docs" / "figures"
COL = {"stochastic": "#2b6cb0", "continuous": "#8a5cc2", "classical": "#d9822b", "incoming": "#9aa0a6"}
NAME = {"stochastic": "Stochastic (quantum)", "continuous": "Continuous (quantum)", "classical": "Classical (LL)"}
MODELS = ("stochastic", "continuous", "classical")
LOS_SUM = {"continuous": 2.84, "stochastic": 2.87}   # Los et al. Model_evidences.csv, summed differences vs classical

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False})


def title(fig, text):
    fig.suptitle(text, x=0.015, ha="left", fontsize=11, fontweight="bold")


def save(fig, name):
    fig.tight_layout()
    check_fits(fig, name)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ animation
def sweep_gif(hold_ms=2200, step_ms=110, steps=8, end_ms=4000):
    """Each a0 is held for hold_ms, then morphs to the next in `steps` frames of step_ms each."""
    import io

    from PIL import Image
    d = json.loads((R / "diagnostics" / "a0_sweep_spectra.json").read_text())
    E = np.array(d["E"])
    a0s = [f["a0"] for f in d["frames"]]
    Y = {m: np.array([f[m]["y"] for f in d["frames"]]) for m in MODELS}
    P = {m: np.array([f[m]["P70"] for f in d["frames"]]) for m in MODELS}
    # timeline: hold on each a0, then interpolate to the next
    t, ms = [], []
    for i in range(len(a0s)):
        t.append(float(i)); ms.append(end_ms if i == len(a0s) - 1 else hold_ms)
        if i < len(a0s) - 1:
            t += list(i + np.arange(1, steps) / steps); ms += [step_ms] * (steps - 1)

    def at(arr, s):
        i = int(np.floor(s)); f = s - i
        return arr[i] if f == 0 or i + 1 >= len(arr) else (1 - f) * arr[i] + f * arr[i + 1]

    fig = plt.figure(figsize=(9.6, 6.2))
    # legend, a0 label and P70 readout sit in a right-hand margin, so they never cover the curves
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.38, left=0.085, right=0.75, top=0.88, bottom=0.09)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle("Classical radiation reaction sharpens the peak; stochastic emission spreads it",
                 x=0.085, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.085, 0.905, "Ptarmigan runs: the same measured Gemini beam (dashed) collides head-on at focus; "
             "coloured lines are the outgoing spectra as a0 rises", fontsize=9, color="#555")
    ax.plot(E, d["incoming"], ls="--", color=COL["incoming"], lw=1.5, label="Incoming beam")
    lines = {m: ax.plot(E, Y[m][0], color=COL[m], lw=2.4 if m == "stochastic" else 1.8, label=NAME[m])[0] for m in MODELS}
    ymax = max(Y[m].max() for m in MODELS) * 1.05
    ax.set(xlim=(E[0], E[-1]), ylim=(0, ymax), xlabel="Electron energy (MeV)", ylabel="E dN/dE, normalised (1/GeV)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5, borderaxespad=0)
    label = ax.text(1.03, 0.52, "", transform=ax.transAxes, ha="left", fontsize=17, fontweight="bold")
    ax_txt = ax.text(1.03, 0.40, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, family="monospace")
    bx.axhline(d["incoming_P70"], ls="--", color=COL["incoming"], lw=1)
    bx.text(a0s[-1], d["incoming_P70"] - 0.06, "incoming beam", color="#777", fontsize=8, ha="right", va="top")
    for m in MODELS:
        bx.plot(a0s, P[m], "-o", color=COL[m], ms=3, lw=1.5)
    dots = {m: bx.plot([], [], "o", color=COL[m], ms=8, mec="white", mew=1.2)[0] for m in MODELS}
    cur = bx.axvline(a0s[0], color="#333", lw=1, alpha=0.5)
    bx.set(xlabel="Laser strength a0", ylabel="P70 (1/GeV)", xticks=a0s[::1] if len(a0s) < 14 else a0s[::2])

    def draw(s):
        a0 = at(np.array(a0s, float), s)
        for m in MODELS:
            lines[m].set_ydata(at(Y[m], s))
            dots[m].set_data([a0], [at(P[m], s)])
        cur.set_xdata([a0, a0])
        label.set_text(f"a0 = {a0:.1f}" if s % 1 else f"a0 = {a0:.0f}")
        ax_txt.set_text("P70\n" + "\n".join(f"{m[:10]:<10s} {at(P[m], s):.2f}" for m in ("classical", "continuous", "stochastic")))
        return [*lines.values(), *dots.values(), cur, label, ax_txt]

    draw(0.0)
    check_fits(fig, "rr_sweep.gif")
    frames = []
    for s_ in t:
        draw(s_)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        rgb = Image.open(buf).convert("RGB")
        pal = frames[0] if frames else None                    # one shared palette: no colour flicker
        frames.append(rgb.quantize(palette=pal, dither=Image.Dither.NONE) if pal else
                      rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(OUT / "rr_sweep.gif", save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    draw(float(a0s.index(8)) if 8 in a0s else 0.0)
    fig.savefig(OUT / "rr_sweep_still.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "rr_sweep.gif")


# ------------------------------------------------------------------ P70 on the real shots
def p70_shots():
    d = json.loads((R / "diagnostics" / "p70" / "summary.json").read_text())
    inc = {a["shot"]: a["incoming_P70_median"] for a in d["A"]["A3"]}
    rows = d["per_shot_model"]
    shots = sorted({r["shot"] for r in rows})
    lo = min(min(r["mean_only"]["pred_P70_p5_p50_p95"][0], r["obs_P70"], inc[r["shot"]]) for r in rows)
    hi = max(max(r["mean_only"]["pred_P70_p5_p50_p95"][2], r["obs_P70"], inc[r["shot"]]) for r in rows)
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 4.8), sharey=True)
    for ax, mdl in zip(axs, MODELS):
        n_in = 0
        for r in (r for r in rows if r["model"] == mdl):
            s = r["shot"]
            p5, p50, p95 = r["mean_only"]["pred_P70_p5_p50_p95"]
            inside = p5 <= r["obs_P70"] <= p95          # the drawn 90% range
            n_in += inside
            ax.plot([p5, p95], [s, s], color=COL[mdl], lw=7, alpha=0.28, solid_capstyle="round")
            ax.plot([p50, p50], [s - 0.22, s + 0.22], color=COL[mdl], lw=2)
            ax.plot(inc[s], s, "o", mfc="none", mec="#999", ms=7, mew=1.3)
            ax.plot(r["obs_P70"], s, "D", color="black" if inside else "#c62828", ms=6.5)
        ax.set_title(f"{NAME[mdl]}\nmatches {n_in} of {len(shots)} shots", loc="left", fontsize=10, color=COL[mdl], fontweight="bold")
        ax.set_xlim(lo - 0.05, hi + 0.05)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("P70 (1/GeV)")
    axs[0].set(yticks=shots, yticklabels=[f"shot {s}" for s in shots], ylim=(max(shots) + 0.6, -0.6))
    h = [plt.Line2D([], [], marker="D", color="black", ls="", label="Observed, inside predicted range"),
         plt.Line2D([], [], marker="D", color="#c62828", ls="", label="Observed, outside"),
         plt.Line2D([], [], marker="o", mfc="none", mec="#999", ls="", ms=7, label="Predicted incoming beam"),
         plt.Line2D([], [], color="#888", lw=7, alpha=0.35, label="Model's 90% predicted range")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    title(fig, "Every shot leaves the collision broader than it arrived; only stochastic emission predicts that")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    check_fits(fig, "p70_shots.png")
    fig.savefig(OUT / "p70_shots.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "p70_shots.png")


# ------------------------------------------------------------------ Bayes factors
def bayes_factors():
    with open(R / "model_comparison" / "shots.csv") as f:
        rows = list(csv.DictReader(f))
    def total(setup, unc, mdl, obs="mean+P70"):
        z = {(int(r["shot"]), r["model"]): float(r[f"logZ_{obs}"]) for r in rows if r["setup"] == setup and r["beam_unc"] == unc}
        return sum(z[(s, mdl)] - z[(s, "classical")] for s in range(10))
    cfg = [("los", "published", "Published beam spread\noffset fixed"), ("los", "calibrated", "Measured beam error\noffset fixed"),
           ("offset", "published", "Published beam spread\noffset free"), ("offset", "calibrated", "Measured beam error\noffset free")]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels, y = [], 0
    for setup, unc, lab in cfg:
        for j, mdl in enumerate(("stochastic", "continuous")):
            v = total(setup, unc, mdl)
            ax.barh(y + j * 0.36, v, height=0.32, color=COL[mdl])
            ax.text(v + 0.4, y + j * 0.36, f"{v:+.1f}", va="center", fontsize=8.5)
        labels.append((y + 0.18, lab)); y += 1
    for j, mdl in enumerate(("stochastic", "continuous")):
        ax.barh(y + j * 0.36, LOS_SUM[mdl], height=0.32, color=COL[mdl], alpha=0.45)
        ax.text(LOS_SUM[mdl] + 0.4, y + j * 0.36, f"{LOS_SUM[mdl]:+.1f}", va="center", fontsize=8.5)
    labels.append((y + 0.18, "Los et al. (2024)\nfull spectrum"))
    ax.set_yticks([p for p, _ in labels], [lab for _, lab in labels], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    ax.axvline(5, ls="--", color="#999", lw=1)
    ax.text(5.3, -0.45, "strong evidence", color="#777", fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Summed log Bayes factor vs classical, 10 shots (mean energy + P70)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in ("stochastic", "continuous")],
              labels=["Stochastic vs classical", "Continuous vs classical"], loc="lower right", fontsize=8.5)
    title(fig, "The stochastic preference survives realistic uncertainty, about 40% weaker")
    save(fig, "bayes_factors.png")


# ------------------------------------------------------------------ effective a0
def a0_eff():
    post = json.loads((R / "inference" / "los2024_posteriors.json").read_text())["results"]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for r in post:
        if r["setup"] != "los":
            continue
        meas = r["beam_unc"] == "calibrated"
        x = r["shot"] + (0.13 if meas else -0.13)
        c = COL["stochastic"] if meas else COL["incoming"]
        a = r["a0_eff"]
        ax.plot([x, x], [a["p16"], a["p84"]], color=c, lw=2.4, solid_capstyle="round")
        ax.plot(x, a["p50"], "o", color=c, ms=5.5, mec="white", mew=1)
    ax.set(xticks=range(10), xlabel="Collision shot", ylabel="Effective a0 (median, 16–84%)", ylim=(0, None))
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Line2D([], [], color=COL["incoming"], lw=2.4, marker="o"), plt.Line2D([], [], color=COL["stochastic"], lw=2.4, marker="o")],
              labels=["Published incoming-beam spread", "Measured incoming-beam error"], loc="upper right", fontsize=8.5)
    title(fig, "Realistic beam uncertainty widens every field-strength estimate by ~46%")
    save(fig, "a0_eff.png")


# ------------------------------------------------------------------ recovery tests (simulated shots)
def recovery_a0():
    rec = json.loads((R / "inference" / "recovery_focus.json").read_text())
    runs, summ = rec["runs"], rec["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, key, lab, cov in ((axes[0], "a0", "Peak a0", summ["a0_coverage90"]),
                              (axes[1], "a0_eff", "Effective a0", summ["a0_eff_coverage90"])):
        hi = 20.5
        ax.plot([0, hi], [0, hi], color="#999", lw=1, ls="--", zorder=0)
        for r in runs:
            t, q = r[f"true_{key}"], r[key]
            c = COL["stochastic"] if r[f"{key}_in90"] else COL["classical"]
            ax.plot([t, t], [q["p5"], q["p95"]], color=c, lw=1.3, alpha=0.55)
            ax.plot(t, q["p50"], "o", color=c, ms=4, mec="white", mew=0.6)
        ax.set(xlim=(0, hi), ylim=(0, hi), xlabel=f"True {lab.lower()}", ylabel=f"Inferred {lab.lower()} (median, 90%)",
               aspect="equal")
        ax.set_title(f"{lab}: truth inside 90% interval in {cov:.0%}", fontsize=9, loc="left")
    axes[1].legend(handles=[plt.Line2D([], [], color=COL["stochastic"], marker="o", lw=1.3),
                            plt.Line2D([], [], color=COL["classical"], marker="o", lw=1.3)],
                   labels=["truth inside 90% interval", "truth outside"], loc="upper left", fontsize=8)
    title(fig, f"On {summ['n']} simulated shots the effective a0 is recovered; the peak a0 is not pinned down")
    save(fig, "recovery_a0.png")


def model_recovery():
    with open(R / "model_comparison" / "recovery.csv") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, obs, lab in ((axes[0], "mean+P70", "Mean energy + P70"), (axes[1], "mean", "Mean energy only")):
        M = np.zeros((3, 3))
        for r in rows:
            win = max(MODELS, key=lambda m: float(r[f"logZ_{m}_{obs}"]))
            M[MODELS.index(r["true_model"]), MODELS.index(win)] += 1
        F = M / M.sum(1, keepdims=True)
        ax.imshow(F, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{F[i, j]:.0%}", ha="center", va="center", fontsize=9.5,
                        color="white" if F[i, j] > 0.55 else "#222")
        short = ["Stochastic", "Continuous", "Classical"]
        ax.set_xticks(range(3), short, fontsize=8.5)
        ax.set_yticks(range(3), short if ax is axes[0] else [""] * 3, fontsize=8.5)
        ax.set_xlabel("Model with the highest evidence")
        if ax is axes[0]:
            ax.set_ylabel("Model that generated the shot")
        ax.set_title(f"{lab} ({int(M.sum())} shots)", fontsize=9.5, loc="left")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    title(fig, "P70 carries the model information, but classical shots often look stochastic")
    save(fig, "model_recovery.png")


# ------------------------------------------------------------------ one predicted-vs-actual panel per stage
def pipeline_checks():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.0))

    def diag(ax, lo, hi):
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1, ls="--", zorder=0)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")

    # 1 beam model: incoming mean energy on the held-out run
    b = json.loads((R / "inference" / "beam_prediction_error.json").read_text())["mean_window_MeV"]
    ps = b["per_shot"]
    t, m = np.array(ps["measured"]), np.array(ps["median"])
    ax = axes[0, 0]
    ax.vlines(t, ps["p5"], ps["p95"], color=COL["stochastic"], alpha=0.25, lw=1.2)
    ax.plot(t, m, "o", color=COL["stochastic"], ms=3.5, mec="white", mew=0.4)
    lo, hi = min(t.min(), np.min(ps["p5"])) - 10, max(t.max(), np.max(ps["p95"])) + 10
    diag(ax, lo, hi)
    ax.set(xlabel="Measured mean energy ≥305 MeV (MeV)", ylabel="Predicted from diagnostics (MeV)")
    ax.set_title(f"1  Beam model vs measurement\n{len(t)} held-out shots: error {b['rmse']:.0f} MeV, "
                 f"90% bands cover {b['our_coverage90']:.0%}", fontsize=9, loc="left")

    # 2 simulation: classical radiation reaction vs the analytic Landau-Lifshitz solution
    rows = [ln.split() for ln in (R / "ptarmigan_validation" / "validation_summary.txt").read_text().splitlines()[1:6]]
    g0 = 1000.0 / 0.51099895
    a0 = np.array([float(r[0]) for r in rows])
    sim = (g0 - np.array([float(r[1]) for r in rows])) * 0.51099895
    ana = (g0 - np.array([float(r[2]) for r in rows])) * 0.51099895
    ax = axes[0, 1]
    ax.plot(ana, sim, "o", color=COL["classical"], ms=6, mec="white", mew=0.6)
    for x, y, a in zip(ana, sim, a0):
        ax.annotate(f"a0={a:.0f}", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=7.5, color="#555")
    diag(ax, 0, 460)
    ax.set(xlabel="Analytic energy loss, Landau–Lifshitz (MeV)", ylabel="Ptarmigan energy loss (MeV)")
    ax.set_title("2  Simulation vs exact solution\nclassical radiation reaction, 1 GeV electron: agree to 10⁻⁴",
                 fontsize=9, loc="left")

    # 3 surrogate vs Ptarmigan on collisions with unseen beams
    d = json.loads((R / "stage3" / "predictions_beam.json").read_text())["loss_window_MeV"]
    y, p, h = (np.array(d[k]) for k in ("ptarmigan", "surrogate", "halfwidth90"))
    ax = axes[1, 0]
    ax.plot(y, p, ".", color=COL["continuous"], ms=2.5, alpha=0.45)
    diag(ax, min(y.min(), p.min()) - 3, max(y.max(), p.max()) + 3)
    r2 = 1 - np.mean((p - y) ** 2) / np.var(y)
    ax.set(xlabel="Ptarmigan energy loss ≥305 MeV (MeV)", ylabel="Surrogate prediction (MeV)")
    ax.set_title(f"3  Surrogate vs simulation\n{len(y):,} unseen collisions: R² {r2:.3f}, "
                 f"90% intervals cover {np.mean(np.abs(p - y) <= h):.0%}", fontsize=9, loc="left")

    # 4 inference: P70 predicted by each model (fitted to mean energy only) vs measured, real shots
    s4 = json.loads((R / "diagnostics" / "p70" / "summary.json").read_text())["per_shot_model"]
    ax = axes[1, 1]
    for k, mdl in enumerate(("classical", "stochastic")):
        rr = [r for r in s4 if r["model"] == mdl]
        o = np.array([r["obs_P70"] for r in rr])
        q = np.array([r["mean_only"]["pred_P70_p5_p50_p95"] for r in rr])
        x = o + (k - 0.5) * 0.012
        ax.vlines(x, q[:, 0], q[:, 2], color=COL[mdl], alpha=0.35, lw=2)
        inside = np.mean((o >= q[:, 0]) & (o <= q[:, 2]))
        ax.plot(x, q[:, 1], "o", color=COL[mdl], ms=5, mec="white", mew=0.6,
                label=f"{NAME[mdl]}: measured inside 90% range on {inside * 10:.0f}/10")
    diag(ax, 1.2, 3.2)
    ax.set(xlabel="Measured P70 after the collision (1/GeV)", ylabel="Predicted P70 (1/GeV)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.set_title("4  Radiation-reaction models vs measurement\n10 real collision shots, each model fitted "
                 "to mean energy only", fontsize=9, loc="left")
    title(fig, "Every stage checked against something it did not see (dashed line: perfect agreement)")
    save(fig, "pipeline_checks.png")


# ------------------------------------------------------------------ surrogate vs Ptarmigan
def surrogate_accuracy():
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 9.2))
    obs = (("loss_window_MeV", "energy loss ≥305 MeV (MeV)", "{:.1f} MeV"), ("e_P70", "P70 (1/GeV)", "{:.2f}"))
    rows = (("beam", "Unseen incoming beams"), ("a0", "Extrapolation: trained on a0 ≤ 16.5, tested above"))
    for i, (split, lab) in enumerate(rows):
        d = json.loads((R / "stage3" / f"predictions_{split}.json").read_text())
        for j, (c, name, fmt) in enumerate(obs):
            ax = axes[i, j]
            y, p, h, t = (np.array(d[c][k]) for k in ("ptarmigan", "surrogate", "halfwidth90", "lightgbm"))
            if split == "a0":
                ax.plot(y, t, ".", color=COL["incoming"], ms=2.5, alpha=0.35, zorder=1)
            ax.plot(y, p, ".", color=COL["continuous"], ms=2.5, alpha=0.45, zorder=2)
            lo, hi = min(y.min(), p.min(), t.min()), max(y.max(), p.max(), t.max())
            pad = 0.03 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555", lw=1, ls="--", zorder=3)
            ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), aspect="equal",
                   xlabel=f"Ptarmigan {name}", ylabel=f"Predicted {name.split(' (')[0]}")
            txt = (f"{lab if j == 0 else ''}\nMLP: R² {_r2(p, y):.3f}, error " + fmt.format(_rmse(p, y))
                   + f", 90% intervals cover {np.mean(np.abs(p - y) <= h):.0%}"
                   + f"\nLightGBM: R² {_r2(t, y):.3f}, error " + fmt.format(_rmse(t, y)))
            ax.set_title(txt.lstrip("\n") if j else txt, fontsize=8.5, loc="left")
    axes[1, 0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL["continuous"]),
                               plt.Line2D([], [], marker="o", ls="", color=COL["incoming"])],
                      labels=["MLP ensemble (adopted)", "LightGBM (baseline)"], loc="upper left", fontsize=8)
    title(fig, "Surrogate vs Ptarmigan on collisions it never saw (dashed line: perfect agreement)")
    save(fig, "surrogate_accuracy.png")


def _r2(q, y):
    return 1 - np.mean((q - y) ** 2) / np.var(y)


def _rmse(q, y):
    return np.sqrt(np.mean((q - y) ** 2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {"p70_shots": p70_shots, "bayes_factors": bayes_factors, "a0_eff": a0_eff,
            "recovery_a0": recovery_a0, "model_recovery": model_recovery, "pipeline_checks": pipeline_checks, "surrogate_accuracy": surrogate_accuracy, "sweep_gif": sweep_gif}
    for name in sys.argv[1:] or figs:
        figs[name]()
