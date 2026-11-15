"""Learning curve: does the surrogate keep improving with more Ptarmigan runs?

Trains the MLP ensemble (and LightGBM) on 1/8, 1/4, 1/2 and all of the training runs of the new-beam split,
always testing on the same held-out beams. Fits RMSE ~ N^(-alpha) per observable on the larger sizes and
extrapolates to twice the data.
  python scripts/stage3_learning_curve.py           (~5-10 min on a laptop)
  python scripts/stage3_learning_curve.py --quick   (small models, smoke test)
Writes results/stage3/learning_curve.json and learning_curve.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage1 import check_fits
from stage3_surrogate import split_masks

from mlsfqed.surrogate.data import NOISE, build
from mlsfqed.surrogate.models import MLPEnsemble, TreeSurrogate

OUT = ROOT / "results" / "stage3"
FRACTIONS = [0.125, 0.25, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    X, Y, meta, _, _ = build()
    tr, te = split_masks("beam", X, meta)
    cols = list(Y.columns)
    idx = np.flatnonzero(tr)
    rng = np.random.default_rng(0)
    order = rng.permutation(idx)
    res = {"n_test": int(te.sum()), "sizes": [], "mlp_rmse": [], "tree_rmse": []}
    for f in FRACTIONS:
        sub = np.zeros(len(X), bool)
        sub[order[:int(f * len(order))]] = True
        mlp = MLPEnsemble(quick=a.quick).fit(X[sub], Y[sub])
        P, _ = mlp.predict(X[te])
        T = TreeSurrogate(quick=a.quick).fit(X[sub], Y[sub]).predict(X[te])
        yt = Y[te].values
        res["sizes"].append(int(sub.sum()))
        res["mlp_rmse"].append(np.sqrt(np.mean((P - yt) ** 2, axis=0)).tolist())
        res["tree_rmse"].append(np.sqrt(np.mean((T - yt) ** 2, axis=0)).tolist())
        print(f"N = {sub.sum():5d}: MLP RMSE/noise " + "  ".join(
            f"{c.split('_')[0] if c.startswith('loss') else c[:10]} {r / NOISE[c]:.1f}" if c in NOISE else ""
            for c, r in zip(cols, res["mlp_rmse"][-1])), flush=True)
    N = np.array(res["sizes"], float)
    M = np.array(res["mlp_rmse"])
    res["alpha"], res["rmse_at_2x"] = {}, {}
    for j, c in enumerate(cols):
        slope = np.polyfit(np.log(N[1:]), np.log(M[1:, j]), 1)[0]
        res["alpha"][c] = float(-slope)
        res["rmse_at_2x"][c] = float(M[-1, j] * 2 ** slope)
    res["observables"] = cols
    (OUT / f"learning_curve{'_quick' if a.quick else ''}.json").write_text(json.dumps(res, indent=2))
    print(f"\n{'observable':22s}{'alpha':>7s}{'RMSE/noise now':>16s}{'at 2x data':>12s}")
    for j, c in enumerate(cols):
        nz = NOISE.get(c)
        now = M[-1, j] / nz if nz else float("nan")
        later = res["rmse_at_2x"][c] / nz if nz else float("nan")
        print(f"{c:22s}{res['alpha'][c]:7.2f}{now:16.2f}{later:12.2f}")
    print("alpha ~ 0.5: still data-limited (more runs help); alpha ~ 0: saturated (model or noise limited)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [c for c in ("loss_MeV", "e_P70", "log_g_number_per_e", "g_crit_MeV", "g_theta_x_rms_mrad") if c in cols]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for c, col in zip(show, colors):
        j = cols.index(c)
        ax.plot(N, M[:, j] / NOISE[c], "o-", color=col, label=c, lw=2, ms=5)
    ax.axhline(1, color="#52514e", ls="--", lw=1, label="Monte Carlo noise floor")
    ax.set(xscale="log", yscale="log", xlabel="training runs", ylabel="RMSE / Monte Carlo noise")
    ax.set_title("Surrogate learning curve (new beams)", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    check_fits(fig, "learning_curve.png")
    fig.savefig(OUT / f"learning_curve{'_quick' if a.quick else ''}.png", dpi=150)


if __name__ == "__main__":
    main()
