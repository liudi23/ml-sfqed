"""Validate the beam sampler (peak model, per-bin calibrated band) on held-out shots and export sampled beams for Stage 2.

Checks, per split:
  - beam-energy coverage at 50/80/90% and PIT uniformity (KS distance from uniform)
  - per-bin coverage of the 90% spectral band
  - correlation between energy and shape errors (the sampler assumes independence)
  - population level: does the pooled sampled distribution of W and mean energy <E> match the measured
    one on the held-out run? Compared with a naive sampler that resamples training spectra.
Exports results/stage1/beam_samples_<split>.npz (dN/dE per MeV, W, Q) and sampler_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlsfqed.beam_model.stage1 import beam_energy  # noqa: E402
from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser, to_electron_spectrum  # noqa: E402
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split  # noqa: E402

OUT = ROOT / "results" / "stage1"


def mean_energy(Y, E):
    return np.trapezoid(Y * E, E, axis=-1) / np.trapezoid(Y, E, axis=-1)


def validate(name, d, tr, te, n_samples, seed):
    E = d["E_MeV"]
    Y = physical_spectra(d)
    rf = RawFeaturiser().fit(d, tr)
    Xtr, Xte = rf.transform(d, tr), rf.transform(d, te)
    m = PeakBeamModel(E, seed=seed).fit(Xtr, Y[tr]).calibrate()
    samp = m.sample_calibrated(Xte, n_samples=n_samples, seed=seed)  # (n_te, n_s, bins)
    Yte, Wte = Y[te], beam_energy(Y[te], E)
    Ws = beam_energy(samp, E)
    out = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_samples": n_samples}
    # energy calibration
    pit = (Ws < Wte[:, None]).mean(1)
    out["W_coverage"] = {str(lv): float(np.mean(np.abs(pit - 0.5) <= lv / 2)) for lv in (0.5, 0.8, 0.9)}
    out["W_PIT_KS"] = float(np.max(np.abs(np.sort(pit) - (np.arange(len(pit)) + 0.5) / len(pit))))
    # spectral band
    lo, hi = np.percentile(samp, [5, 95], axis=1)
    out["spectral_bin_coverage_90"] = float(np.mean((Yte >= lo) & (Yte <= hi)))
    peak = Yte > 0.2 * Yte.max(1, keepdims=True)                        # bins carrying the beam
    out["spectral_bin_coverage_90_on_beam"] = float(np.mean(((Yte >= lo) & (Yte <= hi))[peak]))
    # energy-shape error correlation on test shots
    Yp, Wp = m.predict(Xte)
    Sp, St = Yp / Wp[:, None], Yte / Wte[:, None]
    e_logW = np.log(Wte) - np.log(Wp)
    e_meanE = mean_energy(St, E) - mean_energy(Sp, E)
    out["corr_logW_error_vs_meanE_error"] = float(np.corrcoef(e_logW, e_meanE)[0, 1])
    # population level vs naive sampler
    rng = np.random.default_rng(seed)
    naive = Y[tr][rng.integers(tr.sum(), size=(te.sum(), n_samples))]
    pops = {}
    for lab, S in (("model", samp), ("naive", naive)):
        pops[lab] = {"W_KS": float(ks_2samp(beam_energy(S, E).ravel(), Wte).statistic),
                     "meanE_KS": float(ks_2samp(mean_energy(S, E).ravel(), mean_energy(Yte, E)).statistic),
                     "W_mean_J": float(beam_energy(S, E).mean())}
    pops["measured"] = {"W_mean_J": float(Wte.mean())}
    out["population"] = pops
    # export for Stage 2
    dNdE, Q = to_electron_spectrum(samp[:, :50], E)
    np.savez_compressed(OUT / f"beam_samples_{name}.npz", E_MeV=E, dNdE_per_MeV=dNdE.astype(np.float32),
                        W_J=Ws[:, :50].astype(np.float32), Q_pC=Q.astype(np.float32),
                        run=d["run"][te], shot=d["shot"][te])
    return out, samp, Yte, Wte


def plot(name, d, samp, Yte, Wte, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf, blue, orange = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"
    plt.rcParams.update({"axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2, "axes.grid": True,
                         "grid.color": grid, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": surf, "axes.facecolor": surf, "font.size": 9})
    E = d["E_MeV"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    Ws = beam_energy(samp, E).ravel()
    bins = np.linspace(0, max(Ws.max(), Wte.max()) * 1.02, 30)
    axs[0].hist(Ws, bins=bins, density=True, color=blue, alpha=0.35, label="sampler (pooled)")
    axs[0].hist(Wte, bins=bins, density=True, histtype="step", color=ink, lw=1.5, label="measured")
    axs[0].set(xlabel="beam energy [J]", ylabel="density", title="Population of held-out shots")
    axs[0].legend(frameon=False, fontsize=8)
    pit = (beam_energy(samp, E) < Wte[:, None]).mean(1)
    axs[1].hist(pit, bins=10, range=(0, 1), color=blue, alpha=0.6)
    axs[1].axhline(len(pit) / 10, color=ink2, ls="--", lw=1, label="perfect calibration")
    axs[1].set(xlabel="PIT of measured beam energy", ylabel="shots", title=f"Calibration (KS {out['W_PIT_KS']:.2f})")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    from run_stage1 import check_fits
    check_fits(fig, f"sampler_{name}.png")
    fig.savefig(OUT / f"sampler_{name}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_all()
    splits = {"paper": (d["split"] == "train", d["split"] == "test"), "time": time_ordered_split(d)}
    allres = {}
    for name, (tr, te) in splits.items():
        out, samp, Yte, Wte = validate(name, d, tr, te, a.samples, a.seed)
        plot(name, d, samp, Yte, Wte, out)
        allres[name] = out
    (OUT / "sampler_validation.json").write_text(json.dumps(allres, indent=2))
    print(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
