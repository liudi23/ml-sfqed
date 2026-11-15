"""Which peak shape describes the Gemini electron spectra? (physics check for the Stage 1 peak model)

  python scripts/peak_shape_study.py

Fits the highest-energy peak of all 645 reference spectra (window -180/+150 MeV, linear background) with
  gauss    symmetric Gaussian (the current Stage 1 peak model)
  emg      Gaussian convolved with an exponential tail toward LOW energy (exponentially modified Gaussian):
           sharp high-energy edge = first-injected electrons at maximum energy; tail = later injection
  caustic  (E_max - E)^(-1/2) on [E_max - depth, E_max] smeared by a Gaussian: pure phase-space-rotation pile-up
and measures the peak asymmetry (low-side / high-side half width). Result (Nov 2026): EMG median R^2 0.991
vs Gaussian 0.959 (unexplained variance 0.010 vs 0.044), better on 100% of shots; every peak is skewed to
low energy. Writes results/stage1/peak_shape_study.json.
"""
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT + "/src")
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import erfc

from mlsfqed.beam_model.stage1 import beam_energy
from mlsfqed.data.gemini import load_all, physical_spectra

d = load_all(); E = d["E_MeV"]; Y = physical_spectra(d); S = Y / beam_energy(Y, E)[:, None]
dE = E[1] - E[0]
def gauss(x, A, mu, s, c0, c1): return A * np.exp(-0.5 * ((x - mu) / s) ** 2) + c0 + c1 * (x - mu)
def emg(x, A, mu, s, tau, c0, c1):          # Gaussian convolved with an exponential tail toward LOW energy
    z = (mu - x) / s; lam = s / tau
    return A * 0.5 * lam * np.exp(0.5 * lam ** 2 - (mu - x) / tau) * erfc((lam - z) / np.sqrt(2)) * np.sqrt(2 * np.pi) / 2 + c0 + c1 * (x - mu)
def caustic(x, A, Emax, s, depth, c0, c1):  # (Emax-E)^-1/2 on [Emax-depth, Emax], smeared by a Gaussian of width s
    g = np.arange(Emax - depth, Emax, 0.5); w = 1 / np.sqrt(Emax - g + 0.25)
    k = np.exp(-0.5 * ((x[:, None] - g[None, :]) / s) ** 2) @ w
    return A * k / k.max() + c0 + c1 * (x - Emax)
res = {"gauss": [], "emg": [], "caustic": []}; skew = []
for s in S:
    pk, _ = find_peaks(s, prominence=0.05 * s.max()); pk = pk[E[pk] > 650]
    if not len(pk): continue
    i = pk[-1]; m = (E > E[i] - 180) & (E < E[i] + 150); x, y = E[m], s[m]
    base = y.min(); A0 = s[i] - base
    fits = {"gauss": (gauss, [A0, E[i], 25, base, 0], ([0, E[i]-60, 4, -1, -1e-2], [10, E[i]+60, 120, 10, 1e-2])),
            "emg": (emg, [A0, E[i], 15, 20, base, 0], ([0, E[i]-60, 2, 1, -1, -1e-2], [10, E[i]+80, 120, 300, 10, 1e-2])),
            "caustic": (caustic, [A0, E[i] + 15, 10, 60, base, 0], ([0, E[i]-40, 2, 5, -1, -1e-2], [10, E[i]+100, 80, 300, 10, 1e-2]))}
    out = {}
    for k, (f, p0, b) in fits.items():
        try:
            p, _ = curve_fit(f, x, y, p0=p0, bounds=b, maxfev=4000)
            out[k] = 1 - np.sum((y - f(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
        except (RuntimeError, ValueError):
            out[k] = np.nan
    if all(np.isfinite(list(out.values()))):
        for k, v in res.items():
            v.append(out[k])
        # empirical asymmetry of the peak: half-width below vs above the maximum (at half height above base)
        half = base + 0.5 * A0; lo = i
        while lo > 0 and s[lo] > half: lo -= 1
        hi = i
        while hi < len(s) - 1 and s[hi] > half: hi += 1
        skew.append((E[i] - E[lo]) / max(E[hi] - E[i], dE))
r = {k: np.array(v) for k, v in res.items()}
print(f"{len(skew)} spectra with a fitted peak; local fit R^2 (median, 10th pct) and unexplained variance:")
for k, v in r.items():
    print(f"   {k:8s} R2 median {np.median(v):.3f}  p10 {np.percentile(v,10):.3f}  mean(1-R2) {np.mean(1-v):.4f}")
print(f"   EMG better than Gaussian on {np.mean(r['emg'] > r['gauss']):.0%} of shots; caustic better on {np.mean(r['caustic'] > r['gauss']):.0%}")
sk = np.array(skew)
print(f"peak asymmetry: low-side / high-side half width, median {np.median(sk):.2f} (25-75%: {np.percentile(sk,25):.2f}-{np.percentile(sk,75):.2f}); >1 on {np.mean(sk>1):.0%} of shots")
summary = {"n": len(skew), **{k: {"R2_median": float(np.median(v)), "R2_p10": float(np.percentile(v, 10)),
                                   "unexplained_mean": float(np.mean(1 - v))} for k, v in r.items()},
           "emg_better_than_gauss": float(np.mean(r["emg"] > r["gauss"])),
           "asymmetry_low_over_high_median": float(np.median(sk)), "asymmetry_gt1_frac": float(np.mean(sk > 1))}
with open(Path(ROOT) / "results" / "stage1" / "peak_shape_study.json", "w") as f:
    json.dump(summary, f, indent=1)
