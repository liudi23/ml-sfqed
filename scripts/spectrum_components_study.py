"""How many electron populations are in a Gemini spectrum, and can the diagnostics predict them?

  python scripts/spectrum_components_study.py

1. Census of local maxima above 320 MeV at three prominence thresholds.
2. Full-spectrum fits (305-1250 MeV): square-root edge (main bunch) + quadratic background, plus 0, 1 or 2
   broad Gaussians (secondary populations); R^2 and BIC per shot.
3. Positions, widths and energy fractions of the extra components; spacing test for a bunch train.
4. Predictability of every component parameter from the diagnostics (bagged LightGBM; R^2 on run 13 and on
   the authors' random split).
Result (15 Nov 2026): a second, broad population (median 521 MeV, sigma ~110 MeV, ~30% of the energy) is in
essentially every shot (full-spectrum R^2 0.922 -> 0.992); a weaker third (~6%) adds little (0.997) and is not
a bunch train. None of the extra components' parameters is predictable (R^2 ~ 0), unlike the edge (0.7).
Writes results/stage1/spectrum_components.json.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from mlsfqed.beam_model.peaks import _EDGE_MAX, _edge_profile, fit_peaks
from mlsfqed.beam_model.stage1 import beam_energy
from mlsfqed.data.gemini import load_all, physical_spectra

d = load_all(); E = d["E_MeV"]; Y = physical_spectra(d); S = Y / beam_energy(Y, E)[:, None]
print("E grid", E[0], E[-1], len(E))
# ---- 1. bump census
for prom in (0.05, 0.10, 0.15):
    n = []
    for s in S:
        pk, _ = find_peaks(s, prominence=prom * s.max(), distance=8); n.append(int((E[pk] > 320).sum()))
    n = np.array(n); print(f"local maxima >320 MeV with prominence >{prom:.0%} of max: " + ", ".join(f"{k}: {int((n == k).sum())}" for k in range(1, 6)))
# ---- 2. full-spectrum fits: edge + background (+ K broad Gaussians)
m = (E >= 305) & (E <= 1250); x = E[m]; n_pts = m.sum()
def edge(x, A, Em, s): return A * _edge_profile((x - Em) / s) / _EDGE_MAX
def g(x, A, mu, s): return A * np.exp(-0.5 * ((x - mu) / s) ** 2)
def bg(x, b0, b1, b2): t = (x - 700) / 400; return b0 + b1 * t + b2 * t * t
def M(K):
    def f(x, *p):
        y = edge(x, *p[:3]) + bg(x, *p[3:6])
        for k in range(K): y = y + g(x, *p[6 + 3 * k: 9 + 3 * k])
        return y
    return f
P = fit_peaks(S, E, shape="sqrt_edge")
res = {K: [] for K in (0, 1, 2)}
t0 = time.time()
for i, s in enumerate(S):
    y = s[m]; A, Em, sg = P[i, :3]
    p0 = [A, Em, sg, y.min(), 0, 0]; lo = [0, Em - 40, 2, -2, -2, -2]; hi = [10, Em + 40, 80, 5, 5, 5]
    best = {}
    for K in (0, 1, 2):
        try:
            p0c = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            p, _ = curve_fit(M(K), x, y, p0=p0c, bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            best[K] = None; break
        r = y - M(K)(x, *p); rss = float(r @ r)
        best[K] = {"p": p.tolist(), "rss": rss, "bic": n_pts * np.log(rss / n_pts) + len(p) * np.log(n_pts),
                   "r2": 1 - rss / float(((y - y.mean()) ** 2).sum())}
        # next component starts at the largest positive residual below the edge
        rr = np.where(x < p[1] - 60, r, -np.inf); j = int(np.argmax(rr))
        p0 = list(p) + [max(rr[j], 1e-3), x[j], 60]; lo = list(lo) + [0, 320, 15]; hi = list(hi) + [5, p[1] - 40, 250]
    for K, lst in res.items():
        lst.append(best.get(K))
print(f"fits done in {time.time()-t0:.0f}s")
ok = [i for i in range(len(S)) if all(res[K][i] for K in res)]
bic = np.array([[res[K][i]["bic"] for K in (0, 1, 2)] for i in ok]); r2 = np.array([[res[K][i]["r2"] for K in (0, 1, 2)] for i in ok])
bestK = bic.argmin(1)
print(f"{len(ok)} shots; full-spectrum R2 median: edge+bg {np.median(r2[:,0]):.3f}, +1 Gaussian {np.median(r2[:,1]):.3f}, +2 Gaussians {np.median(r2[:,2]):.3f}")
print("components preferred by BIC (strong = BIC drop > 10):", {k: int((bestK == k).sum()) for k in (0, 1, 2)},
      "| 2nd comp strongly needed:", int(((bic[:, 0] - bic[:, 1]) > 10).sum()), "| 3rd strongly needed:", int(((bic[:, 1] - bic[:, 2]) > 10).sum()))
J = {"ok": ok, "res": {str(K): [res[K][i] for i in ok] for K in res}}
with open(ROOT / "results" / "stage1" / "spectrum_components.json", "w") as f:
    json.dump(J, f)

# ---- 3, 4. components and their predictability
from mlsfqed.beam_model.twopart import EnergyEnsemble, RawFeaturiser
from mlsfqed.data.gemini import time_ordered_split

R1 = np.array([q["p"] for q in J["res"]["1"]]); R2 = np.array([q["p"] for q in J["res"]["2"]])
def wt(A, s): return A * s * np.sqrt(2 * np.pi)          # area of a Gaussian component (fraction of unit energy, x GeV->MeV)
print("Model edge + background + 1 Gaussian (second population):")
mu2, s2, w2 = R1[:, 7], R1[:, 8], wt(R1[:, 6], R1[:, 8]) / 1e3
for nm, v in (("position MeV", mu2), ("sigma MeV", s2), ("energy fraction", w2)):
    print(f"   {nm:16s} median {np.median(v):7.3f}  10-90%: {np.percentile(v,10):7.3f} - {np.percentile(v,90):7.3f}")
print(f"   at a bound: position {np.mean(np.isclose(mu2, 320, atol=1)) :.0%} low; sigma {np.mean(s2 > 249):.0%} high, {np.mean(s2 < 16):.0%} low")
# two extra Gaussians: order them by position
a, b = R2[:, 6:9], R2[:, 9:12]
lowc = np.where((a[:, 1] < b[:, 1])[:, None], a, b); midc = np.where((a[:, 1] < b[:, 1])[:, None], b, a)
print("Model edge + background + 2 Gaussians:")
for lab, c in (("lower", lowc), ("middle", midc)):
    print(f"   {lab:6s} position median {np.median(c[:,1]):.0f} (10-90% {np.percentile(c[:,1],10):.0f}-{np.percentile(c[:,1],90):.0f}) MeV,"
          f" sigma {np.median(c[:,2]):.0f} MeV, energy fraction {np.median(wt(c[:,0],c[:,2])/1e3):.3f}")
gap = midc[:, 1] - lowc[:, 1]; dist_edge = R2[:, 1] - midc[:, 1]
print(f"   spacing lower->middle {np.median(gap):.0f} MeV, middle->edge {np.median(dist_edge):.0f} MeV (equal spacing would suggest a bunch train)")
print(f"   corr(spacing lower->middle, middle->edge) = {np.corrcoef(gap, dist_edge)[0,1]:.2f}")
idx = np.array(J["ok"]); tr0, te0 = time_ordered_split(d)
sp = {"run13": (tr0[idx], te0[idx]), "paper": ((d["split"] == "train")[idx], (d["split"] == "test")[idx])}
F = {}
for k, (tr, te) in sp.items():
    rf = RawFeaturiser().fit(d, idx[tr]); F[k] = (rf.transform(d, idx[tr]), rf.transform(d, idx[te]))
def r2(y):
    o = []
    for k, (tr, te) in sp.items():
        m = EnergyEnsemble(n_members=10, seed=0).fit(F[k][0], y[tr]); p = m.predict(F[k][1])[0]
        o.append(1 - np.mean((p - y[te]) ** 2) / np.var(y[te]))
    return "/".join(f"{v:.2f}" for v in o)
print("Predictability from diagnostics, R2 [run13/random]:")
print("   edge E_max", r2(R1[:, 1]), "| edge log height", r2(np.log(R1[:, 0])))
print("   2nd population: position", r2(mu2), "| log sigma", r2(np.log(s2)), "| log energy fraction", r2(np.log(w2)))
print("   2-Gaussian model: lower position", r2(lowc[:, 1]), "| middle position", r2(midc[:, 1]), "| lower log fraction", r2(np.log(wt(lowc[:,0],lowc[:,2]))), "| middle log fraction", r2(np.log(wt(midc[:,0],midc[:,2]) + 1e-6)))
