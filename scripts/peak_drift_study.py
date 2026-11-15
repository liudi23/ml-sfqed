"""Can peak drift be corrected online? Width shrinkage + running peak-parameter offsets, Stage 1 peak model.

  python scripts/peak_drift_study.py tune     # train runs 05-10, online on run11: choose settings
  python scripts/peak_drift_study.py test     # train runs 05-12, online on run13

lam: predicted log peak width shrunk toward the training mean (1 = as fitted, 0 = constant width).
alpha: online offset of (log W, log height, position, log width) from earlier shots of the same run
(0 = off, a number = exponential weight, "mean" = running mean). Shots are scored from the 11th on.
Result (Nov 2026): width error -13% and biases removed on run13, but spectral MSE only -1%: the remaining
error is shot-to-shot scatter, not drift. Writes results/stage1/peak_drift_{tune,test}.json.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT + "/src")
from mlsfqed.beam_model.peaks import fit_peaks
from mlsfqed.beam_model.stage1 import beam_energy
from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser
from mlsfqed.data.gemini import load_all, physical_spectra

d = load_all(); E = d["E_MeV"]; Y = physical_spectra(d)
runs = np.array([r.split("/")[-1] for r in d["run"]])
order = ["run05","run06","run08","run09","run10","run11","run12","run13"]
LAMS = [1.0, 0.5, 0.25, 0.0]; ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, "mean"]
SKIP = 10                                      # score shots from the 11th of the run on (all variants alike)

def setup(train_runs, test_run, members):
    tr, te = np.isin(runs, train_runs), runs == test_run
    rf = RawFeaturiser().fit(d, tr); Rtr, Rte = rf.transform(d, tr), rf.transform(d, te)
    m = PeakBeamModel(E, n_members=members, seed=0).fit(Rtr, Y[tr])
    Yt = Y[te]; Wt = beam_energy(Yt, E)
    P = fit_peaks(Yt / Wt[:, None], E)
    true = np.column_stack([np.log(Wt), np.log(P[:, 0]), P[:, 1], np.log(P[:, 2])])
    muW, sdW = m.energy.predict(Rte); mp = [p.predict(Rte) for p in m.peak]
    Cm = np.mean([np.column_stack([r.predict(Rte) for r in mm]) for mm in m.cont.members], 0)
    mu = np.column_stack([muW] + [x[0] for x in mp]); sd = np.column_stack([sdW] + [x[1] for x in mp])
    wmean = float(np.mean(np.log(m.peak_fits[:, 2])))
    return m, Yt, true, mu, sd, Cm, wmean

def run_variant(m, Yt, true, mu0, sd, Cm, wmean, lam, alpha, n_samples=100):
    mu = mu0.copy(); mu[:, 3] = wmean + lam * (mu[:, 3] - wmean)
    off = np.zeros(4); n = len(Yt); adj = np.zeros_like(mu); seen = 0
    for i in range(n):                         # strictly sequential: shot i uses shots < i only
        adj[i] = mu[i] + off
        r = true[i] - adj[i]
        if np.all(np.isfinite(r)) and alpha != 0.0:
            seen += 1
            a = 1.0 / seen if alpha == "mean" else alpha
            off = off + a * (r - off) if seen > 1 or alpha == "mean" else a * r
    point = m._assemble(adj[:, 0], adj[:, 1:], Cm)
    rng = np.random.default_rng(1); j = rng.integers(len(m.res_W), size=(n, n_samples))
    logW = adj[:, :1] + m.res_W[j] * sd[:, :1]
    T = np.stack([adj[:, k + 1][:, None] + m.res_P[j, k] * sd[:, k + 1][:, None] for k in range(3)], -1)
    S = m._assemble(logW, T, Cm[:, None, :] + m.res_C[j])
    s = slice(SKIP, None); win = E > 650
    e = adj[s] - true[s]
    return {"MSE_point": float(((point[s] - Yt[s]) ** 2).mean()), "MSE_mean": float(((S.mean(1)[s] - Yt[s]) ** 2).mean()),
            "peak_height_ratio_point": float(np.median(point[s][:, win].max(1) / Yt[s][:, win].max(1))),
            "rmse_logW": float(np.sqrt(np.nanmean(e[:, 0] ** 2))), "rmse_logA": float(np.sqrt(np.nanmean(e[:, 1] ** 2))),
            "rmse_mu": float(np.sqrt(np.nanmean(e[:, 2] ** 2))), "rmse_logsig": float(np.sqrt(np.nanmean(e[:, 3] ** 2))),
            "bias": np.nanmean(e, 0).round(4).tolist()}

if __name__ == "__main__":
    stage = sys.argv[1]; members = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tr_runs, te_run = (order[:5], "run11") if stage == "tune" else (order[:7], "run13")
    t0 = time.time(); ctx = setup(tr_runs, te_run, members)
    res = {f"{lam}|{al}": run_variant(*ctx, lam, al) for lam, al in itertools.product(LAMS, ALPHAS)}
    with open(f"{ROOT}/results/stage1/peak_drift_{stage}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(stage, te_run, f"{time.time()-t0:.0f}s")
    print(f"{'lam|alpha':12s} {'MSEpt':>7s} {'MSEmean':>7s} {'height':>6s} {'logW':>6s} {'logA':>6s} {'mu':>6s} {'logsig':>6s}")
    for k, v in res.items():
        print(f"{k:12s} {v['MSE_point']:7.4f} {v['MSE_mean']:7.4f} {v['peak_height_ratio_point']:6.2f} {v['rmse_logW']:6.3f} {v['rmse_logA']:6.3f} {v['rmse_mu']:6.1f} {v['rmse_logsig']:6.3f}")
