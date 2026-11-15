"""Data for the demo dashboard: the notebook's results (notebooks/demo.ipynb) as small JSON tables.

  python scripts/demo_data.py            # surrogate, inference and (if the data are present) real-shot parts
  python scripts/demo_data.py --beam     # also the Stage 1 beam-model examples (needs the Gemini data, ~2 min)

Writes results/demo/<name>.json, each {"rows": [...], "meta": [...]} with this script's code in meta.
Uses only the shipped models in models/; the real-shot part needs the Los et al. data (local only) and
publishes derived numbers only.
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlsfqed.inference.posterior import ShotPosterior, run_mcmc
from mlsfqed.simulation.ptarmigan import p70
from mlsfqed.surrogate.api import Surrogate

logging.getLogger("emcee").setLevel(logging.ERROR)
OUT = ROOT / "results" / "demo"
CODE = Path(__file__).read_text()
MODELS = ("stochastic", "continuous", "classical")


def save(name, rows, inputs):
    meta = [{"label": "demo_data.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": inputs}]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, separators=(",", ":")))
    print(name, len(rows))


def r(x, n=3):
    return round(float(x), n)


def surrogate_parts(E, B):
    sur = {m: Surrogate.load("stochastic_4096" if m == "stochastic" else m) for m in MODELS}
    w = E >= 305
    beam_rows = []
    for i, y in enumerate(B):
        n = y / E
        beam_rows.append({"beam": i + 1, "mean_window_MeV": r(np.sum(n[w] * E[w]) / np.sum(n[w]), 1),
                          "P70": r(p70(E[w], n[w], 0)), "energy_J": r(np.trapezoid(y, E / 1e3), 4)})
    save("demo_beams", beam_rows, "models/example_beams.npz (six measured Gemini 2021 spectra, run 13)")
    spec = [{"beam": i + 1, "E_MeV": r(e, 1), "dWdE": r(v, 4)}
            for i, y in enumerate(B) for e, v in zip(E[::2], y[::2]) if 250 <= e <= 1300]
    save("demo_beam_spectra", spec, "models/example_beams.npz")

    a0 = np.round(np.arange(5, 20.01, 0.5), 2)
    rows = []
    for m in MODELS:
        for i in range(len(B)):
            d = sur[m].predict(B[i], a0=a0)
            for _, x in d.iterrows():
                rows.append({"model": m, "beam": i + 1, "a0": r(x.a0, 1),
                             "loss": r(x.loss_window_MeV, 2), "loss_lo": r(x.loss_window_MeV_lo, 2), "loss_hi": r(x.loss_window_MeV_hi, 2),
                             "p70": r(x.e_P70), "p70_lo": r(x.e_P70_lo), "p70_hi": r(x.e_P70_hi),
                             "out_mean": r(x.out_mean_window_MeV, 1), "g_crit": r(x.g_crit_MeV, 2),
                             "photons_per_e": r(10 ** x.log_g_number_per_e, 3)})
    save("demo_playground", rows, "models/surrogate_<model>_4096.pkl and models/example_beams.npz; head-on at focus, "
         "nominal Gemini laser and beam (waist 2.5 um, 45 fs, radius 0.7 um, length 5 um, divergence 2 mrad)")

    dz = np.arange(-250, 250.1, 12.5)
    rr = np.arange(0, 5.01, 0.25)
    DZ, R = np.meshgrid(dz, rr)
    rows = []
    for a in (10.0, 15.0, 20.0):
        d = sur["stochastic"].predict(B[0], a0=a, dz_um=DZ.ravel(), r_offset_um=R.ravel())
        rows += [{"a0": a, "dz_um": r(x, 1), "r_um": r(y, 2), "loss": r(v, 2), "a0_eff": r(e, 2)}
                 for x, y, v, e in zip(DZ.ravel(), R.ravel(), d.loss_window_MeV, d.a0_eff_est)]
    save("demo_geometry", rows, "models/surrogate_stochastic_4096.pkl, example beam 1")

    s = Surrogate.load("stochastic")
    n = 100_000
    t0 = time.time()
    s.predict(B[0], a0=np.random.default_rng(0).uniform(5, 20, n))
    dt = time.time() - t0
    pt = json.loads((ROOT / "results" / "stage3" / "metrics_beam.json").read_text())["ptarmigan_mean_wall_s"]
    save("demo_speed", [{"collisions": n, "surrogate_s": r(dt, 3), "surrogate_us_each": r(dt / n * 1e6, 2),
                         "ptarmigan_s_each": r(pt, 2), "ptarmigan_cpu_hours": r(n * pt / 3600, 0)}],
         "timed on this machine; Ptarmigan time per collision from results/stage3/metrics_beam.json")
    return s, sur


def toy_inference(s, sur, B):
    truth = {"a0": 14.0, "dz_um": 40.0, "length_um": 5.0}
    t = sur["stochastic"].predict(B[2], **truth)
    obs_mean, obs_p70 = float(t.out_mean_window_MeV[0]), float(t.e_P70[0])
    post = ShotPosterior(s.model, B[2:3], obs_mean, obs_p70, meas_sd_mean=10.0, dz_prior="focus")
    chain, _ = run_mcmc(post, nwalkers=32, nsteps=8000, burn=3000)
    a_eff = post.a0_eff(chain)
    summ, hist = [], []
    for name, x, tr in (("Peak a0", chain[:, 0], truth["a0"]), ("Effective a0", a_eff, float(t.a0_eff_est[0]))):
        lo, med, hi = np.percentile(x, [5, 50, 95])
        summ.append({"parameter": name, "truth": r(tr, 2), "median": r(med, 2), "p5": r(lo, 2), "p95": r(hi, 2)})
        c, edges = np.histogram(x, bins=np.arange(0, 20.01, 0.4))
        hist += [{"parameter": name, "lo": r(a, 2), "hi": r(b, 2), "share": r(k / len(x), 4)} for a, b, k in zip(edges[:-1], edges[1:], c)]
    summ.append({"parameter": "Measured mean energy (MeV)", "truth": r(obs_mean, 1), "median": None, "p5": None, "p95": None})
    summ.append({"parameter": "Measured P70", "truth": r(obs_p70, 3), "median": None, "p5": None, "p95": None})
    save("demo_toy_posterior", summ, "toy shot: stochastic surrogate at a0 = 14, 40 um from focus, example beam 3")
    save("demo_toy_hist", hist, "posterior samples of the toy shot (emcee, 32 walkers x 8,000 steps)")
    ev = []
    for m in MODELS:
        p = ShotPosterior(sur[m].model, B[2:3], obs_mean, obs_p70, meas_sd_mean=10.0, dz_prior="focus")
        for obs, use in (("Mean energy + P70", (0, 1)), ("Mean energy only", (0,))):
            ev.append({"model": m, "observables": obs, "logZ": p.log_evidence(n=40000, seed=1, use=use)[0]})
    ref = {e["observables"]: e["logZ"] for e in ev if e["model"] == "classical"}
    rows = [{"model": e["model"], "observables": e["observables"], "log_bf": r(e["logZ"] - ref[e["observables"]], 2)}
            for e in ev if e["model"] != "classical"]
    save("demo_toy_evidence", rows, "toy shot; evidence by Monte Carlo over the prior, 40,000 draws")


def real_shot(s, sur):
    los = ROOT / "data" / "raw" / "los2024"
    if not (los / "mcmc_input_spectra.h5").exists():
        print("Los et al. data not found: skipping the real shot")
        return
    from mlsfqed.inference.los2024 import beams_on_pool_grid, load, spectrum_observables
    data = load()
    err = json.loads((ROOT / "results" / "inference" / "beam_prediction_error.json").read_text())
    rmse = (err["mean_window_MeV"]["rmse"], err["P70"]["rmse"])
    rows = []
    for k in range(10):
        bm = beams_on_pool_grid(data, s.E_MeV)[k][::2]
        o = spectrum_observables(data["E_MeV"], data["shots"][k]["post_dNdE"], smooth_mev=10.0)
        z = {}
        for m in MODELS:
            p = ShotPosterior(sur[m].model, bm, o["mean_window_MeV"], o["P70"], 10.0, beam_rmse=rmse, dz_prior="focus")
            z[m] = p.log_evidence(n=40000, seed=1)[0]
        rows.append({"shot": k, "measured_mean_MeV": r(o["mean_window_MeV"], 1), "measured_P70": r(o["P70"], 3),
                     "bf_stochastic": r(z["stochastic"] - z["classical"], 2), "bf_continuous": r(z["continuous"] - z["classical"], 2)})
    save("demo_real_shots", rows, "Los et al. (2024) public collision data, used locally; derived numbers only. "
         "Incoming beam: 50 of their 100 predictions per shot, widened to the measured error")


def beam_examples():
    from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser
    from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split
    d = load_all()
    Eg, Y = d["E_MeV"], physical_spectra(d)
    tr, te = time_ordered_split(d)
    rf = RawFeaturiser().fit(d, tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PeakBeamModel(Eg, seed=0, shape="sqrt_edge", gp_blend=("position",)).fit(rf.transform(d, tr), Y[tr]).calibrate()
    Xte, Yte = rf.transform(d, te), Y[te]
    rows = []
    for i in (10, 50, 90):
        samp = m.sample_calibrated(Xte[i:i + 1], 200, 0)[0]
        lo, hi = np.percentile(samp, [5, 95], axis=0)
        pt = m.predict_point(Xte[i:i + 1])[0]
        rows += [{"shot": f"run 13, shot {int(d['shot'][te][i])}", "E_MeV": r(e, 1), "measured": r(y, 4), "predicted": r(p, 4),
                  "lo": r(a, 4), "hi": r(b, 4)} for e, y, p, a, b in zip(Eg, Yte[i], pt, lo, hi) if e <= 1300]
    save("demo_beam_model", rows, "Gemini LWFA data (Streeter et al. 2023, CC BY 4.0); adopted beam model trained on runs 05-12")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    b = np.load(ROOT / "models" / "example_beams.npz")
    E, B = b["E_MeV"], b["dWdE_J_per_GeV"].astype(float)
    if "--beam" in sys.argv:
        beam_examples()
    else:
        s, sur = surrogate_parts(E, B)
        toy_inference(s, sur, B)
        real_shot(s, sur)
