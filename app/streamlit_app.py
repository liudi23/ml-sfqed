"""ML-SFQED interactive demo (Streamlit).

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Everything on the first four tabs is computed live by the trained surrogates shipped in models/; the last
tab shows results saved by the pipeline's scripts (results/ and docs/figures/).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlsfqed.inference.posterior import ShotPosterior, run_mcmc
from mlsfqed.simulation.ptarmigan import p70
from mlsfqed.surrogate.api import NOMINAL, TRAINING_RANGE, Surrogate

logging.getLogger("emcee").setLevel(logging.ERROR)
st.set_page_config(page_title="ML-SFQED demo", page_icon="⚛️", layout="wide")

MODELS = ("stochastic", "continuous", "classical")
NAME = {"stochastic": "Stochastic (quantum)", "continuous": "Continuous (quantum)", "classical": "Classical (Landau–Lifshitz)"}
COL = {"stochastic": "#2b6cb0", "continuous": "#8a5cc2", "classical": "#d9822b"}
OBS = {  # surrogate output: (label, unit, format)
    "loss_window_MeV": ("Energy loss ≥305 MeV", "MeV", "{:.1f}"),
    "out_mean_window_MeV": ("Outgoing mean energy ≥305 MeV", "MeV", "{:.1f}"),
    "e_P70": ("P70 (peak sharpness)", "1/GeV", "{:.2f}"),
    "e_std_MeV": ("Energy spread", "MeV", "{:.1f}"),
    "g_crit_MeV": ("Photon critical energy", "MeV", "{:.1f}"),
    "log_g_number_per_e": ("log₁₀ photons per electron", "", "{:.3f}"),
    "g_theta_x_rms_mrad": ("Photon angular spread, polarisation plane", "mrad", "{:.2f}"),
}


@st.cache_resource
def surrogates():
    """The three surrogates trained on the same 4,096 collisions (like-for-like comparison)."""
    return {m: Surrogate.load("stochastic_4096" if m == "stochastic" else m) for m in MODELS}


@st.cache_data
def example_beams():
    b = np.load(ROOT / "models" / "example_beams.npz")
    return b["E_MeV"], b["dWdE_J_per_GeV"].astype(float)


def beam_summary(E, y):
    n, w = y / E, E >= 305
    return np.sum(n[w] * E[w]) / np.sum(n[w]), p70(E[w], n[w], 0)


def layout(fig, title, xlab, ylab, h=380):
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, height=h, margin={"l": 10, "r": 10, "t": 50, "b": 10},
                      legend={"orientation": "h", "y": -0.2}, hovermode="x unified")
    return fig


sur = surrogates()
E, BEAMS = example_beams()

# ---------------------------------------------------------------- sidebar: inputs
st.sidebar.header("Incoming electron beam")
choice = st.sidebar.selectbox("Spectrum", [f"Gemini example {i + 1}" for i in range(len(BEAMS))] + ["Upload my own (CSV)"])
if choice.startswith("Upload"):
    up = st.sidebar.file_uploader("CSV with columns E_MeV, dWdE (any scale)", type="csv")
    if up is None:
        st.sidebar.info("Using example 1 until a file is uploaded.")
        beam = BEAMS[0]
    else:
        df_up = pd.read_csv(up)
        beam = np.interp(E, df_up.iloc[:, 0].values, df_up.iloc[:, 1].values, left=0.0, right=0.0)
else:
    beam = BEAMS[int(choice.split()[-1]) - 1]
bm, bp = beam_summary(E, beam)
st.sidebar.caption(f"Mean energy ≥305 MeV: **{bm:.0f} MeV** · P70: **{bp:.2f}**")

st.sidebar.header("Collision")
c = {}
c["a0"] = st.sidebar.slider("Laser strength a0", 5.0, 20.0, 12.0, 0.5)
c["dz_um"] = st.sidebar.slider("Offset from focus (µm)", -250.0, 250.0, 0.0, 5.0)
c["r_offset_um"] = st.sidebar.slider("Sideways offset (µm)", 0.0, 5.0, 0.0, 0.1)
with st.sidebar.expander("Laser and beam details"):
    c["waist_um"] = st.slider("Laser waist (µm)", *map(float, TRAINING_RANGE["waist_um"]), NOMINAL["waist_um"], 0.05)
    c["fwhm_fs"] = st.slider("Laser duration FWHM (fs)", *map(float, TRAINING_RANGE["fwhm_fs"]), NOMINAL["fwhm_fs"], 1.0)
    c["radius_um"] = st.slider("Beam radius (µm)", *map(float, TRAINING_RANGE["radius_um"]), NOMINAL["radius_um"], 0.05)
    c["length_um"] = st.slider("Bunch length (µm)", *map(float, TRAINING_RANGE["length_um"]), NOMINAL["length_um"], 0.5)
    c["div_mrad"] = st.slider("Divergence (mrad)", *map(float, TRAINING_RANGE["div_mrad"]), NOMINAL["div_mrad"], 0.1)
st.sidebar.caption("Ranges are the surrogate's training ranges. Fixed: 0.8 µm laser, head-on collision, LMA rates.")

# ---------------------------------------------------------------- header
st.title("ML-SFQED: radiation reaction in milliseconds")
st.markdown(
    "A neural surrogate of the **Ptarmigan** QED Monte Carlo, trained on 8,192 simulated laser–electron collisions, "
    "predicts what an electron beam looks like after colliding with an intense laser, for three models of "
    "radiation reaction. Set the beam and the collision on the left; everything below is computed live, with "
    "calibrated 90% intervals. [Code and write-up on GitHub](https://github.com/liudi23/ml-sfqed)."
)
tabs = st.tabs(["Predict", "Laser strength scan", "Timing & alignment", "Beam uncertainty", "Inference", "Results on 2021 data"])

# ---------------------------------------------------------------- 1 predict
with tabs[0]:
    t0 = time.perf_counter()
    pred = {m: sur[m].predict(beam, **c).iloc[0] for m in MODELS}
    dt = (time.perf_counter() - t0) * 1e3
    cols = st.columns(3)
    for col, m in zip(cols, MODELS):
        r = pred[m]
        col.metric(NAME[m], f"{r.loss_window_MeV:.1f} MeV lost")
        col.caption(f"P70 {r.e_P70:.2f} after the collision (incoming {bp:.2f}): "
                    + ("broader" if r.e_P70 < bp else "sharper") + " peak")
    rows = []
    for k, (lab, unit, f) in OBS.items():
        row = {"Output": lab + (f" ({unit})" if unit else "")}
        for m in MODELS:
            r = pred[m]
            if k == "out_mean_window_MeV":
                row[NAME[m]] = f.format(r[k])
            else:
                row[NAME[m]] = f"{f.format(r[k])}  [{f.format(r[k + '_lo'])} – {f.format(r[k + '_hi'])}]"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Three surrogate predictions in {dt:.0f} ms (Ptarmigan: ~8 CPU-seconds each). Brackets: calibrated 90% interval. "
               f"Effective a0 at the beam centre: {pred['stochastic'].a0_eff_est:.1f}.")
    fig = go.Figure(go.Scatter(x=E, y=beam, line={"color": "black"}))
    st.plotly_chart(layout(fig, "Incoming spectrum", "Electron energy (MeV)", "dW/dE (J/GeV)", 260), width="stretch")

# ---------------------------------------------------------------- 2 a0 scan
with tabs[1]:
    a0 = np.linspace(5, 20, 61)
    scan = {m: sur[m].predict(beam, a0=a0, **{k: v for k, v in c.items() if k != "a0"}) for m in MODELS}
    left, right = st.columns(2)
    for col, (k, lab) in zip((left, right), (("loss_window_MeV", "Energy loss ≥305 MeV (MeV)"), ("e_P70", "P70 after the collision (1/GeV)"))):
        fig = go.Figure()
        for m in MODELS:
            d = scan[m]
            fig.add_trace(go.Scatter(x=np.r_[a0, a0[::-1]], y=np.r_[d[k + "_hi"], d[k + "_lo"][::-1]], fill="toself", line={"width": 0},
                                     fillcolor=COL[m], opacity=0.15, hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=a0, y=d[k], name=NAME[m], line={"color": COL[m], "width": 2.5}))
        if k == "e_P70":
            fig.add_hline(y=bp, line_dash="dash", annotation_text="incoming beam", line_color="grey")
        fig.add_vline(x=c["a0"], line_color="grey", opacity=0.5)
        col.plotly_chart(layout(fig, lab, "a0", ""), width="stretch")
    st.markdown("Classical radiation reaction removes the most energy and **sharpens** the peak (P70 rises). Stochastic "
                "quantum emission gives each electron a random loss, so the peak stays **broader**. On the 2021 data this "
                "difference in P70, not the energy loss, is what favours the stochastic model.")

# ---------------------------------------------------------------- 3 geometry
with tabs[2]:
    mdl = st.radio("Model", MODELS, format_func=lambda m: NAME[m], horizontal=True, key="geo_model")
    dz, rr = np.linspace(-250, 250, 81), np.linspace(0, 5, 41)
    DZ, R = np.meshgrid(dz, rr)
    other = {k: v for k, v in c.items() if k not in ("dz_um", "r_offset_um")}
    g = sur[mdl].predict(beam, dz_um=DZ.ravel(), r_offset_um=R.ravel(), **other)
    fig = go.Figure(go.Heatmap(x=dz, y=rr, z=g.loss_window_MeV.values.reshape(DZ.shape), colorscale="Blues",
                               colorbar={"title": "MeV"}, hovertemplate="dz %{x:.0f} µm<br>r %{y:.2f} µm<br>loss %{z:.1f} MeV<extra></extra>"))
    fig.add_trace(go.Scatter(x=[c["dz_um"]], y=[c["r_offset_um"]], mode="markers", marker={"color": "red", "size": 10, "symbol": "x"}, name="your setting"))
    st.plotly_chart(layout(fig, f"Energy loss at a0 = {c['a0']:.1f} ({len(g):,} collisions predicted)", "Offset from focus (µm)", "Sideways offset (µm)", 460),
                    width="stretch")
    st.caption("The field the electrons experience falls quickly with either offset, so energy loss alone pins down only an "
               "effective a0 (a strong laser slightly off the beam looks like a weaker one on it).")

# ---------------------------------------------------------------- 4 beam uncertainty
with tabs[3]:
    st.markdown("On a real collision shot the incoming spectrum is not measured, only predicted from diagnostics, with an "
                "error of about 18 MeV in mean energy. Here the six Gemini examples stand in for that uncertainty: the "
                "same collision, six different beams.")
    rows = []
    for i, y in enumerate(BEAMS):
        mi, pi = beam_summary(E, y)
        for m in MODELS:
            r = sur[m].predict(y, **c).iloc[0]
            rows.append({"beam": f"Example {i + 1}", "model": m, "incoming mean": mi, "outgoing mean": r.out_mean_window_MeV,
                         "P70": r.e_P70, "lo": r.e_P70_lo, "hi": r.e_P70_hi})
    df = pd.DataFrame(rows)
    left, right = st.columns(2)
    fig = go.Figure()
    for m in MODELS:
        d = df[df.model == m]
        fig.add_trace(go.Scatter(x=d["incoming mean"], y=d["outgoing mean"], mode="markers", name=NAME[m], marker={"color": COL[m], "size": 10}))
    left.plotly_chart(layout(fig, "Outgoing vs incoming mean energy", "Incoming (MeV)", "Outgoing (MeV)"), width="stretch")
    fig = go.Figure()
    for m in MODELS:
        d = df[df.model == m]
        fig.add_trace(go.Scatter(x=d["P70"], y=d["beam"], mode="markers", name=NAME[m], marker={"color": COL[m], "size": 10},
                                 error_x={"type": "data", "symmetric": False, "array": d["hi"] - d["P70"], "arrayminus": d["P70"] - d["lo"]}))
    right.plotly_chart(layout(fig, "P70 after the collision (90% interval)", "P70 (1/GeV)", ""), width="stretch")

# ---------------------------------------------------------------- 5 inference
with tabs[4]:
    st.markdown("Make a **toy shot** with known answers: the chosen model's prediction for the true values below, plus "
                "measurement noise. The inference sees only the outgoing **mean energy** and **P70** and the incoming "
                "beam (the spectrum chosen on the left), samples a0, the offset from focus and the bunch length with "
                "`emcee`, and compares the three models by their evidence.")
    a, b, d_ = st.columns(3)
    true_model = a.selectbox("True model", MODELS, format_func=lambda m: NAME[m])
    t_a0 = b.slider("True a0", 5.0, 20.0, 14.0, 0.5, key="t_a0")
    t_dz = d_.slider("True offset from focus (µm)", -150.0, 150.0, 40.0, 5.0, key="t_dz")
    noise = st.checkbox("Add measurement noise (10 MeV on mean energy, 5% on P70)", value=False)
    if st.button("Run inference (~15 s)", type="primary"):
        t = sur[true_model].predict(beam, a0=t_a0, dz_um=t_dz, length_um=5.0).iloc[0]
        rng = np.random.default_rng(0)
        om, op = t.out_mean_window_MeV, t.e_P70
        if noise:
            om, op = om + rng.normal(0, 10.0), op * (1 + rng.normal(0, 0.05))
        with st.spinner("Sampling the posterior and computing evidences…"):
            post = ShotPosterior(sur["stochastic"].model, beam[None, :], om, op, meas_sd_mean=10.0, dz_prior="focus")
            chain, diag = run_mcmc(post, nwalkers=32, nsteps=6000, burn=2000)
            a_eff = post.a0_eff(chain)
            ev = {}
            for m in MODELS:
                p = ShotPosterior(sur[m].model, beam[None, :], om, op, meas_sd_mean=10.0, dz_prior="focus")
                ev[m] = {"both": p.log_evidence(n=40000, seed=1, use=(0, 1))[0], "mean": p.log_evidence(n=40000, seed=1, use=(0,))[0]}
        st.session_state["inf"] = {"chain": chain, "a_eff": a_eff, "ev": ev, "truth": (t_a0, t.a0_eff_est), "obs": (om, op), "model": true_model}
    if "inf" in st.session_state:
        s = st.session_state["inf"]
        st.caption(f"'Measured': mean energy {s['obs'][0]:.1f} MeV, P70 {s['obs'][1]:.2f} (generated by the {NAME[s['model']].lower()} model). "
                   "Posterior computed with the stochastic surrogate.")
        left, right = st.columns(2)
        fig = go.Figure()
        for x, lab, tr, colr in ((s["chain"][:, 0], "Peak a0", s["truth"][0], "#9aa0a6"), (s["a_eff"], "Effective a0", s["truth"][1], COL["stochastic"])):
            lo, med, hi = np.percentile(x, [5, 50, 95])
            fig.add_trace(go.Histogram(x=x, nbinsx=60, name=f"{lab}: {med:.1f} [{lo:.1f}–{hi:.1f}], true {tr:.1f}", marker_color=colr, opacity=0.6, histnorm="probability"))
            fig.add_vline(x=tr, line_color=colr, line_width=3)
        fig.update_layout(barmode="overlay")
        left.plotly_chart(layout(fig, "Posterior (bars) and truth (lines), median and 90%", "a0", "share of samples"), width="stretch")
        ev = s["ev"]
        fig = go.Figure()
        for m in ("stochastic", "continuous"):
            fig.add_trace(go.Bar(name=f"{NAME[m]} vs classical", x=["Mean energy + P70", "Mean energy only"],
                                 y=[ev[m]["both"] - ev["classical"]["both"], ev[m]["mean"] - ev["classical"]["mean"]], marker_color=COL[m]))
        fig.add_hline(y=0, line_color="grey")
        right.plotly_chart(layout(fig, "Log Bayes factor vs classical (positive favours the model)", "", ""), width="stretch")
        st.caption("Only the effective a0 is recovered; the peak a0 stays near its prior. With the mean energy alone the models "
                   "cannot be told apart, because a different a0 compensates; P70 separates them.")

# ---------------------------------------------------------------- 6 saved results
with tabs[5]:
    st.markdown("Results of the full pipeline on the 10 public collision shots of the 2021 Gemini experiment (Los et al. 2024), "
                "saved by the repository's scripts. See the [report](https://github.com/liudi23/ml-sfqed/blob/main/docs/report.md) for details.")
    f = ROOT / "results" / "demo" / "demo_real_shots.json"
    if f.exists():
        d = pd.DataFrame(json.loads(f.read_text())["rows"])
        fig = go.Figure()
        for m in ("stochastic", "continuous"):
            fig.add_trace(go.Bar(name=f"{NAME[m]} vs classical", x=[f"Shot {k}" for k in d["shot"]], y=d[f"bf_{m}"], marker_color=COL[m]))
        st.plotly_chart(layout(fig, f"Per-shot log Bayes factor vs classical (sum: stochastic {d.bf_stochastic.sum():+.1f}, "
                                    f"continuous {d.bf_continuous.sum():+.1f})", "", ""), width="stretch")
    for png, cap in (("pipeline_checks.png", "Every stage checked against something it did not see."),
                     ("p70_shots.png", "Each model fitted to the mean energy alone, then asked to predict P70."),
                     ("bayes_factors.png", "Summed log Bayes factors under four uncertainty set-ups."),
                     ("a0_eff.png", "Effective a0 per shot with published vs measured beam uncertainty.")):
        p = ROOT / "docs" / "figures" / png
        if p.exists():
            st.image(str(p), caption=cap, width="stretch")
