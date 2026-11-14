# Roadmap

Pipeline: real Gemini laser-plasma data → calibrated electron-beam model → Ptarmigan strong-field QED Monte Carlo → fast surrogate → surrogate-accelerated Bayesian inference of collision parameters.

Status key: ✅ done · 🔄 in progress · ⬜ to do.

## Stage 1 – Beam model from real data ✅ (first version)
- 645 reference shots (no collision) from the 2021 Gemini radiation-reaction campaign (Streeter et al. 2023, CC BY 4.0). Beam energy drifts from 0.90 J to 0.52 J across the day.
- Model: dW/dE = beam energy × (continuum + explicit Gaussian peak); bagged LightGBM ensembles; out-of-bag residuals drawn as paired sets; per-bin conformal calibration.
- Evaluation on two splits: the authors' random split (for comparison with the paper) and a time split (train runs 05–12, test run13) for the headline numbers.

| | Spectral MSE | Beam-energy error | 90% band coverage |
|---|---|---|---|
| Streeter et al. (authors' split) | 0.057 | 12.4% | – |
| This work, authors' split | 0.048 | 12.4% | 92% of bins |
| This work, time split | 0.032 | 14% | 84% of bins, 90% on beam |

- Findings: raw ensemble spreads are overconfident (44–68% coverage of a nominal 90% interval) until calibrated; history and physics-based features add nothing because the remaining error is independent shot-to-shot noise; optional online adaptation during a run brings run13 coverage to 89%.
- Scripts: `run_stage1.py`, `validate_sampler.py`, `compare_shape_models.py`, `feature_study.py`, `online_adaptation.py`.

## Stage 2 – Monte Carlo with Ptarmigan 🔄
- ✅ Ptarmigan v1.6.1 build script and validation (`scripts/install_ptarmigan.sh`, `results/ptarmigan_validation/`): classical radiation reaction vs Landau–Lifshitz to ≤0.01%; quantum vs classical and LMA vs LCFA consistent; weak-field Compton edge, a0² scaling and Klein–Nishina-corrected energy loss to ≤1%; Stage 1 sampled beams load through spectrum files.
- ✅ Scan design and runner (`scripts/stage2_scan.py`, `src/mlsfqed/simulation/`): 4,096 Sobol-sampled collisions over a pool of 1,935 beams (645 measured spectra + 2 calibrated Stage 1 draws each); a0 5–20, waist 2–3 µm, 35–55 fs, transverse offset 0–5 µm, longitudinal offset ±250 µm (denser near focus), beam radius, bunch length and divergence. Resumable, parallel, Parquet output with config hash and timing.
- **Rates: LMA.** LCFA matches LMA in energy loss (0.07% at a0 = 20) but overproduces photons 2.4–4.7× in weaker collisions and shifts the critical energy, so LMA is used; its tables cap a0 at 20.
- Observables: electron mean energy (all and ≥305 MeV) and P70 as defined by Los et al. (after 10 MeV smoothing), energy spread; photon yield, mean and total energy, critical energy, energy-weighted angular spread; binned electron and photon spectra. Monte Carlo noise at 5,000 electrons: mean energy ±1–1.5 MeV, P70 ±4%, photon yield ±1%. Energy balance (electron loss vs photon energy) holds to ±2 MeV.
- ⬜ Full scan (~7 CPU-hours), then a beam-chirp sensitivity check.

## Stage 3 – Fast surrogate and inference ⬜
- Surrogates for the observables (tree baseline; MLP / small neural operator for spectra; deep ensembles for uncertainty), target ≥1000× faster than Ptarmigan.
- Demos: propagate real shot-to-shot beam uncertainty to observable distributions; run MCMC over a0, offsets and bunch duration for a shot and compare wall time with the ~19,200 CPU-hours per inference reported by Los et al.

## Stage 4 – Write-up ⬜

## Relation to published work
- **Streeter et al., HPLSE 11, e9 (2023)** – variational neural-network model of the same LWFA; source of the data and the Stage 1 baseline.
- **Los et al., Nature Communications (arXiv:2407.12071) and HPLSE (2024, doi:10.1017/hpl.2024.83)** – the same campaign's radiation-reaction analysis: predicted beams from the Streeter et al. network, a custom LCFA Monte Carlo pre-computed into 4-D lookup tables, and Bayesian model comparison on the 10 brightest collision shots (~19,200 CPU-hours per inference; transverse offset fixed for cost).
- **What this project adds:** calibrated beam uncertainty with a time-split test and drift analysis (they use the uncalibrated ensemble spread); an independent, validated simulator (Ptarmigan, LMA rates); and a learned surrogate aimed at their stated bottleneck – cost, which limited them to 10 shots and forced the transverse offset out of the model.
