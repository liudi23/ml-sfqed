# Roadmap

Pipeline: real Gemini laser-plasma data → calibrated electron-beam model → Ptarmigan strong-field QED Monte Carlo → fast surrogate → surrogate-accelerated Bayesian inference of collision parameters.

Status key: ✅ done · 🔄 in progress · ⬜ to do.

## Stage 1 – Beam model from real data ✅
- 645 reference shots (no collision) from the 2021 Gemini radiation-reaction campaign (Streeter et al. 2023, CC BY 4.0). Beam energy drifts from 0.90 J to 0.52 J across the day.
- Model: dW/dE = beam energy × (data-driven PCA continuum + physics-based square-root-edge peak); bagged LightGBM ensembles (peak position averaged with a Gaussian process); out-of-bag residuals drawn as paired sets; per-bin conformal calibration.
- Evaluation on two splits: the authors' random split (for comparison with the paper) and a time split (train runs 05–12, test run13) for the headline numbers.

| | Spectral MSE | Beam-energy error | 90% band coverage |
|---|---|---|---|
| Streeter et al. (authors' split) | 0.057 | 12.4% | – |
| This work, authors' split | 0.045 | 12.5% | 91% of bins |
| This work, time split | 0.030 | 14.7% | 85% of bins, 89% on peak |

- Findings: raw ensemble spreads are overconfident (44–68% coverage of a nominal 90% interval) until calibrated; history and physics-based features add nothing because the remaining error is independent shot-to-shot noise; optional online adaptation during a run brings run13 coverage to 89%.
- Peak shape: physics-based square-root edge (phase-space-rotation pile-up below E_max), closed form; fits 4.4× better than a Gaussian with the same parameters; adopted (tie on run 11, −3 to −4% MSE on the test sets). See `docs/report.md`.
- Peak position: LightGBM ensemble averaged with a Gaussian process (`GPBlendEnsemble`), chosen on run 11; −3 to −4% spectral MSE. Plotted prediction = spectrum at the predicted peak (keeps ~97% of peak height). Online drift correction studied (`peak_drift_study.py`): −13% width error, −1% MSE, not adopted.
- Spectrum components (`spectrum_components_study.py`): a second, broad electron population (median 521 MeV, ~30% of the energy) is in essentially every shot; its position, width and charge are not predictable from the diagnostics (R² ≈ 0), so it stays in the continuum and accounts for much of the irreducible error.
- Scripts: `run_stage1.py`, `validate_sampler.py`, `compare_shape_models.py`, `feature_study.py`, `online_adaptation.py`, `peak_shape_study.py`, `peak_drift_study.py`, `spectrum_components_study.py`.

## Stage 2 – Monte Carlo with Ptarmigan ✅
- ✅ Ptarmigan v1.6.1 build script and validation (`scripts/install_ptarmigan.sh`, `results/ptarmigan_validation/`): classical radiation reaction vs Landau–Lifshitz to ≤0.01%; quantum vs classical and LMA vs LCFA consistent; weak-field Compton edge, a0² scaling and Klein–Nishina-corrected energy loss to ≤1%; Stage 1 sampled beams load through spectrum files.
- ✅ Scan design and runner (`scripts/stage2_scan.py`, `src/mlsfqed/simulation/`): 4,096 Sobol-sampled collisions over a pool of 1,935 beams (645 measured spectra + 2 calibrated Stage 1 draws each); a0 5–20, waist 2–3 µm, 35–55 fs, transverse offset 0–5 µm, longitudinal offset ±250 µm (denser near focus), beam radius, bunch length and divergence. Resumable, parallel, Parquet output with config hash and timing.
- **Rates: LMA.** LCFA matches LMA in energy loss (0.07% at a0 = 20) but overproduces photons 2.4–4.7× in weaker collisions and shifts the critical energy, so LMA is used; its tables cap a0 at 20.
- Observables: electron mean energy (all and ≥305 MeV) and P70 as defined by Los et al. (after 10 MeV smoothing), energy spread; photon yield, mean and total energy, critical energy, energy-weighted angular spread; binned electron and photon spectra. Monte Carlo noise at 5,000 electrons: mean energy ±1–1.5 MeV, P70 ±4%, photon yield ±1%. Energy balance (electron loss vs photon energy) holds to ±2 MeV.
- ✅ Full scan: 8,192 runs (4,096, then extended with the next 4,096 points of the same Sobol sequence after the learning curve showed photon observables were data-limited); 0 failures; ~45 min per 4,096 on a 12-core laptop.
- ✅ Beam-chirp sensitivity check (`scripts/chirp_check.py`): chirp ±0.2/±0.5 shifts energy loss by ≤0.4 MeV and P70 by ≤0.03, within Monte Carlo noise.
- ✅ Radiation-reaction model scans: the first 4,096 collisions repeated with classical (Landau–Lifshitz) and continuous (Gaunt-factor corrected) radiation reaction, same seeds.

## Stage 3 – Fast surrogate and inference ✅ (first version)
- ✅ Surrogate (`scripts/stage3_surrogate.py`, `results/stage3/`): deep MLP ensemble on the 8,192-run scan, LightGBM baseline. On unseen beams R² 0.98–0.995 for energy loss, photon yield and energy, critical energy and angles (P70 0.79); errors 1.4–2.5× the Monte Carlo noise for energy and angle observables. Split-conformal calibration: 88–92% coverage of 90% intervals. Extrapolation to a0 > 16.5: R² 0.97–0.99 (MLP; trees 0.91–0.95). ~18,000× faster than Ptarmigan per call, ~4×10⁶× batched.
- ✅ Inference on the 10 public collision shots of Los et al. (`scripts/infer_los2024.py`, `results/inference/`): emcee over a0, longitudinal and transverse offset and bunch length; likelihood on outgoing mean energy and P70, marginalised over 50 predicted incoming beams. Recovery test on 48 unseen simulated shots: 90% coverage 0.90 (a0) / 0.83 (effective a0). 42 CPU-s per inference vs ~19,200 CPU-hours (simpler likelihood than theirs).
- **Finding:** the published incoming-beam uncertainty is ~3× too narrow (P70 0.11 vs 0.26; mean energy 6.4 vs 18.1 MeV realistic error on a held-out run, adopted Stage 1 model); calibrating it widens the effective-a0 intervals by ~46% (26% with the transverse offset free). With two summary observables only the effective field strength is constrained; offsets and bunch length are prior-dominated.
- ✅ Effective a0 vs Los et al. (`scripts/a0_consistency_check.py`): their ⟨ã0⟩ is also an effective (experienced) a0, with the transverse offset fixed at zero. Direct Ptarmigan gives 7.2, 8.2, 8.1 on the three shots with 34–43 MeV loss (theirs 6.7, 9.4, 7.2); on the five low-loss shots the loss is within 1–1.5× the 18 MeV incoming-beam error, so electron data alone cannot pin a0, and shot 0 (77 MeV) needs ≈12.5 vs their 7.8. Los et al. also fit the gamma signal; using it here needs their spectrometer response modelled first. (Shot n in their paper = our shot n − 1.)
- ✅ Uncertainty propagation (`scripts/propagate_uncertainty.py`): 11,400 surrogate evaluations in 0.08 s vs ~114 CPU-hours.
- ✅ Model comparison (`scripts/compare_models.py`, `scripts/diagnose_p70.py`): energy loss alone cannot separate the models; spectral broadening (P70) favours stochastic quantum radiation reaction on all 10 shots (summed log Bayes factor +20 over classical with the measured beam error, +34 with the published spread, +9 with the transverse offset also free). Diagnostics rule out definition artefacts, out-of-distribution beams and surrogate error.
- ⬜ Next: full-spectrum likelihood with an explicit incoming-shape error model; add a photon observable once the gamma spectrometer response is modelled

## Stage 4 – Write-up ✅
- [docs/report.md](report.md), figures in `docs/figures/` (`scripts/make_figures.py`, animation from `scripts/a0_sweep.py`).

## Relation to published work
- **Streeter et al., HPLSE 11, e9 (2023)** – variational neural-network model of the same LWFA; source of the data and the Stage 1 baseline.
- **Los et al., Nature Communications (arXiv:2407.12071) and HPLSE (2024, doi:10.1017/hpl.2024.83)** – the same campaign's radiation-reaction analysis: predicted beams from the Streeter et al. network, a custom LCFA Monte Carlo pre-computed into 4-D lookup tables, and Bayesian model comparison on the 10 brightest collision shots (~19,200 CPU-hours per inference; transverse offset fixed for cost).
- **What this project adds:** calibrated beam uncertainty with a time-split test and drift analysis (they use the uncalibrated ensemble spread, ~2.4–2.8× too narrow); a physics-based beam model (square-root-edge peak from phase-space rotation; MSE 0.045 vs their 0.057); an independent, validated simulator (Ptarmigan, LMA rates); a learned surrogate aimed at their stated bottleneck – cost, which limited them to 10 shots and forced the transverse offset out of the model; and a model comparison that separates stochastic from continuous radiation reaction through spectral broadening, with its sensitivity to the beam uncertainty quantified.
