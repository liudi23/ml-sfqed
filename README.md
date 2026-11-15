# ML-SFQED

**Calibrated machine learning and simulation-based inference for strong-field QED experiments**

[![CI](https://github.com/liudi23/ml-sfqed/actions/workflows/ci.yml/badge.svg)](https://github.com/liudi23/ml-sfqed/actions/workflows/ci.yml)

An end-to-end pipeline from real laser–plasma data to Bayesian model comparison: a calibrated ML model of the electron beam, a validated QED Monte Carlo, a neural surrogate, and surrogate-accelerated inference on public collision data from the 2021 Gemini radiation-reaction experiment.

<p align="center">
  <img src="docs/figures/rr_sweep.gif" width="720" alt="Animated outgoing electron spectra for classical, continuous and stochastic radiation reaction as the laser strength a0 rises">
</p>

## Key results

| | |
| --- | --- |
| **~1.6 million× faster inference** | A full Bayesian fit of one collision shot takes **42 CPU-seconds**, against ~19,200 CPU-hours with direct simulation (Los et al. 2024). |
| **Published beam uncertainty is ~3× too narrow** | The incoming-beam error measured on held-out shots (P70 0.29, mean energy 18.8 MeV) is ~3× the published per-shot spread (0.11, 6.4 MeV). Using it widens the field-strength estimates by about half and halves the Bayes factors. |
| **Spectral broadening favours stochastic quantum radiation reaction** | On all 10 shots the outgoing spectrum is broader than the incoming one. Only stochastic emission reproduces this; classical and continuous models miss on 8 of 10. Summed log Bayes factor over classical: **+18** with realistic beam uncertainty. |
| **Better beam prediction, with honest uncertainty** | Spectral MSE **0.045 vs 0.057** for the published network ensemble (22% lower), with a physics-based peak shape; calibrated 90% intervals cover 89–90% of held-out shots (raw ensembles: 44–68%). |
| **Accurate surrogate** | MLP ensemble reproduces Ptarmigan to R² 0.98 (energy loss) at ~2× its Monte Carlo noise; ~18,000× faster per call, ~4×10⁶× batched. |

Full write-up: **[docs/report.md](docs/report.md)**.

## Pipeline

<p align="center">
  <img src="docs/figures/pipeline.png" width="900" alt="Pipeline: Gemini data to Stage 1 beam model, beam pool, Stage 2 Ptarmigan scan, Stage 3 surrogate and Stage 4 inference on the Los et al. collision shots">
</p>

<sub>Diagram source: <code>docs/figures/pipeline.mmd</code> (render with <code>mmdc -i docs/figures/pipeline.mmd -o docs/figures/pipeline.png -s 3 -b white</code>).</sub>

Every stage's uncertainty is checked on data it never saw: conformal calibration and coverage tests (stages 1 and 3), validation against analytic limits (stage 2), and recovery tests on simulated shots with known answers plus posterior predictive checks (stage 4).

## Results

### Spectral broadening separates the radiation-reaction models

Each model is fitted to the outgoing mean energy alone, then asked to predict P70, a measure of how sharp the spectral peak is. Only stochastic emission predicts the observed broadening:

![P70 on the 10 collision shots](docs/figures/p70_shots.png)

A dedicated diagnostic (`scripts/diagnose_p70.py`) rules out a definition artefact, out-of-distribution beams and surrogate error, using direct Ptarmigan runs at the fitted parameters.

### The evidence depends on how honest the beam uncertainty is

![Summed log Bayes factors](docs/figures/bayes_factors.png)

The stochastic preference halves once the measured beam error replaces the published spread, and halves again when the transverse offset is also free, a parameter the original analysis had to fix for cost. Energy loss alone carries no model information, because a free a0 absorbs the difference.

### Realistic beam uncertainty widens every field-strength estimate

![Effective a0 per shot](docs/figures/a0_eff.png)

### Stage 1: beam prediction from diagnostics

| Split | Model | Spectral MSE | Beam-energy error | 90% coverage |
| --- | --- | --- | --- | --- |
| authors' random split | Streeter et al. network ensemble | 0.057 | 12.4% | – |
| authors' random split | **Square-root-edge peak model (this work)** | **0.045** | 12.5% | 0.89 |
| time split (held-out last run) | **Square-root-edge peak model (this work)** | **0.030** | 14.7% | 0.90 |

Each spectrum is modelled as a continuum plus a **physics-based peak**: near dephasing, electrons pile up below the maximum energy E_max as dN/dE ∝ (E_max − E)^(−1/2) (phase-space rotation), smeared by resolution and bunch spread. This shape fits the measured peaks 4.4× better than a Gaussian with the same three parameters, and its parameters are physical: E_max is the dephasing-limited maximum energy. E_max is predicted by a LightGBM ensemble averaged with a Gaussian process, whose errors differ (position error −8%). The thick line is the spectrum at the predicted edge (~97% of the measured peak height); the remaining misses are mostly peak position, which the diagnostics only partly determine.

![Held-out spectra](results/stage1/spectra_time.png)

### Stage 3: surrogate accuracy on unseen beams

| Observable | R² | Error / MC noise | 90% coverage, raw → calibrated |
| --- | --- | --- | --- |
| Mean energy loss (≥305 MeV) | 0.979 | 2.1 | 0.74 → 0.91 |
| Spread P70 | 0.79 | 2.0 | 0.79 → 0.92 |
| Photon critical energy | 0.983 | 1.8 | 0.86 → 0.90 |
| log₁₀ photons per electron | 0.994 | 6.5 | 0.94 → 0.90 |

P70's lower R² reflects its small spread relative to Monte Carlo noise (ceiling ~0.95), not a worse fit.

## Limitations and next steps

- Inference uses two summary observables, which constrain only the effective field strength. A **full-spectrum likelihood** that models the incoming spectrum's shape error is the next step, and the confirmation test for the stochastic preference.
- The inferred effective a0 sits below Los et al.'s on most shots (3–7 vs 7–13); not yet explained.
- Laser waist, duration, beam radius and divergence are fixed in inference.

## Reproduce

```bash
uv venv && uv pip install -e ".[dev]"
bash scripts/install_ptarmigan.sh                  # Ptarmigan v1.6.1 (Rust)
# Gemini data: put the two .pkl files from doi:10.5281/zenodo.7510352 in data/raw/
bash scripts/fetch_los2024.sh                      # Los et al. collision data (local only)

python scripts/run_stage1.py                       # Stage 1 beam model           -> results/stage1/
python scripts/validate_ptarmigan.py               # Stage 2 validation          -> results/ptarmigan_validation/
python scripts/stage2_scan.py design && python scripts/stage2_scan.py run && python scripts/stage2_scan.py collect
                                                   # ~45 min per 4,096 collisions on 12 cores; `extend` adds more
for m in classical continuous; do                  # same collisions with the other radiation-reaction models
  python scripts/stage2_scan.py --model $m run --limit 4096 && python scripts/stage2_scan.py --model $m collect
done
python scripts/stage3_surrogate.py && python scripts/stage3_learning_curve.py   # Stage 3 metrics -> results/stage3/
python scripts/stage3_train_final.py               # surrogate used for inference
for m in stochastic classical continuous; do python scripts/stage3_train_final.py --model $m --max-runs 4096; done
python scripts/beam_prediction_error.py            # measured incoming-beam error
python scripts/infer_los2024.py recovery --n 48 && python scripts/infer_los2024.py shots
python scripts/compare_models.py recovery --n 30 && python scripts/compare_models.py shots
python scripts/diagnose_p70.py                     # P70 diagnostics             -> results/diagnostics/p70/
python scripts/a0_sweep.py && python scripts/make_figures.py   # figures in docs/figures/
```

Runs are seeded and LightGBM is single-threaded, so results are deterministic.

## Repository layout

```
src/mlsfqed/
  data/          Gemini loaders, splits, features
  beam_model/    Stage 1 models, peak fits, calibration, online adaptation
  simulation/    Ptarmigan wrapper, observables, Sobol design
  surrogate/     scan -> features/targets, MLP and tree surrogates
  inference/     Los et al. data, likelihood, posterior, evidence
scripts/         one script per step (see Reproduce)
results/         metrics, figures and validation runs
docs/            report.md, roadmap.md, figures/
tests/           unit tests (pytest), run in CI with ruff
```

## Data and references

- **Gemini LWFA data:** M. J. V. Streeter et al., *Laser wakefield accelerator modelling with variational neural networks*, High Power Laser Sci. Eng. 11, e9 (2023), [doi:10.1017/hpl.2022.47](https://doi.org/10.1017/hpl.2022.47). Data: [doi:10.5281/zenodo.7510352](https://doi.org/10.5281/zenodo.7510352) (CC BY 4.0).
- **Collision shots and model evidences:** E. E. Los et al., *Observation of quantum effects on radiation reaction in strong fields*, Nature Communications, [doi:10.1038/s41467-025-67918-8](https://doi.org/10.1038/s41467-025-67918-8) ([arXiv:2407.12071](https://arxiv.org/abs/2407.12071)); E. E. Los et al., *A Bayesian framework to investigate radiation reaction in strong fields*, High Power Laser Sci. Eng. (2024), [doi:10.1017/hpl.2024.83](https://doi.org/10.1017/hpl.2024.83). Public data from [ELos385/RadiationReaction](https://github.com/ELos385/RadiationReaction), fetched locally and not redistributed here; figures show derived quantities.
- **Ptarmigan:** T. G. Blackburn et al., Phys. Plasmas 30, 093903 (2023).

## Author

Di Liu, formerly postdoctoral researcher in strong-field QED, University of Plymouth. Project plan and progress: [docs/roadmap.md](docs/roadmap.md).
