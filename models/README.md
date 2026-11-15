# Trained surrogates and example beams

| File | Radiation-reaction model | Trained on |
| --- | --- | --- |
| `surrogate.pkl` | stochastic (quantum) | 8,192 Ptarmigan collisions (used for parameter inference) |
| `surrogate_stochastic_4096.pkl` | stochastic (quantum) | first 4,096 collisions |
| `surrogate_continuous_4096.pkl` | continuous (quantum, Gaunt-corrected) | the same 4,096 collisions |
| `surrogate_classical_4096.pkl` | classical (Landau–Lifshitz) | the same 4,096 collisions |
| `example_beams.npz` | – | six measured Gemini 2021 electron spectra (run 13), dW/dE in J/GeV on the surrogate grid |

Each `.pkl` is a dict: the MLP ensemble, its inputs and targets, split-conformal 90% factors, Monte Carlo
noise, and the beam-shape PCA. Load them with `mlsfqed.surrogate.api.Surrogate.load(...)`; they were
saved with scikit-learn 1.9 (other versions load them with a warning) and are produced by `scripts/stage3_train_final.py`.

The example spectra come from the Gemini LWFA data of Streeter et al. (2023),
[doi:10.5281/zenodo.7510352](https://doi.org/10.5281/zenodo.7510352), CC BY 4.0.
