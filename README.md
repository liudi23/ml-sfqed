# ml-sfqed

Uncertainty-aware ML pipeline from real Gemini laser–plasma data to a fast surrogate for strong-field QED Monte Carlo.

```
[Stage 1] Gemini LWFA shots  -> ML beam model (diagnostics -> e- spectrum, with uncertainty)
[Stage 2] Ptarmigan MC        -> gamma spectrum, positron yield over (beam, laser) parameters
[Stage 3] Fast surrogate      -> observables with uncertainty, target >=1000x faster than MC
```

## Status
- **Stage 1 (done, first version):** beam model with an explicit spectral peak + per-bin calibrated beam sampler. On the authors' split: spectral MSE 0.048 vs 0.057 published; beam-energy error 12.4% (a tie); 90% spectral band covers 92% of bins. Held-out last run: 14% beam-energy error; band covers 90% of on-beam bins.
- **Stage 2 prep (done):** Ptarmigan v1.6.1 built and validated against Landau–Lifshitz, quantum RR and weak-field Compton theory (`results/ptarmigan_validation/`).
- **Stage 3:** not started.

## Data
Real shots from the CLF Gemini laser (2021 radiation-reaction campaign): M. J. V. Streeter et al., *Laser wakefield accelerator modelling with variational neural networks*, High Power Laser Sci. Eng. 11, e9 (2023), doi:10.1017/hpl.2022.47. Dataset and code: Zenodo, doi:10.5281/zenodo.7510352 (CC BY 4.0).

Simulations use Ptarmigan (T. G. Blackburn et al., Phys. Plasmas 30, 093903 (2023)); build it with `scripts/install_ptarmigan.sh`.

## Setup
```bash
uv venv && uv pip install -e ".[dev]"
pre-commit install
pytest
```

## Layout
- `src/mlsfqed/` – package (data, beam_model, simulation, surrogate, evaluation)
- `scripts/` – run_stage1.py, generate_mc.py, train_surrogate.py
- `configs/` – YAML configs
- `data/raw/` – raw data (gitignored): put `train_data_20220609.pkl` and `test_data_20220609.pkl` from the Gemini dataset here
- `results/` – metrics, figures and validation runs (sampled-beam `.npz` files are gitignored; regenerate with `scripts/validate_sampler.py`)
- `notebooks/` – EDA only
