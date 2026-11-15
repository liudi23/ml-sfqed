"""Public interface to the trained surrogates: Ptarmigan's summary observables in milliseconds.

    from mlsfqed.surrogate.api import Surrogate
    s = Surrogate.load("stochastic")                  # or "continuous", "classical"
    df = s.predict(dWdE, a0=[8, 12, 16], dz_um=0.0)   # one row per collision, 90% intervals included

Inputs: collision parameters (defaults: the nominal Gemini 2021 set-up) and an incoming electron spectrum
dW/dE (any scale) on the surrogate's energy grid `s.E_MeV`, or on your own grid via `E_MeV=`.
Outputs, per collision: the 9 Ptarmigan observables (mean, 90% interval) plus the outgoing mean energy
above 305 MeV. Valid only inside the training ranges (`s.ranges`); outside them a warning is raised.
Fixed physics: 0.8 um laser, head-on collision, LMA rates, 5,000 macro-electrons, no pair creation.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mlsfqed.surrogate.data import COLLISION, assemble_inputs, describe_beams

ROOT = Path(__file__).resolve().parents[3]
SEARCH = (ROOT / "models", ROOT / "data" / "processed" / "models")
FILES = {"stochastic": "surrogate.pkl", "stochastic_4096": "surrogate_stochastic_4096.pkl",
         "continuous": "surrogate_continuous_4096.pkl", "classical": "surrogate_classical_4096.pkl"}
NOMINAL = {"a0": 12.0, "waist_um": 2.5, "fwhm_fs": 45.0, "r_offset_um": 0.0, "phi_offset": 0.0, "dz_um": 0.0,
           "radius_um": 0.7, "length_um": 5.0, "div_mrad": 2.0}
TRAINING_RANGE = {"a0": (5, 20), "waist_um": (2, 3), "fwhm_fs": (35, 55), "r_offset_um": (0, 5),
                  "dz_um": (-250, 250), "radius_um": (0.5, 2), "length_um": (1, 10), "div_mrad": (1, 4)}


class Surrogate:
    def __init__(self, model: dict):
        self.model = model
        self.E_MeV = np.asarray(model["pool_E_MeV"])
        self.targets = list(model["targets"])
        self.rr_model = model.get("rr_model", "stochastic")
        self.ranges = dict(TRAINING_RANGE)

    @classmethod
    def load(cls, rr_model: str = "stochastic", path: str | Path | None = None) -> Surrogate:
        """rr_model: "stochastic" (8,192 runs), "continuous", "classical" or "stochastic_4096" (the three
        trained on the same 4,096 collisions, for model comparison). path overrides the search."""
        if path is None:
            for d in SEARCH:
                if (d / FILES[rr_model]).exists():
                    path = d / FILES[rr_model]
                    break
            else:
                raise FileNotFoundError(f"{FILES[rr_model]} not found in {[str(d) for d in SEARCH]}")
        with open(path, "rb") as f, warnings.catch_warnings():
            # saved with scikit-learn 1.9; other versions load them fine but warn once per estimator
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
            return cls(pickle.load(f))

    def describe(self, dWdE, E_MeV=None) -> pd.DataFrame:
        """Beam descriptors (charge, energy, mean energies, 6 shape coefficients) of one or more spectra."""
        dWdE = np.atleast_2d(np.asarray(dWdE, float))
        if E_MeV is not None:
            dWdE = np.array([np.interp(self.E_MeV, E_MeV, y, left=0.0, right=0.0) for y in dWdE])
        return describe_beams(dWdE, self.E_MeV, self.model["pool_pca"], self.model["pool_Q_median_pC"])

    def _check(self, c: dict):
        for k, (lo, hi) in self.ranges.items():
            v = np.asarray(c[k], float)
            if np.any(v < lo - 1e-9) or np.any(v > hi + 1e-9):
                warnings.warn(f"{k} outside the training range [{lo}, {hi}]: predictions are extrapolations",
                              stacklevel=3)

    def predict(self, dWdE, E_MeV=None, level: float = 0.9, **collision) -> pd.DataFrame:
        """dWdE: (n_bins,) or (n_beams, n_bins). collision: any of the 9 parameters, scalars or arrays.
        Beams and parameter arrays broadcast against each other (one beam for many collisions, or many
        beams at one collision, or equal lengths). level: only 0.9 is calibrated; others scale it as a
        Gaussian would."""
        unknown = set(collision) - set(COLLISION)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}; expected {COLLISION}")
        c = {k: np.atleast_1d(np.asarray(collision.get(k, NOMINAL[k]), float)) for k in COLLISION}
        b = self.describe(dWdE, E_MeV)
        n = max([len(b)] + [len(v) for v in c.values()])
        if any(len(v) not in (1, n) for v in c.values()) or len(b) not in (1, n):
            raise ValueError("beams and parameter arrays must have length 1 or a common length")
        c = {k: np.broadcast_to(v, n) for k, v in c.items()}
        self._check(c)
        bb = {k: np.broadcast_to(b[k].values, n) for k in b.columns}
        X = assemble_inputs(c, bb)
        mu, sd = self.model["mlp"].predict(X)
        from scipy.stats import norm
        scale = norm.ppf(0.5 + level / 2) / 1.645
        out = pd.DataFrame({k: c[k] for k in COLLISION})
        out["a0_eff_est"] = X["a0_eff_est"].values
        for j, t in enumerate(self.targets):
            hw = scale * self.model["conformal_factor90"][t] * np.sqrt(sd[:, j] ** 2 + self.model["mc_noise"].get(t, 0.0) ** 2)
            out[t], out[f"{t}_lo"], out[f"{t}_hi"] = mu[:, j], mu[:, j] - hw, mu[:, j] + hw
        out["beam_mean_window_MeV"] = bb["beam_mean_window_MeV"]
        out["out_mean_window_MeV"] = out["beam_mean_window_MeV"] - out["loss_window_MeV"]
        return out
