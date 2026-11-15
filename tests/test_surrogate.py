"""Surrogate inputs and the MLP ensemble, on toy data (no scan files needed)."""

import numpy as np

from mlsfqed.surrogate.data import INPUTS, assemble_inputs
from mlsfqed.surrogate.models import MLPEnsemble


def collision(**kw):
    c = {"a0": 10.0, "waist_um": 2.5, "fwhm_fs": 45.0, "r_offset_um": 0.0, "phi_offset": 0.0, "dz_um": 0.0,
         "radius_um": 0.7, "length_um": 5.0, "div_mrad": 2.0}
    c.update(kw)
    return c


BEAM = {"beam_Q_pC": 50.0, "beam_W_J": 0.03, "beam_mean_MeV": 600.0, "beam_mean_window_MeV": 610.0,
        **{f"beam_pc{k}": 0.0 for k in range(1, 7)}}


def test_derived_features():
    zR = np.pi * 2.5 ** 2 / 0.8
    X = assemble_inputs(collision(), BEAM)
    assert list(X.columns) == INPUTS
    assert np.isclose(X.a0_eff_est[0], 10.0)                     # head-on at focus: the full a0
    X = assemble_inputs(collision(dz_um=2 * zR), BEAM)
    assert np.isclose(X.a0_eff_est[0], 10.0 / np.sqrt(2))        # one Rayleigh range of laser travel
    X = assemble_inputs(collision(r_offset_um=[0.0, 1.0, 3.0]), BEAM)
    assert np.all(np.diff(X.a0_eff_est) < 0)                     # transverse offset weakens the field
    gamma = 600.0 / 0.51099895
    assert np.isclose(assemble_inputs(collision(), BEAM).chi_est[0], 2 * gamma * 10.0 * 1.5498 / 0.51099895e6)


def test_mlp_ensemble_learns_and_reports_spread():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 3))
    Y = np.column_stack([np.sin(2 * X[:, 0]) + X[:, 1], X[:, 2] ** 2])
    m = MLPEnsemble(n_members=3, hidden=(32, 32), seed=0).fit(X[:500], Y[:500])
    mu, sd = m.predict(X[500:])
    assert mu.shape == sd.shape == (100, 2)
    assert np.all(sd >= 0)
    r2 = 1 - ((mu - Y[500:]) ** 2).mean(0) / Y[500:].var(0)
    assert np.all(r2 > 0.9)
