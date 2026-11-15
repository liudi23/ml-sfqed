"""The public surrogate interface, on the shipped models (skipped if they are absent)."""

import warnings

import numpy as np
import pytest

from mlsfqed.surrogate.api import FILES, SEARCH, Surrogate

HAVE = any((d / FILES["stochastic"]).exists() for d in SEARCH)
pytestmark = pytest.mark.skipif(not HAVE, reason="trained surrogate not available")


def beams():
    root = SEARCH[0]
    b = np.load(root / "example_beams.npz")
    return b["E_MeV"], b["dWdE_J_per_GeV"]


def test_predict_shapes_and_physics():
    s = Surrogate.load("stochastic")
    _, B = beams()
    df = s.predict(B[0], a0=[6.0, 12.0, 18.0])
    assert len(df) == 3
    assert np.all(np.diff(df.loss_window_MeV) > 0)                         # stronger field, more loss
    assert np.all(df.loss_window_MeV_lo < df.loss_window_MeV) and np.all(df.loss_window_MeV < df.loss_window_MeV_hi)
    assert np.allclose(df.out_mean_window_MeV, df.beam_mean_window_MeV - df.loss_window_MeV)
    assert len(s.predict(B, a0=10.0)) == len(B)                             # many beams, one collision


def test_classical_loses_more_than_stochastic():
    _, B = beams()
    q = Surrogate.load("stochastic_4096").predict(B[0], a0=15.0).loss_window_MeV[0]
    c = Surrogate.load("classical").predict(B[0], a0=15.0).loss_window_MeV[0]
    assert c > q


def test_out_of_range_warns():
    _, B = beams()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Surrogate.load("stochastic").predict(B[0], a0=30.0)
    assert any("outside the training range" in str(x.message) for x in w)
