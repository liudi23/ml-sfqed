"""Observables shared by the simulation, the surrogate targets and the measured collision data."""

import numpy as np

from mlsfqed.beam_model.stage1 import beam_energy
from mlsfqed.inference.los2024 import spectrum_observables
from mlsfqed.simulation.ptarmigan import critical_energy, p70

E = np.arange(300.0, 1200.0, 1.0)  # MeV


def gauss(mu, sig):
    return np.exp(-0.5 * ((E - mu) / sig) ** 2)


def test_beam_energy_units():
    # dW/dE = 1 J/GeV over 0.9 GeV -> 0.9 J
    assert abs(beam_energy(np.ones_like(E), E) - (E[-1] - E[0]) / 1e3) < 1e-9


def test_mean_energy_of_symmetric_peak():
    obs = spectrum_observables(E, gauss(600.0, 20.0))
    assert abs(obs["mean_window_MeV"] - 600.0) < 0.1


def test_p70_is_scale_free_and_falls_with_width():
    narrow, broad = gauss(600.0, 15.0), gauss(600.0, 40.0)
    assert abs(p70(E, narrow, 0) - p70(E, 7.3 * narrow, 0)) < 1e-9     # independent of charge
    assert p70(E, narrow, 0) > 2 * p70(E, broad, 0)                       # sharper peak, larger P70
    assert p70(E, narrow, 10.0) < p70(E, narrow, 0)                      # smoothing lowers the maximum


def test_critical_energy_recovers_synchrotron_scale():
    Eg = np.linspace(0.5, 300.0, 3000)
    for Ec in (15.0, 40.0):
        assert abs(critical_energy(Eg, Eg ** (-2 / 3) * np.exp(-Eg / Ec)) - Ec) < 0.01 * Ec
