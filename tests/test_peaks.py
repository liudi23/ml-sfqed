import numpy as np
from scipy.integrate import quad

from mlsfqed.beam_model.peaks import _edge_profile, fit_peaks, sqrt_edge


def test_edge_profile_matches_integral():
    for z in (-50.0, -12.5, -11.5, -3.0, -0.765, 0.0, 1.0, 3.0):
        c = max(-z, 0.0)
        f = lambda t, z=z: t ** -0.5 * np.exp(-0.5 * (z + t) ** 2)
        num = (quad(f, 0, c, limit=500)[0] if c else 0.0) + quad(f, c, np.inf, limit=500)[0]
        assert abs(num / np.sqrt(np.pi) - float(_edge_profile(z))) < 1e-5 * max(1.0, num)


def test_sqrt_edge_fit_recovers_parameters():
    E = np.linspace(305, 1400, 400)
    s = sqrt_edge(E, 2.0, 900.0, 15.0) + 0.3
    P = fit_peaks(s[None, :], E, shape="sqrt_edge")[0]
    assert abs(P[1] - 900.0) < 2 and abs(P[2] - 15.0) < 2 and P[3] > 0.999
