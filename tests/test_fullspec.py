"""Transfer kernel and full-spectrum likelihood basics (needs torch; skipped without it)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlsfqed.inference.fullspec import EDGES, rebin
from mlsfqed.surrogate.kernel import TransferKernel

E0 = np.linspace(305, 1400, 200)
EDGES10 = np.arange(0, 1500.1, 10)
THETA = np.array([[12.0, 2.5, 45.0, 0.0, 0.0, 0.0, 0.7, 5.0, 2.0]])


def test_kernel_conserves_electrons_and_is_linear():
    k = TransferKernel(E0, EDGES10, seed=0)
    K = k.kernel(THETA).detach().numpy()[0]                     # (incoming points, output bins)
    assert np.allclose(K.sum(1), 1.0, atol=1e-3)                 # every electron lands in some bin
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    pa, pb, pab = (k.predict(THETA, x[None, :])[0] for x in (a, b, a + b))
    wa = a.sum() / (a.sum() + b.sum())
    assert np.allclose(pab, wa * pa + (1 - wa) * pb, atol=1e-6)  # outgoing spectrum linear in the incoming one


def test_rebin_conserves_area():
    E = np.linspace(300, 1100, 4001)
    y = np.exp(-0.5 * ((E - 600) / 40) ** 2)
    inside = (E >= EDGES[0]) & (E <= EDGES[-1])
    assert abs(rebin(E, y).sum() - np.trapezoid(y[inside], E[inside])) < 1e-3 * np.trapezoid(y, E)
