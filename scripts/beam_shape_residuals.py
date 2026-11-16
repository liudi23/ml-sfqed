"""Incoming-spectrum shape errors measured on held-out shots, for the full-spectrum likelihood.

  python scripts/beam_shape_residuals.py        # ~2 min

The adopted Stage 1 beam model is trained on runs 05-12 and predicts the 114 shots of run 13. For each shot the
residual is (measured - predicted) electron spectrum dN/dE on the beam-pool grid (>= 305 MeV), both normalised
to unit area. Adding a randomly drawn residual to a predicted incoming spectrum gives a sample with the
realistic shape error, correlations across energy included.
Writes results/inference/beam_shape_residuals.npz (E_MeV, residuals, measured, predicted).
"""

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser, to_electron_spectrum
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split


def main():
    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    tr, te = time_ordered_split(d)
    rf = RawFeaturiser().fit(d, tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PeakBeamModel(E, seed=0, shape="sqrt_edge", gp_blend=("position",)).fit(rf.transform(d, tr), Y[tr]).calibrate()
    S = m.sample_calibrated(rf.transform(d, te), 100, 0)              # (n, 100, bins) dW/dE
    pred, _ = to_electron_spectrum(np.median(S, 1), E)
    meas, _ = to_electron_spectrum(Y[te], E)
    norm = lambda a: a / np.trapezoid(a, E, axis=1)[:, None]
    pred, meas = norm(pred), norm(meas)
    out = ROOT / "results" / "inference" / "beam_shape_residuals.npz"
    np.savez_compressed(out, E_MeV=E, residuals=(meas - pred).astype(np.float32), measured=meas.astype(np.float32),
                        predicted=pred.astype(np.float32))
    rel = np.abs(meas - pred).sum(1) / np.abs(meas).sum(1)
    print(f"{len(meas)} held-out shots; median relative L1 shape error {np.median(rel):.2f} -> {out}")


if __name__ == "__main__":
    main()
