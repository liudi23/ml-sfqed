"""How well can the incoming beam's P70 and mean energy be predicted from the shot diagnostics?

Trains the Stage 1 peak model on runs 05-12, predicts run 13 (time split, never seen) and compares the
predicted (median of 100 calibrated samples) with the measured P70 and mean energy above 305 MeV.
The resulting RMSEs are the realistic incoming-beam uncertainties used by infer_los2024.py --beam-unc
calibrated, and are compared with the spread of Los et al.'s 100 network predictions per shot.
  python scripts/beam_prediction_error.py      (~1-2 min)   -> results/inference/beam_prediction_error.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlsfqed.beam_model.twopart import PeakBeamModel, RawFeaturiser, to_electron_spectrum
from mlsfqed.data.gemini import load_all, physical_spectra, time_ordered_split
from mlsfqed.inference.los2024 import load, spectrum_observables
from mlsfqed.simulation.ptarmigan import p70

OUT = ROOT / "results" / "inference"


def main():
    d = load_all()
    E, Y = d["E_MeV"], physical_spectra(d)
    tr, te = time_ordered_split(d)
    w = E >= 305
    rf = RawFeaturiser().fit(d, tr)
    # the adopted Stage 1 model: square-root-edge peak, peak position blended with a Gaussian process
    m = PeakBeamModel(E, seed=0, shape="sqrt_edge", gp_blend=("position",)).fit(rf.transform(d, tr), Y[tr]).calibrate()
    N, _ = to_electron_spectrum(m.sample_calibrated(rf.transform(d, te), 100, 0), E)
    Nt, _ = to_electron_spectrum(Y[te], E)

    def obs(n):
        return p70(E[w], n[w], smooth_mev=0), float(np.sum(n[w] * E[w]) / np.sum(n[w]))

    P = np.array([[obs(x) for x in s] for s in N])
    T = np.array([obs(x) for x in Nt])
    res = {"method": "Stage 1 square-root-edge peak model (GP-blended position), train runs 05-12, test run 13; median of 100 calibrated samples"}
    for j, name in enumerate(("P70", "mean_window_MeV")):
        med = np.median(P[:, :, j], 1)
        lo, hi = np.percentile(P[:, :, j], [5, 95], axis=1)
        res[name] = {"rmse": float(np.sqrt(np.mean((med - T[:, j]) ** 2))),
                     "our_spread_median": float(np.median(P[:, :, j].std(1))),
                     "our_coverage90": float(np.mean((T[:, j] >= lo) & (T[:, j] <= hi)))}
    dl = load()
    Ed = dl["E_MeV"]
    sp = np.array([[[spectrum_observables(Ed, p)[k] for k in ("P70", "mean_window_MeV")] for p in s["pre_dNdE"]]
                   for s in dl["shots"]])
    for j, name in enumerate(("P70", "mean_window_MeV")):
        res[name]["los_network_spread_per_shot"] = sp[:, :, j].std(1).round(4).tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "beam_prediction_error.json").write_text(json.dumps(res, indent=2))
    for name in ("P70", "mean_window_MeV"):
        r = res[name]
        print(f"incoming {name}: realistic prediction error {r['rmse']:.3f}; our calibrated spread {r['our_spread_median']:.3f} "
              f"(90% coverage {r['our_coverage90']:.2f}); Los network spread median {np.median(r['los_network_spread_per_shot']):.3f}")


if __name__ == "__main__":
    main()
