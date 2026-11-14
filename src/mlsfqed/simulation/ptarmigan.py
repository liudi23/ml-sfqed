"""Run Ptarmigan for one collision and reduce its output to observables.

Rates: LMA (default), quantum stochastic emission, radiation reaction on, pair creation off (positron
yield is negligible at Gemini parameters). LMA is used because LCFA, while matching LMA in energy loss
(0.07% at a0 = 20), overproduces photons 2.4-4.7x in weaker collisions and shifts the critical energy.
Ptarmigan's LMA tables stop near a local a ~ 20, so scans keep a0 <= 20; LMA costs ~11/36/97 s per
5000 electrons at a0 = 10/15/20 (aligned), vs ~3.5 s for LCFA (rates="lcfa").

Observables follow Los et al. (2024/2025) where possible: charge-weighted mean electron energy and P70
in the spectrometer window (E >= 305 MeV; P70 after 10 MeV smoothing), plus photon yield, mean and total energy, critical energy
and energy-weighted angular spread.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

E_BINS_E = np.linspace(0.0, 1500.0, 151)        # electron energy histogram edges [MeV]
E_BINS_G = np.geomspace(0.1, 1500.0, 121)       # photon energy histogram edges [MeV], log-spaced
WINDOW_MEV = 305.0                              # lower edge of the measured electron spectrum
DEFAULT_BIN = Path.home() / "build" / "ptarmigan" / "target" / "release" / "ptarmigan"


def ptarmigan_bin() -> Path:
    return Path(os.environ.get("PTARMIGAN_BIN", DEFAULT_BIN))


def render_yaml(p: dict, spectrum_file: str, E_grid: np.ndarray, n_electrons: int, seed: int,
                rates: str = "lma") -> str:
    """p: a0, waist_um, fwhm_fs, dx_um, dy_um, dz_um, radius_um, length_um, div_mrad, charge_pC, chirp."""
    step = E_grid[1] - E_grid[0]
    return f"""control:
  lcfa: {"true" if rates == "lcfa" else "false"}
  radiation_reaction: true
  pair_creation: false
  rng_seed: {seed}
laser:
  a0: {p['a0']:.6g}
  wavelength: 0.8 * micro
  waist: {p['waist_um']:.6g} * micro
  polarization: linear
  envelope: gaussian
  fwhm_duration: {p['fwhm_fs']:.6g} * femto
beam:
  n: {n_electrons}
  species: electron
  charge: {p['charge_pC']:.6g} * pico
  radius: {p['radius_um']:.6g} * micro
  length: {p['length_um']:.6g} * micro
  rms_divergence: {p['div_mrad']:.6g} * milli
  energy_chirp: {p.get('chirp', 0.0):.6g}
  offset: [{p['dx_um']:.6g} * micro, {p['dy_um']:.6g} * micro, {p['dz_um']:.6g} * micro]
  spectrum:
    file: {spectrum_file}
    min: {E_grid[0]:.6f} * MeV / (me * c^2)
    max: {E_grid[-1]:.6f} * MeV / (me * c^2)
    step: {step:.6f} * MeV / (me * c^2)
output:
  ident: run
  units: hep
  file_format: plain_text
  coordinate_system: beam
  electron:
    - energy:(1500;auto;energy in 0.0, 1500.0 * MeV)
  photon:
    - energy:(15000;auto;energy in 0.0, 1500.0 * MeV)
stats:
  photon:
    - total number
    - variance angle_x`energy
    - variance angle_y`energy
"""


def _read_hist(path):
    d = np.loadtxt(path, skiprows=2)
    return d[:, 0] * 1e3, d[:, 1]                 # hep units: GeV -> MeV; density per GeV (weighted)


def _read_stats(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = float(v.split("[")[0])
    return out


P70_SMOOTH_MEV = 10.0     # Gaussian smoothing before taking the maximum (spectrometer-like resolution)


def p70(E, dQdE, smooth_mev=P70_SMOOTH_MEV):
    """Los et al. P70: normalise the energy-weighted spectrum, find the energy below which 70% of the
    beam energy lies, return the spectrum's maximum above that energy (in 1/GeV). The simulated spectrum
    is first smoothed with a Gaussian of `smooth_mev`; without it, Monte Carlo spikes make the maximum
    noisy (+-8% at 5000 electrons)."""
    if smooth_mev:
        dE = E[1] - E[0]
        k = np.arange(-4 * smooth_mev, 4 * smooth_mev + dE, dE)
        g = np.exp(-0.5 * (k / smooth_mev) ** 2)
        dQdE = np.convolve(np.pad(dQdE, len(k) // 2, mode="edge"), g / g.sum(), mode="valid")[:len(E)]
    Eg = E / 1e3
    w = dQdE * Eg
    tot = np.trapezoid(w, Eg)
    if tot <= 0:
        return np.nan
    w = w / tot
    cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(Eg))])
    e70 = np.interp(0.7, cdf, Eg)
    return float(w[Eg >= e70].max())


def critical_energy(Eg, dNdE):
    """Fit dN/dE ~ E^(-2/3) exp(-E/Ec) to the photon spectrum above 1 MeV (log-likelihood-free LSQ in log)."""
    m = (Eg > 1.0) & (dNdE > 0)
    if m.sum() < 5:
        return np.nan
    y = np.log(dNdE[m]) + (2 / 3) * np.log(Eg[m])
    slope, _ = np.polyfit(Eg[m], y, 1, w=np.sqrt(dNdE[m] * Eg[m]))   # weight towards where energy is
    return float(-1.0 / slope) if slope < 0 else np.nan


def observables(e_E, e_h, g_E, g_h, st, n_electrons_real):
    """Reduce histograms to observables. e_h, g_h are number densities per GeV (weights = real particles)."""
    dE_e = np.diff(E_BINS_E)
    Ne = np.histogram(e_E, E_BINS_E, weights=e_h * np.gradient(e_E) / 1e3)[0]   # electrons per bin
    Ng = np.histogram(g_E, E_BINS_G, weights=g_h * np.gradient(g_E) / 1e3)[0]   # photons per bin
    Ec = 0.5 * (E_BINS_E[1:] + E_BINS_E[:-1])
    win = Ec >= WINDOW_MEV
    o = {}
    o["e_mean_MeV"] = float(np.sum(Ne * Ec) / Ne.sum())
    o["e_std_MeV"] = float(np.sqrt(np.sum(Ne * (Ec - o["e_mean_MeV"]) ** 2) / Ne.sum()))
    o["e_mean_window_MeV"] = float(np.sum(Ne[win] * Ec[win]) / Ne[win].sum()) if Ne[win].sum() > 0 else np.nan
    o["e_frac_in_window"] = float(Ne[win].sum() / Ne.sum())
    o["e_P70"] = p70(Ec[win], Ne[win] / dE_e[win])
    Gc = np.sqrt(E_BINS_G[1:] * E_BINS_G[:-1])
    o["g_number"] = float(st.get("photon: total number", np.nan))
    o["g_number_per_e"] = o["g_number"] / n_electrons_real
    o["g_total_energy_MeV_per_e"] = float(np.sum(Ng * Gc)) / n_electrons_real
    o["g_mean_MeV"] = float(np.sum(Ng * Gc) / Ng.sum()) if Ng.sum() > 0 else np.nan
    o["g_crit_MeV"] = critical_energy(Gc, Ng / np.diff(E_BINS_G))
    for ax in ("x", "y"):
        key = [k for k in st if k.startswith(f"photon: variance angle_{ax}")]
        o[f"g_theta_{ax}_rms_mrad"] = float(np.sqrt(st[key[0]]) * 1e3) if key else np.nan
    return o, Ne, Ng


def run_one(p: dict, spectrum: np.ndarray, E_grid: np.ndarray, n_electrons: int, seed: int, workdir=None,
            rates: str = "lma"):
    """Run one collision. spectrum: dN/dE on E_grid (any normalisation). Returns (record, Ne, Ng)."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        np.savetxt(tmp / "spectrum.txt", spectrum, fmt="%.6e")
        y = render_yaml(p, "spectrum.txt", E_grid, n_electrons, seed, rates)
        (tmp / "run.yml").write_text(y)
        t0 = time.time()
        r = subprocess.run([str(ptarmigan_bin()), "run.yml"], check=False, cwd=tmp, capture_output=True, text=True)
        wall = time.time() - t0
        if r.returncode != 0:
            err = (r.stdout + r.stderr).strip().splitlines()
            return {"ok": False, "error": err[-1] if err else "unknown", "wall_s": wall,
                    "config_hash": hashlib.sha1(y.encode()).hexdigest()[:12]}, None, None
        e_E, e_h = _read_hist(tmp / "run_electron_energy_energy-cut.dat")
        g_E, g_h = _read_hist(tmp / "run_photon_energy_energy-cut.dat")
        st = _read_stats(tmp / "run_stats.txt")
    n_real = p["charge_pC"] * 1e-12 / 1.602176634e-19
    o, Ne, Ng = observables(e_E, e_h, g_E, g_h, st, n_real)
    o.update({"ok": True, "wall_s": wall, "config_hash": hashlib.sha1(y.encode()).hexdigest()[:12]})
    return o, Ne, Ng
