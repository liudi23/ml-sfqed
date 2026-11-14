"""Analyse the Ptarmigan validation runs in results/ptarmigan_validation against theory.

Run the four input files there first (ptarmigan <file>.yml for A, B_*, C, D, D_noRR), then:
    cd results/ptarmigan_validation && python ../../scripts/validate_ptarmigan.py

A  classical RR (LMA classical rate + Landau-Lifshitz) vs the analytic plane-wave LL solution
B  quantum RR: LMA vs LCFA vs Gaunt-corrected classical; quantum/LL suppression; stochastic spread
C  weak-field Compton: first-harmonic edge, a0^2 yield scaling, radiated energy vs LL x Klein-Nishina
"""
import glob
import re

import numpy as np
from scipy.integrate import quad

alpha=1/137.035999; me=0.51099895e6; lam=0.8e-6; c=299792458; hbar_eVs=6.582119569e-16
w0=2*np.pi*c/lam; hw=hbar_eVs*w0; tau=30e-15
g0=1000e6/me; u0=g0*(1+np.sqrt(1-1/g0**2))          # u- = gamma(1+beta), lightfront momentum / m
Ig2=w0*tau*np.sqrt(np.pi/(4*np.log(2)))               # int g^2 dphi for Gaussian intensity FWHM tau
def gamma_f(a0):
    h=1+(2/3)*alpha*(u0*hw/me)*(a0**2/2)*Ig2           # LP: <cos^2> = 1/2
    u=u0/h; return (u+1/u)/2
def read(f,key):
    with open(f) as fh:
        m=re.search(re.escape(key)+r' = ([0-9.eE+-]+)',fh.read())
    return float(m.group(1))
print('a0  ptarmigan_gamma  LL_analytic  rel_diff')
for f in sorted(glob.glob('A_a0_*_stats.txt'),key=lambda s: float(s.split('_')[2])):
    a0=float(f.split('_')[2]); gp=read(f,'electron: mean gamma'); ga=gamma_f(a0)
    print(f'{a0:4.0f}  {gp:10.2f}  {ga:10.2f}  {(gp-ga)/(g0-ga):+.4f} of energy loss')

print()
def gaunt(chi): return (1+4.8*(1+chi)*np.log(1+1.7*chi)+2.44*chi**2)**(-2/3)
print('a0  chi_peak  loss[MeV]: LL   classical+Gaunt  quantum-LCFA  quantum-LMA | LMA/LCFA  quantum/LL  g(chi_peak)  rel.spread(LMA)')
for a0 in (5,10,15):
    r={k:read(f'{k}_a0_{a0:.3f}_stats.txt','electron: mean gamma') for k in ('B_quantum_lma','B_quantum_lcfa','B_classical_gaunt')}
    sd=np.sqrt(read(f'B_quantum_lma_a0_{a0:.3f}_stats.txt','electron: variance gamma'))
    L={k:(g0-v)*me/1e6 for k,v in r.items()}; LL=(g0-gamma_f(a0))*me/1e6
    chi=u0*hw/me*a0
    print(f"{a0:3d}  {chi:6.3f}   {LL:7.1f}  {L['B_classical_gaunt']:7.1f}  {L['B_quantum_lcfa']:7.1f}  {L['B_quantum_lma']:7.1f} | {L['B_quantum_lma']/L['B_quantum_lcfa']:.3f}  {L['B_quantum_lma']/LL:.3f}  {gaunt(chi):.3f}  {sd/r['B_quantum_lma']:.3f}")

print()
n_e=1_000_000
Emax0=u0**2*hw/(1+2*u0*hw/me)/1e6                  # first-harmonic edge [MeV] at a -> 0
phi=np.linspace(-6,6,4001)*w0*tau; g2=np.exp(-4*np.log(2)*phi**2/(w0*tau)**2)
res={}
for a0 in (0.25,0.5):
    f=f'C_a0_{a0:.3f}_stats.txt'; N=read(f,'photon: total number'); Em=read(f,'photon: mean energy')
    dE_LL=(g0-gamma_f(a0))*me/1e6
    chi=u0*hw/me*a0*np.sqrt(g2)/np.sqrt(2)              # cycle-averaged (rms) chi along the pulse
    gbar=np.sum(g2*gaunt(chi))/np.sum(g2)
    dE_pt=N*Em/n_e
    E_edge_peak=u0**2*hw/(1+2*u0*hw/me+a0**2/2)/1e6
    d=np.loadtxt(f'C_a0_{a0:.3f}_photon_energy_energy-cut.dat',skiprows=2); E=d[:,0]*1e3; dn=d[:,1]
    below=dn[(E>E_edge_peak-3)&(E<E_edge_peak-0.5)].mean(); above=dn[(E>Emax0+0.5)&(E<Emax0+3)].mean()
    res[a0]=N
    print(f'a0={a0}: photons/e {N/n_e:.5f} (+-{np.sqrt(N)/n_e:.5f}), radiated {dE_pt*1e3:.1f} keV vs LL x <g> {dE_LL*gbar*1e3:.1f} keV (LL {dE_LL*1e3:.1f}, <g>={gbar:.3f}) ratio {dE_pt/(dE_LL*gbar):.3f}+-{1/np.sqrt(N):.3f}')
    print(f'   edge: theory {E_edge_peak:.2f} (peak a0) - {Emax0:.2f} MeV (a->0); dN/dE drops x{below/above:.0f} across it; fraction above 23.2 MeV {read(f,"photon: total number (energy [MeV] > 2.320000e1)")/N:.3f}')
print(f'yield ratio N(0.5)/N(0.25) = {res[0.5]/res[0.25]:.3f} (a0^2 scaling: 4)')

# Klein-Nishina correction to radiated energy (linear Compton backscattering), x = 4 E w0 / m^2
def f(x):
    ym=x/(1+x)
    def ds(y):
        r=y/(x*(1-y)); return (1/(1-y)+1-y-4*r*(1-r))/x
    return quad(lambda y: y*ds(y),0,ym,limit=200)[0]
x=2*u0*hw/me
kn=(f(x)/x)/(f(1e-6)/1e-6)
print(f'x = {x:.4f}; Klein-Nishina energy-loss factor = {kn:.4f}')
for a0,ratio_ll in ((0.25,128.5/135.7),(0.5,511.4/542.5)):
    print(f'a0={a0}: Ptarmigan/LL = {ratio_ll:.3f}  vs Klein-Nishina factor {kn:.3f}  -> residual {ratio_ll/kn-1:+.3f}')
