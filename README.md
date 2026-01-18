# Coherent-Orbital-Precession

**Unified Informational Spin Theory (TGU) – Orbital Precession Corrections and Computational Coherence Simulations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![QuTiP](https://img.shields.io/badge/QuTiP-4.7+-green)](https://qutip.org/)
[![Torch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)

This repository implements the **Unified Informational Spin Theory (TGU)** introduced in the paper:

> **A Unified Informational Model for Orbital Precession: Spin-Informational Corrections from the TGU Framework**  
> Henry Matuchaki (2025)  
> Preprint DOI: [10.20944/preprints202502.0514.v3](https://doi.org/10.20944/preprints202502.0514.v3) (updated versions available)

TGU proposes that gravitational phenomena, such as perihelion precession, emerge from **informational coherence** in spin-structured systems — a pre-physical coherence field preceding classical observables.

The core phenomenological correction is:

$$
\alpha = 1 + k \cdot \frac{e}{a}
$$

$$
\Delta\phi_{\text{TGU}} = \alpha \cdot \Delta\phi_{\text{GR}}
$$

where:
- \(e\) = orbital eccentricity
- \(a\) = semi-major axis (in AU)
- \(k \approx 0.0881\) — the **Matuchaki Parameter** (informational coherence constant)

This repository contains:
- Empirical validation scripts for Solar System planets, asteroid Icarus, and high-eccentricity exoplanets
- Computational derivation of \(k\) via resonant eigenvalues in spin networks
- Early prototypes for **ICOER** (Informational Coherence Index) and **AYA-NODE** spin-coherent architectures

## Key Features

- **Orbital Precession Predictions**  
  Reproduces TGU-adjusted precession values for Mercury, Venus, Earth, Mars, Icarus, WASP-12b, HD 80606b, HAT-P-2b, etc.

- **Spin Network Simulations**  
  Models orbital systems as coupled spin chains (QuTiP) → derives \(k\) as dominant eigenvalue of coherence/resonance operator  
  Achieves near-exact match: |λ - 0.0881| < 10⁻⁵ in optimized regimes

- **ICOER & Coherence Activation**  
  Torch-based modules for measuring and maximizing informational coherence  
  Applications to AI efficiency, biological resonance, and gravitational analogs

## Repository Structure
