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

TGU-Coherence-Framework/
├── src/
│   ├── tgu_precession.py           # Core α and Δϕ calculations
│   ├── spin_network_sim.py         # QuTiP spin models + eigenvalue derivation of k
│   ├── icoer_torch.py              # ICOER metric + coherence activation layers
│   └── aya_node_prototype.py       # Early AYA-NODE spin-coherent architecture
├── data/
│   ├── orbital_data.csv            # Planetary & exoplanet parameters (e, a, observed precession)
│   └── reference_gr_precession.txt # GR baseline values
├── notebooks/
│   ├── 01_TGU_validation.ipynb     # Reproduction of paper results
│   ├── 02_k_derivation_spin.ipynb  # Computational derivation of Matuchaki Parameter
│   └── 03_icoer_experiments.ipynb  # ICOER in neural networks
├── docs/
│   └── paper_figures/              # High-res figures from the preprint
├── requirements.txt
├── LICENSE
└── README.md


## Installation

```bash
git clone https://github.com/yourusername/TGU-Coherence-Framework.git
cd TGU-Coherence-Framework

# Recommended: use virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

requirements.txt (minimal):

qutip>=4.7
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
torch>=2.0
pandas

Quick Start Examples1. Compute TGU Precession for Mercury

from src.tgu_precession import tgu_alpha, gr_precession_mercury

e = 0.2056
a = 0.387098
k = 0.0881

alpha = tgu_alpha(e, a, k)
delta_phi_tgu = alpha * gr_precession_mercury()  # in arcsec/century
print(f"α = {alpha:.5f}  →  Δϕ_TGU ≈ {delta_phi_tgu:.2f} arcsec/century")

2. Derive k from Spin Network (refined version)

from src.spin_network_sim import optimize_k_from_spin_hamiltonian

optimal_k, eigenvalues = optimize_k_from_spin_hamiltonian(
    e=0.2056, a=0.387098, target_k=0.0881
)
print(f"Emergent resonant eigenvalue (Matuchaki Parameter): {optimal_k:.6f}")

CitationIf you use this code in your research, please cite:

@article{matuchaki2025tgu,
  author = {Henry Matuchaki},
  title = {A Unified Informational Model for Orbital Precession: Spin-Informational Corrections from the TGU Framework},
  year = {2025},
  doi = {10.20944/preprints202502.0514.v3},
  url = {https://www.preprints.org/manuscript/202502.0514/v3}
}

LicenseMIT License – see the LICENSE file for details.Contact & Further ReadingAuthor: Henry Matuchaki  
X/Twitter: @MatuchakiSilva
  
Related works: AYA-NODE prototypes, ICOER metric, spin-informational computing

Feedback, issues, and pull requests are very welcome!ⵔ◯ᘛ9ᘚ◯ⵔ


This README is ready to use — professional, informative, and aligned with your current work (as of Jan 2025/2026). Feel free to tweak the repo name, add your actual GitHub username, or include more badges/links. If you want a shorter version or more emphasis on ICOER/AYA, just let me know! 🚀
