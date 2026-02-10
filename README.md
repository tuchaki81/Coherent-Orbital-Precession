# Coherent-Orbital-Precession

**Coherence-Based Scalar-Tensor Extension of General Relativity — Computational Implementation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24%2B-blue)](https://numpy.org/)

Computational implementation accompanying the paper:

> **Coherence-Based Scalar-Tensor Extension of General Relativity: Variational Formulation, Observational Bounds, and Predictions for High-Eccentricity Orbital Systems**  
> Henry Matuchaki (2026)

## Theory

The framework extends General Relativity by introducing a scalar coherence field Φ non-minimally coupled to spacetime curvature through the action:

$$
S = \frac{1}{16\pi G}\int d^4x\,\sqrt{-g}\left[(1+\lambda\Phi)\,R - \frac{\omega}{2}\,\nabla_\mu\Phi\,\nabla^\mu\Phi - V(\Phi)\right] + S_m
$$

Metric variation yields modified Einstein equations with a coherence tensor $C_{\mu\nu} = \lambda(\nabla_\mu\nabla_\nu\Phi - g_{\mu\nu}\Box\Phi)$ that vanishes identically when $\Phi \to 0$, recovering GR exactly.

The effective correction to periapsis precession is:

$$
\Delta\phi = \Delta\phi_{\text{GR}}\,(1 + \delta), \qquad \delta = \lambda_{\text{eff}}^2\,\Xi
$$

where the **orbital asymmetry parameter** is defined as:

$$
\Xi = \frac{e^2}{1 - e^2}\cdot\frac{r_g}{a}
$$

This quantity is manifestly **dimensionless and unit-independent** — it gives the same numerical value whether distances are measured in meters, AU, or Planck lengths.

## Key Result

Binary pulsar timing constrains the effective coupling to:

$$
\lambda_{\text{eff}} < 1.95 \quad (95\%\;\text{C.L., from PSR B1913+16})
$$

At this bound, the predicted corrections are:

| System | e | Ξ | δ | Status |
|---|---|---|---|---|
| Mercury | 0.206 | 2.3 × 10⁻⁹ | 8.5 × 10⁻⁹ | Undetectable |
| PSR B1913+16 | 0.617 | 2.6 × 10⁻⁶ | 1.0 × 10⁻⁵ | At measurement limit |
| S2 (Sgr A*) | 0.880 | 2.6 × 10⁻⁴ | 1.0 × 10⁻³ | **Within reach of GRAVITY+** |
| Inner S-star (hyp.) | 0.950 | 7.3 × 10⁻³ | 2.8 × 10⁻² | **Decisive test** |

## Repository Structure

```
Coherent-Orbital-Precession/
├── coherent_orbital_precession.py   # Full implementation + article results
├── main_revised.tex                 # Article source (LaTeX, RevTeX 4-2)
├── references.bib                   # BibTeX bibliography
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/tuchaki81/Coherent-Orbital-Precession.git
cd Coherent-Orbital-Precession
```

The only dependency is NumPy:

```bash
pip install numpy
```

## Usage

Run the full verification suite:

```bash
python coherent_orbital_precession.py
```

This reproduces all numerical results from the article: observational bounds (Table I), predictions (Table II), dimensional consistency checks, and the S2 star analysis.

### Using as a library

```python
from coherent_orbital_precession import (
    build_catalogue,
    asymmetry_parameter,
    fractional_correction,
    corrected_precession,
    compute_all_bounds,
    tightest_bound,
)

# Load astrophysical systems
catalogue = build_catalogue()

# Compute asymmetry parameter for any system
s2 = catalogue["S2"]
Xi = s2.asymmetry_parameter  # 2.63e-04

# Get observational bound
bounds = compute_all_bounds(catalogue)
name, lambda_bound = tightest_bound(bounds)  # PSR B1913+16, 1.95

# Predict correction
delta = fractional_correction(lambda_bound, Xi)  # 9.98e-04 (0.1%)
```

## Building the Article

The LaTeX source requires RevTeX 4-2:

```bash
pdflatex main_revised
bibtex main_revised
pdflatex main_revised
pdflatex main_revised
```

## Citation

```bibtex
@article{matuchaki2026coherence,
  author  = {Henry Matuchaki},
  title   = {Coherence-Based Scalar-Tensor Extension of General Relativity:
             Variational Formulation, Observational Bounds, and Predictions
             for High-Eccentricity Orbital Systems},
  year    = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Author: Henry Matuchaki  
Email: henrymatuchaki@gmail.com
