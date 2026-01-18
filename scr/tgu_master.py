"""
tgu_master.py

MASTER TGU – REFINED ORBITAL PRECESSION MODEL
Author: Henry Matuchaki (@MatuchakiSilva)
Date: January 2026

This module implements the refined TGU model with coherence resistance factor
for orbital precession calculations, ensuring convergence with General Relativity (GR)
in low-strain regimes while preserving informational gain in high-eccentricity orbits.

Core Formula:
Δϕ_TGU = α * Δϕ_GR * ε^{-n}

Where:
α = 1 + k * (e / a)  # Informational gain
ε = 1 + (rs / r)^2    # Informational dilution
r ≈ a (semi-major axis)

Usage:
- Run directly for Mercury example
- Import functions for custom bodies
"""

import numpy as np

# ============================================================
# TGU UNIVERSAL CONSTANTS (Calibrated)
# ============================================================
K = 0.0881                  # Universal Informational Coupling (Matuchaki Parameter)
N = 12                      # Coherence Exponent (Harmonic Structure)
RS_INFORMATIONAL = 0.02391625  # Solar Coherence Radius (AU)


# ============================================================
# CORE FUNCTIONS
# ============================================================
def calculate_alpha(e: float, a: float, k: float = K) -> float:
    """
    Calculate the TGU informational gain factor α.

    Parameters:
    - e: Orbital eccentricity (dimensionless)
    - a: Semi-major axis (AU)
    - k: Informational coupling constant (default: 0.0881)

    Returns:
    - alpha: Gain factor ≥ 1
    """
    return 1.0 + k * (e / a)


def calculate_coherence_factor(r: float, rs: float = RS_INFORMATIONAL, n: float = N) -> float:
    """
    Calculate the coherence resistance factor ε^{-n}.

    Parameters:
    - r: Distance from solar vortex (≈ a in AU)
    - rs: Solar coherence radius (default: 0.02391625 AU)
    - n: Coherence exponent (default: 12)

    Returns:
    - coherence_factor: Resistance factor ∈ (0, 1]
    """
    epsilon = 1.0 + (rs / r) ** 2
    return epsilon ** (-n)


def calculate_tgu_precession(e: float, a: float, delta_phi_gr: float,
                             k: float = K, rs: float = RS_INFORMATIONAL, n: float = N) -> float:
    """
    Calculate the refined TGU precession.

    Parameters:
    - e, a: Orbital parameters
    - delta_phi_gr: GR precession (arcsec/century)
    - k, rs, n: TGU constants

    Returns:
    - delta_phi_tgu: Refined precession (arcsec/century)
    """
    alpha = calculate_alpha(e, a, k)
    coherence_factor = calculate_coherence_factor(a, rs, n)  # r ≈ a
    return delta_phi_gr * alpha * coherence_factor


# ============================================================
# EXAMPLE EXECUTION (Mercury – Stationary State)
# ============================================================
if __name__ == "__main__":
    # Orbital Parameters (Mercury)
    a = 0.387              # Semi-major axis (AU)
    e = 0.2056             # Eccentricity
    period = 0.240846      # Orbital period (years, unused in this calc)
    precessao_rg = 42.98   # GR Reference (arcsec/century)

    # Compute
    alpha = calculate_alpha(e, a)
    coherence_factor = calculate_coherence_factor(a)
    tgu_precession = calculate_tgu_precession(e, a, precessao_rg)

    # Output
    print("TGU MASTER ANALYSIS – MERCURY")
    print("-" * 30)
    print(f"Informational Gain (Alpha):   {alpha:.6f}")
    print(f"Coherence Resistance (eps^-n): {coherence_factor:.6f}")
    print(f"Final TGU Precession:         {tgu_precession:.2f} arcsec/century")
    print(f"Convergence with GR:          {(tgu_precession / precessao_rg) * 100:.2f}%")