"""
tgu_precession.py

Core module for calculating the TGU (Unified Informational Spin Theory) 
orbital precession correction factor α and the adjusted precession Δϕ_TGU.

Formula:
    α = 1 + k * (e / a)
    Δϕ_TGU = α * Δϕ_GR

Where:
    e: orbital eccentricity (dimensionless)
    a: semi-major axis in Astronomical Units (AU)
    k: Matuchaki Parameter ≈ 0.0881 (informational coherence constant)

Author: Henry Matuchaki (@MatuchakiSilva)
Date: January 2026
"""

# Default value of the Matuchaki Parameter (from empirical fit in the paper)
DEFAULT_K = 0.0881


def tgu_alpha(e: float, a: float, k: float = DEFAULT_K) -> float:
    """
    Calculate the TGU correction factor α.

    Parameters:
    -----------
    e : float
        Orbital eccentricity (dimensionless, 0 ≤ e < 1)
    a : float
        Semi-major axis in Astronomical Units (AU)
    k : float, optional
        Informational coherence constant (default: 0.0881)

    Returns:
    --------
    float
        Correction factor α ≥ 1

    Raises:
    -------
    ValueError
        If e < 0, e ≥ 1, or a ≤ 0
    """
    if e < 0 or e >= 1:
        raise ValueError("Eccentricity e must be in [0, 1)")
    if a <= 0:
        raise ValueError("Semi-major axis a must be positive")

    strain = e / a
    alpha = 1.0 + k * strain
    return alpha


def tgu_precession(e: float, a: float, delta_phi_gr: float, k: float = DEFAULT_K) -> float:
    """
    Compute the TGU-adjusted precession Δϕ_TGU in arcsec/century.

    Parameters:
    -----------
    e : float
        Orbital eccentricity
    a : float
        Semi-major axis (AU)
    delta_phi_gr : float
        General Relativity precession (arcsec/century)
    k : float, optional
        Matuchaki Parameter (default: 0.0881)

    Returns:
    --------
    float
        TGU-corrected precession Δϕ_TGU
    """
    alpha = tgu_alpha(e, a, k)
    return alpha * delta_phi_gr


# Reference GR precession values (arcsec/century) from the paper (approximate)
GR_PRECESSION_REFERENCE = {
    "Mercury": 43.03,
    "Venus":   8.6,
    "Earth":   3.84,
    "Mars":    1.35,
    "Icarus":  10.05,      # Asteroid
    # Example exoplanets (theoretical GR values, for illustration)
    "WASP-12b":   0.5,     # Very small due to large a in some models; adjust as needed
    "HD 80606b":  1.2,
    "HAT-P-2b":   2.8,
}


def get_tgu_prediction(body: str, k: float = DEFAULT_K) -> dict:
    """
    Convenience function: Get TGU prediction for a known body.

    Parameters:
    -----------
    body : str
        Name of the body (case-sensitive, see GR_PRECESSION_REFERENCE keys)
    k : float, optional
        Custom Matuchaki Parameter

    Returns:
    --------
    dict
        {'alpha': float, 'delta_phi_gr': float, 'delta_phi_tgu': float}
    """
    if body not in GR_PRECESSION_REFERENCE:
        raise ValueError(f"Unknown body: {body}. Available: {list(GR_PRECESSION_REFERENCE.keys())}")

    # Orbital parameters from standard sources (JPL/NASA, approximate)
    orbital_params = {
        "Mercury":   (0.205630, 0.387098),
        "Venus":     (0.006772, 0.723332),
        "Earth":     (0.016709, 1.000000),
        "Mars":      (0.093412, 1.523679),
        "Icarus":    (0.8269,   1.077),
        "WASP-12b":  (0.049,    0.02293),   # Hot Jupiter example
        "HD 80606b": (0.9332,   0.469),     # High eccentricity
        "HAT-P-2b":  (0.517,    0.0674),    # Another high-e example
    }

    e, a = orbital_params[body]
    delta_phi_gr = GR_PRECESSION_REFERENCE[body]

    alpha = tgu_alpha(e, a, k)
    delta_phi_tgu = tgu_precession(e, a, delta_phi_gr, k)

    return {
        "body": body,
        "e": e,
        "a_AU": a,
        "alpha": round(alpha, 5),
        "delta_phi_GR_arcsec_century": delta_phi_gr,
        "delta_phi_TGU_arcsec_century": round(delta_phi_tgu, 2)
    }


# Example usage (run module directly)
if __name__ == "__main__":
    print("TGU Precession Predictions (k = {:.4f})\n".format(DEFAULT_K))
    
    for body in ["Mercury", "Venus", "Earth", "Mars", "Icarus"]:
        result = get_tgu_prediction(body)
        print(f"{body:10}: α = {result['alpha']:.5f}  |  "
              f"Δϕ_GR = {result['delta_phi_GR_arcsec_century']:6.2f}  →  "
              f"Δϕ_TGU = {result['delta_phi_TGU_arcsec_century']:6.2f} arcsec/century")
    
    print("\nHigh-eccentricity exoplanet example:")
    print(get_tgu_prediction("HD 80606b"))
