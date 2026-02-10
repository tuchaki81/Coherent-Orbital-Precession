"""
COHERENCE-BASED SCALAR-TENSOR EXTENSION OF GENERAL RELATIVITY
==============================================================
Reference: Matuchaki, H. (2026). Coherence-Based Scalar-Tensor Extension
           of General Relativity: Variational Formulation, Observational
           Bounds, and Predictions for High-Eccentricity Orbital Systems.

Computational implementation of the theoretical framework.
Each function corresponds to a specific equation in the article.

Repository: https://github.com/tuchaki81/Coherent-Orbital-Precession
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ============================================================================
# 1. PHYSICAL CONSTANTS (SI)
# ============================================================================

G_SI: float = 6.67430e-11       # Gravitational constant [m³ kg⁻¹ s⁻²]
C_SI: float = 2.99792458e8      # Speed of light [m s⁻¹]
M_SUN: float = 1.989e30         # Solar mass [kg]
AU_SI: float = 1.496e11         # Astronomical unit [m]

# ============================================================================
# 2. ORBITAL SYSTEM DATA
# ============================================================================

@dataclass
class OrbitalSystem:
    """
    A gravitational system with measured orbital parameters.

    Attributes:
        name: System identifier.
        eccentricity: Orbital eccentricity e (dimensionless).
        semi_major_axis_m: Semi-major axis a [m].
        central_mass_kg: Central mass M [kg].
        precession_gr_arcsec: GR-predicted precession [arcsec/century],
                              or None if not applicable.
        precession_obs_arcsec: Observed precession [arcsec/century],
                               or None if unavailable.
        precision_ppm: Measurement precision on precession [ppm],
                       or None if unavailable.
    """
    name: str
    eccentricity: float
    semi_major_axis_m: float
    central_mass_kg: float
    precession_gr_arcsec: Optional[float] = None
    precession_obs_arcsec: Optional[float] = None
    precision_ppm: Optional[float] = None

    @property
    def gravitational_radius_m(self) -> float:
        """Gravitational radius r_g = 2GM/c² [m]."""
        return 2.0 * G_SI * self.central_mass_kg / C_SI**2

    @property
    def compactness(self) -> float:
        """Gravitational compactness r_g / a (dimensionless)."""
        return self.gravitational_radius_m / self.semi_major_axis_m

    @property
    def eccentricity_factor(self) -> float:
        """Eccentricity enhancement e²/(1 - e²) (dimensionless)."""
        return self.eccentricity**2 / (1.0 - self.eccentricity**2)

    @property
    def asymmetry_parameter(self) -> float:
        """
        Eq. (9): Orbital asymmetry parameter

            Ξ = e²/(1 - e²) · r_g/a

        Dimensionless. Governs the magnitude of coherence corrections.
        """
        return self.eccentricity_factor * self.compactness

    @property
    def semi_major_axis_au(self) -> float:
        """Semi-major axis in AU (for display)."""
        return self.semi_major_axis_m / AU_SI


# ============================================================================
# 3. CATALOGUE OF ASTROPHYSICAL SYSTEMS
# ============================================================================

def _semi_major_axis_from_period(period_days: float, total_mass_kg: float) -> float:
    """Kepler's third law: a = (G M P² / 4π²)^(1/3)."""
    P_s = period_days * 86400.0
    return (G_SI * total_mass_kg * P_s**2 / (4.0 * np.pi**2))**(1.0 / 3.0)


def build_catalogue() -> Dict[str, OrbitalSystem]:
    """
    Returns a dictionary of astrophysical systems used in the article.

    Sources:
        Mercury–Mars: Pitjeva & Pitjev (2013), Will (2014).
        Icarus: Shapiro et al. (1971).
        PSR B1913+16: Weisberg & Huang (2016).
        PSR J0737-3039: Kramer et al. (2021).
        S2: GRAVITY Collaboration (2020, 2022).
        HD 80606b: Hébrard et al. (2010).
    """
    catalogue = {}

    # --- Solar System planets ---
    solar = [
        ("Mercury",  0.2056, 0.3871, 42.98, 42.98, 1000),
        ("Venus",    0.0068, 0.7233, 8.60,  8.60,  5000),
        ("Earth",    0.0167, 1.0000, 3.84,  3.84,  5000),
        ("Mars",     0.0934, 1.5237, 1.35,  1.35, 10000),
    ]
    for name, e, a_au, pgr, pobs, prec in solar:
        catalogue[name] = OrbitalSystem(
            name=name,
            eccentricity=e,
            semi_major_axis_m=a_au * AU_SI,
            central_mass_kg=M_SUN,
            precession_gr_arcsec=pgr,
            precession_obs_arcsec=pobs,
            precision_ppm=prec,
        )

    # --- Asteroid 1566 Icarus ---
    catalogue["Icarus"] = OrbitalSystem(
        name="Icarus",
        eccentricity=0.8269,
        semi_major_axis_m=1.0770 * AU_SI,
        central_mass_kg=M_SUN,
        precession_gr_arcsec=10.05,
    )

    # --- Binary pulsars ---
    a_psr1 = _semi_major_axis_from_period(0.323, 2.828 * M_SUN)
    catalogue["PSR B1913+16"] = OrbitalSystem(
        name="PSR B1913+16",
        eccentricity=0.617,
        semi_major_axis_m=a_psr1,
        central_mass_kg=2.828 * M_SUN,
        precision_ppm=10,
    )

    a_psr2 = _semi_major_axis_from_period(0.1023, 2.587 * M_SUN)
    catalogue["PSR J0737-3039"] = OrbitalSystem(
        name="PSR J0737-3039",
        eccentricity=0.088,
        semi_major_axis_m=a_psr2,
        central_mass_kg=2.587 * M_SUN,
        precision_ppm=40,
    )

    # --- S2 star at Galactic Center ---
    catalogue["S2"] = OrbitalSystem(
        name="S2",
        eccentricity=0.88,
        semi_major_axis_m=1031.0 * AU_SI,
        central_mass_kg=4.0e6 * M_SUN,
    )

    # --- Exoplanet HD 80606b ---
    catalogue["HD 80606b"] = OrbitalSystem(
        name="HD 80606b",
        eccentricity=0.9332,
        semi_major_axis_m=0.469 * AU_SI,
        central_mass_kg=0.98 * M_SUN,
    )

    # --- Hypothetical inner S-star ---
    catalogue["Inner S-star (hyp.)"] = OrbitalSystem(
        name="Inner S-star (hyp.)",
        eccentricity=0.95,
        semi_major_axis_m=100.0 * AU_SI,
        central_mass_kg=4.0e6 * M_SUN,
    )

    return catalogue


# ============================================================================
# 4. CORE EQUATIONS OF THE COHERENCE FRAMEWORK
# ============================================================================

def coherence_tensor(
    grad_grad_phi: np.ndarray,
    box_phi: float,
    metric: np.ndarray,
    lambda_coupling: float,
) -> np.ndarray:
    """
    Eq. (5): Coherence tensor

        C_μν = λ (∇_μ ∇_ν Φ  −  g_μν □Φ)

    Arises from metric variation of the non-minimal coupling λΦR
    in the action (Eq. 1). Vanishes identically when Φ → 0.

    Args:
        grad_grad_phi: Second covariant derivatives ∇_μ ∇_ν Φ  [4×4 array].
        box_phi: d'Alembertian □Φ (scalar).
        metric: Metric tensor g_μν [4×4 array].
        lambda_coupling: Coherence coupling constant λ.

    Returns:
        4×4 numpy array: C_μν.
    """
    return lambda_coupling * (grad_grad_phi - metric * box_phi)


def asymmetry_parameter(eccentricity: float, r_g: float, a: float) -> float:
    """
    Eq. (9): Orbital asymmetry parameter

        Ξ = e² / (1 − e²)  ·  r_g / a

    Dimensionless combination that governs coherence corrections to
    periapsis precession. Vanishes for circular orbits (e = 0) and
    grows with both eccentricity and gravitational compactness.

    Args:
        eccentricity: Orbital eccentricity e.
        r_g: Gravitational radius 2GM/c² [m].
        a: Semi-major axis [m].

    Returns:
        float: Ξ (dimensionless).
    """
    return eccentricity**2 / (1.0 - eccentricity**2) * r_g / a


def fractional_correction(lambda_eff: float, Xi: float) -> float:
    """
    Eq. (10): Fractional coherence correction to precession

        δ = λ_eff² · Ξ

    The corrected precession is Δφ = Δφ_GR · (1 + δ).

    Args:
        lambda_eff: Effective coherence coupling (dimensionless).
        Xi: Orbital asymmetry parameter Ξ.

    Returns:
        float: δ (dimensionless fractional correction).
    """
    return lambda_eff**2 * Xi


def corrected_precession(
    precession_gr: float,
    lambda_eff: float,
    Xi: float,
) -> float:
    """
    Eq. (8): Corrected periapsis precession

        Δφ_TGU = Δφ_GR · (1 + δ)

    where δ = λ_eff² · Ξ (Eq. 10).

    Args:
        precession_gr: GR precession Δφ_GR [any consistent units].
        lambda_eff: Effective coherence coupling.
        Xi: Orbital asymmetry parameter Ξ.

    Returns:
        float: Corrected precession [same units as input].
    """
    delta = fractional_correction(lambda_eff, Xi)
    return precession_gr * (1.0 + delta)


def gr_precession_per_orbit(M_kg: float, a_m: float, e: float) -> float:
    """
    Eq. (7): GR periapsis advance per orbit [radians]

        Δφ_GR = 6π G M / [c² a (1 − e²)]

    Args:
        M_kg: Central mass [kg].
        a_m: Semi-major axis [m].
        e: Eccentricity.

    Returns:
        float: Precession per orbit [radians].
    """
    return 6.0 * np.pi * G_SI * M_kg / (C_SI**2 * a_m * (1.0 - e**2))


# ============================================================================
# 5. OBSERVATIONAL BOUNDS
# ============================================================================

def lambda_bound_from_system(system: OrbitalSystem) -> Optional[float]:
    """
    Derive upper bound on λ_eff from a system's measurement precision.

        |δ| < δ_max   →   λ_eff < sqrt(δ_max / Ξ)

    Args:
        system: An OrbitalSystem with known precision_ppm.

    Returns:
        Upper bound on λ_eff, or None if precision is unknown.
    """
    if system.precision_ppm is None:
        return None
    delta_max = system.precision_ppm * 1.0e-6
    Xi = system.asymmetry_parameter
    if Xi <= 0:
        return None
    return np.sqrt(delta_max / Xi)


def compute_all_bounds(catalogue: Dict[str, OrbitalSystem]) -> Dict[str, float]:
    """
    Compute λ_eff bounds from all systems with known precision.

    Returns:
        Dictionary {system_name: lambda_bound}.
    """
    bounds = {}
    for name, sys in catalogue.items():
        bound = lambda_bound_from_system(sys)
        if bound is not None:
            bounds[name] = bound
    return bounds


def tightest_bound(bounds: Dict[str, float]) -> Tuple[str, float]:
    """Return the system providing the tightest (smallest) λ_eff bound."""
    name = min(bounds, key=bounds.get)
    return name, bounds[name]


# ============================================================================
# 6. PREDICTIONS
# ============================================================================

def predict_corrections(
    catalogue: Dict[str, OrbitalSystem],
    lambda_eff: float,
) -> List[Dict[str, Any]]:
    """
    Compute predicted coherence corrections for all systems.

    Args:
        catalogue: Dictionary of OrbitalSystem objects.
        lambda_eff: Effective coupling to use (typically the tightest bound).

    Returns:
        List of dictionaries with predictions for each system.
    """
    results = []
    for name, sys in catalogue.items():
        Xi = sys.asymmetry_parameter
        delta = fractional_correction(lambda_eff, Xi)

        entry = {
            "name": name,
            "eccentricity": sys.eccentricity,
            "semi_major_axis_au": sys.semi_major_axis_au,
            "compactness": sys.compactness,
            "Xi": Xi,
            "delta": delta,
            "delta_percent": delta * 100.0,
        }

        if sys.precession_gr_arcsec is not None:
            entry["precession_gr"] = sys.precession_gr_arcsec
            entry["precession_tgu"] = corrected_precession(
                sys.precession_gr_arcsec, lambda_eff, Xi
            )
            entry["excess_arcsec"] = (
                entry["precession_tgu"] - entry["precession_gr"]
            )

        results.append(entry)

    # Sort by decreasing Ξ
    results.sort(key=lambda r: r["Xi"], reverse=True)
    return results


# ============================================================================
# 7. SCALAR-TENSOR FIELD EQUATIONS (NUMERICAL UTILITIES)
# ============================================================================

def phi_static_spherical(
    r: np.ndarray,
    lambda_coupling: float,
    omega: float,
    M_kg: float,
) -> np.ndarray:
    """
    Approximate static, spherically symmetric solution for Φ(r)
    in the weak-field limit.

    From Eq. (6):  ω □Φ + λR = 0
    In Schwarzschild background with R ≈ 0 (vacuum), the leading
    contribution is from the trace of the matter source.
    For a point mass: Φ(r) ≈ −(λ / ω) · (G M) / (c² r)

    Args:
        r: Radial coordinate(s) [m].
        lambda_coupling: λ.
        omega: Kinetic coefficient ω.
        M_kg: Source mass [kg].

    Returns:
        Array of Φ values at each r.
    """
    return -(lambda_coupling / omega) * G_SI * M_kg / (C_SI**2 * r)


def effective_ppn_gamma(lambda_coupling: float, omega: float) -> float:
    """
    PPN parameter γ in the scalar-tensor framework.

    For Brans-Dicke-type theories:
        γ = (ω + 1) / (ω + 2)

    The Cassini bound requires |γ − 1| < 2.3 × 10⁻⁵.

    Args:
        lambda_coupling: λ (not used directly; included for interface).
        omega: Kinetic coefficient ω (maps to ω_BD).

    Returns:
        float: γ_eff.
    """
    return (omega + 1.0) / (omega + 2.0)


# ============================================================================
# 8. DIMENSIONAL CONSISTENCY VERIFICATION
# ============================================================================

def verify_dimensional_consistency(system: OrbitalSystem) -> Dict[str, Any]:
    """
    Verify that Ξ is independent of the unit system.

    Computes Ξ using SI and then using rescaled units to confirm
    that the result is identical (dimensionless).

    Args:
        system: An OrbitalSystem.

    Returns:
        Dictionary with verification results.
    """
    # Compute in SI
    Xi_si = asymmetry_parameter(
        system.eccentricity,
        system.gravitational_radius_m,
        system.semi_major_axis_m,
    )

    # Compute in CGS (factor-of-100 on lengths cancels)
    r_g_cgs = system.gravitational_radius_m * 100.0
    a_cgs = system.semi_major_axis_m * 100.0
    Xi_cgs = asymmetry_parameter(system.eccentricity, r_g_cgs, a_cgs)

    # Compute in AU (custom length unit)
    r_g_au = system.gravitational_radius_m / AU_SI
    a_au = system.semi_major_axis_m / AU_SI
    Xi_au = asymmetry_parameter(system.eccentricity, r_g_au, a_au)

    # Compute in Planck lengths
    L_PLANCK = 1.616e-35
    r_g_pl = system.gravitational_radius_m / L_PLANCK
    a_pl = system.semi_major_axis_m / L_PLANCK
    Xi_pl = asymmetry_parameter(system.eccentricity, r_g_pl, a_pl)

    return {
        "system": system.name,
        "Xi_SI": Xi_si,
        "Xi_CGS": Xi_cgs,
        "Xi_AU": Xi_au,
        "Xi_Planck": Xi_pl,
        "max_relative_deviation": max(
            abs(Xi_cgs - Xi_si),
            abs(Xi_au - Xi_si),
            abs(Xi_pl - Xi_si),
        ) / (abs(Xi_si) + 1e-300),
        "consistent": np.allclose([Xi_si], [Xi_cgs, Xi_au, Xi_pl], rtol=1e-12),
    }


# ============================================================================
# 9. MAIN: REPRODUCE ALL ARTICLE RESULTS
# ============================================================================

def main():
    """
    Reproduce the numerical results presented in the article.
    Prints Tables I and II and the observational bounds.
    """
    print("=" * 76)
    print("COHERENCE-BASED SCALAR-TENSOR EXTENSION OF GENERAL RELATIVITY")
    print("Computational verification of article results")
    print("=" * 76)

    # --- Build catalogue ---
    catalogue = build_catalogue()

    # ----------------------------------------------------------------
    # TABLE I: Observational bounds on λ_eff
    # ----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("TABLE I — Observational bounds on λ_eff")
    print("-" * 76)
    print(f"{'System':<20} {'e':>6} {'Ξ':>14} {'|δ|_max':>12} {'λ_eff^max':>12}")
    print("-" * 76)

    bounds = compute_all_bounds(catalogue)
    for name in ["Mercury", "Venus", "Earth", "Mars",
                  "PSR B1913+16", "PSR J0737-3039"]:
        sys = catalogue[name]
        Xi = sys.asymmetry_parameter
        bnd = bounds.get(name)
        d_max = sys.precision_ppm * 1e-6 if sys.precision_ppm else None
        print(
            f"{name:<20} {sys.eccentricity:>6.4f} "
            f"{Xi:>14.3e} "
            f"{d_max:>12.1e} " if d_max else f"{'—':>12} ",
            f"{bnd:>12.2f}" if bnd else f"{'—':>12}",
        )

    best_name, best_bound = tightest_bound(bounds)
    print("-" * 76)
    print(f"Tightest bound: λ_eff < {best_bound:.2f}  (from {best_name})")

    # ----------------------------------------------------------------
    # TABLE II: Predictions at the tightest bound
    # ----------------------------------------------------------------
    print("\n" + "-" * 76)
    print(f"TABLE II — Predictions with λ_eff = {best_bound:.2f}")
    print("-" * 76)
    print(
        f"{'System':<22} {'e':>6} {'r_g/a':>12} "
        f"{'Ξ':>14} {'δ':>14} {'δ (%)':>10}"
    )
    print("-" * 76)

    predictions = predict_corrections(catalogue, best_bound)
    for p in predictions:
        print(
            f"{p['name']:<22} {p['eccentricity']:>6.4f} "
            f"{p['compactness']:>12.3e} "
            f"{p['Xi']:>14.3e} {p['delta']:>14.3e} "
            f"{p['delta_percent']:>9.4f}%"
        )

    # ----------------------------------------------------------------
    # DIMENSIONAL CONSISTENCY CHECK
    # ----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("DIMENSIONAL CONSISTENCY VERIFICATION")
    print("Ξ must be identical in any unit system")
    print("-" * 76)
    print(f"{'System':<20} {'Ξ (SI)':>14} {'Ξ (CGS)':>14} {'Ξ (AU)':>14} {'Ξ (Planck)':>14} {'OK':>4}")
    print("-" * 76)

    for name in ["Mercury", "PSR B1913+16", "S2"]:
        v = verify_dimensional_consistency(catalogue[name])
        ok = "✓" if v["consistent"] else "✗"
        print(
            f"{name:<20} {v['Xi_SI']:>14.6e} {v['Xi_CGS']:>14.6e} "
            f"{v['Xi_AU']:>14.6e} {v['Xi_Planck']:>14.6e} {ok:>4}"
        )

    # ----------------------------------------------------------------
    # S2 STAR: KEY PREDICTION
    # ----------------------------------------------------------------
    s2 = catalogue["S2"]
    Xi_s2 = s2.asymmetry_parameter
    delta_s2 = fractional_correction(best_bound, Xi_s2)

    print("\n" + "-" * 76)
    print("KEY PREDICTION: S2 STAR AT GALACTIC CENTER")
    print("-" * 76)
    print(f"  Eccentricity:          e  = {s2.eccentricity}")
    print(f"  Semi-major axis:       a  = {s2.semi_major_axis_au:.1f} AU")
    print(f"  Central mass:          M  = {s2.central_mass_kg/M_SUN:.1e} M_sun")
    print(f"  Gravitational radius:  r_g = {s2.gravitational_radius_m:.3e} m")
    print(f"  Compactness:           r_g/a = {s2.compactness:.3e}")
    print(f"  Asymmetry parameter:   Ξ  = {Xi_s2:.3e}")
    print(f"  Predicted correction:  δ  = {delta_s2:.3e}  ({delta_s2*100:.2f}%)")
    print(f"  Status: {'Within reach of GRAVITY+/ELT' if delta_s2 > 1e-4 else 'Below current sensitivity'}")

    # ----------------------------------------------------------------
    # COMPARISON WITH ORIGINAL (FLAWED) FORMULA
    # ----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("COMPARISON: REVISED vs ORIGINAL FORMULA")
    print("Demonstrates that the original α = 1 + k·(e/a) is unit-dependent")
    print("-" * 76)

    k_old = 0.0881
    for name in ["Mercury", "PSR B1913+16", "S2"]:
        sys = catalogue[name]
        # Original formula with a in AU (as used in the old article)
        alpha_au = 1.0 + k_old * (sys.eccentricity / sys.semi_major_axis_au)
        # Original formula with a in meters (SHOULD give same result if covariant)
        alpha_m = 1.0 + k_old * (sys.eccentricity / sys.semi_major_axis_m)
        # New formula (unit-independent)
        Xi = sys.asymmetry_parameter
        delta_new = fractional_correction(best_bound, Xi)

        print(f"\n  {name}:")
        print(f"    OLD α (a in AU):     {alpha_au:.10f}")
        print(f"    OLD α (a in meters): {alpha_m:.15f}")
        print(f"    UNIT DISCREPANCY:    {abs(alpha_au - alpha_m):.6e}")
        print(f"    NEW δ (any units):   {delta_new:.10e}  [unit-independent ✓]")

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"""
  Framework: Scalar-tensor theory with non-minimal coupling λΦR
  Action:    S = ∫d⁴x√(-g)[(1+λΦ)R − ω/2(∂Φ)² − V(Φ)]/(16πG) + S_m
  GR limit:  Exact when Φ → 0 (C_μν → 0)

  Key equation:
    Δφ = Δφ_GR · (1 + λ_eff² · Ξ)
    where Ξ = e²/(1−e²) · r_g/a  [dimensionless, unit-independent]

  Observational bound:
    λ_eff < {best_bound:.2f}  (from {best_name}, 95% C.L.)

  Predictions (at bound):
    Mercury:      δ = {fractional_correction(best_bound, catalogue['Mercury'].asymmetry_parameter):.2e}  (undetectable)
    PSR B1913+16: δ = {fractional_correction(best_bound, catalogue['PSR B1913+16'].asymmetry_parameter):.2e}  (at measurement limit)
    S2:           δ = {fractional_correction(best_bound, catalogue['S2'].asymmetry_parameter):.2e}  (within reach of GRAVITY+)
    Inner S-star: δ = {fractional_correction(best_bound, catalogue['Inner S-star (hyp.)'].asymmetry_parameter):.2e}  (decisive test)
    """)
    print("=" * 76)


if __name__ == "__main__":
    main()
