"""
Test suite for coherent_orbital_precession.py

Verifies:
  - Dimensional consistency (Ξ is unit-independent)
  - Physical limits (δ → 0 for e → 0, GR recovery for λ → 0)
  - Observational bounds from catalogue systems
  - Numerical agreement with article Tables I and II
  - Symmetry and edge-case behavior

Run:  pytest test_coherent_orbital_precession.py -v
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from coherent_orbital_precession import (
    # Constants
    G_SI, C_SI, M_SUN, AU_SI,
    # Data
    OrbitalSystem,
    build_catalogue,
    # Core equations
    asymmetry_parameter,
    fractional_correction,
    corrected_precession,
    gr_precession_per_orbit,
    coherence_tensor,
    # Bounds and predictions
    lambda_bound_from_system,
    compute_all_bounds,
    tightest_bound,
    predict_corrections,
    # Utilities
    phi_static_spherical,
    effective_ppn_gamma,
    verify_dimensional_consistency,
)


# ============================================================
# DIMENSIONAL CONSISTENCY TESTS
# ============================================================

class TestDimensionalConsistency:
    """Ξ = e²/(1-e²) · r_g/a must be unit-independent."""

    def test_xi_unit_independent(self):
        catalogue = build_catalogue()
        for name in ["Mercury", "PSR B1913+16", "S2"]:
            result = verify_dimensional_consistency(catalogue[name])
            assert result["consistent"], (
                f"{name}: Ξ differs across unit systems — "
                f"max deviation = {result['max_relative_deviation']:.2e}"
            )

    def test_xi_si_equals_cgs(self):
        """r_g and a both scale by 100 in CGS → ratio unchanged."""
        e, r_g, a = 0.5, 3000.0, 1e11
        xi_si = asymmetry_parameter(e, r_g, a)
        xi_cgs = asymmetry_parameter(e, r_g * 100, a * 100)
        assert abs(xi_si - xi_cgs) < 1e-15

    def test_xi_arbitrary_scale(self):
        """Multiplying both lengths by any factor leaves Ξ unchanged."""
        e, r_g, a = 0.3, 5000.0, 2e10
        xi_base = asymmetry_parameter(e, r_g, a)
        for factor in [1e-10, 0.001, 7.77, 1e6, 1e20]:
            xi_scaled = asymmetry_parameter(e, r_g * factor, a * factor)
            assert abs(xi_base - xi_scaled) / (abs(xi_base) + 1e-300) < 1e-12


# ============================================================
# PHYSICAL LIMIT TESTS
# ============================================================

class TestPhysicalLimits:
    """Theory must satisfy known physical constraints."""

    def test_circular_orbit_no_correction(self):
        """e = 0 ⟹ Ξ = 0 ⟹ δ = 0: no correction for circular orbits."""
        xi = asymmetry_parameter(0.0, 3000.0, 1e11)
        assert xi == 0.0

    def test_gr_recovery_lambda_zero(self):
        """λ_eff = 0 ⟹ δ = 0: exact GR recovery."""
        delta = fractional_correction(lambda_eff=0.0, Xi=1e-4)
        assert delta == 0.0

    def test_corrected_precession_reduces_to_gr(self):
        """When δ = 0, corrected precession equals GR precession."""
        prec_gr = 42.98
        prec_tgu = corrected_precession(prec_gr, lambda_eff=0.0, Xi=1e-4)
        assert prec_tgu == prec_gr

    def test_xi_positive_for_nonzero_eccentricity(self):
        """Ξ > 0 for any e > 0 and finite a."""
        xi = asymmetry_parameter(0.01, 3000.0, 1e14)
        assert xi > 0.0

    def test_delta_positive(self):
        """δ = λ² · Ξ ≥ 0 always (squared coupling)."""
        delta = fractional_correction(lambda_eff=-3.0, Xi=1e-4)
        assert delta > 0.0

    def test_xi_increases_with_eccentricity(self):
        """Ξ is monotonically increasing in e for fixed r_g/a."""
        r_g, a = 3000.0, 1e11
        xi_prev = 0.0
        for e in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            xi = asymmetry_parameter(e, r_g, a)
            assert xi > xi_prev
            xi_prev = xi

    def test_xi_increases_with_compactness(self):
        """Ξ is monotonically increasing in r_g/a for fixed e."""
        e = 0.5
        xi_prev = 0.0
        for ratio in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            a = 1.0  # arbitrary
            r_g = ratio * a
            xi = asymmetry_parameter(e, r_g, a)
            assert xi > xi_prev
            xi_prev = xi


# ============================================================
# COHERENCE TENSOR TESTS
# ============================================================

class TestCoherenceTensor:
    """C_μν = λ(∇_μ∇_ν Φ − g_μν □Φ) must vanish when Φ → 0."""

    def test_vanishes_for_zero_field(self):
        """C_μν = 0 when all field derivatives are zero."""
        grad_grad = np.zeros((4, 4))
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        C = coherence_tensor(grad_grad, box_phi=0.0, metric=metric, lambda_coupling=1.0)
        assert np.allclose(C, 0.0)

    def test_vanishes_for_zero_coupling(self):
        """C_μν = 0 when λ = 0 regardless of field."""
        grad_grad = np.random.randn(4, 4)
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        C = coherence_tensor(grad_grad, box_phi=1.5, metric=metric, lambda_coupling=0.0)
        assert np.allclose(C, 0.0)

    def test_linearity_in_lambda(self):
        """C_μν scales linearly with λ."""
        grad_grad = np.random.randn(4, 4)
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        C1 = coherence_tensor(grad_grad, 0.5, metric, lambda_coupling=1.0)
        C2 = coherence_tensor(grad_grad, 0.5, metric, lambda_coupling=3.0)
        assert np.allclose(C2, 3.0 * C1)


# ============================================================
# GR PRECESSION FORMULA TESTS
# ============================================================

class TestGRPrecession:
    """Verify Δφ_GR = 6πGM / [c² a(1-e²)]."""

    def test_mercury_precession_per_orbit(self):
        """Mercury: ~0.1035 arcsec/orbit (known value)."""
        prec_rad = gr_precession_per_orbit(M_SUN, 0.3871 * AU_SI, 0.2056)
        prec_arcsec = prec_rad * (180 / np.pi) * 3600
        # Mercury: ~0.1035 arcsec/orbit
        assert abs(prec_arcsec - 0.1035) < 0.002

    def test_precession_decreases_with_distance(self):
        """More distant orbits have smaller precession."""
        prec_inner = gr_precession_per_orbit(M_SUN, 0.4 * AU_SI, 0.2)
        prec_outer = gr_precession_per_orbit(M_SUN, 1.0 * AU_SI, 0.2)
        assert prec_inner > prec_outer

    def test_precession_increases_with_eccentricity(self):
        """Higher eccentricity → larger precession (via 1-e² denominator)."""
        prec_low = gr_precession_per_orbit(M_SUN, 1.0 * AU_SI, 0.01)
        prec_high = gr_precession_per_orbit(M_SUN, 1.0 * AU_SI, 0.9)
        assert prec_high > prec_low


# ============================================================
# CATALOGUE AND BOUNDS TESTS
# ============================================================

class TestCatalogue:
    """Verify the astrophysical catalogue and derived bounds."""

    def test_catalogue_completeness(self):
        """All systems from the article are present."""
        catalogue = build_catalogue()
        expected = [
            "Mercury", "Venus", "Earth", "Mars", "Icarus",
            "PSR B1913+16", "PSR J0737-3039", "S2", "HD 80606b",
            "Inner S-star (hyp.)",
        ]
        for name in expected:
            assert name in catalogue, f"Missing system: {name}"

    def test_mercury_asymmetry_parameter(self):
        """Ξ(Mercury) ≈ 2.25 × 10⁻⁹ (Table I)."""
        cat = build_catalogue()
        xi = cat["Mercury"].asymmetry_parameter
        assert abs(xi - 2.25e-9) / 2.25e-9 < 0.01  # 1% tolerance

    def test_psr_b1913_asymmetry_parameter(self):
        """Ξ(PSR B1913+16) ≈ 2.64 × 10⁻⁶ (Table I)."""
        cat = build_catalogue()
        xi = cat["PSR B1913+16"].asymmetry_parameter
        assert abs(xi - 2.64e-6) / 2.64e-6 < 0.01

    def test_s2_asymmetry_parameter(self):
        """Ξ(S2) ≈ 2.63 × 10⁻⁴ (Table II)."""
        cat = build_catalogue()
        xi = cat["S2"].asymmetry_parameter
        assert abs(xi - 2.63e-4) / 2.63e-4 < 0.01

    def test_tightest_bound_is_psr_b1913(self):
        """PSR B1913+16 must provide the tightest bound on λ_eff."""
        bounds = compute_all_bounds(build_catalogue())
        name, value = tightest_bound(bounds)
        assert name == "PSR B1913+16"

    def test_lambda_bound_value(self):
        """λ_eff < 1.95 from PSR B1913+16 (Eq. 12)."""
        bounds = compute_all_bounds(build_catalogue())
        _, value = tightest_bound(bounds)
        assert abs(value - 1.95) < 0.01

    def test_bound_returns_none_without_precision(self):
        """Systems without precision_ppm yield no bound."""
        sys = OrbitalSystem("test", 0.5, 1e11, M_SUN, precision_ppm=None)
        assert lambda_bound_from_system(sys) is None


# ============================================================
# PREDICTIONS TESTS
# ============================================================

class TestPredictions:
    """Verify article Table II predictions."""

    def test_mercury_correction_negligible(self):
        """δ(Mercury) < 10⁻⁷ at λ_eff = 1.95."""
        cat = build_catalogue()
        delta = fractional_correction(1.95, cat["Mercury"].asymmetry_parameter)
        assert delta < 1e-7

    def test_s2_correction_order_of_magnitude(self):
        """δ(S2) ≈ 10⁻³ at λ_eff = 1.95."""
        cat = build_catalogue()
        delta = fractional_correction(1.95, cat["S2"].asymmetry_parameter)
        assert 5e-4 < delta < 5e-3

    def test_inner_sstar_correction_percent_level(self):
        """δ(inner S-star) ≈ 2.8% at λ_eff = 1.95."""
        cat = build_catalogue()
        delta = fractional_correction(1.95, cat["Inner S-star (hyp.)"].asymmetry_parameter)
        assert 0.01 < delta < 0.10

    def test_predictions_sorted_by_xi(self):
        """predict_corrections returns systems sorted by decreasing Ξ."""
        cat = build_catalogue()
        preds = predict_corrections(cat, 1.95)
        xi_values = [p["Xi"] for p in preds]
        assert xi_values == sorted(xi_values, reverse=True)

    def test_corrected_precession_exceeds_gr(self):
        """Δφ_TGU ≥ Δφ_GR for any λ_eff and Ξ (since δ ≥ 0)."""
        prec_gr = 42.98
        prec_tgu = corrected_precession(prec_gr, lambda_eff=1.95, Xi=1e-4)
        assert prec_tgu >= prec_gr


# ============================================================
# SCALAR FIELD UTILITIES TESTS
# ============================================================

class TestScalarField:
    """Approximate Φ(r) solution and PPN γ."""

    def test_phi_decays_with_distance(self):
        """Φ(r) ∝ 1/r → decays at larger radii."""
        r = np.array([1e10, 1e11, 1e12])
        phi = phi_static_spherical(r, lambda_coupling=1.0, omega=40000.0, M_kg=M_SUN)
        assert np.all(np.diff(np.abs(phi)) < 0)

    def test_phi_proportional_to_mass(self):
        """Φ ∝ M at fixed r."""
        r = np.array([1e11])
        phi_1 = phi_static_spherical(r, 1.0, 40000.0, M_SUN)
        phi_2 = phi_static_spherical(r, 1.0, 40000.0, 2.0 * M_SUN)
        assert np.allclose(phi_2, 2.0 * phi_1)

    def test_ppn_gamma_approaches_one(self):
        """γ → 1 as ω → ∞ (GR limit)."""
        gamma = effective_ppn_gamma(lambda_coupling=1.0, omega=1e8)
        assert abs(gamma - 1.0) < 1e-7

    def test_ppn_gamma_cassini_bound(self):
        """ω > 50000 ⟹ |γ − 1| < 2.3 × 10⁻⁵ (Cassini bound)."""
        gamma = effective_ppn_gamma(lambda_coupling=1.0, omega=50000.0)
        assert abs(gamma - 1.0) < 2.3e-5


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    """Guard against division by zero and invalid inputs."""

    def test_eccentricity_near_one(self):
        """e → 1 gives large but finite Ξ."""
        xi = asymmetry_parameter(0.9999, 3000.0, 1e11)
        assert np.isfinite(xi)
        assert xi > 0

    def test_eccentricity_exactly_zero(self):
        """e = 0 gives Ξ = 0 exactly."""
        xi = asymmetry_parameter(0.0, 3000.0, 1e11)
        assert xi == 0.0

    def test_very_small_semi_major_axis(self):
        """Compact orbit gives large but finite Ξ."""
        xi = asymmetry_parameter(0.5, 1e4, 1e5)
        assert np.isfinite(xi)
        assert xi > 0

    def test_fractional_correction_large_lambda(self):
        """Large λ gives large but finite δ."""
        delta = fractional_correction(lambda_eff=1000.0, Xi=1e-4)
        assert np.isfinite(delta)
        assert delta > 0
