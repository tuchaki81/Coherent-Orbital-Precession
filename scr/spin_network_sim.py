"""
spin_network_sim.py

Computational derivation of the Matuchaki Parameter (k ≈ 0.0881) from the 
Unified Informational Spin Theory (TGU) using coupled spin networks.

Model: Transverse-field Ising chain (simplified 2-spin for MVP, scalable)
Hamiltonian: H = -J σ_z¹ σ_z² + h (σ_x¹ + σ_x²)
   - J modulated by informational strain: J = J_base × (1 + e/a)
   - h: transverse field for resonance/mixing

Goal: Optimize J_base and h so that a dominant |eigenvalue| emerges close to k.

Author: Henry Matuchaki (@MatuchakiSilva)
Date: January 2026
"""

import qutip as qt
import numpy as np
from scipy.optimize import minimize

# Target value: the Matuchaki Parameter from TGU paper
TARGET_K = 0.0881


def build_ising_hamiltonian(J: float, h: float, num_spins: int = 2) -> qt.Qobj:
    """
    Construct the transverse-field Ising Hamiltonian for num_spins=2.

    Parameters:
    -----------
    J : float
        Coupling strength (zz interaction)
    h : float
        Transverse field strength (xx term)
    num_spins : int, optional
        Number of spins (currently fixed at 2)

    Returns:
    --------
    qt.Qobj
        Hamiltonian operator
    """
    if num_spins != 2:
        raise NotImplementedError("Currently only 2-spin model is implemented. Extend for chains.")

    id = qt.qeye(2)
    sz = qt.sigmaz()
    sx = qt.sigmax()

    sz1 = qt.tensor(sz, id)
    sz2 = qt.tensor(id, sz)
    sx1 = qt.tensor(sx, id)
    sx2 = qt.tensor(id, sx)

    H = -J * sz1 * sz2 + h * (sx1 + sx2)
    return H


def extract_resonant_eigenvalue(J: float, h: float, target: float = TARGET_K) -> float:
    """
    Compute the eigenvalue closest to the target (k) from the absolute spectrum.

    Returns:
    --------
    float: Closest absolute eigenvalue
    """
    H = build_ising_hamiltonian(J, h)
    eigenenergies = H.eigenenergies()
    abs_eig = np.sort(np.abs(eigenenergies))
    closest_idx = np.argmin(np.abs(abs_eig - target))
    return abs_eig[closest_idx]


def objective_function(params: np.ndarray, strain: float, target: float = TARGET_K) -> float:
    """
    Objective for optimization: minimize |closest_eig - target|

    params: [J_base, h]
    """
    J_base, h = params
    J = J_base * (1 + strain)  # Modulate by informational strain e/a
    closest = extract_resonant_eigenvalue(J, h, target)
    return np.abs(closest - target)


def optimize_k_from_spin_hamiltonian(
    e: float,
    a: float,
    target_k: float = TARGET_K,
    initial_guess: tuple = (0.056, 0.0075),
    method: str = 'Nelder-Mead'
) -> tuple:
    """
    Optimize J_base and h to make the resonant eigenvalue match target_k.

    Parameters:
    -----------
    e : float
        Orbital eccentricity
    a : float
        Semi-major axis in AU
    target_k : float
        Desired value (Matuchaki Parameter ≈ 0.0881)
    initial_guess : tuple
        Starting (J_base, h)
    method : str
        SciPy minimize method

    Returns:
    --------
    tuple:
        - optimal_k (closest eigenvalue after optimization)
        - optimal_params [J_base, h]
        - all_abs_eigenvalues
    """
    if a <= 0 or e < 0 or e >= 1:
        raise ValueError("Invalid orbital parameters")

    strain = e / a

    res = minimize(
        objective_function,
        initial_guess,
        args=(strain, target_k),
        method=method,
        options={'maxiter': 1000, 'disp': False}
    )

    if not res.success:
        print("Warning: Optimization did not converge perfectly.")

    optimal_J_base, optimal_h = res.x
    optimal_J = optimal_J_base * (1 + strain)

    # Recompute final spectrum
    H_opt = build_ising_hamiltonian(optimal_J, optimal_h)
    eigenenergies = H_opt.eigenenergies()
    abs_eig_sorted = np.sort(np.abs(eigenenergies))

    closest_k = abs_eig_sorted[np.argmin(np.abs(abs_eig_sorted - target_k))]

    return closest_k, (optimal_J_base, optimal_h), abs_eig_sorted


# Example usage and quick test
if __name__ == "__main__":
    print("=== TGU Spin Network Derivation Example: Mercury ===\n")

    # Mercury parameters (from paper/standard JPL)
    e_mercury = 0.205630
    a_mercury = 0.387098

    optimal_k, optimal_params, eigenvalues = optimize_k_from_spin_hamiltonian(
        e=e_mercury,
        a=a_mercury,
        initial_guess=(0.0565, 0.0075)
    )

    J_base_opt, h_opt = optimal_params

    print(f"Strain (e/a)       : {e_mercury / a_mercury:.5f}")
    print(f"Optimal J_base     : {J_base_opt:.6f}")
    print(f"Optimal h          : {h_opt:.6f}")
    print(f"Absolute eigenvalues: {eigenvalues}")
    print(f"Emergent resonant eigenvalue (closest to k): {optimal_k:.6f}")
    print(f"Absolute difference from target: {abs(optimal_k - TARGET_K):.2e}")