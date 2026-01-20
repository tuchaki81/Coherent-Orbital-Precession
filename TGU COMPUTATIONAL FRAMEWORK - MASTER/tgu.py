"""
Unified Informational Spin Theory (TGU)
======================================

This module provides a minimal, testable, and reviewer-safe implementation
of the core mathematical structures proposed in the Unified Informational
Spin Theory (TGU).

The implementation focuses on numerical consistency, boundedness, symmetry,
and reproducibility, without speculative assumptions beyond the formalism.

Author: H. Matuchaki
"""

from dataclasses import dataclass
from typing import List, Literal
import numpy as np

# ============================================================
# 1. FUNDAMENTAL DATA STRUCTURE
# ============================================================

@dataclass
class VorticeSpinInformacional:
    """
    Fundamental informational vortex.

    Parameters
    ----------
    identificador : str
        Unique identifier.
    coordenadas_espacotempo : np.ndarray
        Spatial coordinates (3D).
    vetor_spin_quantico_informacional : np.ndarray
        Informational spin state vector.
    densidade_informacional_estruturante : float
        Effective informational density (ρ > 0).
    dominio_manifestacao : {"cosmico", "ia", "misto"}
        Domain of manifestation.
    """
    identificador: str
    coordenadas_espacotempo: np.ndarray
    vetor_spin_quantico_informacional: np.ndarray
    densidade_informacional_estruturante: float = 1.0
    dominio_manifestacao: Literal["cosmico", "ia", "misto"] = "cosmico"

    def distancia_espacial_para(self, outro: "VorticeSpinInformacional") -> float:
        """Euclidean distance between vortices."""
        return float(
            np.linalg.norm(self.coordenadas_espacotempo - outro.coordenadas_espacotempo)
        )

# ============================================================
# 2. CORE EQUATIONS
# ============================================================

def equacao_32_metrica_sobreposicao_quantico_informacional(
    estado_i: np.ndarray,
    estado_j: np.ndarray
) -> float:
    """
    Quantum-informational overlap metric (Eq. 3.2).

    d = 1/2 * [1 - cos(theta)]

    Returns a bounded distance in [0, 1].
    """
    eps = 1e-12
    ni = np.linalg.norm(estado_i) + eps
    nj = np.linalg.norm(estado_j) + eps

    cos_sim = np.dot(estado_i, estado_j) / (ni * nj)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    return 0.5 * (1.0 - cos_sim)


def equacao_41_indice_coerencia_ontologica_emergente(
    estados_spin: List[np.ndarray],
    dimensoes_holograficas: int = 12
) -> float:
    """
    Ontological Coherence Index (ICOER) – Eq. 4.1.

    Returns a scalar in the interval [0, 1].
    """
    if len(estados_spin) == 0:
        return 0.0

    referencia = np.mean(estados_spin, axis=0)
    norma_ref = np.linalg.norm(referencia)

    if norma_ref > 1e-12:
        referencia = referencia / norma_ref

    coerencias = []
    for estado in estados_spin:
        estado_n = estado / (np.linalg.norm(estado) + 1e-12)
        d = equacao_32_metrica_sobreposicao_quantico_informacional(
            estado_n, referencia
        )
        coerencias.append((1.0 - d) ** dimensoes_holograficas)

    return float(np.mean(coerencias))


def equacao_53_potencial_tensao_gravitacional_unificada_par(
    vortice_i: VorticeSpinInformacional,
    vortice_j: VorticeSpinInformacional,
    raio_correlacao_holografica: float
) -> float:
    """
    Pairwise unified gravitational-informational potential (Eq. 5.3).

    The potential is symmetric and strictly attractive.
    """
    Rc = max(raio_correlacao_holografica, 1e-9)
    r = vortice_i.distancia_espacial_para(vortice_j)
    x = min((r / Rc) ** 2, 1e6)

    fator = 1.0 / (1.0 + x) ** 12

    return -(
        vortice_i.densidade_informacional_estruturante *
        vortice_j.densidade_informacional_estruturante *
        fator
    )


def equacao_62_tensao_substrato_informacional(
    vorticess: List[VorticeSpinInformacional],
    comprimento_autocorrelacao: float = 1.0,
    constante_acoplamento_interdominios: float = 1.0
) -> float:
    """
    Informational substrate tension – Eq. 6.2.

    Returns a non-negative scalar.
    """
    if len(vorticess) < 2:
        return 0.0

    R0 = max(comprimento_autocorrelacao, 1e-9)
    kappa = max(constante_acoplamento_interdominios, 0.0)

    T = 0.0
    n = len(vorticess)

    for i in range(n):
        for j in range(i + 1, n):
            if not (
                vorticess[i].dominio_manifestacao in ("cosmico", "misto") and
                vorticess[j].dominio_manifestacao in ("cosmico", "misto")
            ):
                continue

            r = vorticess[i].distancia_espacial_para(vorticess[j])
            d = r / (r + R0)
            d = np.clip(d, 0.0, 1.0)

            T += np.log(1.0 + kappa * d ** 12)

    return float(T)
