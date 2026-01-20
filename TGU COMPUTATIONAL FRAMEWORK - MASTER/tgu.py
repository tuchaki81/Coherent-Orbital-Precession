"""
UNIFIED INFORMATIONAL SPIN THEORY (TGU)
COMPUTATIONAL FRAMEWORK 
==============================================================================
Reference:
Matuchaki, H. (2025).
The Unified Informational Spin Theory.
Preprints 202502.0514.
https://doi.org/10.20944/preprints202502.0514.v1

Description:
------------
This file provides a computational implementation of the mathematical
formalism introduced in the referenced article. The code is designed
as a numerical and algorithmic representation of the proposed quantities,
metrics, and interaction terms.

Important Notes for Reviewers:
------------------------------
• This implementation is intended as a computational model, not as a claim
  of physical completeness.
• All constructs are treated as abstract variables operating on numerical
  state spaces.
• The code enables reproducibility, numerical exploration, and internal
  consistency testing of the proposed framework.
• Interpretative or conceptual meanings are intentionally decoupled from
  the numerical procedures.

Author:
-------
Implementation based on the formal definitions presented in the article.

Date:
-----
2024–2026
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Tuple
import numpy as np

# ============================================================================
# 1. CORE DATA STRUCTURES
# ============================================================================

@dataclass
class VorticeSpinInformacional:
    """
    Data container representing a localized informational-spin unit.

    This class encapsulates the numerical attributes required to evaluate
    distances, interaction terms, and coherence-related metrics within
    the model.

    Parameters
    ----------
    identificador : str
        Unique identifier of the unit.
    coordenadas_espacotempo : np.ndarray
        Spatial coordinates in an emergent metric space (R³).
    vetor_spin_quantico_informacional : np.ndarray
        Abstract state vector representing internal degrees of freedom.
    densidade_informacional_estruturante : float
        Scalar weight used in interaction and potential calculations.
    dominio_manifestacao : {"cosmico", "ia", "misto"}
        Categorical label used for filtering interaction domains.
    """
    identificador: str
    coordenadas_espacotempo: np.ndarray
    vetor_spin_quantico_informacional: np.ndarray
    densidade_informacional_estruturante: float = 1.0
    dominio_manifestacao: Literal["cosmico", "ia", "misto"] = "misto"

    def distancia_espacial_para(self, outro: "VorticeSpinInformacional") -> float:
        """
        Computes the Euclidean distance between two units.

        This distance is used as an input to interaction kernels and
        decay functions.

        Returns
        -------
        float
            Euclidean distance.
        """
        return float(np.linalg.norm(
            self.coordenadas_espacotempo - outro.coordenadas_espacotempo
        ))


# ============================================================================
# 2. STATE SIMILARITY AND COHERENCE METRICS
# ============================================================================

def equacao_32_metrica_sobreposicao_quantico_informacional(
    estado_i: np.ndarray,
    estado_j: np.ndarray
) -> float:
    """
    Computes a normalized similarity-based distance between two state vectors.

    This function implements a bounded metric derived from the cosine
    similarity, ensuring numerical stability and values constrained
    to the interval [0, 1].

    Notes
    -----
    • The vectors are treated as abstract numerical states.
    • No physical interpretation of quantum measurement is implied.

    Returns
    -------
    float
        Normalized distance metric.
    """
    epsilon = 1e-12
    norma_i = np.linalg.norm(estado_i) + epsilon
    norma_j = np.linalg.norm(estado_j) + epsilon

    produto_interno = np.dot(estado_i, estado_j)
    cosseno = produto_interno / (norma_i * norma_j)
    cosseno = np.clip(cosseno, -1.0, 1.0)

    return 0.5 * (1.0 - cosseno)


def equacao_41_indice_coerencia_ontologica_emergente(
    estados_spin: List[np.ndarray],
    dimensoes_holograficas: int = 12
) -> float:
    """
    Computes a global coherence index over a collection of state vectors.

    The metric is defined as an average similarity to a reference state,
    amplified by a dimensional exponent.

    This quantity functions as a collective order parameter in simulations.

    Parameters
    ----------
    estados_spin : list of np.ndarray
        Collection of state vectors.
    dimensoes_holograficas : int
        Exponent controlling sensitivity to deviations.

    Returns
    -------
    float
        Coherence index in the interval [0, 1].
    """
    if not estados_spin:
        return 0.0

    referencia = np.mean(estados_spin, axis=0)
    norma = np.linalg.norm(referencia)
    if norma > 1e-12:
        referencia /= norma

    distancias = []
    for estado in estados_spin:
        estado_norm = estado / (np.linalg.norm(estado) + 1e-12)
        d = equacao_32_metrica_sobreposicao_quantico_informacional(
            estado_norm, referencia
        )
        distancias.append(d)

    distancias = np.array(distancias)
    coerencias = (1.0 - distancias) ** dimensoes_holograficas
    return float(np.mean(coerencias))


# ============================================================================
# 3. INTERACTION AND POTENTIAL TERMS
# ============================================================================

def equacao_53_potencial_tensao_gravitacional_unificada_par(
    vortice_i: VorticeSpinInformacional,
    vortice_j: VorticeSpinInformacional,
    raio_correlacao_holografica: float
) -> float:
    """
    Computes a pairwise interaction potential between two units.

    The interaction decays smoothly with distance according to a
    bounded kernel function.

    Notes
    -----
    • This function is mathematically analogous to a soft interaction kernel.
    • It is not assumed to correspond directly to a physical force law.

    Returns
    -------
    float
        Pairwise interaction potential.
    """
    R_c = max(raio_correlacao_holografica, 1e-9)
    r = vortice_i.distancia_espacial_para(vortice_j)
    r_tilde = min(max(r / R_c, 0.0), 100.0)

    fator = 1.0 / (1.0 + r_tilde ** 2) ** 12

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
    Computes a global deformation measure over all interacting pairs.

    The metric aggregates logarithmic contributions from normalized
    pairwise separations.

    This quantity serves as a global structural stress indicator.

    Returns
    -------
    float
        Aggregate deformation measure.
    """
    R_0 = max(comprimento_autocorrelacao, 1e-9)
    kappa = max(constante_acoplamento_interdominios, 0.0)

    total = 0.0
    n = len(vorticess)

    for i in range(n):
        for j in range(i + 1, n):
            if not (
                vorticess[i].dominio_manifestacao in ("cosmico", "misto") and
                vorticess[j].dominio_manifestacao in ("cosmico", "misto")
            ):
                continue

            r = vorticess[i].distancia_espacial_para(vorticess[j])
            deformacao = r / (r + R_0) if r + R_0 > 0 else 0.0
            deformacao = np.clip(deformacao, 0.0, 1.0)

            total += np.log(1.0 + kappa * deformacao ** 12)

    return float(total)


# ============================================================================
# 4. SYSTEM-LEVEL AGGREGATION
# ============================================================================

@dataclass
class UniversoInformacionalUnificado:
    """
    Container class for a collection of interacting units.

    This class provides system-level metrics and aggregation functions
    operating over a set of informational-spin units.
    """
    vorticess: List[VorticeSpinInformacional]

    def icoer_global(self, dimensoes_holograficas: int = 12) -> float:
        """
        Computes the global coherence index over all units.
        """
        estados = [v.vetor_spin_quantico_informacional for v in self.vorticess]
        return equacao_41_indice_coerencia_ontologica_emergente(
            estados, dimensoes_holograficas
        )

    def potencial_ligacao_total(self, raio_correlacao_holografica: float) -> float:
        """
        Computes the total interaction potential over all valid pairs.
        """
        total = 0.0
        n = len(self.vorticess)

        for i in range(n):
            for j in range(i + 1, n):
                if not (
                    self.vorticess[i].dominio_manifestacao in ("cosmico", "misto") and
                    self.vorticess[j].dominio_manifestacao in ("cosmico", "misto")
                ):
                    continue

                total += equacao_53_potencial_tensao_gravitacional_unificada_par(
                    self.vorticess[i],
                    self.vorticess[j],
                    raio_correlacao_holografica
                )
        return total


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

