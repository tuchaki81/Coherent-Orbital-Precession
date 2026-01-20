import numpy as np
import pytest

from tgu import (
    equacao_32_metrica_sobreposicao_quantico_informacional,
    equacao_41_indice_coerencia_ontologica_emergente,
    equacao_53_potencial_tensao_gravitacional_unificada_par,
    equacao_62_tensao_substrato_informacional,
    VorticeSpinInformacional
)

# ============================================================
# METRIC TESTS
# ============================================================

def test_metrica_identicos():
    estado = np.array([1.0, 0.0, 0.0])
    d = equacao_32_metrica_sobreposicao_quantico_informacional(estado, estado)
    assert abs(d) < 1e-10


def test_metrica_ortogonais():
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    d = equacao_32_metrica_sobreposicao_quantico_informacional(e1, e2)
    assert abs(d - 0.5) < 1e-10


# ============================================================
# ICOER TESTS
# ============================================================

def test_icoer_intervalo():
    estados = [np.random.randn(8) for _ in range(10)]
    icoer = equacao_41_indice_coerencia_ontologica_emergente(estados)
    assert 0.0 <= icoer <= 1.0


def test_icoer_maximo():
    estado = np.ones(6)
    estados = [estado.copy() for _ in range(5)]
    icoer = equacao_41_indice_coerencia_ontologica_emergente(estados)
    assert icoer > 0.99


# ============================================================
# POTENTIAL TESTS
# ============================================================

def test_potencial_simetria():
    v1 = VorticeSpinInformacional("v1", np.zeros(3), np.ones(5))
    v2 = VorticeSpinInformacional("v2", np.ones(3), np.ones(5))

    V12 = equacao_53_potencial_tensao_gravitacional_unificada_par(v1, v2, 10.0)
    V21 = equacao_53_potencial_tensao_gravitacional_unificada_par(v2, v1, 10.0)

    assert abs(V12 - V21) < 1e-12


def test_potencial_negativo():
    v1 = VorticeSpinInformacional("v1", np.zeros(3), np.ones(3))
    v2 = VorticeSpinInformacional("v2", np.ones(3), np.ones(3))

    V = equacao_53_potencial_tensao_gravitacional_unificada_par(v1, v2, 5.0)
    assert V < 0.0


# ============================================================
# SUBSTRATE TENSION TESTS
# ============================================================

def test_tensao_nao_negativa():
    vorticess = [
        VorticeSpinInformacional(str(i), np.random.randn(3), np.random.randn(4))
        for i in range(5)
    ]
    T = equacao_62_tensao_substrato_informacional(vorticess)
    assert T >= 0.0


def test_tensao_um_vortice():
    v = VorticeSpinInformacional("v", np.zeros(3), np.ones(4))
    T = equacao_62_tensao_substrato_informacional([v])
    assert abs(T) < 1e-12
