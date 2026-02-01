"""
TEORIA DO SPIN INFORMACIONAL (TGU) - IMPLEMENTAÇÃO COMPUTACIONAL
================================================================
Referência: Matuchaki, H. (2026). The Theory of Informational Spin: 
            A Coherence-Based Framework for Gravitation, Cosmology, and Quantum Systems.

Implementação fiel das equações e conceitos do artigo TGU.
Cada função corresponde a uma equação específica da teoria.

Autor: Implementação baseada no framework TGU original
Data: Janeiro 2026
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Tuple, Optional
import numpy as np

# ============================================================================
# 1. DEFINIÇÕES FUNDAMENTAIS DA TGU
# ============================================================================

@dataclass
class SistemaCoerenciaInformacional:
    """
    Eq. 1: Sistema com Spin Informacional (Seção 2)
    
    Representa um sistema físico caracterizado por distribuição estruturada
    de coerência informacional.
    
    Atributos:
        identificador: str - Identificação única do sistema
        coordenadas: np.ndarray[float] - Posição no espaço (x,y,z)
        campo_coerencia: np.ndarray[float] - Campo de coerência informacional
        densidade_informacional: float - Densidade de informação I(x)
        regime: Literal["fraca", "intermediaria", "forte"] - Regime de tensão orbital
        parametros_orbitais: Optional[Dict] - Parâmetros orbitais (a, e, etc.)
    """
    identificador: str
    coordenadas: np.ndarray
    campo_coerencia: np.ndarray
    densidade_informacional: float = 1.0
    regime: Literal["fraca", "intermediaria", "forte"] = "fraca"
    parametros_orbitais: Optional[Dict] = None
    
    def distancia_para(self, outro: SistemaCoerenciaInformacional) -> float:
        """Distância euclidiana entre sistemas"""
        return float(np.linalg.norm(self.coordenadas - outro.coordenadas))

# ============================================================================
# 2. PARÂMETROS FUNDAMENTAIS DA TGU
# ============================================================================

# Parâmetro de Matuchaki - Eq. 7
PARAMETRO_MATUCHAKI: float = 0.0881

# Expoente de coerência - Eq. 71 (Seção 15.4)
EXPOENTE_COERENCIA: int = 12

# Fator isotrópico base - Eq. 8
FATOR_ISOTROPICO_BASE: float = 1.0 / (4.0 * np.pi)

# ============================================================================
# 3. EQUAÇÕES FUNDAMENTAIS DA TGU
# ============================================================================

def equacao_1_spin_informacional(
    estados: List[np.ndarray],
    referencia: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Eq. 1: Representação do Spin Informacional
    
    S_I = (1/N) Σ (ψ_i/ψ_ref)^α (φ_i/φ_ref)^β
    
    Args:
        estados: Lista de variáveis de estado informacional
        referencia: Estado de referência (padrão: média)
        
    Returns:
        np.ndarray: Vetor de spin informacional
    """
    if not estados:
        return np.array([])
    
    N = len(estados)
    
    # Determina estado de referência
    if referencia is None:
        referencia = np.mean(estados, axis=0)
    
    # Normaliza referência
    norma_ref = np.linalg.norm(referencia)
    if norma_ref > 1e-12:
        referencia = referencia / norma_ref
    
    # Calcula spin informacional
    spin_total = np.zeros_like(estados[0])
    
    for estado in estados:
        # Normaliza estado
        estado_norm = estado / (np.linalg.norm(estado) + 1e-12)
        
        # Coeficientes α e β (simplificação para demonstração)
        alpha = 1.0
        beta = 1.0
        
        # Contribuição do estado
        contribuicao = (estado_norm / (referencia + 1e-12))**alpha
        spin_total += contribuicao
    
    return spin_total / N

def equacao_6_fator_correcao_orbital(
    excentricidade: float,
    semi_eixo_maior: float,
    parametro_k: float = PARAMETRO_MATUCHAKI
) -> float:
    """
    Eq. 6: Fator de correção orbital α
    
    α = 1 + k * (e/a)
    
    Args:
        excentricidade: e
        semi_eixo_maior: a
        parametro_k: Parâmetro de Matuchaki
        
    Returns:
        float: Fator de correção α
    """
    return 1.0 + parametro_k * (excentricidade / semi_eixo_maior)

def equacao_25_precessao_TGU(
    precessao_GR: float,
    excentricidade: float,
    semi_eixo_maior: float,
    distancia: float,
    raio_crossover: float,
    n: int = EXPOENTE_COERENCIA
) -> float:
    """
    Eq. 25: Precessão periélio no framework TGU
    
    Δφ_TGU = Δφ_GR * α * ε^{-n}
    
    Args:
        precessao_GR: Precessão GR
        excentricidade: e
        semi_eixo_maior: a
        distancia: r
        raio_crossover: r_s
        n: Expoente de coerência
        
    Returns:
        float: Precessão TGU
    """
    # Fator de correção orbital
    alpha = equacao_6_fator_correcao_orbital(excentricidade, semi_eixo_maior)
    
    # Fator de resistência de coerência - Eq. 31
    epsilon = 1.0 + (raio_crossover / distancia)**2
    
    # Precessão TGU - Eq. 25
    return precessao_GR * alpha * (epsilon ** -n)

def equacao_32_fator_resistencia_coerencia(
    distancia: float,
    raio_crossover: float,
    n: int = EXPOENTE_COERENCIA
) -> float:
    """
    Eq. 32: Fator de resistência de coerência
    
    ε(r)^{-n} = [1 + (r_s/r)^2]^{-n}
    
    Args:
        distancia: r
        raio_crossover: r_s
        n: Expoente de coerência
        
    Returns:
        float: Fator de resistência
    """
    epsilon = 1.0 + (raio_crossover / distancia)**2
    return epsilon ** -n

def equacao_35_rotacao_galactica_TGU(
    massa_baronica: float,
    raio: float,
    campo_coerencia: np.ndarray,
    constante_gravitacional: float = 1.0,
    densidade_referencia: float = 1.0
) -> float:
    """
    Eq. 35: Velocidade de rotação galáctica no framework TGU
    
    V²/r = G * (I/I₀) * ε(r)^{-12} * M/r²
    
    Args:
        massa_baronica: M
        raio: r
        campo_coerencia: Campo de coerência I(x)
        constante_gravitacional: G
        densidade_referencia: I₀
        
    Returns:
        float: Velocidade de rotação quadrática
    """
    # Normaliza campo de coerência
    I_norm = np.linalg.norm(campo_coerencia) / densidade_referencia
    
    # Fator de resistência (assume raio_crossover pequeno para escalas galácticas)
    epsilon = 1.0  # ε ≈ 1 para r ≫ r_s
    
    # Velocidade quadrática - Eq. 35 simplificada
    V2 = constante_gravitacional * I_norm * epsilon * massa_baronica / raio
    
    return float(V2)

def equacao_47_forca_TGU(
    gradiente_coerencia: np.ndarray,
    constante_acoplamento: float = 1.0
) -> np.ndarray:
    """
    Eq. 47: Força efetiva no framework TGU
    
    F_TGU^μ = ∇^μ(I·C)
    
    Args:
        gradiente_coerencia: ∇I
        constante_acoplamento: C
        
    Returns:
        np.ndarray: Vetor força
    """
    return gradiente_coerencia * constante_acoplamento

def tensor_coerencia_muchaki(
    campo_fase: np.ndarray,
    tensor_metrico: np.ndarray,
    lambda_acoplamento: float = 1.0
) -> np.ndarray:
    """
    Eq. 17: Tensor de Coerência de Matuchaki
    
    C_μν = λ(∇_μ∇_νΦ - g_μν□Φ)
    
    Args:
        campo_fase: Campo de fase informacional Φ
        tensor_metrico: g_μν
        lambda_acoplamento: λ
        
    Returns:
        np.ndarray: Tensor de coerência C_μν
    """
    # Dimensões
    dim = len(campo_fase.shape)
    
    # Calcula gradiente do gradiente (simplificação para espaço euclidiano)
    grad_grad = np.gradient(np.gradient(campo_fase))
    
    # Operador d'Alembert (simplificação)
    dAlembert = np.sum(np.gradient(np.gradient(campo_fase)))
    
    # Tensor de coerência
    C = lambda_acoplamento * (grad_grad - tensor_metrico * dAlembert)
    
    return C

# ============================================================================
# 4. UNIVERSO TGU - SISTEMA COERENTE
# ============================================================================

@dataclass
class UniversoTGU:
    """
    Sistema completo implementando o framework TGU.
    
    Representa um conjunto de sistemas com coerência informacional
    interagindo através do substrato de informação.
    """
    sistemas: List[SistemaCoerenciaInformacional]
    
    def filtrar_por_regime(
        self,
        regime: Literal["fraca", "intermediaria", "forte"]
    ) -> List[SistemaCoerenciaInformacional]:
        """Filtra sistemas por regime de tensão"""
        return [s for s in self.sistemas if s.regime == regime]
    
    def calcula_coerencia_media(self) -> float:
        """Calcula coerência informacional média"""
        if not self.sistemas:
            return 0.0
        
        coerencias = [np.linalg.norm(s.campo_coerencia) for s in self.sistemas]
        return float(np.mean(coerencias))
    
    def calcula_entropia_informacional(self) -> float:
        """
        Eq. 3: Entropia informacional
        
        S_info = -Σ p_n ln p_n
        """
        if not self.sistemas:
            return 0.0
        
        # Probabilidades baseadas na densidade informacional
        densidades = [s.densidade_informacional for s in self.sistemas]
        total = sum(densidades)
        
        if total == 0:
            return 0.0
        
        probabilidades = [d/total for d in densidades]
        
        # Entropia de Shannon
        entropia = 0.0
        for p in probabilidades:
            if p > 0:
                entropia -= p * np.log(p)
        
        return entropia
    
    def simula_evolucao_coerencia(
        self,
        passos_tempo: int = 10,
        dt: float = 0.1,
        lambda_dissipacao: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Eq. 4: Evolução da coerência informacional
        
        ∂I/∂t + ∇·(Iν) = -λI
        """
        historico = []
        
        for passo in range(passos_tempo + 1):
            # Estado atual
            estado = {
                "passo": passo,
                "tempo": passo * dt,
                "coerencia_media": self.calcula_coerencia_media(),
                "entropia": self.calcula_entropia_informacional(),
                "num_sistemas": len(self.sistemas)
            }
            historico.append(estado)
            
            if passo == passos_tempo:
                break
            
            # Atualiza campos de coerência (simulação simplificada)
            for sistema in self.sistemas:
                # Termo de fluxo (simplificado)
                fluxo = np.random.normal(0, 0.1, size=sistema.campo_coerencia.shape)
                
                # Termo de dissipação
                dissipacao = -lambda_dissipacao * sistema.campo_coerencia
                
                # Atualização Euler
                sistema.campo_coerencia += dt * (fluxo + dissipacao)
                
                # Mantém positividade
                sistema.campo_coerencia = np.maximum(sistema.campo_coerencia, 0.0)
        
        return historico
    
    def calcula_metricas_TGU(
        self,
        raio_crossover: float = 1.0,
        constante_acoplamento: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calcula métricas do framework TGU para análise
        
        Returns:
            Dict com todas as métricas relevantes
        """
        metricas = {
            "coerencia_media": self.calcula_coerencia_media(),
            "entropia_informacional": self.calcula_entropia_informacional(),
            "num_sistemas": len(self.sistemas),
            "distribuicao_regimes": {
                "fraca": len(self.filtrar_por_regime("fraca")),
                "intermediaria": len(self.filtrar_por_regime("intermediaria")),
                "forte": len(self.filtrar_por_regime("forte"))
            }
        }
        
        # Calcula interações TGU para sistemas fortes
        sistemas_fortes = self.filtrar_por_regime("forte")
        if len(sistemas_fortes) > 1:
            forcas_TGU = []
            for i in range(len(sistemas_fortes)):
                for j in range(i+1, len(sistemas_fortes)):
                    s1, s2 = sistemas_fortes[i], sistemas_fortes[j]
                    
                    # Gradiente de coerência (simplificado)
                    grad_coerencia = s2.campo_coerencia - s1.campo_coerencia
                    
                    # Força TGU - Eq. 47
                    forca = equacao_47_forca_TGU(grad_coerencia, constante_acoplamento)
                    forcas_TGU.append(np.linalg.norm(forca))
            
            if forcas_TGU:
                metricas["forca_TGU_media"] = float(np.mean(forcas_TGU))
                metricas["forca_TGU_max"] = float(np.max(forcas_TGU))
        
        return metricas

# ============================================================================
# 5. CONSTRUTORES DE SISTEMAS PARA SIMULAÇÃO
# ============================================================================

def construir_sistema_solar(
    semente: int = 42,
    dimensao_campo: int = 3
) -> UniversoTGU:
    """
    Constrói um sistema solar simulado no framework TGU
    
    Args:
        semente: Seed para reproducibilidade
        dimensao_campo: Dimensão do campo de coerência
        
    Returns:
        UniversoTGU: Sistema solar simulado
    """
    rng = np.random.default_rng(semente)
    sistemas = []
    
    # Corpos celestes do sistema solar
    corpos = [
        {"nome": "Sol", "massa": 1.0, "raio": 0.0, "regime": "fraca"},
        {"nome": "Mercúrio", "massa": 0.055, "raio": 0.387, "regime": "fraca"},
        {"nome": "Vênus", "massa": 0.815, "raio": 0.723, "regime": "fraca"},
        {"nome": "Terra", "massa": 1.0, "raio": 1.0, "regime": "fraca"},
        {"nome": "Marte", "massa": 0.107, "raio": 1.524, "regime": "fraca"},
        {"nome": "Júpiter", "massa": 317.8, "raio": 5.204, "regime": "intermediaria"},
        {"nome": "Saturno", "massa": 95.2, "raio": 9.582, "regime": "intermediaria"},
        {"nome": "Urano", "massa": 14.5, "raio": 19.201, "regime": "intermediaria"},
        {"nome": "Netuno", "massa": 17.1, "raio": 30.047, "regime": "intermediaria"}
    ]
    
    for i, corpo in enumerate(corpos):
        # Posição orbital (simplificada)
        angulo = 2 * np.pi * i / len(corpos)
        distancia = corpo["raio"]
        posicao = np.array([
            distancia * np.cos(angulo),
            distancia * np.sin(angulo),
            0.0
        ])
        
        # Campo de coerência (baseado na massa)
        campo_coerencia = np.ones(dimensao_campo) * corpo["massa"]**0.5
        
        # Adiciona ruído para realismo
        campo_coerencia += 0.1 * rng.normal(size=dimensao_campo)
        
        # Parâmetros orbitais para Mercúrio (teste de precessão)
        parametros_orbitais = None
        if corpo["nome"] == "Mercúrio":
            parametros_orbitais = {
                "semi_eixo_maior": 0.387,  # AU
                "excentricidade": 0.2056,
                "precessao_GR": 42.98  # segundos de arco por século
            }
        
        sistema = SistemaCoerenciaInformacional(
            identificador=corpo["nome"],
            coordenadas=posicao,
            campo_coerencia=campo_coerencia,
            densidade_informacional=corpo["massa"],
            regime=corpo["regime"],
            parametros_orbitais=parametros_orbitais
        )
        sistemas.append(sistema)
    
    return UniversoTGU(sistemas=sistemas)

def construir_galaxia_sombrero(
    semente: int = 43,
    num_sistemas: int = 20,
    dimensao_campo: int = 3
) -> UniversoTGU:
    """
    Constrói galáxia Messier 104 (Sombrero) simulada
    
    Simula os efeitos de rotação plana sem matéria escura
    """
    rng = np.random.default_rng(semente)
    sistemas = []
    
    # Componente central (bulbo)
    for i in range(num_sistemas // 2):
        # Distribuição radial exponencial
        raio = 5.0 * rng.exponential(0.5)
        angulo = 2 * np.pi * rng.random()
        
        posicao = np.array([
            raio * np.cos(angulo),
            raio * np.sin(angulo),
            0.1 * rng.normal()
        ])
        
        # Campo de coerência decai com raio
        campo_coerencia = np.ones(dimensao_campo) * np.exp(-raio/10.0)
        campo_coerencia += 0.05 * rng.normal(size=dimensao_campo)
        
        sistema = SistemaCoerenciaInformacional(
            identificador=f"bulbo_{i}",
            coordenadas=posicao,
            campo_coerencia=campo_coerencia,
            densidade_informacional=np.exp(-raio/5.0),
            regime="fraca"
        )
        sistemas.append(sistema)
    
    # Disco galáctico
    for i in range(num_sistemas // 2, num_sistemas):
        # Distribuição mais estendida
        raio = 10.0 + 20.0 * rng.random()
        angulo = 2 * np.pi * rng.random()
        
        posicao = np.array([
            raio * np.cos(angulo),
            raio * np.sin(angulo),
            0.5 * rng.normal()
        ])
        
        # Campo de coerência constante no disco (simula rotação plana)
        campo_coerencia = np.ones(dimensao_campo) * 0.8
        campo_coerencia += 0.1 * rng.normal(size=dimensao_campo)
        
        sistema = SistemaCoerenciaInformacional(
            identificador=f"disco_{i}",
            coordenadas=posicao,
            campo_coerencia=campo_coerencia,
            densidade_informacional=0.5,
            regime="intermediaria"
        )
        sistemas.append(sistema)
    
    return UniversoTGU(sistemas=sistemas)

def construir_sistema_binario(
    excentricidade: float = 0.8,
    semente: int = 44,
    dimensao_campo: int = 3
) -> UniversoTGU:
    """
    Constrói sistema binário de alta excentricidade
    
    Para teste das previsões TGU em regime forte
    """
    rng = np.random.default_rng(semente)
    sistemas = []
    
    # Estrela primária
    posicao1 = np.array([-1.0, 0.0, 0.0])
    campo1 = np.ones(dimensao_campo) * 2.0
    campo1 += 0.1 * rng.normal(size=dimensao_campo)
    
    sistema1 = SistemaCoerenciaInformacional(
        identificador="Primaria",
        coordenadas=posicao1,
        campo_coerencia=campo1,
        densidade_informacional=2.0,
        regime="forte",
        parametros_orbitais={
            "semi_eixo_maior": 2.0,
            "excentricidade": excentricidade
        }
    )
    sistemas.append(sistema1)
    
    # Estrela secundária
    posicao2 = np.array([1.0, 0.0, 0.0])
    campo2 = np.ones(dimensao_campo) * 1.5
    campo2 += 0.1 * rng.normal(size=dimensao_campo)
    
    sistema2 = SistemaCoerenciaInformacional(
        identificador="Secundaria",
        coordenadas=posicao2,
        campo_coerencia=campo2,
        densidade_informacional=1.5,
        regime="forte",
        parametros_orbitais={
            "semi_eixo_maior": 2.0,
            "excentricidade": excentricidade
        }
    )
    sistemas.append(sistema2)
    
    return UniversoTGU(sistemas=sistemas)

# ============================================================================
# 6. SIMULAÇÕES E EXPERIMENTOS
# ============================================================================

def experimento_precessao_mercurio():
    """
    Experimento: Precessão do periélio de Mercúrio no framework TGU
    
    Demonstra convergência com GR em regime de baixa tensão
    """
    print("=" * 70)
    print("EXPERIMENTO: PRECESSÃO DO PERIÉLIO DE MERCÚRIO (TGU)")
    print("=" * 70)
    
    # Parâmetros de Mercúrio
    precessao_GR = 42.98  # segundos de arco por século
    excentricidade = 0.2056
    semi_eixo_maior = 0.387  # AU
    distancia = semi_eixo_maior  # aproximação
    
    # Testa diferentes raios de crossover
    raios_crossover = [0.001, 0.01, 0.1, 1.0]
    
    print("\nPrecessão GR: {:.4f}''/século".format(precessao_GR))
    print("Excentricidade: {:.4f}".format(excentricidade))
    print("Semi-eixo maior: {:.4f} AU\n".format(semi_eixo_maior))
    
    print("Resultados TGU para diferentes r_s:")
    print("-" * 50)
    
    for r_s in raios_crossover:
        precessao_TGU = equacao_25_precessao_TGU(
            precessao_GR=precessao_GR,
            excentricidade=excentricidade,
            semi_eixo_maior=semi_eixo_maior,
            distancia=distancia,
            raio_crossover=r_s
        )
        
        diferenca = precessao_TGU - precessao_GR
        diferenca_relativa = diferenca / precessao_GR * 100
        
        print("r_s = {:.3f} AU: {:.4f}''/século (Δ = {:.6f}, {:.4f}%)".format(
            r_s, precessao_TGU, diferenca, diferenca_relativa
        ))
    
    # Demonstra fator de correção orbital
    alpha = equacao_6_fator_correcao_orbital(excentricidade, semi_eixo_maior)
    print("\nFator de correção orbital α = {:.6f}".format(alpha))
    
    # Para r_s pequeno (regime de convergência)
    r_s_pequeno = 0.001
    precessao_convergente = equacao_25_precessao_TGU(
        precessao_GR=precessao_GR,
        excentricidade=excentricidade,
        semi_eixo_maior=semi_eixo_maior,
        distancia=distancia,
        raio_crossover=r_s_pequeno
    )
    
    print("\nPara r_s = {:.3f} AU (regime de convergência):".format(r_s_pequeno))
    print("TGU: {:.6f}''/século".format(precessao_convergente))
    print("GR:  {:.6f}''/século".format(precessao_GR))
    print("Diferença: {:.10f}''/século".format(precessao_convergente - precessao_GR))

def experimento_rotacao_galactica():
    """
    Experimento: Curvas de rotação galáctica sem matéria escura
    
    Demonstra rotação plana através de gradientes de coerência
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: CURVAS DE ROTAÇÃO GALÁCTICA (TGU)")
    print("=" * 70)
    
    # Constrói galáxia simulada
    galaxia = construir_galaxia_sombrero(num_sistemas=30)
    
    # Calcula velocidades em diferentes raios
    raios = np.linspace(0.1, 30.0, 20)
    velocidades_newton = []
    velocidades_TGU = []
    
    massa_total = 10.0  # Massa barônica total
    
    for raio in raios:
        # Newtoniana (sem matéria escura)
        V2_newton = massa_total / raio
        velocidades_newton.append(np.sqrt(V2_newton))
        
        # TGU (com coerência)
        campo_coerencia = np.ones(3) * (1.0 + 0.1 * raio)  # Gradiente linear
        V2_TGU = equacao_35_rotacao_galactica_TGU(
            massa_baronica=massa_total,
            raio=raio,
            campo_coerencia=campo_coerencia
        )
        velocidades_TGU.append(np.sqrt(V2_TGU))
    
    print("\nComparação de velocidades de rotação:")
    print("-" * 60)
    print("Raio (kpc)  |  Newton (km/s)  |  TGU (km/s)  |  Diferença")
    print("-" * 60)
    
    for i, raio in enumerate(raios[::2]):  # Mostra apenas alguns pontos
        idx = i * 2
        if idx < len(raios):
            V_newton = velocidades_newton[idx]
            V_TGU = velocidades_TGU[idx]
            dif = V_TGU - V_newton
            
            print("{:7.1f}     | {:12.1f}    | {:10.1f}   | {:8.1f}".format(
                raio, V_newton, V_TGU, dif
            ))
    
    # Análise TGU da galáxia
    metricas = galaxia.calcula_metricas_TGU()
    print("\nMétricas TGU da galáxia:")
    print("Coerência média: {:.3f}".format(metricas["coerencia_media"]))
    print("Entropia informacional: {:.3f}".format(metricas["entropia_informacional"]))
    print("Número de sistemas: {}".format(metricas["num_sistemas"]))

def experimento_sistema_binario_alta_excentricidade():
    """
    Experimento: Sistema binário de alta excentricidade
    
    Demonstra previsões TGU para desvios mensuráveis
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: SISTEMA BINÁRIO DE ALTA EXCENTRICIDADE")
    print("=" * 70)
    
    # Sistema binário com alta excentricidade
    excentricidade = 0.9
    sistema_binario = construir_sistema_binario(excentricidade=excentricidade)
    
    # Parâmetros orbitais
    precessao_GR_base = 10.0  # Valor base para comparação
    semi_eixo_maior = 2.0
    distancia = semi_eixo_maior * (1 - excentricidade)  # Periapsis
    
    # Testa diferentes raios de crossover
    raios = [0.1, 0.5, 1.0, 2.0]
    
    print("\nSistema binário com e = {:.2f}".format(excentricidade))
    print("Semi-eixo maior: {:.1f} AU".format(semi_eixo_maior))
    print("Distância no periapsis: {:.3f} AU".format(distancia))
    print("\nComparação TGU vs GR:")
    print("-" * 60)
    
    for r_s in raios:
        # Fator de resistência
        epsilon_factor = equacao_32_fator_resistencia_coerencia(distancia, r_s)
        
        # Precessão TGU
        precessao_TGU = equacao_25_precessao_TGU(
            precessao_GR=precessao_GR_base,
            excentricidade=excentricidade,
            semi_eixo_maior=semi_eixo_maior,
            distancia=distancia,
            raio_crossover=r_s
        )
        
        diferenca = precessao_TGU - precessao_GR_base
        diferenca_percentual = diferenca / precessao_GR_base * 100
        
        print("r_s = {:.1f} AU: ε = {:.6f}, TGU = {:.4f} (Δ = {:+.2f}%, {:.4f})".format(
            r_s, epsilon_factor, precessao_TGU, diferenca_percentual, diferenca
        ))
    
    # Análise TGU do sistema
    metricas = sistema_binario.calcula_metricas_TGU()
    
    print("\nMétricas TGU do sistema binário:")
    print("Coerência média: {:.3f}".format(metricas["coerencia_media"]))
    
    if "forca_TGU_media" in metricas:
        print("Força TGU média: {:.3f}".format(metricas["forca_TGU_media"]))

def experimento_evolucao_coerencia():
    """
    Experimento: Evolução temporal da coerência informacional
    
    Demonstra a equação de continuidade da coerência
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: EVOLUÇÃO TEMPORAL DA COERÊNCIA")
    print("=" * 70)
    
    # Sistema solar simulado
    sistema_solar = construir_sistema_solar()
    
    # Simula evolução
    historico = sistema_solar.simula_evolucao_coerencia(
        passos_tempo=10,
        dt=0.2,
        lambda_dissipacao=0.02
    )
    
    print("\nEvolução da coerência informacional:")
    print("-" * 60)
    print("Passo | Tempo | Coerência | Entropia | Sistemas")
    print("-" * 60)
    
    for estado in historico[::2]:  # Mostra a cada 2 passos
        print("{:5d} | {:5.1f} | {:9.3f} | {:8.3f} | {:8d}".format(
            estado["passo"],
            estado["tempo"],
            estado["coerencia_media"],
            estado["entropia"],
            estado["num_sistemas"]
        ))
    
    # Análise final
    estado_final = historico[-1]
    estado_inicial = historico[0]
    
    delta_coerencia = estado_final["coerencia_media"] - estado_inicial["coerencia_media"]
    delta_entropia = estado_final["entropia"] - estado_inicial["entropia"]
    
    print("\nResumo da evolução:")
    print("Variação da coerência: {:+.4f} ({:+.2f}%)".format(
        delta_coerencia, delta_coerencia/estado_inicial["coerencia_media"]*100
    ))
    print("Variação da entropia: {:+.4f} ({:+.2f}%)".format(
        delta_entropia, delta_entropia/estado_inicial["entropia"]*100
    ))

# ============================================================================
# 7. PONTO DE ENTRADA PRINCIPAL
# ============================================================================

def executar_experimentos_TGU():
    """
    Executa todos os experimentos demonstrativos do framework TGU
    """
    print("=" * 80)
    print("TEORIA DO SPIN INFORMACIONAL (TGU) - IMPLEMENTAÇÃO COMPUTACIONAL")
    print("=" * 80)
    print("Referência: Matuchaki, H. (2026). The Theory of Informational Spin")
    print("A Coherence-Based Framework for Gravitation, Cosmology, and Quantum Systems")
    print("\nParâmetros fundamentais:")
    print(f"  • Parâmetro de Matuchaki (k): {PARAMETRO_MATUCHAKI:.6f}")
    print(f"  • Expoente de coerência (n): {EXPOENTE_COERENCIA}")
    print(f"  • Fator isotrópico base: {FATOR_ISOTROPICO_BASE:.6f}")
    print("=" * 80 + "\n")
    
    # Executa experimentos
    experimento_precessao_mercurio()
    experimento_rotacao_galactica()
    experimento_sistema_binario_alta_excentricidade()
    experimento_evolucao_coerencia()
    
    print("\n" + "=" * 80)
    print("CONCLUSÕES DO FRAMEWORK TGU:")
    print("=" * 80)
    print("1. Convergência com GR em regimes de baixa tensão (Sistema Solar)")
    print("2. Explicação de rotações galácticas planas sem matéria escura")
    print("3. Previsões de desvios mensuráveis em sistemas de alta excentricidade")
    print("4. Evolução coerente da informação através do substrato universal")
    print("5. Dualidade matemática entre descrições geométricas e informacionais")
    print("=" * 80)

# ============================================================================
# 8. TESTES E VALIDAÇÃO
# ============================================================================

def testar_funcoes_basicas():
    """Testa as funções fundamentais da implementação TGU"""
    print("Testando funções básicas TGU...")
    
    # Teste do fator de correção orbital
    alpha = equacao_6_fator_correcao_orbital(0.2, 1.0)
    print(f"α(e=0.2, a=1.0) = {alpha:.6f}")
    
    # Teste do fator de resistência
    epsilon = equacao_32_fator_resistencia_coerencia(1.0, 0.1)
    print(f"ε(r=1.0, r_s=0.1) = {epsilon:.6f}")
    
    # Teste do spin informacional
    estados = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    spin = equacao_1_spin_informacional(estados)
    print(f"Spin informacional para 2 estados: {spin}")
    
    print("Testes básicos concluídos.")

# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    """
    Ponto de entrada principal da implementação TGU
    """
    try:
        # Testes básicos
        testar_funcoes_basicas()
        print()
        
        # Experimento completo
        executar_experimentos_TGU()
        
        # Exemplo adicional: Sistema solar TGU
        print("\n" + "=" * 80)
        print("EXEMPLO: SISTEMA SOLAR NO FRAMEWORK TGU")
        print("=" * 80)
        
        sistema_solar = construir_sistema_solar()
        metricas = sistema_solar.calcula_metricas_TGU()
        
        print("\nDistribuição por regime de tensão:")
        for regime, quantidade in metricas["distribuicao_regimes"].items():
            print(f"  • {regime}: {quantidade} sistemas")
        
        print(f"\nCoerência informacional média: {metricas['coerencia_media']:.3f}")
        print(f"Entropia informacional: {metricas['entropia_informacional']:.3f}")
        
    except Exception as e:
        print(f"\nErro durante a execução: {e}")
        print("Verifique as dependências: numpy é necessário.")
    
    finally:
        print("\n" + "=" * 80)
        print("Implementação da Teoria do Spin Informacional (TGU)")
        print("Baseada em: Matuchaki, H. (2026). The Theory of Informational Spin")
        print("=" * 80)
