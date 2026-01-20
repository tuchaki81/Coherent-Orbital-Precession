"""
icoer_torch.py

PyTorch implementation of the Informational Coherence Index (ICOER) 
and coherence-aware activation functions for the Unified Informational Spin Theory (TGU).

ICOER measures the degree of informational alignment/resonance between states,
inspired by the coherence fields in orbital dynamics and extended to neural networks.

Key components:
- ICOER metric: 1 - normalized divergence (KL + cosine distance)
- Coherence activation: modulated ReLU-like with resonance gate
- Loss regularization term to promote high ICOER

Author: Henry Matuchaki (@MatuchakiSilva)
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICOER(nn.Module):
    """
    Informational Coherence Index (ICOER) module.
    
    Computes coherence between two tensors (e.g., reference state and current state).
    ICOER ∈ [0, 1], higher = better coherence / lower informational strain.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Compute ICOER between x (current activations) and ref (reference/coherent state).
        
        Parameters:
        -----------
        x : torch.Tensor     shape [batch, features] or higher
        ref : torch.Tensor   same shape as x (reference distribution/state)
        
        Returns:
        --------
        icoer : torch.Tensor  scalar or per-batch coherence score ∈ [0, 1]
        """
        # Flatten if needed (for higher dims)
        x_flat = x.view(x.size(0), -1)
        ref_flat = ref.view(ref.size(0), -1)

        # Softmax to treat as distributions (if not probabilities)
        x_prob = F.softmax(x_flat, dim=-1) + self.eps
        ref_prob = F.softmax(ref_flat, dim=-1) + self.eps

        # KL divergence (symmetrized Jensen-Shannon style)
        kl = F.kl_div(x_prob.log(), ref_prob, reduction='batchmean') + \
             F.kl_div(ref_prob.log(), x_prob, reduction='batchmean')

        # Cosine similarity (1 - distance)
        cos_sim = F.cosine_similarity(x_flat, ref_flat, dim=-1).mean()

        # Normalized divergence (KL bounded roughly [0, some value])
        divergence = torch.clamp(kl / (kl + 1.0), 0.0, 1.0)

        # ICOER = 1 - divergence + bonus from cosine
        icoer = (1.0 - divergence) * 0.7 + (cos_sim.clamp(-1, 1) + 1) / 2 * 0.3

        return icoer.clamp(0.0, 1.0)


def coherence_activation(x: torch.Tensor, icoer: torch.Tensor = None, threshold: float = 0.35) -> torch.Tensor:
    """
    Coherence-modulated activation function.
    
    Amplifies signal when ICOER is high, suppresses when low (informational strain).
    
    Parameters:
    -----------
    x : torch.Tensor         input activations
    icoer : torch.Tensor     precomputed ICOER (scalar or broadcastable)
    threshold : float        minimum coherence to amplify (default 0.35)
    
    Returns:
    --------
    torch.Tensor             activated output
    """
    if icoer is None:
        # If no ICOER provided, use simple ReLU (fallback)
        return F.relu(x)

    # Resonance gate: sigmoid transition around threshold
    gate = torch.sigmoid(10.0 * (icoer - threshold))  # sharp transition

    # Amplify positive signals when coherent
    activated = F.relu(x) * (1.0 + 0.5 * gate)  # up to 50% amplification

    return activated


class ResonanceGateLayer(nn.Module):
    """
    Learnable resonance gate layer that modulates activations based on learned ICOER.
    Useful for ICOER-aware neural networks (AYA-NODE style).
    """
    def __init__(self, features: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(features) * 0.35)  # initial threshold-like
        self.icoer_metric = ICOER()

    def forward(self, x: torch.Tensor, reference: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional reference state.
        If no reference, uses mean as pseudo-reference (self-coherence).
        """
        if reference is None:
            reference = x.mean(dim=0, keepdim=True).expand_as(x)

        icoer = self.icoer_metric(x, reference)
        
        # Modulate gate with learned parameter
        modulated_gate = torch.sigmoid(10.0 * (icoer - self.gate))

        return F.relu(x) * (1.0 + 0.5 * modulated_gate.mean())


def icoer_regularization_loss(icoer: torch.Tensor, target: float = 0.85, weight: float = 0.1) -> torch.Tensor:
    """
    Regularization term to encourage high ICOER during training.
    
    Loss = weight * (target - icoer)^2
    """
    return weight * (target - icoer).pow(2).mean()


# Example usage (run module directly)
if __name__ == "__main__":
    print("=== ICOER Torch Example ===\n")

    # Dummy activations
    torch.manual_seed(42)
    batch_size, features = 32, 128
    x = torch.randn(batch_size, features)
    ref = torch.randn(batch_size, features) * 0.2 + 1.0  # somewhat coherent reference

    # Compute ICOER
    icoer_module = ICOER()
    icoer_value = icoer_module(x, ref)
    print(f"ICOER between x and ref: {icoer_value.item():.4f}")

    # Coherence activation
    activated = coherence_activation(x, icoer_value)
    print(f"Activated shape: {activated.shape}, mean: {activated.mean().item():.4f}")

    # Regularization example
    reg_loss = icoer_regularization_loss(icoer_value, target=0.90)
    print(f"ICOER regularization loss: {reg_loss.item():.6f}")

    # Learnable resonance layer
    layer = ResonanceGateLayer(features)
    output = layer(x, ref)
    print(f"ResonanceGateLayer output mean: {output.mean().item():.4f}")
