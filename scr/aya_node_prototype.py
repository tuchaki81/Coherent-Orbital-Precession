"""
aya_node_prototype.py

Prototype implementation of AYA-NODE: Spin-Informational Coherent Node Architecture
for the Unified Informational Spin Theory (TGU) extension to computational systems.

Each AYA-NODE acts as a "vortex" that:
- Processes input with coherence-modulated activation
- Maintains internal reference state (pseudo-vortex center)
- Synchronizes with neighbors via ICOER-driven resonance

Key features:
- Learnable resonance gate
- ICOER-based synchronization loss
- Modular design for stacking multiple nodes (future network)

Author: Henry Matuchaki (@MatuchakiSilva)
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.icoer_torch import ICOER, coherence_activation, icoer_regularization_loss


class AyaNode(nn.Module):
    """
    Single AYA-NODE: a coherence-aware processing unit.
    
    Behaves like an informational vortex:
    - Internal reference state (learnable or dynamic)
    - Coherence-modulated forward pass
    - Optional synchronization with other nodes
    """
    def __init__(self, in_features: int, hidden_features: int = 64, 
                 resonance_init: float = 0.35, eps: float = 1e-8):
        super().__init__()
        
        self.in_features = in_features
        self.hidden = hidden_features
        
        # Linear transformation (core processing)
        self.linear = nn.Linear(in_features, hidden_features)
        
        # Learnable resonance threshold (like a dynamic gate)
        self.resonance_threshold = nn.Parameter(torch.tensor(resonance_init))
        
        # ICOER calculator
        self.icoer_metric = ICOER(eps=eps)
        
        # Internal reference state (pseudo-vortex center, learnable)
        self.register_parameter(
            "internal_ref", 
            nn.Parameter(torch.randn(1, hidden_features) * 0.1)
        )

    def forward(self, x: torch.Tensor, external_ref: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of a single AYA-NODE.
        
        Parameters:
        -----------
        x : torch.Tensor [batch, in_features]
        external_ref : torch.Tensor [batch, hidden_features], optional
                       External reference for inter-node synchronization
        
        Returns:
        --------
        output : torch.Tensor [batch, hidden_features]
        icoer_score : torch.Tensor  (scalar or per-batch)
        """
        # Project input
        z = self.linear(x)
        
        # Choose reference: external > internal
        ref = external_ref if external_ref is not None else self.internal_ref.expand_as(z)
        
        # Compute ICOER (coherence with reference)
        icoer = self.icoer_metric(z, ref)
        
        # Apply coherence-modulated activation
        activated = coherence_activation(z, icoer, threshold=self.resonance_threshold)
        
        return activated, icoer


class AyaNodeLayer(nn.Module):
    """
    Layer of multiple AYA-NODEs (for stacking or parallel processing).
    Includes optional synchronization loss between nodes.
    """
    def __init__(self, in_features: int, num_nodes: int = 4, hidden_features: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.nodes = nn.ModuleList([
            AyaNode(in_features, hidden_features) for _ in range(num_nodes)
        ])

    def forward(self, x: torch.Tensor):
        """
        Process input through all nodes, compute inter-node ICOER sync.
        
        Returns:
        --------
        outputs : list of tensors [batch, hidden]
        sync_icoer_mean : average ICOER between nodes (for monitoring/loss)
        """
        node_outputs = []
        icoers = []
        
        for node in self.nodes:
            out, icoer = node(x)
            node_outputs.append(out)
            icoers.append(icoer)
        
        # Inter-node synchronization: pairwise ICOER
        sync_scores = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pair_icoer = self.nodes[0].icoer_metric(  # reuse from first node
                    node_outputs[i], node_outputs[j]
                )
                sync_scores.append(pair_icoer)
        
        sync_icoer_mean = torch.stack(sync_scores).mean() if sync_scores else torch.tensor(1.0)
        
        # Stack outputs (concat or mean depending on downstream)
        combined = torch.cat(node_outputs, dim=-1)  # example: concat
        
        return combined, sync_icoer_mean


# Example training loop snippet (minimal)
def simple_aya_training_example():
    print("=== Minimal AYA-NODE Training Demo ===\n")
    
    torch.manual_seed(42)
    batch_size, in_dim, hidden = 32, 16, 32
    model = AyaNodeLayer(in_dim, num_nodes=3, hidden_features=hidden)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    x = torch.randn(batch_size, in_dim)
    
    for step in range(50):
        optimizer.zero_grad()
        out, sync_icoer = model(x)
        
        # Dummy task: reconstruct input after projection
        recon_loss = F.mse_loss(out.mean(dim=-1), x.mean(dim=-1))
        
        # Encourage high inter-node coherence
        coherence_reg = icoer_regularization_loss(sync_icoer, target=0.80, weight=0.2)
        
        loss = recon_loss + coherence_reg
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:2d} | Recon loss: {recon_loss.item():.4f} | "
                  f"Sync ICOER: {sync_icoer.item():.4f} | Total loss: {loss.item():.4f}")


if __name__ == "__main__":
    simple_aya_training_example()