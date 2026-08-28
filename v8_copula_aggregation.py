"""
v8_copula_aggregation.py
------------------------
Copula-Based Synergistic Aggregation PyTorch Module (v8 Architecture - Phase 4).

Replaces the legacy independent probabilistic union formula:
    Drift = 1.0 - prod(1.0 - S_i)
with an Archimedean Gumbel Copula aggregation layer modeling upper-tail dependence
(lambda_U = 2 - 2^(1/theta)) and synergistic multi-ability shock amplification.

Key Features:
1. GumbelCopulaAggregator: PyTorch nn.Module with constrained trainable parameter theta in [1, inf).
2. Upper-Tail Dependence Calculation: Computes lambda_U = 2 - 2^(1/theta) dynamically.
3. Robust Numerical Stability: Clamping (eps = 1e-7) preventing log(0) or 0^(1/alpha) autograd traps.
4. Flexible Inputs: Handles 1D (single shock), 2D (bivariate pair), and multivariate shock arrays.
5. Integration with Phase 2/3 outputs.
"""

import math
from typing import Dict, List, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 0. META-NETWORK ADJACENCY MATRIX (PHASE 2 CROSS-AGENT COUNTER EFFECTS)
# ============================================================================

META_ADJACENCY: Dict[str, Dict[str, float]] = {
    "Phoenix": {"Sage": 0.15, "Cypher": -0.05},
    "Neon": {"Cypher": 0.20, "Breach": 0.10, "Killjoy": 0.15},
    "Jett": {"Cypher": 0.15, "Sova": 0.10, "Killjoy": 0.10},
    "Raze": {"Deadlock": 0.15, "Cypher": 0.10},
    "Iso": {"Omen": 0.10, "Fade": 0.10},
    "Viper": {"Harbor": 0.15, "Astra": 0.10},
    "Cypher": {"Sova": -0.10, "Raze": -0.15},
}


def apply_meta_network_adjacency(
    drift_dict: Dict[str, torch.Tensor],
    meta_adjacency: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, torch.Tensor]:
    """
    Phase 2: Propagates cross-agent Meta-Network Effects via static adjacency weights
    without requiring a Graph Neural Network.
    
    For each agent with an active drift, propagates drift * weight to target counter-agents.
    Ensures final adjusted drift indices are clipped between [0.0, 1.0].
    """
    adj = meta_adjacency if meta_adjacency is not None else META_ADJACENCY
    if not adj:
        return drift_dict
        
    # Shallow copy of tensors to preserve original inputs
    adjusted: Dict[str, torch.Tensor] = {k: v.clone() for k, v in drift_dict.items()}
    
    for src_agent, src_drift in drift_dict.items():
        if src_agent in adj:
            for target_agent, weight in adj[src_agent].items():
                drift_propagation = src_drift * float(weight)
                if target_agent in adjusted:
                    adjusted[target_agent] = adjusted[target_agent] + drift_propagation
                else:
                    adjusted[target_agent] = torch.zeros_like(src_drift) + drift_propagation
                    
    # Clip all drift scores between 0.0 and 1.0
    for agent in list(adjusted.keys()):
        adjusted[agent] = torch.clamp(adjusted[agent], min=0.0, max=1.0)
        
    return adjusted


# ============================================================================
# 1. GUMBEL COPULA AGGREGATOR MODULE
# ============================================================================

class GumbelCopulaAggregator(nn.Module):
    """
    Archimedean Gumbel Copula Aggregator.
    
    Models upper-tail dependence and synergistic coupling between combined ability shocks on an agent.
    
    Mathematical Formulation:
        theta = 1.0 + Softplus(raw_theta),  theta in [1.0, inf)
        alpha = 1.0 / theta,                alpha in (0, 1.0]
        u_i = 1.0 - clamp(S_i, 0, 1 - eps)
        psi(u_i) = (-ln(u_i))^alpha
        C(u_1, ..., u_d; alpha) = exp( - ( sum_i (-ln(u_i))^alpha )^(1/alpha) )
        Concept Drift Index = 1.0 - C(u_1, ..., u_d; alpha)
        
    Behavior:
        - When theta = 1.0 (alpha = 1.0), lambda_U = 0, matching independent probabilistic union.
        - When theta > 1.0 (alpha < 1.0), lambda_U > 0, synergistically amplifying combined shocks.
    """
    def __init__(
        self,
        init_theta: float = 1.0,
        eps: float = 1e-7,
        trainable: bool = True
    ):
        super().__init__()
        self.eps = eps

        # Enforce theta >= 1.0 via Softplus parameterization:
        # theta = 1.0 + Softplus(raw_theta)
        target_diff = max(init_theta - 1.0, 0.0)
        if target_diff > 1e-4:
            raw_val = math.log(math.expm1(target_diff))
        else:
            raw_val = -15.0

        if trainable:
            self.raw_theta = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))
        else:
            self.register_buffer("raw_theta", torch.tensor(raw_val, dtype=torch.float32))

    @property
    def theta(self) -> torch.Tensor:
        """Returns dependence parameter theta constrained to [1.0, inf)."""
        return 1.0 + F.softplus(self.raw_theta)

    @property
    def upper_tail_dependence(self) -> torch.Tensor:
        """Computes upper-tail dependence coefficient: lambda_U = 2 - 2^(1/theta)."""
        th = self.theta
        return 2.0 - torch.pow(2.0, 1.0 / th)

    def forward(self, shocks: torch.Tensor) -> torch.Tensor:
        """
        Aggregates marginal shock probabilities into a final Concept Drift Index.
        
        Args:
            shocks: Tensor of shape (d,) or (B, d) or (M, 1) representing marginal shock values in [0, 1).
            
        Returns:
            Concept Drift Index tensor of shape (1, 1) or (B, 1).
        """
        th = self.theta
        alpha = 1.0 / th
        
        # Ensure 2D shape (B, d)
        if shocks.dim() == 1:
            shocks_2d = shocks.unsqueeze(0)
        else:
            shocks_2d = shocks

        # 1. Transform marginal shocks S_i to survival probabilities u_i = 1 - S_i
        shocks_clamped = torch.clamp(shocks_2d, min=0.0, max=1.0 - self.eps)
        u = 1.0 - shocks_clamped
        u_clamped = torch.clamp(u, min=self.eps, max=1.0 - self.eps)

        d = u_clamped.shape[-1]
        
        # Single shock boundary case (d=1): Copula collapses to identity u_1
        if d == 1:
            return shocks_clamped

        # 2. Archimedean Generator: psi(u) = (-ln(u))^alpha
        neg_log_u = -torch.log(u_clamped)
        psi_u = torch.pow(neg_log_u, alpha)

        # 3. Sum generator outputs across features/abilities
        sum_psi = torch.sum(psi_u, dim=-1, keepdim=True)  # Shape: (B, 1)

        # 4. Inverse Generator: psi_inv(s) = exp( - s^(1/alpha) )
        sum_psi_clamped = torch.clamp(sum_psi, min=self.eps)
        copula_val = torch.exp(-torch.pow(sum_psi_clamped, 1.0 / alpha))

        # 5. Concept Drift Index = 1.0 - Copula survival probability
        concept_drift_index = 1.0 - copula_val
        return concept_drift_index


# ============================================================================
# 2. AGENT GROUPED COPULA AGGREGATION UTILITY
# ============================================================================

class AgentGroupedCopulaAggregator(nn.Module):
    """
    Wrapper module that groups Phase 2/3 extracted ability shocks by Agent name,
    applies GumbelCopulaAggregator to each agent's shock array, and propagates
    cross-agent Meta-Network Effects (Phase 2).
    """
    def __init__(self, init_theta: float = 1.0):
        super().__init__()
        self.copula = GumbelCopulaAggregator(init_theta=init_theta)

    def forward(
        self,
        agent_names: List[str],
        shocks: torch.Tensor,
        enable_meta_network: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            agent_names: List of string agent names corresponding to rows in shocks (length M)
            shocks: Tensor of shape (M, 1) or (M, embed_dim) representing shock magnitudes
            enable_meta_network: If True, propagates cross-agent adjacency shifts.

        Returns:
            Dict mapping agent name -> aggregated Concept Drift Index tensor (shape (1, 1))
        """
        # Squeeze embedding to scalar shock probability via Sigmoid of sum
        if shocks.dim() > 1 and shocks.shape[-1] > 1:
            scalar_shocks = torch.sigmoid(shocks.sum(dim=-1, keepdim=True))
        elif shocks.dim() == 1:
            scalar_shocks = torch.sigmoid(shocks).unsqueeze(1)
        else:
            scalar_shocks = torch.sigmoid(shocks)

        # Group rows by agent name
        agent_map: Dict[str, List[int]] = {}
        for idx, name in enumerate(agent_names):
            agent_map.setdefault(name, []).append(idx)

        results: Dict[str, torch.Tensor] = {}
        for agent, indices in agent_map.items():
            agent_shock_tensor = scalar_shocks[indices].squeeze(-1).unsqueeze(0)  # Shape: (1, d)
            drift_index = self.copula(agent_shock_tensor)
            results[agent] = drift_index

        # Phase 2: Apply Meta-Network Effects Adjacency Propagation
        if enable_meta_network:
            results = apply_meta_network_adjacency(results)

        return results


# ============================================================================
# CLI DEMONSTRATION ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    print("--- TESTING V8 COPULA-BASED SYNERGISTIC AGGREGATION ---")

    copula_indep = GumbelCopulaAggregator(init_theta=1.0)
    copula_synergy = GumbelCopulaAggregator(init_theta=2.0)

    # Jett Smoke (0.3) + Dash (0.4) combined nerfs
    jett_shocks = torch.tensor([[0.3, 0.4]])

    drift_indep = copula_indep(jett_shocks)
    drift_synergy = copula_synergy(jett_shocks)

    print("\n1. Jett Combined Nerfs (0.3 Smoke, 0.4 Dash):")
    print(f"Independent Probabilistic Union (theta=1.0): {drift_indep.item():.4f}")
    print(f"Synergistic Gumbel Copula     (theta=2.0): {drift_synergy.item():.4f}")
    print(f"Upper-Tail Dependence (lambda_U at theta=2.0): {copula_synergy.upper_tail_dependence.item():.4f}")

    # Autograd Verification
    loss = drift_synergy.sum()
    loss.backward()
    print("\n2. Autograd Verification:")
    print("raw_theta grad:", copula_synergy.raw_theta.grad.item())
    print("Gradient verification SUCCESSFUL!")
