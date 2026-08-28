"""
v8_differentiable_base.py
-------------------------
Differentiable Base & Attention Gating PyTorch Module (v8 Architecture).

Replaces legacy static heuristic weighting (combat=1.2, ability=1.0, Signature=0.40, etc.)
with a differentiable PyTorch neural network layer that learns category elasticities and
ability power budgets while applying context-aware attention gating:

    beta_dynamic = sigmoid(W_attn * X_context + b)

Integrates seamlessly with the Phase 1 NLP JSON schema (PatchExtractionPayload / PatchChangeItem).
"""

import math
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing Phase 1 Pydantic models if available
try:
    from v8_patch_parser import PatchExtractionPayload, PatchChangeItem
except ImportError:
    PatchExtractionPayload = None
    PatchChangeItem = None


# ============================================================================
# 1. CONSTANTS & CATEGORICAL MAPPINGS
# ============================================================================

CATEGORY_MAP = {
    "combat": 0,
    "ability": 1,
    "movement": 2,
    "economy": 3,
    "general": 4
}

ABILITY_TYPE_MAP = {
    "signature": 0,
    "ultimate": 1,
    "basic": 2,
    "passive": 3,
    "general": 4
}

# Baseline initializations reflecting legacy expert system heuristics
DEFAULT_CATEGORY_ELASTICITIES = [1.2, 1.0, 1.0, 0.8, 0.5]  # combat, ability, movement, economy, general
DEFAULT_POWER_BUDGET_WEIGHTS = [0.40, 0.30, 0.15, 0.10, 0.05]  # signature, ultimate, basic, passive, general


# ============================================================================
# 2. TENSOR BUILDER (PHASE 1 JSON -> PYTORCH TENSORS)
# ============================================================================

class PatchTensorBuilder:
    """
    Transforms Phase 1 NLP JSON extracted patch items into structured PyTorch tensors
    suitable for consumption by PatchEmbeddingBase.
    """
    @staticmethod
    def infer_category(stat_modified: str, ability: str) -> int:
        """Heuristically infers category index if not explicitly tagged."""
        combined = (stat_modified + " " + ability).lower()
        if any(k in combined for k in ["damage", "fire rate", "recoil", "headshot", "kill", "health", "shield"]):
            return CATEGORY_MAP["combat"]
        elif any(k in combined for k in ["slide", "speed", "dash", "velocity", "satchel", "teleport", "movement"]):
            return CATEGORY_MAP["movement"]
        elif any(k in combined for k in ["cred", "cost", "econ", "buy", "price"]):
            return CATEGORY_MAP["economy"]
        elif ability.lower() not in ["general", "global", "passive"]:
            return CATEGORY_MAP["ability"]
        return CATEGORY_MAP["general"]

    @staticmethod
    def infer_ability_type(ability: str) -> int:
        """Heuristically infers ability type index if not explicitly tagged."""
        ab_lower = ability.lower()
        if any(k in ab_lower for k in ["ult", "ultimate", "x"]):
            return ABILITY_TYPE_MAP["ultimate"]
        elif any(k in ab_lower for k in ["signature", "e", "dash"]):
            return ABILITY_TYPE_MAP["signature"]
        elif any(k in ab_lower for k in ["passive"]):
            return ABILITY_TYPE_MAP["passive"]
        elif ab_lower in ["general", "global"]:
            return ABILITY_TYPE_MAP["general"]
        return ABILITY_TYPE_MAP["basic"]

    @classmethod
    def extract_features(cls, change_item: Dict[str, Any]) -> Tuple[int, int, float, float, float]:
        """
        Extracts numerical and categorical tuple for a single PatchChangeItem.
        Returns: (category_idx, ability_type_idx, raw_delta, is_mechanical_removal_float, has_numeric_transition_float)
        """
        # Read fields from dict or Pydantic object
        if hasattr(change_item, "model_dump"):
            item_dict = change_item.model_dump()
        elif isinstance(change_item, dict):
            item_dict = change_item
        else:
            item_dict = change_item.__dict__

        stat_modified = str(item_dict.get("stat_modified", ""))
        ability = str(item_dict.get("ability", "General"))
        old_val = item_dict.get("old_value")
        new_val = item_dict.get("new_value")
        is_mech = bool(item_dict.get("is_mechanical_removal", False))

        cat_idx = cls.infer_category(stat_modified, ability)
        ab_type_idx = cls.infer_ability_type(ability)

        # Compute normalized numeric delta
        raw_delta = 0.0
        has_numeric = 0.0

        if old_val is not None and new_val is not None:
            try:
                old_f = float(old_val)
                new_f = float(new_val)
                has_numeric = 1.0
                if abs(old_f) > 1e-6:
                    raw_delta = (new_f - old_f) / abs(old_f)
                else:
                    raw_delta = new_f - old_f
            except (ValueError, TypeError):
                raw_delta = 1.0 if new_val != old_val else 0.0
                has_numeric = 0.0
        elif is_mech:
            # Mechanical removals without explicit numeric transition carry default shock magnitude 1.0
            raw_delta = -1.0
            has_numeric = 0.0

        mech_float = 1.0 if is_mech else 0.0
        return (cat_idx, ab_type_idx, float(raw_delta), mech_float, has_numeric)

    @classmethod
    def payload_to_tensors(
        cls,
        payload_or_changes: Any,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Converts a list of changes or a PatchExtractionPayload into PyTorch tensors.
        """
        if hasattr(payload_or_changes, "changes"):
            changes_list = payload_or_changes.changes
        elif isinstance(payload_or_changes, list):
            changes_list = payload_or_changes
        else:
            changes_list = [payload_or_changes]

        cat_list = []
        ab_list = []
        delta_list = []
        mech_list = []
        has_num_list = []

        for chg in changes_list:
            c, a, d, m, hn = cls.extract_features(chg)
            cat_list.append(c)
            ab_list.append(a)
            delta_list.append(d)
            mech_list.append(m)
            has_num_list.append(hn)

        if not cat_list:
            # Fallback dummy single row
            cat_list = [0]
            ab_list = [0]
            delta_list = [0.0]
            mech_list = [0.0]
            has_num_list = [0.0]

        t_cat = torch.tensor(cat_list, dtype=torch.long, device=device)
        t_ab = torch.tensor(ab_list, dtype=torch.long, device=device)
        t_delta = torch.tensor(delta_list, dtype=torch.float32, device=device).unsqueeze(1)
        t_mech = torch.tensor(mech_list, dtype=torch.float32, device=device).unsqueeze(1)
        t_has_num = torch.tensor(has_num_list, dtype=torch.float32, device=device).unsqueeze(1)

        return {
            "category_indices": t_cat,
            "ability_indices": t_ab,
            "deltas": t_delta,
            "is_mechanical_removal": t_mech,
            "has_numeric_transition": t_has_num
        }


# ============================================================================
# 3. DIFFERENTIABLE BASE & ATTENTION GATING MODULE
# ============================================================================

class PatchEmbeddingBase(nn.Module):
    """
    Differentiable PyTorch Module replacing static scalar heuristics.
    
    Learns:
    1. category_elasticities: Parameter matrix representing dynamic category weights (combat, ability, movement, etc.).
    2. power_budget_weights: Parameter matrix representing ability tier power budgets (signature, ultimate, etc.).
    3. W_attn & b_attn: Linear attention gating layers computing beta_dynamic = Sigmoid(W_attn * X_context + b).
    """
    def __init__(
        self,
        embed_dim: int = 16,
        num_categories: int = len(CATEGORY_MAP),
        num_ability_types: int = len(ABILITY_TYPE_MAP),
        context_dim: int = 32
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        self.num_ability_types = num_ability_types
        self.context_dim = context_dim

        # Trainable Parameter Matrices replacing static elasticities & power budget weights
        # Initialized around the baseline heuristics for stable initial convergence
        cat_init = torch.tensor(DEFAULT_CATEGORY_ELASTICITIES, dtype=torch.float32).unsqueeze(1).repeat(1, embed_dim)
        self.category_elasticities = nn.Parameter(cat_init + torch.randn_like(cat_init) * 0.02)

        ab_init = torch.tensor(DEFAULT_POWER_BUDGET_WEIGHTS, dtype=torch.float32).unsqueeze(1).repeat(1, embed_dim)
        self.power_budget_weights = nn.Parameter(ab_init + torch.randn_like(ab_init) * 0.02)

        # Categorical Embeddings for context representation
        self.cat_context_embed = nn.Embedding(num_categories, embed_dim)
        self.ab_context_embed = nn.Embedding(num_ability_types, embed_dim)

        # Context Projection Layer (projects categorical + continuous numerical features into d_ctx)
        # Numerical feature dim: delta (1) + is_mechanical_removal (1) + has_numeric_transition (1) = 3
        in_feature_dim = embed_dim * 2 + 3
        self.context_projection = nn.Sequential(
            nn.Linear(in_feature_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )

        # Attention Gating Mechanism: beta_dynamic = Sigmoid(W_attn * X_context + b)
        self.W_attn = nn.Linear(context_dim, 1)

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing differentiable shock vectors and dynamic attention gates.

        Args:
            input_tensors: Dict containing:
                - 'category_indices': Tensor of shape (M,)
                - 'ability_indices': Tensor of shape (M,)
                - 'deltas': Tensor of shape (M, 1)
                - 'is_mechanical_removal': Tensor of shape (M, 1)
                - 'has_numeric_transition': Tensor of shape (M, 1)

        Returns:
            Dict containing:
                - 'beta_dynamic': Attention gate tensor of shape (M, 1)
                - 'raw_shocks': Ungated shock embeddings of shape (M, embed_dim)
                - 'gated_shocks': Final dynamically gated shock embeddings of shape (M, embed_dim)
                - 'scalar_shocks': Aggregated scalar shock values of shape (M, 1)
        """
        cat_idx = input_tensors["category_indices"]
        ab_idx = input_tensors["ability_indices"]
        deltas = input_tensors["deltas"]
        is_mech = input_tensors["is_mechanical_removal"]
        has_num = input_tensors["has_numeric_transition"]

        # 1. Lookup learned category elasticity and power budget weights
        beta_cat = self.category_elasticities[cat_idx]  # Shape: (M, embed_dim)
        w_ab = self.power_budget_weights[ab_idx]         # Shape: (M, embed_dim)

        # 2. Build Context Tensor X_context
        cat_emb = self.cat_context_embed(cat_idx)        # Shape: (M, embed_dim)
        ab_emb = self.ab_context_embed(ab_idx)          # Shape: (M, embed_dim)

        raw_context_inputs = torch.cat([cat_emb, ab_emb, deltas, is_mech, has_num], dim=-1)
        X_context = self.context_projection(raw_context_inputs)  # Shape: (M, context_dim)

        # 3. Compute Attention Gate: beta_dynamic = Sigmoid(W_attn * X_context + b)
        beta_dynamic = torch.sigmoid(self.W_attn(X_context))      # Shape: (M, 1)

        # 4. Compute Raw Base Shock Vector: S_base = deltas * (w_ab * beta_cat)
        # Apply mechanical removal weighting rule:
        # Non-mechanical bug fixes receive zero weight, mechanical removals & balance tweaks receive full weight.
        # Weight multiplier = (1.0 - is_mech_bug) where non-mech bug has is_mech=0.
        effective_delta = torch.where(
            is_mech > 0.5,
            torch.tensor(-1.0, device=deltas.device),  # Fixed shock direction for mechanical exploit removal
            deltas
        )

        raw_shocks = effective_delta * (w_ab * beta_cat)  # Shape: (M, embed_dim)

        # 5. Apply Attention Gating: S_gated = beta_dynamic * raw_shocks
        gated_shocks = beta_dynamic * raw_shocks         # Shape: (M, embed_dim)

        # 6. Aggregate to scalar shock per change item for downstream prediction graph
        scalar_shocks = gated_shocks.sum(dim=-1, keepdim=True)  # Shape: (M, 1)

        return {
            "beta_dynamic": beta_dynamic,
            "raw_shocks": raw_shocks,
            "gated_shocks": gated_shocks,
            "scalar_shocks": scalar_shocks
        }


# ============================================================================
# CLI MOCK / DEMONSTRATION EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("--- TESTING V8 DIFFERENTIABLE BASE & ATTENTION GATING MODULE ---")

    # Sample extracted patch change dictionaries (Phase 1 output format)
    sample_changes = [
        {
            "agent": "Neon",
            "ability": "High Gear",
            "stat_modified": "Slide Speed",
            "old_value": 1.0,
            "new_value": 0.8,
            "is_mechanical_removal": False
        },
        {
            "agent": "Neon",
            "ability": "High Gear",
            "stat_modified": "Unintended Double Slide Boost Removal",
            "old_value": None,
            "new_value": None,
            "is_mechanical_removal": True
        },
        {
            "agent": "Omen",
            "ability": "Dark Cover",
            "stat_modified": "Round-End Audio Loop Bug Fix",
            "old_value": None,
            "new_value": None,
            "is_mechanical_removal": False
        }
    ]

    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)
    model = PatchEmbeddingBase(embed_dim=8, context_dim=16)

    output = model(tensors)

    print("\n1. Dynamic Attention Gates (beta_dynamic):")
    print(output["beta_dynamic"].detach().numpy())

    print("\n2. Gated Shock Embeddings Shape:")
    print(output["gated_shocks"].shape)

    print("\n3. Scalar Shock Values per change:")
    print(output["scalar_shocks"].detach().numpy())

    # Gradient Verification Test
    loss = output["gated_shocks"].sum()
    loss.backward()

    print("\n4. Gradient Verification:")
    print("Category Elasticities Grad Norm:", model.category_elasticities.grad.norm().item())
    print("W_attn Weight Grad Norm:", model.W_attn.weight.grad.norm().item())
    print("Gradient verification SUCCESSFUL!")
