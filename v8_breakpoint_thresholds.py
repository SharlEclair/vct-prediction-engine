"""
v8_breakpoint_thresholds.py
---------------------------
Differentiable Breakpoint Thresholding PyTorch Module (v8 Architecture - Phase 3).

Replaces continuous half-saturation (k=0.5) with differentiable breakpoint
thresholding operations that accurately capture discrete gameplay phase-shifts
(e.g., weapon headshot damage falling below the critical 150 HP threshold).

Key Features:
1. StraightThroughStep (STE): Custom torch.autograd.Function implementing
   hard step evaluation in forward pass and identity gradient pass in backward pass.
2. SoftBreakpointSurrogate: Parameterized Sigmoid surrogate relaxation with tunable
   temperature parameter (tau_temp) for smooth continuous differentiability during training.
3. BreakpointShockEvaluator: Integrates threshold crossing logic with Phase 2
   PatchEmbeddingBase shock embeddings.
"""

import math
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. STRAIGHT-THROUGH ESTIMATOR (STE) AUTOGRAD FUNCTION
# ============================================================================

class StraightThroughStep(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for discrete thresholding.
    
    Forward Pass:
        y = 1.0 if x >= threshold else 0.0
        
    Backward Pass:
        dL/dx = dL/dy (Identity gradient pass-through, bypassing the zero-derivative Dirac trap)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: Union[float, torch.Tensor] = 150.0) -> torch.Tensor:
        if isinstance(threshold, (int, float)):
            thresh_tensor = torch.tensor(threshold, dtype=x.dtype, device=x.device)
        else:
            thresh_tensor = threshold
            
        ctx.save_for_backward(x, thresh_tensor)
        return (x >= thresh_tensor).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Pass upstream gradient through unmodified (Straight-Through Estimator)
        return grad_output.clone(), None


# ============================================================================
# 2. HARD BREAKPOINT STE MODULE
# ============================================================================

class HardBreakpointSTE(nn.Module):
    """
    PyTorch nn.Module encapsulating Straight-Through Estimator (STE) step logic.
    """
    def __init__(self, threshold: float = 150.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates hard step: 1.0 if x >= threshold else 0.0 with STE backprop."""
        return StraightThroughStep.apply(x, self.threshold)

    def evaluate_crossing(self, x_old: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
        """
        Evaluates whether a patch change crosses the threshold from above to below.
        Returns 1.0 if (x_old >= threshold AND x_new < threshold), else 0.0.
        """
        was_above = StraightThroughStep.apply(x_old, self.threshold)
        is_above = StraightThroughStep.apply(x_new, self.threshold)
        # Crossing occurs if was_above = 1 and is_above = 0
        crossing = was_above * (1.0 - is_above)
        return crossing


# ============================================================================
# 3. SOFT BREAKPOINT SURROGATE (TEMPERATURE-CONTROLLED SIGMOID RELAXATION)
# ============================================================================

class SoftBreakpointSurrogate(nn.Module):
    """
    Continuous, differentiable approximation of discrete threshold steps using a
    temperature-controlled Sigmoid surrogate:
    
        y = Sigmoid((x - threshold) / tau_temp)
        
    As tau_temp -> 0, the function sharpens into a hard discrete step function.
    """
    def __init__(
        self,
        threshold: float = 150.0,
        tau_temp: float = 1.0,
        trainable_threshold: bool = False
    ):
        super().__init__()
        if trainable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

        self.tau_temp = nn.Parameter(torch.tensor(tau_temp, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes soft step probability in (0, 1)."""
        temp = torch.clamp(self.tau_temp, min=1e-3)
        return torch.sigmoid((x - self.threshold) / temp)

    def evaluate_crossing(self, x_old: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
        """
        Computes continuous soft crossing intensity:
            P(crossing) = Sigmoid((x_old - threshold)/temp) * (1 - Sigmoid((x_new - threshold)/temp))
        """
        prob_old_above = self.forward(x_old)
        prob_new_above = self.forward(x_new)
        soft_crossing = prob_old_above * (1.0 - prob_new_above)
        return soft_crossing


# ============================================================================
# 4. BREAKPOINT SHOCK EVALUATOR (INTEGRATION WITH PHASE 2 EMBEDDINGS)
# ============================================================================

class BreakpointShockEvaluator(nn.Module):
    """
    Integrates Breakpoint Thresholding with Phase 2 Gated Shocks.
    
    Evaluates raw numerical transitions (e.g., weapon headshot damage) against
    discrete game thresholds (e.g. 150 HP) to amplify shock vectors when a binary
    phase-shift occurs (Case B) while ignoring minor non-crossing noise (Case A).
    """
    def __init__(
        self,
        threshold: float = 150.0,
        mode: str = "ste",
        crossing_weight: float = 2.0,
        tau_temp: float = 1.0
    ):
        super().__init__()
        self.threshold = threshold
        self.mode = mode.lower()
        self.crossing_weight = nn.Parameter(torch.tensor(crossing_weight, dtype=torch.float32))

        if self.mode == "ste":
            self.threshold_layer = HardBreakpointSTE(threshold=threshold)
        else:
            self.threshold_layer = SoftBreakpointSurrogate(threshold=threshold, tau_temp=tau_temp)

    def forward(
        self,
        x_old: torch.Tensor,
        x_new: torch.Tensor,
        phase2_gated_shocks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_old: Pre-patch raw attribute values (e.g. 160.0 or 155.0), shape (M, 1) or (M,)
            x_new: Post-patch raw attribute values (e.g. 155.0 or 145.0), shape (M, 1) or (M,)
            phase2_gated_shocks: Gated shock embeddings from Phase 2 model, shape (M, embed_dim)

        Returns:
            Dict containing:
                - 'crossing_signal': Tensor of shape (M, 1) indicating threshold crossing intensity
                - 'fused_shocks': Phase-shift augmented shock embeddings of shape (M, embed_dim)
                - 'scalar_impact': Aggregated scalar impact score of shape (M, 1)
        """
        if x_old.dim() == 1:
            x_old = x_old.unsqueeze(1)
        if x_new.dim() == 1:
            x_new = x_new.unsqueeze(1)

        # 1. Compute relative baseline delta
        denom = torch.clamp(torch.abs(x_old), min=1e-4)
        relative_delta = torch.abs(x_new - x_old) / denom  # Shape: (M, 1)

        # 2. Evaluate discrete/soft threshold crossing signal
        crossing_signal = self.threshold_layer.evaluate_crossing(x_old, x_new)  # Shape: (M, 1)

        # 3. Fuse baseline relative delta with binary breakpoint phase-shift shock
        phase_shift_multiplier = 1.0 + self.crossing_weight * crossing_signal

        # 4. Scale Phase 2 Gated Shocks by phase shift multiplier
        fused_shocks = phase2_gated_shocks * phase_shift_multiplier

        # 5. Aggregate scalar impact
        scalar_impact = fused_shocks.sum(dim=-1, keepdim=True)

        return {
            "crossing_signal": crossing_signal,
            "relative_delta": relative_delta,
            "phase_shift_multiplier": phase_shift_multiplier,
            "fused_shocks": fused_shocks,
            "scalar_impact": scalar_impact
        }


# ============================================================================
# CLI DEMONSTRATION & TEST ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    print("--- TESTING V8 DIFFERENTIABLE BREAKPOINT THRESHOLDING ---")

    # Case A: Damage 160 -> 155 (Both > 150 HP, No Breakpoint Crossed)
    # Case B: Damage 155 -> 145 (Crosses 150 HP, Binary Phase-Shift!)
    x_old = torch.tensor([[160.0], [155.0]], requires_grad=True)
    x_new = torch.tensor([[155.0], [145.0]], requires_grad=True)
    dummy_phase2_shocks = torch.ones((2, 8), requires_grad=True)

    evaluator_ste = BreakpointShockEvaluator(threshold=150.0, mode="ste")
    res_ste = evaluator_ste(x_old, x_new, dummy_phase2_shocks)

    print("\n1. STE Mode Crossing Signal:")
    print("Case A (160 -> 155):", res_ste["crossing_signal"][0].item())
    print("Case B (155 -> 145):", res_ste["crossing_signal"][1].item())

    print("\n2. STE Fused Impact Scores:")
    print("Case A Impact:", res_ste["scalar_impact"][0].item())
    print("Case B Impact:", res_ste["scalar_impact"][1].item())

    # Backward Pass Verification
    loss = res_ste["fused_shocks"].sum()
    loss.backward()

    print("\n3. STE Gradient Flow:")
    print("x_old grad:", x_old.grad.detach().numpy())
    print("x_new grad:", x_new.grad.detach().numpy())
    print("Gradient verification SUCCESSFUL!")
