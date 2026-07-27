"""
test_v8_breakpoint_thresholds.py
---------------------------------
Unit and Integration tests for v8_breakpoint_thresholds.py.
Verifies STE autograd function, SoftBreakpointSurrogate relaxation,
and BreakpointShockEvaluator detection of Case A vs Case B phase shifts.
"""

import pytest
import torch
import torch.nn as nn
from v8_breakpoint_thresholds import (
    StraightThroughStep,
    HardBreakpointSTE,
    SoftBreakpointSurrogate,
    BreakpointShockEvaluator
)
from v8_differentiable_base import PatchEmbeddingBase, PatchTensorBuilder


def test_straight_through_step_autograd():
    """
    Verifies that StraightThroughStep:
    1. Evaluates strict discrete step in forward pass.
    2. Passes non-zero gradients straight through in backward pass without Dirac zero trap.
    """
    x = torch.tensor([140.0, 150.0, 160.0], requires_grad=True)
    threshold = 150.0

    out = StraightThroughStep.apply(x, threshold)

    # Forward pass: 140 < 150 -> 0.0, 150 >= 150 -> 1.0, 160 >= 150 -> 1.0
    assert torch.equal(out, torch.tensor([0.0, 1.0, 1.0]))

    # Backward pass: STE identity gradient
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.equal(x.grad, torch.tensor([1.0, 1.0, 1.0]))


def test_soft_breakpoint_surrogate_temperature():
    """
    Verifies that SoftBreakpointSurrogate:
    1. Outputs continuous values in (0, 1).
    2. Sharpens towards a hard step as tau_temp decreases.
    """
    x = torch.tensor([149.0, 150.0, 151.0])
    threshold = 150.0

    soft_wide = SoftBreakpointSurrogate(threshold=threshold, tau_temp=5.0)
    soft_sharp = SoftBreakpointSurrogate(threshold=threshold, tau_temp=0.1)

    val_wide = soft_wide(x)
    val_sharp = soft_sharp(x)

    # At x = 150.0, sigmoid(0) is exactly 0.5
    assert pytest.approx(val_wide[1].item(), 0.001) == 0.5
    assert pytest.approx(val_sharp[1].item(), 0.001) == 0.5

    # As tau_temp sharpens (0.1), x=149 gives near 0 and x=151 gives near 1
    assert val_sharp[0].item() < 0.01
    assert val_sharp[2].item() > 0.99


def test_case_a_vs_case_b_phase_shift():
    """
    Verifies the core breakpoint thesis:
    Case A: Damage 160 -> 155 (No 150 HP crossing) -> Zero threshold shock.
    Case B: Damage 155 -> 145 (Crosses 150 HP threshold) -> Full phase shift shock multiplier!
    """
    x_old = torch.tensor([[160.0], [155.0]], requires_grad=True)
    x_new = torch.tensor([[155.0], [145.0]], requires_grad=True)
    dummy_shocks = torch.ones((2, 4))

    evaluator = BreakpointShockEvaluator(threshold=150.0, mode="ste", crossing_weight=2.0)
    out = evaluator(x_old, x_new, dummy_shocks)

    crossing_signal = out["crossing_signal"]
    multiplier = out["phase_shift_multiplier"]

    # Case A crossing signal must be 0.0, Case B must be 1.0
    assert crossing_signal[0].item() == 0.0
    assert crossing_signal[1].item() == 1.0

    # Multiplier for Case A: 1.0, Case B: 1.0 + 2.0*1.0 = 3.0
    assert multiplier[0].item() == 1.0
    assert multiplier[1].item() == 3.0

    # Case B impact score is 3x Case A impact score!
    assert out["scalar_impact"][1].item() == 3.0 * out["scalar_impact"][0].item()


def test_full_pipeline_integration_phase2_plus_phase3():
    """
    Integrates Phase 1 NLP + Phase 2 Differentiable Base + Phase 3 Breakpoint Thresholding.
    """
    sample_changes = [
        {
            "agent": "Vandal",
            "ability": "Primary Fire",
            "stat_modified": "Headshot damage decreased from 160 to 155",
            "old_value": 160.0,
            "new_value": 155.0,
            "is_mechanical_removal": False
        },
        {
            "agent": "Phantom",
            "ability": "Primary Fire",
            "stat_modified": "Headshot damage decreased from 155 to 145",
            "old_value": 155.0,
            "new_value": 145.0,
            "is_mechanical_removal": False
        }
    ]

    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)
    phase2_model = PatchEmbeddingBase(embed_dim=8, context_dim=16)
    phase2_out = phase2_model(tensors)

    x_old = tensors["deltas"] + 155.0  # mock raw scale
    x_old[0] = 160.0
    x_old[1] = 155.0

    x_new = torch.tensor([[155.0], [145.0]])

    phase3_evaluator = BreakpointShockEvaluator(threshold=150.0, mode="ste")
    phase3_out = phase3_evaluator(x_old, x_new, phase2_out["gated_shocks"])

    # Backprop test across entire pipeline
    loss = phase3_out["fused_shocks"].sum()
    loss.backward()

    assert phase2_model.category_elasticities.grad is not None
    assert phase2_model.category_elasticities.grad.norm().item() > 0.0

    print("PHASE 2 + PHASE 3 INTEGRATION TEST PASSED!")


if __name__ == "__main__":
    test_straight_through_step_autograd()
    test_soft_breakpoint_surrogate_temperature()
    test_case_a_vs_case_b_phase_shift()
    test_full_pipeline_integration_phase2_plus_phase3()
    print("ALL BREAKPOINT THRESHOLD TESTS PASSED SUCCESSFULLY!")
