"""
test_v8_differentiable_base.py
-------------------------------
Unit and Integration tests for v8_differentiable_base.py.
Verifies PyTorch tensor builder, PatchEmbeddingBase forward pass,
attention gating calculation (beta_dynamic = sigmoid(W_attn * X_context + b)),
and backward pass gradient propagation.
"""

import pytest
import torch
import torch.nn as nn
from v8_differentiable_base import PatchEmbeddingBase, PatchTensorBuilder, CATEGORY_MAP, ABILITY_TYPE_MAP
from v8_patch_parser import V8PatchParser, PatchExtractionPayload


def test_tensor_builder_conversion():
    """Verifies that PatchTensorBuilder cleanly converts dictionary/Pydantic payloads into PyTorch tensors."""
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
            "agent": "Raze",
            "ability": "Blast Pack",
            "stat_modified": "Satchel double jump velocity boost removal",
            "old_value": None,
            "new_value": None,
            "is_mechanical_removal": True
        }
    ]

    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)

    assert isinstance(tensors["category_indices"], torch.Tensor)
    assert isinstance(tensors["deltas"], torch.Tensor)
    assert tensors["category_indices"].shape == (2,)
    assert tensors["deltas"].shape == (2, 1)
    assert tensors["is_mechanical_removal"].shape == (2, 1)

    # First change is slide movement (delta = (0.8 - 1.0)/1.0 = -0.2)
    assert pytest.approx(tensors["deltas"][0].item(), 0.01) == -0.2
    assert tensors["is_mechanical_removal"][0].item() == 0.0

    # Second change is mechanical exploit removal (is_mechanical_removal = 1.0)
    assert tensors["is_mechanical_removal"][1].item() == 1.0


def test_patch_embedding_base_forward_pass():
    """Verifies PatchEmbeddingBase forward pass outputs and attention gate values."""
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
            "agent": "Omen",
            "ability": "Dark Cover",
            "stat_modified": "Audio loop glitch fix",
            "old_value": None,
            "new_value": None,
            "is_mechanical_removal": False
        }
    ]

    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)
    model = PatchEmbeddingBase(embed_dim=8, context_dim=16)

    output = model(tensors)

    assert "beta_dynamic" in output
    assert "raw_shocks" in output
    assert "gated_shocks" in output
    assert "scalar_shocks" in output

    beta_dynamic = output["beta_dynamic"]
    assert beta_dynamic.shape == (2, 1)
    # Sigmoid output must be strictly bounded in (0, 1)
    assert torch.all(beta_dynamic > 0.0) and torch.all(beta_dynamic < 1.0)

    assert output["gated_shocks"].shape == (2, 8)
    assert output["scalar_shocks"].shape == (2, 1)


def test_gradient_flow_backpropagation():
    """Verifies that trainable parameter matrices receive non-zero gradients during backprop."""
    sample_changes = [
        {
            "agent": "Jett",
            "ability": "Tailwind",
            "stat_modified": "Dash duration decreased from 12s to 8s",
            "old_value": 12.0,
            "new_value": 8.0,
            "is_mechanical_removal": False
        },
        {
            "agent": "Cypher",
            "ability": "Trapwire",
            "stat_modified": "Re-arm timer increased from 1s to 2s",
            "old_value": 1.0,
            "new_value": 2.0,
            "is_mechanical_removal": False
        }
    ]

    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)
    model = PatchEmbeddingBase(embed_dim=12, context_dim=24)

    # Forward pass
    output = model(tensors)
    loss = output["gated_shocks"].pow(2).mean()

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Check parameter gradients
    assert model.category_elasticities.grad is not None
    assert model.category_elasticities.grad.norm().item() > 0.0

    assert model.power_budget_weights.grad is not None
    assert model.power_budget_weights.grad.norm().item() > 0.0

    assert model.W_attn.weight.grad is not None
    assert model.W_attn.weight.grad.norm().item() > 0.0

    print("GRADIENT FLOW VERIFIED SUCCESSFULLY!")


def test_end_to_end_nlp_to_pytorch():
    """End-to-end integration test: Wikitext -> Phase 1 NLP Payload -> TensorBuilder -> PatchEmbeddingBase."""
    sample_wikitext = """
    {{Infobox patch
    | version = 8.11
    | date = June 11, 2024
    }}
    == Agent Updates ==
    === Neon ===
    * High Gear
    ** Slide speed decreased from 1.0 >>> 0.8.
    ** Fixed a bug where Neon could execute an unintended double slide boost when cancelling animation.
    """

    parser = V8PatchParser(force_offline_mock=True)
    payload = parser.parse_wikitext(sample_wikitext, version="8.11")

    # Convert payload directly to tensors
    tensors = PatchTensorBuilder.payload_to_tensors(payload)
    model = PatchEmbeddingBase(embed_dim=16, context_dim=32)

    output = model(tensors)

    assert output["gated_shocks"].shape[0] == len(payload.changes)
    assert output["beta_dynamic"].shape[0] == len(payload.changes)
    print("END-TO-END INTEGRATION TEST PASSED!")


if __name__ == "__main__":
    test_tensor_builder_conversion()
    test_patch_embedding_base_forward_pass()
    test_gradient_flow_backpropagation()
    test_end_to_end_nlp_to_pytorch()
    print("ALL PYTORCH MODULE TESTS PASSED SUCCESSFULLY!")
