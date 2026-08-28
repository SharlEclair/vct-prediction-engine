"""
test_v8_copula_aggregation.py
------------------------------
Unit and Integration tests for v8_copula_aggregation.py.
Verifies baseline theta=1.0 independent equivalence, theta > 1.0 synergy amplification,
upper-tail dependence lambda_U calculation, numerical stability, and autograd gradient flow.
"""

import pytest
import torch
import torch.nn as nn
from v8_copula_aggregation import GumbelCopulaAggregator, AgentGroupedCopulaAggregator
from v8_differentiable_base import PatchEmbeddingBase, PatchTensorBuilder
from v8_breakpoint_thresholds import BreakpointShockEvaluator


def test_gumbel_copula_baseline_independent():
    """
    Verifies that when theta = 1.0 (independent baseline):
    The Gumbel Copula output matches standard probabilistic union: 1.0 - (1-S1)*(1-S2)
    """
    copula_indep = GumbelCopulaAggregator(init_theta=1.0, trainable=False)
    shocks = torch.tensor([[0.3, 0.4]])

    out_copula = copula_indep(shocks)

    # Probabilistic Union: 1.0 - (1 - 0.3)*(1 - 0.4) = 1.0 - 0.7 * 0.6 = 0.58
    expected_prob_union = 1.0 - (1.0 - 0.3) * (1.0 - 0.4)

    assert out_copula.item() == pytest.approx(expected_prob_union, abs=1e-3)
    assert copula_indep.upper_tail_dependence.item() == pytest.approx(0.0, abs=1e-3)


def test_gumbel_copula_synergistic_amplification():
    """
    Verifies that when theta > 1.0 (synergistic coupling):
    The combined Concept Drift Index strictly exceeds independent probabilistic union.
    """
    copula_synergy = GumbelCopulaAggregator(init_theta=2.0, trainable=False)
    shocks = torch.tensor([[0.3, 0.4]])

    out_synergy = copula_synergy(shocks)
    prob_union_baseline = 1.0 - (1.0 - 0.3) * (1.0 - 0.4)  # 0.58

    assert out_synergy.item() > prob_union_baseline
    # For theta=2.0, upper-tail dependence lambda_U = 2 - sqrt(2) = 0.5858
    expected_lambda_u = 2.0 - (2.0 ** 0.5)
    assert copula_synergy.upper_tail_dependence.item() == pytest.approx(expected_lambda_u, abs=1e-3)


def test_single_and_multivariate_dimensions():
    """Verifies copula evaluation for single (d=1), bivariate (d=2), and multivariate (d=4) shock inputs."""
    copula = GumbelCopulaAggregator(init_theta=1.5)

    single_shock = torch.tensor([[0.25]])
    bivariate_shocks = torch.tensor([[0.25, 0.35]])
    multivariate_shocks = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    out_1 = copula(single_shock)
    out_2 = copula(bivariate_shocks)
    out_4 = copula(multivariate_shocks)

    assert out_1.shape == (1, 1)
    assert out_2.shape == (1, 1)
    assert out_4.shape == (1, 1)

    # Single shock should pass through identity: 0.25
    assert out_1.item() == pytest.approx(0.25, abs=1e-3)


def test_numerical_stability_and_autograd():
    """Verifies that autograd backprop flows cleanly through raw_theta without NaN/Inf crashes."""
    copula = GumbelCopulaAggregator(init_theta=1.8, trainable=True)
    shocks = torch.tensor([[0.5, 0.5]], requires_grad=True)

    drift = copula(shocks)
    loss = drift.sum()

    copula.zero_grad()
    loss.backward()

    assert not torch.isnan(drift).any()
    assert not torch.isinf(drift).any()
    assert copula.raw_theta.grad is not None
    assert not torch.isnan(copula.raw_theta.grad).any()
    assert copula.raw_theta.grad.item() != 0.0


def test_end_to_end_phase2_3_4_integration():
    """
    Integrates Phase 2 (Differentiable Base) + Phase 3 (Breakpoint Thresholding) + Phase 4 (Copula Aggregation).
    """
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
        }
    ]

    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)
    phase2_model = PatchEmbeddingBase(embed_dim=8, context_dim=16)
    phase2_out = phase2_model(tensors)

    x_old = torch.tensor([[160.0], [155.0]])
    x_new = torch.tensor([[155.0], [145.0]])

    phase3_evaluator = BreakpointShockEvaluator(threshold=150.0, mode="ste")
    phase3_out = phase3_evaluator(x_old, x_new, phase2_out["gated_shocks"])

    # Phase 4 Agent Grouped Copula Aggregation
    phase4_grouped_copula = AgentGroupedCopulaAggregator(init_theta=1.5)
    agent_names = ["Neon", "Neon"]
    drift_dict = phase4_grouped_copula(agent_names, phase3_out["fused_shocks"])

    assert "Neon" in drift_dict
    neon_drift = drift_dict["Neon"]
    assert neon_drift.shape == (1, 1)

    loss = neon_drift.sum()
    loss.backward()

    assert phase2_model.category_elasticities.grad is not None
    assert phase4_grouped_copula.copula.raw_theta.grad is not None
    print("END-TO-END PHASE 2+3+4 INTEGRATION TEST PASSED!")


def test_meta_network_adjacency_propagation():
    """
    Phase 2: Verifies that cross-agent Meta-Network Adjacency shifts drift
    to counter-meta agents and clips between [0.0, 1.0].
    """
    grouped_copula = AgentGroupedCopulaAggregator(init_theta=1.0)
    agent_names = ["Phoenix"]
    # Single shock of magnitude 0.40 on Phoenix
    shocks = torch.tensor([[0.40]])

    # Without meta-network: Phoenix = sigmoid(0.40) ≈ 0.5987, Sage not present
    expected_phoenix_drift = torch.sigmoid(torch.tensor(0.40)).item()
    raw_drift = grouped_copula(agent_names, shocks, enable_meta_network=False)
    assert raw_drift["Phoenix"].item() == pytest.approx(expected_phoenix_drift, abs=1e-3)
    assert "Sage" not in raw_drift

    # With meta-network: Phoenix = 0.5987, Sage = 0.5987 * 0.15 = 0.0898, Cypher = max(0, -0.05 * 0.5987) = 0.0
    meta_drift = grouped_copula(agent_names, shocks, enable_meta_network=True)
    expected_sage_drift = expected_phoenix_drift * 0.15
    assert meta_drift["Phoenix"].item() == pytest.approx(expected_phoenix_drift, abs=1e-3)
    assert "Sage" in meta_drift
    assert meta_drift["Sage"].item() == pytest.approx(expected_sage_drift, abs=1e-3)
    assert meta_drift["Cypher"].item() == pytest.approx(0.0, abs=1e-3)
    assert 0.0 <= meta_drift["Sage"].item() <= 1.0


if __name__ == "__main__":
    test_gumbel_copula_baseline_independent()
    test_gumbel_copula_synergistic_amplification()
    test_single_and_multivariate_dimensions()
    test_numerical_stability_and_autograd()
    test_end_to_end_phase2_3_4_integration()
    test_meta_network_adjacency_propagation()
    print("ALL COPULA AGGREGATION TESTS PASSED SUCCESSFULLY!")

