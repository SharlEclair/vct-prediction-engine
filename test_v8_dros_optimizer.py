"""
test_v8_dros_optimizer.py
--------------------------
Unit and Integration tests for v8_dros_optimizer.py.
Verifies DRos optimistic shrinkage variance bounding, asymptotic equivalence as lambda -> inf,
Seq2Seq Direct Method reward prediction, and end-to-end Phase 1-5 pipeline gradient flow.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from v8_dros_optimizer import Seq2SeqRewardPredictor, DRosObjective
from v8_differentiable_base import PatchEmbeddingBase, PatchTensorBuilder
from v8_breakpoint_thresholds import BreakpointShockEvaluator
from v8_copula_aggregation import AgentGroupedCopulaAggregator


def test_dros_variance_bounding_under_extreme_propensity():
    """
    Verifies that optimistic shrinkage prevents infinite variance:
    Even when pi_0 -> 0.00001 (causing raw w -> 50,000!),
    w_lambda remains strictly bounded by sqrt(lambda) / 2.0.
    """
    lambda_shrink = 5.0
    dros_opt = DRosObjective(lambda_shrink=lambda_shrink)

    # Target policy pi_e = 0.5, extreme logging behavior policy pi_0 = 0.00001
    pi_e = torch.tensor([[0.5]])
    pi_0 = torch.tensor([[0.00001]])

    w_raw, w_lambda = dros_opt.compute_shrunk_weights(pi_e, pi_0)

    # Raw IPS weight is 50,000!
    assert w_raw.item() == pytest.approx(50000.0, abs=1.0)

    # Theoretical maximum of w_lambda is sqrt(lambda)/2 = sqrt(5)/2 ~ 1.11803
    max_theoretical_limit = math.sqrt(lambda_shrink) / 2.0

    assert w_lambda.item() <= max_theoretical_limit + 1e-4
    assert not torch.isnan(w_lambda).any()
    assert not torch.isinf(w_lambda).any()


def test_dros_asymptotic_equivalence_to_standard_dr():
    """
    Verifies that as lambda -> inf:
    w_lambda -> w_raw, recovering standard Doubly Robust IPS.
    """
    pi_e = torch.tensor([[0.6, 0.4]])
    pi_0 = torch.tensor([[0.3, 0.7]])

    # Large lambda (1e6) minimizes shrinkage
    dros_large_lambda = DRosObjective(lambda_shrink=1e6)
    w_raw, w_lambda = dros_large_lambda.compute_shrunk_weights(pi_e, pi_0)

    assert w_lambda[0, 0].item() == pytest.approx(w_raw[0, 0].item(), abs=1e-3)
    assert w_lambda[0, 1].item() == pytest.approx(w_raw[0, 1].item(), abs=1e-3)


def test_seq2seq_reward_predictor_dimensions():
    """Verifies Seq2Seq (LSTM) direct reward predictor outputs probabilities in (0, 1)."""
    B, seq_len, input_dim, A = 4, 12, 16, 4
    predictor = Seq2SeqRewardPredictor(input_dim=input_dim, hidden_dim=32, num_actions=A)

    state_seq = torch.randn(B, seq_len, input_dim)
    q_hat = predictor(state_seq)

    assert q_hat.shape == (B, A)
    assert (q_hat >= 0.0).all() and (q_hat <= 1.0).all()


def test_end_to_end_phase1_to_phase5_pipeline():
    """
    Full End-to-End Pipeline Verification across Phase 1 -> 2 -> 3 -> 4 -> 5:
    1. Phase 1 Mock Patch Payload
    2. Phase 2 PatchEmbeddingBase Gated Shocks
    3. Phase 3 BreakpointShockEvaluator STE Thresholds
    4. Phase 4 AgentGroupedCopulaAggregator Synergistic Drift
    5. Phase 5 DRos Objective Optimization Loss & Backprop
    """
    sample_changes = [
        {
            "agent": "Jett",
            "ability": "Cloudburst",
            "stat_modified": "Smoke duration decreased from 4.5s to 2.5s",
            "old_value": 4.5,
            "new_value": 2.5,
            "is_mechanical_removal": False
        },
        {
            "agent": "Jett",
            "ability": "Tailwind",
            "stat_modified": "Dash windup delay increased",
            "old_value": 0.0,
            "new_value": 0.75,
            "is_mechanical_removal": False
        }
    ]

    # Phase 1 & 2
    tensors = PatchTensorBuilder.payload_to_tensors(sample_changes)
    phase2_model = PatchEmbeddingBase(embed_dim=8, context_dim=16)
    phase2_out = phase2_model(tensors)

    # Phase 3
    x_old = torch.tensor([[4.5], [0.0]])
    x_new = torch.tensor([[2.5], [0.75]])
    phase3_evaluator = BreakpointShockEvaluator(threshold=3.0, mode="ste")
    phase3_out = phase3_evaluator(x_old, x_new, phase2_out["gated_shocks"])

    # Phase 4
    agent_names = ["Jett", "Jett"]
    phase4_copula = AgentGroupedCopulaAggregator(init_theta=1.8)
    drift_dict = phase4_copula(agent_names, phase3_out["fused_shocks"])
    jett_drift = drift_dict["Jett"]  # Shape: (1, 1)

    # Phase 5: DRos Optimization
    seq_predictor = Seq2SeqRewardPredictor(input_dim=16, hidden_dim=32, num_actions=2)
    mock_seq = torch.randn(1, 12, 16)
    q_hat = seq_predictor(mock_seq)

    pi_e = torch.cat([1.0 - jett_drift, jett_drift], dim=-1)  # Target policy
    pi_0 = torch.tensor([[0.001, 0.999]])  # Logging behavior policy with extreme shift
    logged_actions = torch.tensor([0])
    logged_rewards = torch.tensor([0.75])

    dros_opt = DRosObjective(lambda_shrink=5.0)
    dros_out = dros_opt(pi_e, pi_0, logged_actions, logged_rewards, q_hat)

    # Loss computation & backward pass
    loss = dros_opt.compute_loss(jett_drift, dros_out)
    phase2_model.zero_grad()
    phase4_copula.zero_grad()
    loss.backward()

    # Verify non-zero gradient backprop across whole graph
    assert phase2_model.category_elasticities.grad is not None
    assert phase4_copula.copula.raw_theta.grad is not None
    assert not torch.isnan(phase2_model.category_elasticities.grad).any()
    print("FULL END-TO-END PHASE 1-5 PIPELINE BACKPROP PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dros_variance_bounding_under_extreme_propensity()
    test_dros_asymptotic_equivalence_to_standard_dr()
    test_seq2seq_reward_predictor_dimensions()
    test_end_to_end_phase1_to_phase5_pipeline()
    print("ALL DROS OPTIMIZER TESTS PASSED SUCCESSFULLY!")
