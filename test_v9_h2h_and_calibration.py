"""
test_v9_h2h_and_calibration.py
-------------------------------
Comprehensive unit test suite for v9_h2h_and_calibration.py.

Verifies:
1. Scaled H2H Weight:
   - Dynamic sigmoid activation starting at N=2 maps (w_h2h = 0.35).
   - Approaching max_weight (0.70) as sample size grows (N >= 5).
2. Team Elo Proxy Multiplier:
   - Correct EV inflation for favored teams (+400 Elo -> ~1.114x).
   - Correct EV deflation for outclassed teams (-400 Elo -> ~0.886x).
   - Bounded between [0.85, 1.15].
   - Applied strictly when player H2H data is sparse (N < 2).
3. Post-GW Calibration Pass (Momentum Learning Loop):
   - Proportional prior update based on prediction error delta (Actual - Predicted).
   - Correct handling of breakout and underperforming players.
   - Preserves prior for unplayed players.
"""

import pytest
import numpy as np
from v9_h2h_and_calibration import (
    calculate_scaled_h2h_weight,
    compute_team_elo_proxy_multiplier,
    combine_h2h_prior_and_elo_proxy,
    execute_post_gw_calibration_pass,
    H2HEloBlendResult,
    CalibrationPassResult
)


def test_scaled_h2h_weight_activation():
    """
    Verifies that H2H blending weight activates smoothly at N=2 maps
    and approaches 0.70 ceiling as N grows.
    """
    w_0 = calculate_scaled_h2h_weight(n_maps=0.0)
    w_1 = calculate_scaled_h2h_weight(n_maps=1.0)
    w_2 = calculate_scaled_h2h_weight(n_maps=2.0)
    w_5 = calculate_scaled_h2h_weight(n_maps=5.0)
    w_10 = calculate_scaled_h2h_weight(n_maps=10.0)
    
    # N=0 maps yields 0.0 weight
    assert w_0 == 0.0
    
    # N=1 map yields small weight (< 0.20)
    assert 0.0 < w_1 < 0.20
    
    # N=2 maps yields exactly 0.35 (35% weight)
    assert w_2 == pytest.approx(0.35)
    
    # N=5 maps approaches 0.70 ceiling (> 0.68)
    assert w_5 > 0.68
    assert w_10 == pytest.approx(0.70, abs=1e-3)
    
    # Monotonicity check
    assert w_0 < w_1 < w_2 < w_5 <= w_10


def test_team_elo_proxy_multiplier():
    """
    Verifies Elo proxy multiplier behavior for equal, favored, and outclassed teams.
    """
    # Equal teams -> Multiplier = 1.0
    m_equal = compute_team_elo_proxy_multiplier(elo_team_a=1500.0, elo_team_b=1500.0)
    assert m_equal == pytest.approx(1.0)
    
    # Favored team (+400 Elo diff) -> ~1.114x multiplier (+11.4% EV boost)
    m_favored = compute_team_elo_proxy_multiplier(elo_team_a=1800.0, elo_team_b=1400.0, gamma=0.15)
    expected_favored = 1.0 + 0.15 * np.tanh(1.0)
    assert m_favored == pytest.approx(expected_favored)
    assert m_favored > 1.10
    
    # Outclassed team (-400 Elo diff) -> ~0.886x multiplier (-11.4% EV penalty)
    m_underdog = compute_team_elo_proxy_multiplier(elo_team_a=1400.0, elo_team_b=1800.0, gamma=0.15)
    expected_underdog = 1.0 + 0.15 * np.tanh(-1.0)
    assert m_underdog == pytest.approx(expected_underdog)
    assert m_underdog < 0.90
    
    # Extreme Elo (+2000 Elo) is strictly bounded by 1.0 + gamma = 1.15
    m_extreme = compute_team_elo_proxy_multiplier(elo_team_a=3000.0, elo_team_b=1000.0, gamma=0.15)
    assert m_extreme <= 1.15


def test_combine_h2h_prior_and_elo_proxy_sparse_vs_dense():
    """
    Verifies that Elo proxy multiplier is applied when N < 2,
    and bypassed when N >= 2.
    """
    ev_h2h = 25.0
    ev_prior = 20.0
    elo_favored = 1800.0
    elo_opponent = 1400.0
    
    # Case 1: Sparse H2H (N = 0) -> Elo proxy applied to prior EV
    res_n0 = combine_h2h_prior_and_elo_proxy(
        ev_h2h=ev_h2h, ev_prior=ev_prior, n_maps=0.0,
        elo_team_a=elo_favored, elo_team_b=elo_opponent
    )
    assert res_n0.w_h2h == 0.0
    assert res_n0.ev_blended == pytest.approx(ev_prior)
    assert res_n0.ev_final == pytest.approx(ev_prior * res_n0.m_proxy)
    assert res_n0.ev_final > ev_prior
    
    # Case 2: Dense H2H (N = 3) -> Elo proxy bypassed, blended EV used directly
    res_n3 = combine_h2h_prior_and_elo_proxy(
        ev_h2h=ev_h2h, ev_prior=ev_prior, n_maps=3.0,
        elo_team_a=elo_favored, elo_team_b=elo_opponent
    )
    assert res_n3.ev_final == pytest.approx(res_n3.ev_blended)


def test_post_gw_calibration_pass_momentum_updates():
    """
    Verifies that post-GW calibration updates player priors proportionally
    to prediction error deltas (Actual - Predicted).
    """
    predicted_evs = {
        "player_breakout": 20.0,
        "player_slump": 25.0,
        "player_benched": 18.0
    }
    actual_points = {
        "player_breakout": 30.0,  # Delta = +10.0
        "player_slump": 15.0       # Delta = -10.0
    }
    prior_mus = {
        "player_breakout": 20.0,
        "player_slump": 25.0,
        "player_benched": 18.0
    }
    
    res = execute_post_gw_calibration_pass(
        predicted_evs=predicted_evs,
        actual_points=actual_points,
        prior_mus=prior_mus,
        alpha=0.20
    )
    
    # Breakout player: 20.0 + 0.20 * (+10.0) = 22.0
    assert res.updated_priors["player_breakout"] == pytest.approx(22.0)
    assert res.error_deltas["player_breakout"] == pytest.approx(10.0)
    
    # Slumping player: 25.0 + 0.20 * (-10.0) = 23.0
    assert res.updated_priors["player_slump"] == pytest.approx(23.0)
    assert res.error_deltas["player_slump"] == pytest.approx(-10.0)
    
    # Benched / Unplayed player retains existing prior 18.0
    assert res.updated_priors["player_benched"] == pytest.approx(18.0)
    assert "player_benched" not in res.error_deltas
    
    # MAE = (|10| + |-10|) / 2 = 10.0
    assert res.mean_absolute_error == pytest.approx(10.0)


def test_post_gw_calibration_adaptive_jump_diffusion():
    """
    Phase 4: Verifies Jump-Diffusion adaptive momentum when consecutive large errors occur.
    """
    predicted_evs = {
        "player_innovator": 18.0,
        "player_standard": 20.0
    }
    actual_points = {
        "player_innovator": 32.0,  # Delta = +14.0 (> 10.0 innovation threshold)
        "player_standard": 32.0    # Delta = +12.0 (> 10.0 innovation threshold, but no sustain history)
    }
    prior_mus = {
        "player_innovator": 18.0,
        "player_standard": 20.0
    }
    previous_errors = {
        "player_innovator": 8.0,   # Sustained prior error (> 5.0 sustain threshold)
        "player_standard": 2.0     # Low prior error (<= 5.0)
    }
    
    res = execute_post_gw_calibration_pass(
        predicted_evs=predicted_evs,
        actual_points=actual_points,
        prior_mus=prior_mus,
        alpha=0.20,
        previous_errors=previous_errors,
        innovation_threshold=10.0,
        sustain_threshold=5.0,
        adaptive_alpha=0.60
    )
    
    # Innovator triggers jump-diffusion (alpha = 0.60): 18.0 + 0.60 * 14.0 = 18.0 + 8.4 = 26.4
    assert "player_innovator" in res.innovating_players
    assert res.applied_alphas["player_innovator"] == pytest.approx(0.60)
    assert res.updated_priors["player_innovator"] == pytest.approx(26.4)
    
    # Standard player stays at baseline momentum (alpha = 0.20): 20.0 + 0.20 * 12.0 = 20.0 + 2.4 = 22.4
    assert "player_standard" not in res.innovating_players
    assert res.applied_alphas["player_standard"] == pytest.approx(0.20)
    assert res.updated_priors["player_standard"] == pytest.approx(22.4)


if __name__ == "__main__":
    pytest.main([__file__])

