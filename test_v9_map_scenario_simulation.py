"""
test_v9_map_scenario_simulation.py
-----------------------------------
Comprehensive unit test suite for v9_map_scenario_simulation.py.

Verifies:
1. Discrete Margin Buckets:
   - Gaussian CDF probability mapping for Margin 5-9, 10+, and 13-0 / 0-13 sweeps.
   - Monotonic probability shifts with increasing expected round differential (mu_margin).
2. Single Map Expected Value:
   - Base win payout (+1.0) and margin bonus/penalty points summation.
3. BO3 Series EV & Map Cap Rule:
   - 2-0 Sweep bonus (+2.0 * P(2-0)).
   - Map 3 conditional execution probability.
   - Map Cap rule (subtraction of lowest map EV when 3 maps are played).
"""

import pytest
import numpy as np
from v9_map_scenario_simulation import (
    compute_map_margin_probabilities,
    compute_single_map_ev,
    compute_bo3_series_ev,
    MapMarginProbabilities,
    MapEVResult,
    SeriesEVResult
)


def test_margin_probabilities_monotonicity():
    """Verifies that higher expected round margin (mu_margin) increases win/sweep probabilities."""
    probs_low = compute_map_margin_probabilities(mu_margin=1.0, p_win=0.55)
    probs_high = compute_map_margin_probabilities(mu_margin=6.0, p_win=0.80)
    probs_blowout = compute_map_margin_probabilities(mu_margin=11.0, p_win=0.95)
    
    # Margin 5-9 probability should be significantly higher at mu=6.0 than mu=1.0
    assert probs_high.p_margin_5_to_9 > probs_low.p_margin_5_to_9
    
    # 13-0 sweep probability should increase with extreme positive margin
    assert probs_blowout.p_sweep_13_0 > probs_high.p_sweep_13_0
    
    # Net margin bonus should strictly increase with higher mu_margin
    assert probs_blowout.ev_margin_bonus > probs_high.ev_margin_bonus > probs_low.ev_margin_bonus


def test_single_map_ev_calculation():
    """Verifies single map EV calculation combining win probability and margin bonuses."""
    # Case A: High margin win expected (mu = +7.0, p_win = 0.85)
    map_high = compute_single_map_ev(p_win=0.85, mu_margin=7.0, map_name="Haven")
    
    # Base win EV = 0.85 * 1.0 = 0.85
    assert map_high.base_win_ev == pytest.approx(0.85)
    assert map_high.margin_ev > 0.0
    assert map_high.total_map_ev == pytest.approx(map_high.base_win_ev + map_high.margin_ev)
    
    # Case B: Severe loss expected (mu = -8.0, p_win = 0.15)
    map_low = compute_single_map_ev(p_win=0.15, mu_margin=-8.0, map_name="Bind")
    assert map_low.margin_ev < 0.0
    assert map_low.total_map_ev < map_low.base_win_ev


def test_bo3_series_sweep_bonus():
    """Verifies that a guaranteed 2-0 sweep yields exactly the +2.0 sweep bonus."""
    # 100% win probability on Map 1 and Map 2
    res = compute_bo3_series_ev(
        maps_p_win=[1.0, 1.0, 0.5],
        maps_mu_margin=[5.0, 5.0, 0.0],
        team_name="Fnatic",
        opponent_name="Sentinels"
    )
    
    assert res.p_2_0_sweep == pytest.approx(1.0)
    assert res.ev_sweep_bonus == pytest.approx(2.0)
    assert res.p_map3_played == pytest.approx(0.0)
    assert res.ev_map3_played == pytest.approx(0.0)
    assert res.map_cap_discount == pytest.approx(0.0)
    
    # Total EV should equal Map 1 EV + Map 2 EV + 2.0 Sweep Bonus
    expected_total = res.ev_map1 + res.ev_map2 + 2.0
    assert res.total_series_ev == pytest.approx(expected_total)


def test_bo3_series_map_cap_discount():
    """
    Verifies that when 3 maps are played, the lowest map score is properly discounted
    according to the VFL Map Cap rule.
    """
    # Guaranteed split on first two maps (p1=1.0, p2=0.0) -> Map 3 played with 100% certainty
    res = compute_bo3_series_ev(
        maps_p_win=[1.0, 0.0, 0.8],
        maps_mu_margin=[6.0, -6.0, 4.0],
        team_name="Paper Rex",
        opponent_name="DRX"
    )
    
    assert res.p_2_0_sweep == pytest.approx(0.0)
    assert res.ev_sweep_bonus == pytest.approx(0.0)
    assert res.p_map3_played == pytest.approx(1.0)
    
    # Map Cap discount should equal 1.0 * min(ev_m1, ev_m2, ev_m3)
    min_ev = min(res.ev_map1, res.ev_map2, res.maps[2].total_map_ev)
    assert res.map_cap_discount == pytest.approx(min_ev)
    
    # Total series EV should equal sum of all 3 maps minus min map EV
    expected_series_ev = res.ev_map1 + res.ev_map2 + res.maps[2].total_map_ev - min_ev
    assert res.total_series_ev == pytest.approx(expected_series_ev)


def test_bo3_series_probabilistic_split():
    """Verifies intermediate BO3 probability dynamics (e.g. 60% win on each map)."""
    p_win = [0.60, 0.60, 0.50]
    mu_margin = [2.0, 2.0, 0.0]
    
    res = compute_bo3_series_ev(p_win, mu_margin)
    
    # P(2-0) = 0.6 * 0.6 = 0.36
    assert res.p_2_0_sweep == pytest.approx(0.36)
    
    # P(Map 3 Played) = 0.6*(1-0.6) + (1-0.6)*0.6 = 0.24 + 0.24 = 0.48
    assert res.p_map3_played == pytest.approx(0.48)
    
    # Map Cap discount should be non-zero and reduce total series EV
    assert res.map_cap_discount > 0.0
    
    # Total series EV equation check
    calc_total = res.ev_map1 + res.ev_map2 + res.ev_map3_played + res.ev_sweep_bonus - res.map_cap_discount
    assert res.total_series_ev == pytest.approx(calc_total)


if __name__ == "__main__":
    pytest.main([__file__])
