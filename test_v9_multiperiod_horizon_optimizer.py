"""
test_v9_multiperiod_horizon_optimizer.py
----------------------------------------
Unit tests for v9 Multi-Period Horizon Optimizer and Stochastic Bracket Simulator.
"""

import pytest
import numpy as np
from v9_bracket_monte_carlo import StochasticBracketSimulator, compute_survival_probability
from v9_multiperiod_horizon_optimizer import execute_multiperiod_horizon_optimization
from v9_fantasy_engine import generate_v9_horizon_optimal_plan


def test_survival_probability_calculation():
    assert compute_survival_probability(3) == 1.0
    assert compute_survival_probability(2) == 0.75  # 1 - (0.5)^2
    assert compute_survival_probability(1) == 0.50  # Knockout match 50%
    assert compute_survival_probability(0) == 0.00


def test_stochastic_ev_matrix_generation():
    players = [
        {"name": "PlayerA", "team": "TeamUpper", "role": "Duelist", "price": 10.0, "ppg": 20.0},
        {"name": "PlayerB", "team": "TeamLower", "role": "Sentinel", "price": 8.0, "ppg": 20.0},
    ]
    simulator = StochasticBracketSimulator(stage_preset="Double Elimination Playoffs")
    simulator.set_team_lives("TeamUpper", lives=2)
    simulator.set_team_lives("TeamLower", lives=1)

    ev_matrix = simulator.calculate_stochastic_player_ev_matrix(
        players, horizon_weeks=4, known_schedule_weeks=2, risk_bias_mode="Balanced"
    )

    assert ev_matrix.shape == (2, 4)
    # Week 0 and 1 are deterministic: full 20.0 Pts
    assert ev_matrix[0, 0] == 20.0
    assert ev_matrix[1, 0] == 20.0
    # Week 2 (unassigned week 1): TeamLower (1 life = 0.50 survival) gets 20.0 * 0.50 = 10.0 Pts
    assert ev_matrix[1, 2] == 10.0
    # TeamUpper (2 lives = 0.75 survival) gets 20.0 * 0.75 = 15.0 Pts
    assert ev_matrix[0, 2] == 15.0


def test_multiperiod_horizon_optimization_execution():
    teams = ["Paper Rex", "Sentinels", "Fnatic", "DRX", "Gen.G", "LEVIATÁN"]
    roles = ["Duelist", "Initiator", "Controller", "Sentinel"]

    players = []
    pid = 1
    for t in teams:
        for r in roles:
            for i in range(2):
                players.append({
                    "name": f"P{pid}_{t}_{r}",
                    "team": t,
                    "role": r,
                    "price": 5.0 + (pid % 8),
                    "ppg": 10.0 + (pid % 12)
                })
                pid += 1

    res = execute_multiperiod_horizon_optimization(
        players=players,
        current_roster=None,
        horizon_weeks=3,
        budget_cap=100.0,
        roster_size=11,
        min_role_count=2,
        max_team_count=2,
        max_transfers_per_week=3
    )

    assert res.success is True
    assert len(res.weekly_rosters) == 3
    for w_roster in res.weekly_rosters:
        assert len(w_roster) == 11

    assert len(res.core_anchors) + len(res.swing_slots) > 0


def test_top_level_horizon_plan_api():
    plan = generate_v9_horizon_optimal_plan(
        horizon_weeks=4,
        budget_cap=100.0,
        stage_preset="Double Elimination Playoffs",
        risk_bias_mode="Balanced"
    )

    assert plan["success"] is True
    assert "total_horizon_ev" in plan
    assert len(plan["weekly_evs"]) == 4
