"""
test_v9_fantasy_engine.py
--------------------------
Comprehensive unit and integration test suite for v9_fantasy_engine.py.

Verifies:
1. End-to-End Pipeline Integration:
   - Sequential integration of Step A (Telemetry & Decay), Step B (H2H & Elo Proxy),
     Step C (Map Scenario EV), and Step D (2N MILP Optimizer).
   - Valid output dictionary matching Streamlit UI table requirements.
2. Exact Roster & IGL Properties:
   - 11 roster players, exactly 1 IGL player, budget cap <= 100.0 VP.
3. Total EV Summation:
   - EV_total = EV_kill + EV_map for each player.
   - Total projected points equals sum of 11 roster EV_total values + 1x extra IGL EV.
4. Backward Compatibility:
   - Calling legacy alias `optimize_roster()` returns valid optimal result.
"""

import pytest
import numpy as np
from v9_fantasy_engine import (
    generate_v9_optimal_roster,
    optimize_roster
)


@pytest.fixture
def mock_integration_pool():
    """Generates a realistic VFL player pool of 28 players across 4 roles and 7 VCT teams."""
    teams = ["Paper Rex", "Sentinels", "Fnatic", "DRX", "Team Liquid", "Gen.G", "LEVIATÁN"]
    roles = ["Duelist", "Initiator", "Controller", "Sentinel"]
    
    players = []
    pid = 1
    
    for role_idx, role in enumerate(roles):
        for team_idx, team in enumerate(teams):
            price = round(5.0 + (pid % 7) * 1.0, 1)      # Costs between 5.0 and 11.0 VP
            ppg = round(14.0 + (pid % 8) * 2.0 + role_idx * 1.5, 1) # PPGs between 14.0 and 32.0 pts
            
            players.append({
                "player_name": f"Player_{pid}",
                "name": f"Player_{pid}",
                "role": role,
                "team": team,
                "team_name": team,
                "price": price,
                "cost": price,
                "ppg": ppg,
                "adr": 135.0 + (pid % 5) * 8.0,
                "kast": 0.73 + (pid % 4) * 0.03,
                "fd": 0.09 + (pid % 3) * 0.02,
                "scores_history": [ppg - 2.0, ppg + 1.0, ppg + 3.0]
            })
            pid += 1
            
    return players


def test_v9_fantasy_engine_e2e_integration(mock_integration_pool):
    """
    Verifies full Phase 1-5 pipeline integration from player ingestion to MILP optimization.
    """
    matchup_pairs = [("Paper Rex", "Sentinels"), ("Fnatic", "DRX")]
    team_elos = {"Paper Rex": 1780.0, "Sentinels": 1700.0, "Fnatic": 1750.0, "DRX": 1660.0}
    
    result = generate_v9_optimal_roster(
        players=mock_integration_pool,
        budget_cap=100.0,
        matchup_pairs=matchup_pairs,
        team_elos=team_elos
    )
    
    # Verify top-level status & schema
    assert result["solver_status"] == "optimal"
    assert result["total_cost"] <= 100.0
    assert result["projected_points"] > 0.0
    assert result["igl_player"] is not None
    
    roster = result["optimal_roster"]
    assert len(roster) == 11
    
    # Verify player schema keys expected by Streamlit UI
    required_keys = [
        "player_name", "team", "role", "price", "ppg", "ev_total",
        "ev_kill", "ev_map", "floor", "ceiling", "z_kast", "z_adr",
        "z_fd", "is_igl", "is_wildcard"
    ]
    for p in roster:
        for k in required_keys:
            assert k in p, f"Missing required key '{k}' in output player dictionary."
            
    # Verify exact IGL count = 1
    igl_count = sum(1 for p in roster if p["is_igl"])
    assert igl_count == 1
    
    # Verify total projected points equation (Sum of 11 roster EVs + 1x extra IGL EV)
    base_roster_sum = sum(p["ev_total"] for p in roster)
    igl_ev = next(p["ev_total"] for p in roster if p["is_igl"])
    assert result["projected_points"] == pytest.approx(base_roster_sum + igl_ev, abs=1e-2)


def test_v9_fantasy_engine_ev_kill_map_summation(mock_integration_pool):
    """Verifies that EV_total = EV_kill + EV_map for every player in the roster."""
    result = generate_v9_optimal_roster(players=mock_integration_pool)
    assert result["solver_status"] == "optimal"
    
    for p in result["optimal_roster"]:
        assert p["ev_total"] == pytest.approx(p["ev_kill"] + p["ev_map"], abs=0.03)


def test_v9_fantasy_engine_backward_compatibility_alias(mock_integration_pool):
    """Verifies that calling legacy optimize_roster alias yields valid optimal output."""
    result = optimize_roster(players=mock_integration_pool, budget_cap=95.0)
    
    assert result["solver_status"] == "optimal"
    assert result["total_cost"] <= 95.0
    assert len(result["optimal_roster"]) == 11


if __name__ == "__main__":
    pytest.main([__file__])
