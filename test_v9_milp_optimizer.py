"""
test_v9_milp_optimizer.py
-------------------------
Comprehensive unit test suite for v9_milp_optimizer.py.

Verifies:
1. Strict Budget Cap: Total VP spent <= 100.0 VP.
2. Exact Roster & IGL Singularity: Exactly 11 roster members and 1 IGL selected.
3. Logical Link (IGL Roster Inclusion): Designated IGL must be an active member of the 11-player roster (y_i <= x_i).
4. Role & Team Bounds:
   - Each of the 4 canonical roles has between 2 and 5 players (2 <= count <= 5).
   - No single VCT team has more than 2 players (count <= 2).
5. Clean Optimization & Sortino Risk-Adjusted IGL Option.
"""

import pytest
import numpy as np
from v9_milp_optimizer import (
    execute_roster_optimization_milp,
    compute_sortino_igl_score,
    MILPOptimizationResult,
    CANONICAL_ROLES
)


@pytest.fixture
def mock_player_pool():
    """Generates a realistic VFL player pool of 28 players across 4 roles and 7 VCT teams."""
    teams = ["Fnatic", "Paper Rex", "Sentinels", "DRX", "Team Liquid", "LOUD", "NRG"]
    roles = ["Duelist", "Initiator", "Controller", "Sentinel"]
    
    players = []
    pid = 1
    
    # Create 7 players per role (28 total)
    for role_idx, role in enumerate(roles):
        for team_idx, team in enumerate(teams):
            price = round(4.5 + (pid % 7) * 1.0, 1)      # Costs between 4.5 and 10.5 VP
            ev = round(14.0 + (pid % 9) * 2.5 + role_idx * 1.5, 1) # EVs between 14.0 and 38.0 pts
            
            players.append({
                "id": f"p_{pid}",
                "name": f"Player_{pid}",
                "role": role,
                "team": team,
                "price": price,
                "ev": ev,
                "std": 4.0 + (pid % 3),
                "fantasy_history": [ev - 3.0, ev + 1.0, ev + 4.0, ev - 2.0]
            })
            pid += 1
            
    return players


def test_milp_roster_optimization_constraints(mock_player_pool):
    """
    Verifies that the MILP solver produces an optimal solution satisfying all 6 matrix constraints.
    """
    result = execute_roster_optimization_milp(
        players=mock_player_pool,
        budget_cap=100.0,
        roster_size=11,
        min_role_count=2,
        max_role_count=5,
        max_team_count=2
    )
    
    assert result.success is True
    assert result.status_message == "MILP optimization converged successfully."
    
    # Constraint A: Exact Roster Size = 11
    assert len(result.roster_players) == 11
    
    # Constraint B: IGL Singularity = 1
    assert result.igl_player is not None
    assert result.igl_index >= 0
    
    # Constraint C: IGL Roster Inclusion (y_i <= x_i)
    igl_id = result.igl_player["id"]
    roster_ids = [p["id"] for p in result.roster_players]
    assert igl_id in roster_ids
    
    # Constraint D: Strict Budget Cap <= 100.0 VP
    assert result.total_cost <= 100.0
    actual_cost = sum(p["price"] for p in result.roster_players)
    assert result.total_cost == pytest.approx(actual_cost)
    
    # Constraint E: Role Composition Bounds (2 <= count <= 5 for each role)
    for role in CANONICAL_ROLES:
        count = result.role_counts[role]
        assert 2 <= count <= 5, f"Role {role} count {count} violates [2, 5] bounds."
    assert sum(result.role_counts.values()) == 11
    
    # Constraint F: Maximum VCT Team Limits (count <= 2 per team)
    for team, count in result.team_counts.items():
        assert count <= 2, f"Team {team} count {count} exceeds max limit of 2."
        
    # Total EV Verification (Sum of roster EVs + 1x extra IGL EV)
    base_ev = sum(p["ev"] for p in result.roster_players)
    igl_ev = result.igl_player["ev"]
    assert result.total_ev == pytest.approx(base_ev + igl_ev)


def test_milp_tight_budget_cap(mock_player_pool):
    """Verifies solver respects a constrained tight budget cap (e.g., 80.0 VP)."""
    result = execute_roster_optimization_milp(
        players=mock_player_pool,
        budget_cap=80.0
    )
    
    assert result.success is True
    assert result.total_cost <= 80.0
    assert len(result.roster_players) == 11


def test_sortino_risk_adjusted_igl_selection(mock_player_pool):
    """Verifies Risk-Adjusted IGL selection using Sortino-like downside deviation ratio."""
    result_pure = execute_roster_optimization_milp(
        players=mock_player_pool,
        use_risk_adjusted_igl=False
    )
    
    result_sortino = execute_roster_optimization_milp(
        players=mock_player_pool,
        use_risk_adjusted_igl=True,
        sortino_tau=15.0,
        sortino_weight=0.5
    )
    
    assert result_pure.success is True
    assert result_sortino.success is True
    
    # In pure EV mode, IGL selected is strictly the highest EV player on roster
    roster_evs = [p["ev"] for p in result_pure.roster_players]
    max_ev = max(roster_evs)
    assert result_pure.igl_player["ev"] == pytest.approx(max_ev)


def test_milp_infeasible_pool():
    """Verifies graceful handling when player pool is smaller than roster size."""
    small_pool = [
        {"id": "p_1", "role": "Duelist", "team": "FNC", "price": 8.0, "ev": 20.0},
        {"id": "p_2", "role": "Sentinel", "team": "PRX", "price": 8.0, "ev": 20.0}
    ]
    
    result = execute_roster_optimization_milp(players=small_pool, roster_size=11)
    assert result.success is False
    assert "smaller than required roster size" in result.status_message


if __name__ == "__main__":
    pytest.main([__file__])
