"""
Unit Tests for Phase 21: Opponent-Aware Predictions & Bracket Match Inference Engine
"""

import pytest
from fantasy_engine import (
    compute_all_players_opponent_stats, blend_ev, optimize_roster, RAW_DIR
)
from bracket_matchup_predictor import BracketMatchupPredictor, MatchupPrediction


def test_blend_ev_fallback():
    global_stats = {"TenZ": {"ppg": 14.5, "sigma": 2.5}}
    h2h_stats = {"TenZ": {"LOUD": {"ppg": 20.0, "sigma": 1.5, "n_maps": 2}}}
    
    # Under 3 maps threshold -> fall back to global
    res = blend_ev(global_stats, h2h_stats, "LOUD", "TenZ")
    assert res["h2h_used"] is False
    assert res["ppg"] == 14.5


def test_blend_ev_weighting():
    global_stats = {"TenZ": {"ppg": 14.0, "sigma": 3.0}}
    h2h_stats = {"TenZ": {"LOUD": {"ppg": 20.0, "sigma": 2.0, "n_maps": 5}}}
    
    # 5 maps -> 50% H2H, 50% global -> blended PPG = 0.5*20 + 0.5*14 = 17.0
    res = blend_ev(global_stats, h2h_stats, "LOUD", "TenZ")
    assert res["h2h_used"] is True
    assert res["ppg"] == 17.0


def test_bracket_matchup_predictor_group_stage():
    predictor = BracketMatchupPredictor()
    groups = {
        "Group A": ["Fnatic", "Team Vitality", "BBL Esports", "Karmine Corp"],
        "Group B": ["Sentinels", "LOUD", "Paper Rex", "DRX"]
    }
    predictions = predictor.predict_group_stage(groups)
    
    # 4 teams in Group A -> 4*3/2 = 6 pairings; 4 teams in Group B -> 6 pairings. Total = 12
    assert len(predictions) == 12
    
    # Check that teams only play within group
    for p in predictions:
        assert isinstance(p, MatchupPrediction)
        t_a_in_grp_a = p.team_a in groups["Group A"]
        t_b_in_grp_a = p.team_b in groups["Group A"]
        assert t_a_in_grp_a == t_b_in_grp_a, "Teams across different groups should not be paired"


def test_compute_all_players_opponent_stats():
    opp_stats = compute_all_players_opponent_stats(RAW_DIR)
    assert isinstance(opp_stats, dict)
    # Check if any player has opponent entries
    if opp_stats:
        first_player = next(iter(opp_stats))
        assert isinstance(opp_stats[first_player], dict)


def test_optimize_roster_with_h2h_matchups():
    mock_players = [
        {"player_name": "something", "team_name": "Paper Rex", "role": "Duelist", "price": 10, "ppg": 15.0},
        {"player_name": "f0rsakeN", "team_name": "Paper Rex", "role": "Initiator", "price": 9, "ppg": 14.0},
        {"player_name": "TenZ", "team_name": "Sentinels", "role": "Controller", "price": 10, "ppg": 14.0},
        {"player_name": "johnqt", "team_name": "Sentinels", "role": "Sentinel", "price": 9, "ppg": 12.0},
        {"player_name": "Derke", "team_name": "Fnatic", "role": "Duelist", "price": 10, "ppg": 14.5},
        {"player_name": "Leo", "team_name": "Fnatic", "role": "Initiator", "price": 9, "ppg": 13.0},
    ]
    matchup_pairs = [("Paper Rex", "Sentinels"), ("Fnatic", "Team Liquid")]
    h2h_stats = {
        "TenZ": {"Paper Rex": {"ppg": 22.0, "sigma": 2.0, "n_maps": 6}}
    }
    
    res = optimize_roster(
        vfl_players=mock_players,
        salary_cap=60,
        roster_size=6,
        survival_threshold=0.0,
        matchup_pairs=matchup_pairs,
        h2h_stats=h2h_stats
    )
    
    assert res["solver_status"] == "optimal"
    roster_names = [p["player_name"] for p in res["optimal_roster"]]
    assert "TenZ" in roster_names


def test_prepare_player_slate_h2h_blending():
    from knapsack_solver import prepare_player_slate
    h2h_stats = {
        "Demon1": {
            "Paper Rex": {"n_maps": 5, "ppg": 30.0, "sigma": 4.0}
        }
    }
    matchup_pairs = [("ENVY", "Paper Rex")]
    df_meta, df_fused = prepare_player_slate(
        num_iterations=10000,
        matchup_pairs=matchup_pairs,
        h2h_stats=h2h_stats
    )
    assert not df_meta.empty
    demon1_rows = df_meta[df_meta["name"] == "Demon1"]
    if not demon1_rows.empty:
        assert demon1_rows.iloc[0]["h2h_used"] == True
        assert demon1_rows.iloc[0]["opponent"] == "Paper Rex"
