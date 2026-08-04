"""
v9_fantasy_engine.py
--------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 5: Full Pipeline Integration & Application Wiring.

Unified pipeline wrapper that hooks up Phase 1 (v9_historical_stats.py),
Phase 2 (v9_map_scenario_simulation.py), Phase 3 (v9_h2h_and_calibration.py),
and Phase 4 (v9_milp_optimizer.py) to provide a clean top-level API
`generate_v9_optimal_roster()` consumable by Streamlit WebUI and downstream applications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import glob
import logging
import numpy as np

# Import Phase 1-4 Core Mathematical Modules
from v9_historical_stats import (
    compute_exponential_decay_stats,
    compute_logistic_decay_stats,
    compute_bayesian_shrinkage_stats,
    compute_telemetry_zscores,
    apply_telemetry_modifiers,
    DecayStatsResult,
    ModifiedStatsResult,
    DEFAULT_ROLE_BENCHMARKS
)
from v9_map_scenario_simulation import (
    compute_map_margin_probabilities,
    compute_single_map_ev,
    compute_bo3_series_ev,
    MapEVResult,
    SeriesEVResult
)
from v9_h2h_and_calibration import (
    calculate_scaled_h2h_weight,
    compute_team_elo_proxy_multiplier,
    combine_h2h_prior_and_elo_proxy,
    execute_post_gw_calibration_pass,
    H2HEloBlendResult,
    CalibrationPassResult
)
from v9_milp_optimizer import (
    execute_roster_optimization_milp,
    compute_sortino_igl_score,
    MILPOptimizationResult,
    CANONICAL_ROLES
)

logger = logging.getLogger("v9_fantasy_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PROCESSED_DIR = "./data/processed"
RAW_DIR = "./data/raw"


def _normalize_price(price_val: Any) -> float:
    """Normalizes raw price into standard VP units (e.g. 8500 -> 8.5 VP, 8.5 -> 8.5 VP)."""
    try:
        p = float(price_val)
        if p > 100.0:
            return round(p / 1000.0, 1)
        return round(p, 1)
    except (ValueError, TypeError):
        return 8.0


def _load_default_player_pool() -> List[Dict[str, Any]]:
    """Loads player pool from processed json databases or constructs fallback."""
    json_paths = [
        os.path.join(PROCESSED_DIR, "vfl_players_db.json"),
        os.path.join(PROCESSED_DIR, "vfl_players.json"),
        os.path.join(PROCESSED_DIR, "global_player_ledger.json")
    ]
    
    for path in json_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "players" in data:
                        return data["players"]
                    elif isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Could not load player pool from {path}: {e}")
                
    # Default fallback pool if files are absent
    logger.info("Using internal fallback player pool for v9 engine.")
    teams = ["Paper Rex", "Sentinels", "Fnatic", "DRX", "Team Liquid", "Gen.G", "LEVIATÁN"]
    roles = ["Duelist", "Initiator", "Controller", "Sentinel"]
    pool = []
    pid = 1
    for role in roles:
        for team in teams:
            pool.append({
                "player_name": f"Player_{pid}",
                "name": f"Player_{pid}",
                "team": team,
                "team_name": team,
                "role": role,
                "price": round(5.0 + (pid % 7) * 1.0, 1),
                "ppg": round(14.0 + (pid % 8) * 2.0, 1),
                "adr": 130.0 + (pid % 5) * 10,
                "kast": 0.72 + (pid % 4) * 0.03,
                "fd": 0.10 + (pid % 3) * 0.02
            })
            pid += 1
    return pool


def generate_v9_optimal_roster(
    players: Optional[List[Dict[str, Any]]] = None,
    budget_cap: float = 100.0,
    matchup_pairs: Optional[List[Tuple[str, str]]] = None,
    team_elos: Optional[Dict[str, float]] = None,
    player_h2h_records: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    map_veto_probs: Optional[Dict[str, List[float]]] = None,
    map_margins: Optional[Dict[str, List[float]]] = None,
    use_risk_adjusted_igl: bool = False,
    sortino_tau: float = 12.0
) -> Dict[str, Any]:
    """
    Top-Level v9 Architecture Integration API.
    
    Sequential Data Flow:
    1. Step A (v9_historical_stats.py): Replaces legacy unweighted PPG with Bayesian decay stats
       and role-normalized telemetry modifiers (ADR, KAST%, FD) to compute EV_kill, Floor (CVaR_10),
       and Ceiling (CVaR_90).
    2. Step B (v9_h2h_and_calibration.py): Blends historical H2H data and applies cross-regional
       Team Elo proxy multipliers when player H2H samples are sparse (N < 2).
    3. Step C (v9_map_scenario_simulation.py): Evaluates discrete map margin probabilities, 2-0 BO3
       sweep bonus (+2 pts), and VFL Map Cap rules to compute EV_map. Yields EV_total = EV_kill + EV_map.
    4. Step D (v9_milp_optimizer.py): Solves the 2N decision vector MILP (x_i for roster, y_i for IGL)
       subject to all 6 matrix constraints (Roster=11, IGL=1, Budget<=100 VP, Role bounds [2,5], Team<=2).
    5. Step E: Formats output dictionary compatible with Streamlit UI visualization tables.
    """
    raw_players = players if players is not None else _load_default_player_pool()
    if not raw_players:
        return {
            "solver_status": "no_players",
            "total_cost": 0.0,
            "projected_points": 0.0,
            "igl_player": None,
            "optimal_roster": []
        }

    elos = team_elos or {
        "Paper Rex": 1750.0, "Sentinels": 1720.0, "Fnatic": 1740.0,
        "DRX": 1680.0, "Team Liquid": 1650.0, "Gen.G": 1700.0, "LEVIATÁN": 1660.0
    }
    
    processed_player_pool: List[Dict[str, Any]] = []

    for p in raw_players:
        pname = str(p.get("player_name") or p.get("name") or "Unknown").strip()
        pteam = str(p.get("team") or p.get("team_name") or p.get("team_short") or "FreeAgent").strip()
        prole = str(p.get("role") or "Duelist").strip()
        pprice = _normalize_price(p.get("price") if p.get("price") is not None else p.get("cost", 8.0))
        
        # ---------------------------------------------------------------------
        # STEP A: HISTORICAL STATS & TELEMETRY FUSION
        # ---------------------------------------------------------------------
        history_scores = p.get("scores_history") or p.get("fantasy_history") or [p.get("ppg", 15.0)]
        days_elapsed = p.get("days_elapsed") or list(range(len(history_scores) - 1, -1, -1))
        
        # Bayesian Shrinkage Baseline
        decay_result = compute_bayesian_shrinkage_stats(
            fantasy_points=history_scores,
            days_elapsed=days_elapsed,
            mu_prior=float(p.get("ppg", 20.0)),
            sigma_prior_sq=25.0
        )
        
        # Raw Telemetry & Z-Scores
        raw_adr = float(p.get("adr", DEFAULT_ROLE_BENCHMARKS.get(prole, DEFAULT_ROLE_BENCHMARKS["Global"])["adr_mean"]))
        raw_kast = float(p.get("kast", DEFAULT_ROLE_BENCHMARKS.get(prole, DEFAULT_ROLE_BENCHMARKS["Global"])["kast_mean"]))
        raw_fd = float(p.get("fd", DEFAULT_ROLE_BENCHMARKS.get(prole, DEFAULT_ROLE_BENCHMARKS["Global"])["fd_mean"]))
        
        z_adr, z_kast, z_fd = compute_telemetry_zscores(raw_adr, raw_kast, raw_fd, role=prole)
        
        # Telemetry-Modified EV & CVaR bounds
        mod_result = apply_telemetry_modifiers(
            base_ev=decay_result,
            z_kast=z_kast,
            z_adr=z_adr,
            z_fd=z_fd,
            beta_fd=0.5,
            beta_kast=1.0,
            beta_adr=1.0
        )
        
        ev_kill_base = mod_result.ev_modified
        cvar_10_floor = mod_result.cvar_10_modified
        cvar_90_ceiling = mod_result.cvar_90_modified
        
        # ---------------------------------------------------------------------
        # STEP B: H2H BLENDING & CROSS-REGIONAL ELO PROXIES
        # ---------------------------------------------------------------------
        opponent_team = None
        if matchup_pairs:
            for (t_a, t_b) in matchup_pairs:
                if pteam.lower() in str(t_a).lower() or str(t_a).lower() in pteam.lower():
                    opponent_team = str(t_b).strip()
                    break
                elif pteam.lower() in str(t_b).lower() or str(t_b).lower() in pteam.lower():
                    opponent_team = str(t_a).strip()
                    break
                    
        elo_a = elos.get(pteam, 1500.0)
        elo_b = elos.get(opponent_team, 1500.0) if opponent_team else 1500.0
        
        # Extract H2H records if available
        n_h2h_maps = 0.0
        ev_h2h_val = ev_kill_base
        if player_h2h_records and pname in player_h2h_records and opponent_team:
            opp_rec = player_h2h_records[pname].get(opponent_team, {})
            n_h2h_maps = float(opp_rec.get("n_maps", 0.0))
            ev_h2h_val = float(opp_rec.get("ev_h2h", ev_kill_base))
            
        h2h_blend_result = combine_h2h_prior_and_elo_proxy(
            ev_h2h=ev_h2h_val,
            ev_prior=ev_kill_base,
            n_maps=n_h2h_maps,
            elo_team_a=elo_a,
            elo_team_b=elo_b
        )
        
        ev_kill = h2h_blend_result.ev_final
        
        # ---------------------------------------------------------------------
        # STEP C: DISCRETE SCENARIO SIMULATION FOR MAP EV
        # ---------------------------------------------------------------------
        veto_p_wins = (map_veto_probs or {}).get(pteam, [0.55, 0.55, 0.50])
        veto_margins = (map_margins or {}).get(pteam, [2.0, 2.0, 0.0])
        
        series_ev_result = compute_bo3_series_ev(
            maps_p_win=veto_p_wins,
            maps_mu_margin=veto_margins,
            team_name=pteam,
            opponent_name=opponent_team or "Opponent"
        )
        
        ev_map = series_ev_result.total_series_ev
        
        # Total Expected Value summation: EV_total = EV_kill + EV_map
        ev_total = ev_kill + ev_map
        
        processed_player_pool.append({
            "player_name": pname,
            "name": pname,
            "team": pteam,
            "team_name": pteam,
            "role": prole,
            "price": pprice,
            "cost": pprice,
            "vp": pprice,
            "ev": ev_total,
            "ev_total": ev_total,
            "ppg": ev_total,
            "ev_kill": ev_kill,
            "ev_map": ev_map,
            "floor": cvar_10_floor,
            "cvar_10": cvar_10_floor,
            "ceiling": cvar_90_ceiling,
            "cvar_90": cvar_90_ceiling,
            "z_kast": z_kast,
            "z_adr": z_adr,
            "z_fd": z_fd,
            "opponent": opponent_team,
            "raw_dict": p
        })

    # -------------------------------------------------------------------------
    # STEP D: 2N MILP ROSTER OPTIMIZATION
    # -------------------------------------------------------------------------
    milp_result: MILPOptimizationResult = execute_roster_optimization_milp(
        players=processed_player_pool,
        budget_cap=budget_cap,
        roster_size=11,
        min_role_count=2,
        max_role_count=5,
        max_team_count=2,
        use_risk_adjusted_igl=use_risk_adjusted_igl,
        sortino_tau=sortino_tau
    )

    if not milp_result.success:
        return {
            "solver_status": "infeasible",
            "total_cost": 0.0,
            "projected_points": 0.0,
            "igl_player": None,
            "optimal_roster": []
        }

    # -------------------------------------------------------------------------
    # STEP E: UI COMPATIBILITY SCHEMA FORMATTING
    # -------------------------------------------------------------------------
    optimal_roster_formatted = []
    
    # Calculate role counts to determine wildcards
    role_counts = {}
    for p in milp_result.roster_players:
        r = p["role"]
        role_counts[r] = role_counts.get(r, 0) + 1
        
    role_slots_filled = {r: 0 for r in CANONICAL_ROLES}

    for p in milp_result.roster_players:
        pname = p["player_name"]
        is_igl = (milp_result.igl_player is not None and pname == milp_result.igl_player["player_name"])
        
        prole = p["role"]
        # Mark as wildcard if the core 2 mandatory role slots are already filled
        if role_slots_filled.get(prole, 0) >= 2:
            is_wildcard = True
        else:
            is_wildcard = False
            role_slots_filled[prole] = role_slots_filled.get(prole, 0) + 1

        optimal_roster_formatted.append({
            "player_name": pname,
            "name": pname,
            "team": p["team"],
            "team_name": p["team"],
            "role": prole,
            "price": p["price"],
            "cost": p["price"],
            "vp": p["price"],
            "ppg": round(p["ev_total"], 2),
            "ev_total": round(p["ev_total"], 2),
            "ev_kill": round(p["ev_kill"], 2),
            "ev_map": round(p["ev_map"], 2),
            "floor": round(p["floor"], 2),
            "cvar_10": round(p["cvar_10"], 2),
            "ceiling": round(p["ceiling"], 2),
            "cvar_90": round(p["cvar_90"], 2),
            "z_kast": round(p["z_kast"], 2),
            "z_adr": round(p["z_adr"], 2),
            "z_fd": round(p["z_fd"], 2),
            "is_igl": is_igl,
            "is_wildcard": is_wildcard,
            "opponent": p.get("opponent")
        })

    igl_name = milp_result.igl_player["player_name"] if milp_result.igl_player else None

    return {
        "solver_status": "optimal",
        "total_cost": round(milp_result.total_cost, 2),
        "projected_points": round(milp_result.total_ev, 2),
        "igl_player": igl_name,
        "optimal_roster": optimal_roster_formatted
    }


# Backwards compatibility alias for app.py
def optimize_roster(
    players: Optional[List[Dict[str, Any]]] = None,
    budget_cap: float = 100.0,
    matchup_pairs: Optional[List[Tuple[str, str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Backward-compatible wrapper mapping legacy optimize_roster calls to v9 engine."""
    return generate_v9_optimal_roster(
        players=players,
        budget_cap=budget_cap,
        matchup_pairs=matchup_pairs
    )


def generate_v9_horizon_optimal_plan(
    players: Optional[List[Dict[str, Any]]] = None,
    current_roster: Optional[List[Dict[str, Any]]] = None,
    horizon_weeks: int = 4,
    budget_cap: float = 100.0,
    roster_size: int = 11,
    min_role_count: int = 2,
    max_team_count: int = 2,
    max_transfers_per_week: int = 3,
    stage_preset: str = "Double Elimination Playoffs",
    lower_bracket_teams: Optional[List[str]] = None,
    risk_bias_mode: str = "Balanced"
) -> Dict[str, Any]:
    """
    v9 Architecture Horizon Plan Integration API.
    Computes multi-period dynamic transfer path over K gameweeks using Stochastic Bracket Simulation
    and scipy.optimize.milp multi-stage decision vector optimization.
    """
    from v9_bracket_monte_carlo import StochasticBracketSimulator
    from v9_multiperiod_horizon_optimizer import execute_multiperiod_horizon_optimization

    raw_players = players if players is not None else _load_default_player_pool()
    if not raw_players:
        return {
            "success": False,
            "status_message": "No players available.",
            "total_horizon_ev": 0.0,
            "weekly_evs": [],
            "weekly_rosters": [],
            "weekly_transfers_in": [],
            "weekly_transfers_out": [],
            "core_anchors": [],
            "swing_slots": []
        }

    # Normalize players
    norm_players = []
    for p in raw_players:
        np_item = dict(p)
        np_item['name'] = str(p.get("player_name") or p.get("name") or "Unknown").strip()
        np_item['team'] = str(p.get("team_name") or p.get("team") or p.get("team_short") or "FreeAgent").strip()
        np_item['role'] = str(p.get("role") or "Duelist").strip()
        np_item['price'] = _normalize_price(p.get("price") if p.get("price") is not None else p.get("cost", 8.0))
        np_item['computed_ppg'] = float(p.get("computed_ppg") or p.get("ppg") or 10.0)
        norm_players.append(np_item)

    # Configure Tier-1 Bracket Simulator
    simulator = StochasticBracketSimulator(stage_preset=stage_preset)
    all_teams = sorted(list(set(p['team'] for p in norm_players)))
    simulator.configure_tier1_presets(all_teams, lower_bracket_teams=lower_bracket_teams)

    ev_matrix = simulator.calculate_stochastic_player_ev_matrix(
        players=norm_players,
        horizon_weeks=horizon_weeks,
        known_schedule_weeks=2,
        risk_bias_mode=risk_bias_mode
    )

    res = execute_multiperiod_horizon_optimization(
        players=norm_players,
        current_roster=current_roster,
        horizon_weeks=horizon_weeks,
        budget_cap=budget_cap,
        roster_size=roster_size,
        min_role_count=min_role_count,
        max_team_count=max_team_count,
        max_transfers_per_week=max_transfers_per_week,
        ev_matrix=ev_matrix,
        stage_preset=stage_preset,
        risk_bias_mode=risk_bias_mode
    )

    return {
        "success": res.success,
        "status_message": res.status_message,
        "total_horizon_ev": res.total_horizon_ev,
        "weekly_evs": res.weekly_evs,
        "weekly_rosters": res.weekly_rosters,
        "weekly_igls": res.weekly_igls,
        "weekly_transfers_in": res.weekly_transfers_in,
        "weekly_transfers_out": res.weekly_transfers_out,
        "core_anchors": res.core_anchors,
        "swing_slots": res.swing_slots
    }

