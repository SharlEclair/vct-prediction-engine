"""
Knapsack Solver Integration Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 4).

Executes Mixed-Integer Linear Programming (MILP) using PuLP to generate the optimal 6-man VFL roster.
Maximizes GPP tournament upside (Ceiling_p85) subject to salary cap, team roster caps, role requirements,
and IGL multiplier logic. Performs portfolio validation against 10,000 Monte Carlo simulation iterations.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import pulp

from copula_fusion import get_top_down_predictions, generate_independent_marginals, run_iman_conover_fusion, validate_and_extract_metrics
from covariance_profiler import extract_simulation_matrix, compute_spearman_covariance

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_player_slate() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingest Phase 3 fused projections and matrix, and construct rich slate metadata (salaries, roles, teams).
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Player Metadata DF, Fused Simulation Matrix DF)
    """
    logger.info("Ingesting Phase 3 fused copula outputs for slate setup...")
    
    # Run Phase 1 -> Phase 2 -> Phase 3 pipeline
    predictions_td = get_top_down_predictions()
    df_sim_matrix = extract_simulation_matrix(num_iterations=10000, seed=42)
    df_target_corr, _ = compute_spearman_covariance(df_sim_matrix)
    df_marginal = generate_independent_marginals(predictions_td, num_iterations=10000, seed=42)
    df_fused = run_iman_conover_fusion(df_marginal, df_target_corr)
    projections = validate_and_extract_metrics(df_marginal, df_fused, df_target_corr)
    
    # Expand player pool metadata across 4 distinct VCT teams (Sentinels, EDG, Fnatic, Paper Rex)
    # to satisfy the <= 2 player team cap constraint for a 6-man roster.
    metadata = [
        {"player_id": "P0_TeamA", "name": "Aspas",      "team": "Sentinels", "role": "Duelist",    "salary": 9.8},
        {"player_id": "P1_TeamA", "name": "Leo",        "team": "Sentinels", "role": "Initiator",  "salary": 8.5},
        {"player_id": "P2_TeamA", "name": "Chronicle",  "team": "Fnatic",    "role": "Flex",       "salary": 7.6},
        {"player_id": "P3_TeamA", "name": "Boaster",    "team": "Fnatic",    "role": "Controller", "salary": 7.0},
        {"player_id": "P4_TeamA", "name": "Alfa",       "team": "Fnatic",    "role": "Sentinel",   "salary": 6.5},
        {"player_id": "P5_TeamB", "name": "ZmjjKK",     "team": "EDG",       "role": "Duelist",    "salary": 9.5},
        {"player_id": "P6_TeamB", "name": "Nobody",     "team": "EDG",       "role": "Initiator",  "salary": 8.2},
        {"player_id": "P7_TeamB", "name": "CHICHOO",    "team": "EDG",       "role": "Controller", "salary": 7.2},
        {"player_id": "P8_TeamB", "name": "fORSKEN",    "team": "PRX",       "role": "Flex",       "salary": 7.5},
        {"player_id": "P9_TeamB", "name": "mindfreak",  "team": "PRX",       "role": "Sentinel",   "salary": 6.2},
    ]
    
    df_meta = pd.DataFrame(metadata)
    
    # Attach Phase 3 metrics (EV, Floor_p15, Ceiling_p85) to metadata
    df_meta["EV"] = df_meta["player_id"].map(lambda pid: projections[pid]["EV"])
    df_meta["Floor_p15"] = df_meta["player_id"].map(lambda pid: projections[pid]["Floor_p15"])
    df_meta["Ceiling_p85"] = df_meta["player_id"].map(lambda pid: projections[pid]["Ceiling_p85"])
    
    logger.info("Task 4.1 Complete: Ingested %d players into optimization slate.", len(df_meta))
    return df_meta, df_fused


def solve_vfl_knapsack(
    df_meta: pd.DataFrame, 
    salary_cap: float = 50.0, 
    igl_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Execute Tasks 4.1 - 4.5: Formulate and solve the MILP optimization model using PuLP.
    
    Args:
        df_meta (pd.DataFrame): Player metadata containing roles, teams, salaries, and Ceiling_p85.
        salary_cap (float): Maximum salary cap (50.0 VP).
        igl_multiplier (float): Multiplier bonus for designated IGL (1.5x).
        
    Returns:
        Dict[str, Any]: Optimization solution containing drafted lineup, roles, IGL, and stats.
    """
    logger.info("Formulating MILP Knapsack problem in PuLP...")
    
    players = df_meta["player_id"].tolist()
    n_players = len(players)
    
    # Define PuLP Problem (Maximize GPP Ceiling)
    prob = pulp.LpProblem("VFL_Knapsack_Optimizer", pulp.LpMaximize)
    
    # Task 4.1: Decision variables x_i (drafted) and y_i (IGL)
    x = pulp.LpVariable.dicts("draft", players, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("igl", players, cat=pulp.LpBinary)
    
    # Map attributes for quick solver lookup
    ceilings = dict(zip(df_meta["player_id"], df_meta["Ceiling_p85"]))
    salaries = dict(zip(df_meta["player_id"], df_meta["salary"]))
    roles = dict(zip(df_meta["player_id"], df_meta["role"]))
    teams = dict(zip(df_meta["player_id"], df_meta["team"]))
    
    # Task 4.5: Objective Function (Maximize Ceiling + IGL Multiplier bonus)
    # Total Ceiling = Sum(Ceiling_i * x_i + (IGL_Multiplier - 1.0) * Ceiling_i * y_i)
    prob += pulp.lpSum([ceilings[p] * x[p] + (igl_multiplier - 1.0) * ceilings[p] * y[p] for p in players]), "Total_GPP_Ceiling"
    
    # Task 4.2: Roster & Salary Constraints
    # 1. Exact 6-man lineup
    prob += pulp.lpSum([x[p] for p in players]) == 6, "Lineup_Size_6"
    
    # 2. Salary Cap <= 50.0 VP
    prob += pulp.lpSum([salaries[p] * x[p] for p in players]) <= salary_cap, "Salary_Cap_50VP"
    
    # 3. Max 2 players per real-world VCT team
    unique_teams = set(teams.values())
    for t in unique_teams:
        team_players = [p for p in players if teams[p] == t]
        prob += pulp.lpSum([x[p] for p in team_players]) <= 2, f"Team_Cap_{t}"
        
    # Task 4.3: Positional Role Constraints (1 Duelist, 1 Initiator, 1 Controller, 1 Sentinel, 2 Flex)
    role_counts = {"Duelist": 1, "Initiator": 1, "Controller": 1, "Sentinel": 1, "Flex": 2}
    for r, count in role_counts.items():
        role_players = [p for p in players if roles[p] == r]
        prob += pulp.lpSum([x[p] for p in role_players]) == count, f"Role_Req_{r}"
        
    # Task 4.4: IGL Multiplier Constraints
    # Exactly 1 IGL
    prob += pulp.lpSum([y[p] for p in players]) == 1, "Exactly_One_IGL"
    # Dependency: y_i <= x_i (Can only be IGL if drafted)
    for p in players:
        prob += y[p] <= x[p], f"IGL_Dependency_{p}"
        
    # Solve MILP model
    logger.info("Solving MILP model using COIN-OR / CBC solver...")
    solver_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    status_str = pulp.LpStatus[solver_status]
    logger.info("Solver Status: %s", status_str)
    assert status_str == "Optimal", f"Solver failed to find optimal solution! Status: {status_str}"
    
    # Extract optimal selection
    drafted_players = [p for p in players if x[p].varValue > 0.5]
    igl_player = [p for p in players if y[p].varValue > 0.5][0]
    
    total_salary = sum([salaries[p] for p in drafted_players])
    total_projected_ceiling = pulp.value(prob.objective)
    
    lineup_details = []
    for p in drafted_players:
        row = df_meta[df_meta["player_id"] == p].iloc[0]
        lineup_details.append({
            "player_id": p,
            "name": row["name"],
            "team": row["team"],
            "role": row["role"],
            "salary": row["salary"],
            "EV": row["EV"],
            "Ceiling_p85": row["Ceiling_p85"],
            "is_igl": (p == igl_player)
        })
        
    solution = {
        "status": status_str,
        "total_salary": total_salary,
        "projected_gpp_ceiling": total_projected_ceiling,
        "igl_player": igl_player,
        "lineup": lineup_details
    }
    
    return solution


def run_portfolio_simulation(
    solution: Dict[str, Any], 
    df_fused: pd.DataFrame, 
    igl_multiplier: float = 1.5
) -> Dict[str, float]:
    """
    Task 4.6: Portfolio Simulation Validation.
    Maps optimal lineup back against 10,000 raw simulation iterations to calculate true slate performance.
    
    Args:
        solution (Dict[str, Any]): Optimization solution.
        df_fused (pd.DataFrame): 10,000 x 10 fused simulation matrix.
        igl_multiplier (float): Multiplier for IGL (1.5x).
        
    Returns:
        Dict[str, float]: Aggregate portfolio metrics (Simulated Mean, Sim Ceiling p85, Sim Floor p15).
    """
    logger.info("Running Portfolio Simulation validation against 10,000 iterations...")
    
    drafted_pids = [item["player_id"] for item in solution["lineup"]]
    igl_pid = solution["igl_player"]
    
    # Compute aggregate lineup score for every iteration
    # Lineup score = sum of drafted player scores + (igl_multiplier - 1) * igl score
    sim_scores = df_fused[drafted_pids].sum(axis=1) + (igl_multiplier - 1.0) * df_fused[igl_pid]
    
    sim_metrics = {
        "simulated_lineup_mean": float(sim_scores.mean()),
        "simulated_lineup_floor_p15": float(np.percentile(sim_scores, 15)),
        "simulated_lineup_ceiling_p85": float(np.percentile(sim_scores, 85)),
        "simulated_lineup_max": float(sim_scores.max())
    }
    
    logger.info("Portfolio Validation Complete: Simulated Lineup Mean = %.2f | Ceiling (p85) = %.2f", 
                sim_metrics["simulated_lineup_mean"], sim_metrics["simulated_lineup_ceiling_p85"])
    return sim_metrics


def print_optimal_lineup_summary(solution: Dict[str, Any], sim_metrics: Dict[str, float]) -> None:
    """
    Task 4.5 Console Output: Print formatted optimal lineup and optimization summary.
    """
    df_lineup = pd.DataFrame(solution["lineup"])
    df_lineup["IGL"] = df_lineup["is_igl"].map(lambda x: "YES (1.5x)" if x else "")
    display_cols = ["name", "team", "role", "salary", "EV", "Ceiling_p85", "IGL"]
    
    print("\n" + "="*70)
    print("      HYBRID VALORANT DFS MICRO ENGINE (v6) - OPTIMAL GPP LINEUP")
    print("="*70)
    print(df_lineup[display_cols].to_string(index=False))
    print("-" * 70)
    print(f"Total Salary Used           : {solution['total_salary']:.1f} / 50.0 VP")
    print(f"Designated In-Game Leader   : {solution['lineup'][0]['name'] if solution['lineup'][0]['is_igl'] else [p['name'] for p in solution['lineup'] if p['is_igl']][0]}")
    print(f"Total Projected GPP Ceiling : {solution['projected_gpp_ceiling']:.2f} Pts")
    print("-" * 70)
    print(f"Simulated Lineup EV (Mean)  : {sim_metrics['simulated_lineup_mean']:.2f} Pts")
    print(f"Simulated Lineup Floor (p15): {sim_metrics['simulated_lineup_floor_p15']:.2f} Pts")
    print(f"Simulated Lineup Ceiling(p85: {sim_metrics['simulated_lineup_ceiling_p85']:.2f} Pts")
    print(f"Simulated Tournament Max    : {sim_metrics['simulated_lineup_max']:.2f} Pts")
    print("="*70 + "\n")


if __name__ == "__main__":
    df_meta_slate, df_fused_slate = prepare_player_slate()
    opt_solution = solve_vfl_knapsack(df_meta_slate, salary_cap=50.0, igl_multiplier=1.5)
    portfolio_results = run_portfolio_simulation(opt_solution, df_fused_slate, igl_multiplier=1.5)
    print_optimal_lineup_summary(opt_solution, portfolio_results)
