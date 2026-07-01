"""
Knapsack Solver Integration Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 4 & 5).

Executes Mixed-Integer Linear Programming (MILP) using PuLP to generate the optimal 6-man VFL roster.
Maximizes GPP tournament upside (Ceiling_p85) subject to salary cap, team roster caps, role requirements,
and IGL multiplier logic. Performs portfolio validation against 10,000 Monte Carlo simulation iterations.

Decoupled to read static configuration from config.yaml and slate metadata from current_slate.json.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import pulp

from copula_fusion import get_top_down_predictions, generate_independent_marginals, run_iman_conover_fusion, validate_and_extract_metrics
from covariance_profiler import extract_simulation_matrix, compute_spearman_covariance
from utils import load_config, load_slate_payload

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_player_slate(num_iterations: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingest Phase 3 fused projections and matrix, and construct slate metadata dynamically from current_slate.json.
    
    Args:
        num_iterations (int): Monte Carlo simulation depth.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Player Metadata DF, Fused Simulation Matrix DF)
    """
    logger.info("Ingesting Phase 3 fused copula outputs and dynamic slate JSON payload...")
    
    # Run Phase 1 -> Phase 2 -> Phase 3 pipeline
    predictions_td = get_top_down_predictions()
    df_sim_matrix = extract_simulation_matrix(num_iterations=num_iterations, seed=42)
    
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent
    slate_path = root_dir / "data" / "processed" / "current_slate.json"
    
    player_ids = list(predictions_td.keys())
    k = len(player_ids)
    
    if k == 10:
        df_target_corr, _ = compute_spearman_covariance(df_sim_matrix)
    else:
        # Dynamically build a k x k target correlation matrix matching slate dimensions
        import json
        with open(slate_path, "r", encoding="utf-8") as f:
            slate_data = json.load(f)
        id_to_team = {item["player_id"]: item["team"] for item in slate_data}
        id_to_team.update({item["name"]: item["team"] for item in slate_data})
        
        teams = list(set(id_to_team.values()))
        C = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                t_i = id_to_team.get(player_ids[i])
                t_j = id_to_team.get(player_ids[j])
                if t_i is not None and t_j is not None:
                    if t_i == t_j:
                        C[i, j] = 0.45
                        C[j, i] = 0.45
                    else:
                        if len(teams) == 2:
                            C[i, j] = -0.40
                            C[j, i] = -0.40
                        else:
                            C[i, j] = -0.05
                            C[j, i] = -0.05
        df_target_corr = pd.DataFrame(C, index=player_ids, columns=player_ids)
        
    df_marginal = generate_independent_marginals(predictions_td, num_iterations=num_iterations, seed=42)
    df_fused = run_iman_conover_fusion(df_marginal, df_target_corr)
    projections = validate_and_extract_metrics(df_marginal, df_fused, df_target_corr)
    
    # Load dynamic slate payload (current_slate.json)
    metadata = load_slate_payload()
    df_meta = pd.DataFrame(metadata)
    
    # Attach Phase 3 metrics (EV, Floor_p15, Ceiling_p85) to metadata
    df_meta["EV"] = df_meta["player_id"].map(lambda pid: projections[pid]["EV"])
    df_meta["Floor_p15"] = df_meta["player_id"].map(lambda pid: projections[pid]["Floor_p15"])
    df_meta["Ceiling_p85"] = df_meta["player_id"].map(lambda pid: projections[pid]["Ceiling_p85"])
    
    logger.info("Task 5.3 Complete: Ingested %d players dynamically from current_slate.json.", len(df_meta))
    return df_meta, df_fused


def solve_vfl_knapsack(
    df_meta: pd.DataFrame, 
    salary_cap: float = None, 
    igl_multiplier: float = None
) -> Dict[str, Any]:
    """
    Execute MILP optimization using dynamic parameters loaded from config.yaml.
    
    Args:
        df_meta (pd.DataFrame): Player metadata containing roles, teams, salaries, and Ceiling_p85.
        salary_cap (float, optional): Maximum salary cap. Loaded from config.yaml if None.
        igl_multiplier (float, optional): Multiplier bonus for designated IGL. Loaded from config.yaml if None.
        
    Returns:
        Dict[str, Any]: Optimization solution containing drafted lineup, roles, IGL, and stats.
    """
    logger.info("Formulating MILP Knapsack problem in PuLP using configuration rules...")
    
    config = load_config()
    dfs_constraints = config.get("DFS_CONSTRAINTS", {})
    
    if salary_cap is None:
        salary_cap = float(dfs_constraints.get("salary_cap", 50.0))
    if igl_multiplier is None:
        igl_multiplier = float(dfs_constraints.get("igl_multiplier", 1.5))
        
    lineup_size = int(dfs_constraints.get("lineup_size", 6))
    max_per_team = int(dfs_constraints.get("max_players_per_team", 2))
    role_counts = dfs_constraints.get("role_counts", {"Duelist": 1, "Initiator": 1, "Controller": 1, "Sentinel": 1, "Flex": 2})
    
    players = df_meta["player_id"].tolist()
    
    # Define PuLP Problem (Maximize GPP Ceiling)
    prob = pulp.LpProblem("VFL_Knapsack_Optimizer", pulp.LpMaximize)
    
    # Decision variables x_i (drafted) and y_i (IGL)
    x = pulp.LpVariable.dicts("draft", players, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("igl", players, cat=pulp.LpBinary)
    
    ceilings = dict(zip(df_meta["player_id"], df_meta["Ceiling_p85"]))
    salaries = dict(zip(df_meta["player_id"], df_meta["salary"]))
    roles = dict(zip(df_meta["player_id"], df_meta["role"]))
    teams = dict(zip(df_meta["player_id"], df_meta["team"]))
    
    # Objective Function: Maximize Ceiling + IGL Multiplier bonus
    prob += pulp.lpSum([ceilings[p] * x[p] + (igl_multiplier - 1.0) * ceilings[p] * y[p] for p in players]), "Total_GPP_Ceiling"
    
    # Roster & Salary Constraints
    # 1. Lineup size constraint
    prob += pulp.lpSum([x[p] for p in players]) == lineup_size, f"Lineup_Size_{lineup_size}"
    
    # 2. Salary Cap constraint
    prob += pulp.lpSum([salaries[p] * x[p] for p in players]) <= salary_cap, f"Salary_Cap_{salary_cap}VP"
    
    # 3. Max players per real-world VCT team
    unique_teams = set(teams.values())
    for t in unique_teams:
        team_players = [p for p in players if teams[p] == t]
        prob += pulp.lpSum([x[p] for p in team_players]) <= max_per_team, f"Team_Cap_{t}"
        
    # Positional Role Constraints
    for r, count in role_counts.items():
        if r.lower() == "flex":
            continue
        role_players = [p for p in players if roles[p] == r]
        prob += pulp.lpSum([x[p] for p in role_players]) >= count, f"Role_Req_{r}"
        
    # IGL Multiplier Constraints
    prob += pulp.lpSum([y[p] for p in players]) == 1, "Exactly_One_IGL"
    for p in players:
        prob += y[p] <= x[p], f"IGL_Dependency_{p}"
        
    # Solve MILP model
    logger.info("Solving MILP model using COIN-OR / CBC solver...")
    solver_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    status_str = pulp.LpStatus[solver_status]
    logger.info("Solver Status: %s", status_str)
    assert status_str == "Optimal", f"Solver failed to find optimal solution! Status: {status_str}"
    
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
        "salary_cap": salary_cap,
        "projected_gpp_ceiling": total_projected_ceiling,
        "igl_player": igl_player,
        "lineup": lineup_details
    }
    
    return solution


def run_portfolio_simulation(
    solution: Dict[str, Any], 
    df_fused: pd.DataFrame, 
    igl_multiplier: float = None
) -> Dict[str, float]:
    """
    Task 4.6: Portfolio Simulation Validation.
    Maps optimal lineup back against 10,000 raw simulation iterations using config igl_multiplier.
    """
    logger.info("Running Portfolio Simulation validation against 10,000 iterations...")
    
    if igl_multiplier is None:
        config = load_config()
        igl_multiplier = float(config.get("DFS_CONSTRAINTS", {}).get("igl_multiplier", 1.5))
        
    drafted_pids = [item["player_id"] for item in solution["lineup"]]
    igl_pid = solution["igl_player"]
    
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
    Console Output: Print formatted optimal lineup and optimization summary.
    """
    df_lineup = pd.DataFrame(solution["lineup"])
    df_lineup["IGL"] = df_lineup["is_igl"].map(lambda x: "YES (1.5x)" if x else "")
    display_cols = ["name", "team", "role", "salary", "EV", "Ceiling_p85", "IGL"]
    
    sal_cap = solution.get("salary_cap", 50.0)
    
    print("\n" + "="*70)
    print("      HYBRID VALORANT DFS MICRO ENGINE (v6) - OPTIMAL GPP LINEUP")
    print("="*70)
    print(df_lineup[display_cols].to_string(index=False))
    print("-" * 70)
    print(f"Total Salary Used           : {solution['total_salary']:.1f} / {sal_cap:.1f} VP")
    print(f"Designated In-Game Leader   : {[p['name'] for p in solution['lineup'] if p['is_igl']][0]}")
    print(f"Total Projected GPP Ceiling : {solution['projected_gpp_ceiling']:.2f} Pts")
    print("-" * 70)
    print(f"Simulated Lineup EV (Mean)  : {sim_metrics['simulated_lineup_mean']:.2f} Pts")
    print(f"Simulated Lineup Floor (p15): {sim_metrics['simulated_lineup_floor_p15']:.2f} Pts")
    print(f"Simulated Lineup Ceiling(p85: {sim_metrics['simulated_lineup_ceiling_p85']:.2f} Pts")
    print(f"Simulated Tournament Max    : {sim_metrics['simulated_lineup_max']:.2f} Pts")
    print("="*70 + "\n")


if __name__ == "__main__":
    df_meta_slate, df_fused_slate = prepare_player_slate()
    opt_solution = solve_vfl_knapsack(df_meta_slate)
    portfolio_results = run_portfolio_simulation(opt_solution, df_fused_slate)
    print_optimal_lineup_summary(opt_solution, portfolio_results)
