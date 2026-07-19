"""
Knapsack Solver Integration Module for Hybrid Valorant DFS Micro Engine (v7.9).

Two-pass Benders-style Right-Tail CVaR optimizer:
  Pass 1 (Master): Binary PuLP knapsack on EV to select candidate lineup.
  Pass 2 (Subproblem): All binaries fixed → pure continuous Rockafellar-Uryasev LP
    over the full 10,000 scenario matrix using scipy.optimize.milp (HiGHS interior-point).
    Correctly minimizes CVaR of NEGATIVE returns to target the right-tail ceiling.

Decoupled to read static configuration from config.yaml and slate metadata from current_slate.json.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import pulp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

from copula_fusion import get_top_down_predictions, generate_independent_marginals, run_iman_conover_fusion, validate_and_extract_metrics
from archive.covariance_profiler import extract_simulation_matrix, compute_spearman_covariance
from utils.utils import load_config, load_slate_payload, filter_slate_by_teams

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_player_slate(num_iterations: int = 10000, allowed_teams: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingest Phase 3 fused projections and matrix, and construct slate metadata dynamically from current_slate.json.
    Optionally filters by allowed_teams.
    
    Args:
        num_iterations (int): Monte Carlo simulation depth.
        allowed_teams (Optional[Any]): Optional set/list of allowed team names.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Player Metadata DF, Fused Simulation Matrix DF)
    """
    logger.info("Ingesting Phase 3 fused copula outputs and dynamic slate JSON payload...")
    
    # Run Phase 1 -> Phase 2 -> Phase 3 pipeline
    predictions_td = get_top_down_predictions(allowed_teams=allowed_teams)
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
        if allowed_teams:
            slate_data = filter_slate_by_teams(slate_data, allowed_teams)
            
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
    if allowed_teams:
        metadata = filter_slate_by_teams(metadata, allowed_teams)
        
    df_meta = pd.DataFrame(metadata)
    
    # Attach Phase 3 metrics (EV, Floor_p15, Ceiling_p85, CVaR 90, CVaR 10) to metadata
    df_meta["EV"] = df_meta["player_id"].map(lambda pid: projections[pid]["EV"])
    df_meta["Floor_p15"] = df_meta["player_id"].map(lambda pid: projections[pid]["Floor_p15"])
    df_meta["Ceiling_p85"] = df_meta["player_id"].map(lambda pid: projections[pid]["Ceiling_p85"])
    df_meta["cvar_90"] = df_meta["player_id"].map(lambda pid: projections[pid]["cvar_90"])
    df_meta["cvar_10"] = df_meta["player_id"].map(lambda pid: projections[pid]["cvar_10"])
    
    logger.info("Task 5.3 Complete: Ingested %d players dynamically from current_slate.json.", len(df_meta))
    return df_meta, df_fused


def _solve_right_tail_cvar_subproblem(
    drafted_players: List[str],
    igl_player: str,
    df_fused: pd.DataFrame,
    igl_multiplier: float,
    beta: float = 0.90
) -> float:
    """
    Pass 2 (Subproblem): Rockafellar-Uryasev right-tail CVaR LP.

    With all binary player selections fixed (x_p known), this reduces to a pure
    continuous LP solvable by HiGHS in milliseconds over S=10,000 scenarios.

    Correctly targets the RIGHT tail (top 1-beta = top 10% scenarios) by
    minimizing the CVaR of NEGATIVE simulated lineup returns:

        min_{alpha, y_s}  alpha + 1/((1-beta)*S) * sum(y_s)
        s.t.  y_s >= -R_s - alpha    for all s
              y_s >= 0               for all s

    where R_s = sum_p(x_p * Score_{p,s}) + (igl_mult-1) * Score_{igl,s}
    is the total lineup return in scenario s.
    """
    if df_fused is None or len(drafted_players) == 0:
        return 0.0

    # Build scenario return vector R_s (shape: S,)
    available = [p for p in drafted_players if p in df_fused.columns]
    R = df_fused[available].sum(axis=1).values
    if igl_player in df_fused.columns:
        R = R + (igl_multiplier - 1.0) * df_fused[igl_player].values
    S = len(R)

    # Decision vars: [alpha (1), y_s (S)]
    # Objective: alpha + 1/((1-beta)*S) * sum(y_s)  → minimize
    c = np.zeros(1 + S)
    c[0] = 1.0                                # alpha coefficient
    c[1:] = 1.0 / ((1.0 - beta) * S)         # y_s coefficients

    # Constraint: y_s >= -R_s - alpha  ↔  alpha + y_s >= -R_s
    # Written as: [1, 0...1...0] * [alpha, y_s] >= -R_s
    # scipy LinearConstraint: lb <= A @ x <= ub
    # Row s: alpha + y_s >= -R_s  →  lb = -R_s, ub = +inf
    # Build constraint matrix directly in sparse COO format.
    # Each row s has exactly 2 non-zeros: A[s,0]=1 (alpha) and A[s,1+s]=1 (y_s).
    # The dense Python loop previously materialized a full (S, 1+S) NumPy array before
    # calling csc_matrix — allocating ~800MB for S=10,000. Building COO triplets directly
    # costs O(S) time and O(S) memory (~160KB for S=10,000).
    row_idx = np.concatenate([np.arange(S, dtype=np.int32), np.arange(S, dtype=np.int32)])
    col_idx = np.concatenate([np.zeros(S, dtype=np.int32), np.arange(1, S + 1, dtype=np.int32)])
    data    = np.ones(2 * S, dtype=np.float64)
    A = csc_matrix((data, (row_idx, col_idx)), shape=(S, 1 + S))
    lb = -R                     # lower bound per constraint row
    ub = np.full(S, np.inf)    # no upper bound

    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(
        lb=np.concatenate([[-np.inf], np.zeros(S)]),   # alpha unbounded, y_s >= 0
        ub=np.full(n_vars, np.inf)
    )
    # All variables continuous (integrality=0)
    integrality = np.zeros(n_vars)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.success:
        alpha_opt = res.x[0]
        # Right-tail CVaR of lineup = -objective (we minimized CVaR of -R)
        right_tail_cvar = -res.fun
        logger.info("Right-tail CVaR (p90) subproblem: VaR α=%.3f, CVaR_90=%.3f", alpha_opt, right_tail_cvar)
        return float(right_tail_cvar)
    else:
        logger.warning("CVaR subproblem did not converge; falling back to EV.")
        return float(R.mean())


def solve_vfl_knapsack(
    df_meta: pd.DataFrame,
    salary_cap: float = None,
    igl_multiplier: float = None,
    lineup_size: int = None,
    max_per_team: int = None,
    role_counts: Dict[str, int] = None,
    df_fused: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Two-pass Benders-style Right-Tail CVaR Knapsack Optimizer.

    Pass 1 (Master): Binary PuLP knapsack using EV selects the optimal integer lineup.
    Pass 2 (Subproblem): With all player binaries fixed, the Rockafellar-Uryasev
      right-tail CVaR LP is solved over the full 10,000 scenario matrix via
      scipy.optimize.milp (HiGHS interior-point). This is a pure continuous LP
      (no integer branching) and resolves in milliseconds.

    Args:
        df_meta (pd.DataFrame): Player metadata.
        salary_cap (float, optional): Maximum salary cap.
        igl_multiplier (float, optional): IGL score multiplier.
        lineup_size (int, optional): Roster size.
        max_per_team (int, optional): Max players from same team.
        role_counts (Dict[str, int], optional): Minimum role counts.
        df_fused (pd.DataFrame, optional): Full 10k Monte Carlo scenario matrix.

    Returns:
        Dict[str, Any]: Optimal lineup solution with right-tail CVaR score.
    """
    logger.info("Pass 1: Formulating binary EV knapsack in PuLP...")

    config = load_config()
    dfs_constraints = config.get("DFS_CONSTRAINTS", {})

    if salary_cap is None:
        salary_cap = float(dfs_constraints.get("salary_cap", 50.0))
    if igl_multiplier is None:
        igl_multiplier = float(dfs_constraints.get("igl_multiplier", 2.0))
    if lineup_size is None:
        lineup_size = int(dfs_constraints.get("lineup_size", 6))
    if max_per_team is None:
        max_per_team = int(dfs_constraints.get("max_players_per_team", 2))
    if role_counts is None:
        role_counts = dfs_constraints.get("role_counts", {"Duelist": 1, "Initiator": 1, "Controller": 1, "Sentinel": 1, "Flex": 2})

    players = df_meta["player_id"].tolist()
    ev_scores = dict(zip(df_meta["player_id"], df_meta["EV"]))
    salaries = dict(zip(df_meta["player_id"], df_meta["salary"]))
    roles = dict(zip(df_meta["player_id"], df_meta["role"]))
    teams = dict(zip(df_meta["player_id"], df_meta["team"]))

    # --- Pass 1: Binary integer master problem on EV ---
    prob = pulp.LpProblem("VFL_Knapsack_EV_Master", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("draft", players, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("igl", players, cat=pulp.LpBinary)

    # Maximize expected value in master to keep integer solve tractable
    prob += pulp.lpSum([ev_scores[p] * x[p] + (igl_multiplier - 1.0) * ev_scores[p] * y[p] for p in players]), "EV_Master"

    prob += pulp.lpSum([x[p] for p in players]) == lineup_size, f"Lineup_Size_{lineup_size}"
    prob += pulp.lpSum([salaries[p] * x[p] for p in players]) <= salary_cap, f"Salary_Cap_{salary_cap}VP"

    unique_teams = set(teams.values())
    for t in unique_teams:
        team_players = [p for p in players if teams[p] == t]
        prob += pulp.lpSum([x[p] for p in team_players]) <= max_per_team, f"Team_Cap_{t}"

    for r, count in role_counts.items():
        if r.lower() == "flex":
            continue
        role_players = [p for p in players if roles[p] == r]
        prob += pulp.lpSum([x[p] for p in role_players]) >= count, f"Role_Req_{r}"

    prob += pulp.lpSum([y[p] for p in players]) == 1, "Exactly_One_IGL"
    for p in players:
        prob += y[p] <= x[p], f"IGL_Dependency_{p}"

    solver_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[solver_status]
    logger.info("Pass 1 Solver Status: %s", status_str)
    assert status_str == "Optimal", f"Pass 1 failed: {status_str}"

    drafted_players = [p for p in players if x[p].varValue > 0.5]
    igl_player = [p for p in players if y[p].varValue > 0.5][0]

    # --- Pass 2: Right-tail CVaR continuous LP subproblem on full scenario matrix ---
    if df_fused is not None:
        logger.info("Pass 2: Running right-tail Rockafellar-Uryasev CVaR subproblem over %d scenarios...", len(df_fused))
        projected_score = _solve_right_tail_cvar_subproblem(
            drafted_players, igl_player, df_fused, igl_multiplier, beta=0.90
        )
    else:
        projected_score = sum(ev_scores[p] for p in drafted_players) + (igl_multiplier - 1.0) * ev_scores.get(igl_player, 0.0)

    total_salary = sum([salaries[p] for p in drafted_players])

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
            "Ceiling_p85": row.get("Ceiling_p85", row["EV"]),
            "is_igl": (p == igl_player)
        })

    solution = {
        "status": status_str,
        "total_salary": total_salary,
        "salary_cap": salary_cap,
        "projected_gpp_ceiling": projected_score,
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
        igl_multiplier = float(config.get("DFS_CONSTRAINTS", {}).get("igl_multiplier", 2.0))
        
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
    df_lineup["IGL"] = df_lineup["is_igl"].map(lambda x: "YES (2x)" if x else "")
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
