"""
v9_milp_optimizer.py
--------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 4: Multi-dimensional Stochastic Knapsack MILP & Dynamic IGL Selection.

This module reformulates roster optimization using scipy.optimize.milp with an expanded
2N decision variable vector x = [x_1, ..., x_N, y_1, ..., y_N]^T to natively handle dynamic
IGL doubling and enforce strict 11-player, 100 VP budget, role, and team constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


CANONICAL_ROLES: List[str] = ["Duelist", "Initiator", "Controller", "Sentinel"]


@dataclass
class MILPOptimizationResult:
    """Result container for the 2N MILP roster optimization solver."""
    success: bool
    status_message: str
    total_ev: float                          # Total projected VFL points (including 2x IGL)
    total_cost: float                        # Total VP spent
    roster_players: List[Dict[str, Any]]     # The 11 drafted players
    igl_player: Optional[Dict[str, Any]]     # Designated IGL player
    igl_index: int                           # Pool index of the designated IGL
    role_counts: Dict[str, int]              # Count of drafted players per canonical role
    team_counts: Dict[str, int]              # Count of drafted players per VCT team
    decision_vector: np.ndarray              # Raw 2N binary decision vector


def compute_sortino_igl_score(
    ev_total: float,
    scores_history: Optional[List[float]] = None,
    std_dev: Optional[float] = None,
    tau: float = 12.0,
    cvar_90: Optional[float] = None,
    eps: float = 1e-4
) -> float:
    """
    Computes a Sortino-like Risk-Adjusted ratio for IGL dynamic upside selection.
    
    Formula:
        Sortino = (EV_total - tau) / sigma_down
        where sigma_down = sqrt( E[ min(0, x - tau)^2 ] )
    """
    if scores_history and len(scores_history) > 0:
        arr = np.asarray(scores_history, dtype=np.float64)
        downside_diffs = np.minimum(0.0, arr - tau)
        sigma_down = float(np.sqrt(np.mean(downside_diffs ** 2)))
    elif std_dev is not None and std_dev > 0:
        # Approximate downside deviation from symmetric total standard deviation
        sigma_down = float(std_dev / 2.0)
    else:
        sigma_down = 3.0  # Default nominal downside deviation
        
    sigma_down = max(sigma_down, eps)
    sortino_ratio = (ev_total - tau) / sigma_down
    
    # Phase 3 Skill-Ceiling Elasticity: If CVaR_90 upside ceiling is elevated, provide an additional upside premium
    if cvar_90 is not None and ev_total > 0:
        upside_ratio = max((float(cvar_90) - ev_total) / ev_total, 0.0)
        sortino_ratio = sortino_ratio * (1.0 + 0.20 * upside_ratio)
        
    return float(sortino_ratio)


def _normalize_role(raw_role: str) -> str:
    """Helper to map raw role string to one of the 4 canonical VFL roles."""
    role_str = str(raw_role).strip().title()
    for canonical in CANONICAL_ROLES:
        if canonical.lower() in role_str.lower():
            return canonical
    return "Duelist"  # Fallback canonical role


def execute_roster_optimization_milp(
    players: List[Dict[str, Any]],
    budget_cap: float = 100.0,
    roster_size: int = 11,
    min_role_count: int = 2,
    max_role_count: int = 5,
    max_team_count: int = 2,
    use_risk_adjusted_igl: bool = False,
    sortino_tau: float = 12.0,
    sortino_weight: float = 0.5
) -> MILPOptimizationResult:
    """
    Executes Multi-dimensional Stochastic Knapsack MILP optimization using scipy.optimize.milp.
    
    Expanded Decision Vector (length 2N):
        x = [x_1, ..., x_N, y_1, ..., y_N]^T
        x_i in {0, 1}: Drafted on roster
        y_i in {0, 1}: Designated as IGL
        
    Cost Vector:
        c = -[EV_total, ..., EV_total, EV_igl_score, ..., EV_igl_score]^T
        Drafted & IGL player i receives -2 * EV_total (or Risk-Adjusted EV) in the objective.
        
    Matrix Constraints:
        Constraint A: sum(x_i) = 11 (Exact Roster Size)
        Constraint B: sum(y_i) = 1  (IGL Singularity)
        Constraint C: y_i - x_i <= 0 for all i (IGL Roster Inclusion)
        Constraint D: sum(P_i * x_i) <= 100.0 VP (Strict Budget Cap)
        Constraint E: 2 <= sum(R_role,i * x_i) <= 5 for 4 roles (Role Composition Bounds)
        Constraint F: sum(T_team,i * x_i) <= 2 for all teams (Max Team Limits)
    """
    n = len(players)
    if n < roster_size:
        return MILPOptimizationResult(
            success=False,
            status_message=f"Player pool size ({n}) is smaller than required roster size ({roster_size}).",
            total_ev=0.0,
            total_cost=0.0,
            roster_players=[],
            igl_player=None,
            igl_index=-1,
            role_counts={},
            team_counts={},
            decision_vector=np.zeros(2 * n)
        )
        
    # Extract player properties
    evs = np.array([float(p.get("ev", p.get("ev_total", p.get("points", 0.0)))) for p in players], dtype=np.float64)
    costs = np.array([float(p.get("price", p.get("cost", p.get("salary", p.get("vp", 0.0))))) for p in players], dtype=np.float64)
    roles = [_normalize_role(p.get("role", "Duelist")) for p in players]
    teams = [str(p.get("team", "FreeAgent")).strip() for p in players]
    
    # Evaluate IGL score vector
    igl_scores = np.copy(evs)
    if use_risk_adjusted_igl:
        for i, p in enumerate(players):
            history = p.get("fantasy_history", p.get("scores_history", None))
            std_dev = p.get("std", p.get("std_dev", None))
            cvar_90_val = p.get("cvar_90", p.get("ceiling", None))
            sortino = compute_sortino_igl_score(
                evs[i],
                scores_history=history,
                std_dev=std_dev,
                tau=sortino_tau,
                cvar_90=cvar_90_val
            )
            # Risk-adjusted upside multiplier for IGL position
            igl_scores[i] = evs[i] * (1.0 + sortino_weight * np.clip(sortino, -0.5, 2.0))
            
    # Objective cost vector c (minimize c^T x -> negative for maximization)
    # c has length 2N: [-evs, -igl_scores]
    c = -np.concatenate([evs, igl_scores])
    
    # -------------------------------------------------------------------------
    # MATRIX CONSTRAINTS CONSTRUCTION
    # -------------------------------------------------------------------------
    constraint_rows: List[np.ndarray] = []
    lbs: List[float] = []
    ubs: List[float] = []
    
    # Constraint A: Exact Roster Size (sum(x_i) = 11)
    row_a = np.zeros(2 * n, dtype=np.float64)
    row_a[:n] = 1.0
    constraint_rows.append(row_a)
    lbs.append(float(roster_size))
    ubs.append(float(roster_size))
    
    # Constraint B: IGL Singularity (sum(y_i) = 1)
    row_b = np.zeros(2 * n, dtype=np.float64)
    row_b[n:] = 1.0
    constraint_rows.append(row_b)
    lbs.append(1.0)
    ubs.append(1.0)
    
    # Constraint C: IGL Roster Inclusion (y_i - x_i <= 0 for all i)
    for i in range(n):
        row_c = np.zeros(2 * n, dtype=np.float64)
        row_c[i] = -1.0
        row_c[n + i] = 1.0
        constraint_rows.append(row_c)
        lbs.append(-1e9)  # Equivalent to -infinity
        ubs.append(0.0)
        
    # Constraint D: Strict Budget Cap (sum(P_i * x_i) <= budget_cap)
    row_d = np.zeros(2 * n, dtype=np.float64)
    row_d[:n] = costs
    constraint_rows.append(row_d)
    lbs.append(-1e9)
    ubs.append(float(budget_cap))
    
    # Constraint E: Role Composition Bounds (2 <= sum(R_role,i * x_i) <= 5 for 4 core roles)
    for canonical in CANONICAL_ROLES:
        row_e = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            if roles[i] == canonical:
                row_e[i] = 1.0
        constraint_rows.append(row_e)
        lbs.append(float(min_role_count))
        ubs.append(float(max_role_count))
        
    # Constraint F: Maximum VCT Team Limits (sum(T_team,i * x_i) <= 2 for each team)
    unique_teams = sorted(list(set(teams)))
    for team_name in unique_teams:
        row_f = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            if teams[i] == team_name:
                row_f[i] = 1.0
        constraint_rows.append(row_f)
        lbs.append(-1e9)
        ubs.append(float(max_team_count))
        
    # Assemble A matrix, lb, ub
    A = np.vstack(constraint_rows)
    linear_constraints = LinearConstraint(A, lbs, ubs)
    
    # All 2N decision variables are binary integer variables in {0, 1}
    integrality = np.ones(2 * n, dtype=np.int32)
    bounds = Bounds(0.0, 1.0)
    
    # Solve MILP
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=linear_constraints
    )
    
    if not res.success or res.x is None:
        return MILPOptimizationResult(
            success=False,
            status_message=f"MILP solver failed or returned infeasible: {res.status}",
            total_ev=0.0,
            total_cost=0.0,
            roster_players=[],
            igl_player=None,
            igl_index=-1,
            role_counts={},
            team_counts={},
            decision_vector=np.zeros(2 * n)
        )
        
    dec_vec = np.round(res.x).astype(int)
    x_vars = dec_vec[:n]
    y_vars = dec_vec[n:]
    
    drafted_indices = np.where(x_vars == 1)[0]
    igl_indices = np.where(y_vars == 1)[0]
    
    igl_idx = int(igl_indices[0]) if len(igl_indices) > 0 else -1
    igl_player = players[igl_idx] if igl_idx >= 0 else None
    
    roster_players = [players[i] for i in drafted_indices]
    total_cost = float(np.sum(costs[drafted_indices]))
    
    # Total Projected EV = sum of drafted player EVs + IGL extra EV
    base_roster_ev = float(np.sum(evs[drafted_indices]))
    extra_igl_ev = float(evs[igl_idx]) if igl_idx >= 0 else 0.0
    total_ev = base_roster_ev + extra_igl_ev
    
    # Role & Team breakdown counts
    role_counts = {role: 0 for role in CANONICAL_ROLES}
    team_counts = {team: 0 for team in unique_teams}
    
    for idx in drafted_indices:
        r_norm = roles[idx]
        t_name = teams[idx]
        role_counts[r_norm] = role_counts.get(r_norm, 0) + 1
        team_counts[t_name] = team_counts.get(t_name, 0) + 1
        
    return MILPOptimizationResult(
        success=True,
        status_message="MILP optimization converged successfully.",
        total_ev=total_ev,
        total_cost=total_cost,
        roster_players=roster_players,
        igl_player=igl_player,
        igl_index=igl_idx,
        role_counts=role_counts,
        team_counts=team_counts,
        decision_vector=dec_vec
    )
