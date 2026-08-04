"""
v9_multiperiod_horizon_optimizer.py
-----------------------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 6: Multi-Period Horizon Roster Optimizer & Dynamic Transfer Path Engine.

This module reformulates roster optimization across K gameweeks using scipy.optimize.milp.
Decision Vector length 4N * K:
    x = [x_1, y_1, u_1, v_1,  ...,  x_K, y_K, u_K, v_K]^T
    x_{i, t} in {0, 1}: Drafted on roster in week t
    y_{i, t} in {0, 1}: Designated as IGL in week t
    u_{i, t} in {0, 1}: Transferred IN in week t
    v_{i, t} in {0, 1}: Transferred OUT in week t
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from v9_milp_optimizer import CANONICAL_ROLES, _normalize_role
from v9_bracket_monte_carlo import StochasticBracketSimulator, compute_survival_probability


@dataclass
class HorizonOptimizationResult:
    """Result container for multi-period horizon roster optimization."""
    success: bool
    status_message: str
    total_horizon_ev: float                           # Total accumulated EV across all K weeks
    weekly_evs: List[float]                          # Per-week projected points
    weekly_rosters: List[List[Dict[str, Any]]]        # Roster per week (1..K)
    weekly_igls: List[Optional[Dict[str, Any]]]       # IGL per week (1..K)
    weekly_transfers_in: List[List[Dict[str, Any]]]   # Transfers IN per week (1..K)
    weekly_transfers_out: List[List[Dict[str, Any]]]  # Transfers OUT per week (1..K)
    core_anchors: List[Dict[str, Any]]                # Players held for >= 3 weeks
    swing_slots: List[Dict[str, Any]]                 # Short-term tactical swing players


def execute_multiperiod_horizon_optimization(
    players: List[Dict[str, Any]],
    current_roster: Optional[List[Dict[str, Any]]] = None,
    horizon_weeks: int = 4,
    budget_cap: float = 100.0,
    roster_size: int = 11,
    min_role_count: int = 2,
    max_team_count: int = 2,
    max_transfers_per_week: int = 3,
    ev_matrix: Optional[np.ndarray] = None,
    stage_preset: str = "Double Elimination Playoffs",
    risk_bias_mode: str = "Balanced"
) -> HorizonOptimizationResult:
    """
    Executes Multi-Period Horizon Roster Optimization over K gameweeks.
    """
    N = len(players)
    K = horizon_weeks
    if N == 0:
        return HorizonOptimizationResult(
            success=False,
            status_message="Empty player pool provided.",
            total_horizon_ev=0.0,
            weekly_evs=[],
            weekly_rosters=[],
            weekly_igls=[],
            weekly_transfers_in=[],
            weekly_transfers_out=[],
            core_anchors=[],
            swing_slots=[]
        )

    # Compute or validate EV matrix N x K
    if ev_matrix is None:
        simulator = StochasticBracketSimulator(stage_preset=stage_preset)
        all_teams = sorted(list(set(p.get('team', '') for p in players)))
        simulator.configure_tier1_presets(all_teams)
        ev_matrix = simulator.calculate_stochastic_player_ev_matrix(
            players, horizon_weeks=K, known_schedule_weeks=2, risk_bias_mode=risk_bias_mode
        )

    # Map current roster binary vector x_0 of length N
    x_0 = np.zeros(N, dtype=np.float64)
    if current_roster and len(current_roster) > 0:
        curr_names = set(p.get('player_name', p.get('name', '')).lower().strip() for p in current_roster)
        for i, p in enumerate(players):
            p_name = p.get('name', p.get('player_name', '')).lower().strip()
            if p_name in curr_names:
                x_0[i] = 1.0

    # Decision Vector Layout (length 4N * K):
    # For each week t in 0..K-1:
    #   [x_{0..N-1, t}, y_{0..N-1, t}, u_{0..N-1, t}, v_{0..N-1, t}]
    total_vars = 4 * N * K

    def idx_x(i: int, t: int) -> int: return (4 * N * t) + i
    def idx_y(i: int, t: int) -> int: return (4 * N * t) + N + i
    def idx_u(i: int, t: int) -> int: return (4 * N * t) + 2 * N + i
    def idx_v(i: int, t: int) -> int: return (4 * N * t) + 3 * N + i

    # Cost Vector c (length 4N * K): We MINIMIZE c^T x, so c = -EV
    c = np.zeros(total_vars, dtype=np.float64)
    for t in range(K):
        for i in range(N):
            base_ev = float(ev_matrix[i, t])
            # Objective: Maximize sum(EV * x_{i,t}) + sum(EV * y_{i,t})  (IGL gets 2x total)
            c[idx_x(i, t)] = -base_ev
            c[idx_y(i, t)] = -base_ev

    # Bounds: all decision variables are binary in {0, 1}
    integrality = np.ones(total_vars, dtype=np.int32)
    bounds = Bounds(lb=0.0, ub=1.0)

    # Construct Linear Constraints
    constraints_A = []
    lb_list = []
    ub_list = []

    def add_constraint(row_dict: Dict[int, float], lb: float, ub: float):
        row = np.zeros(total_vars, dtype=np.float64)
        for var_idx, val in row_dict.items():
            row[var_idx] = val
        constraints_A.append(row)
        lb_list.append(lb)
        ub_list.append(ub)

    # 1. State Transition Constraints for each week t
    for t in range(K):
        for i in range(N):
            # x_{i, t} - x_{i, t-1} - u_{i, t} + v_{i, t} = 0
            row = {idx_x(i, t): 1.0, idx_u(i, t): -1.0, idx_v(i, t): 1.0}
            if t == 0:
                # x_{i, 0} - u_{i, 0} + v_{i, 0} = x_0[i]
                add_constraint(row, lb=x_0[i], ub=x_0[i])
            else:
                row[idx_x(i, t - 1)] = -1.0
                add_constraint(row, lb=0.0, ub=0.0)

    # 2. Per-Week Roster Rules & Transfer Caps
    teams = sorted(list(set(p.get('team', '') for p in players)))
    
    for t in range(K):
        # A. Exact Roster Size: sum_i x_{i, t} = roster_size
        add_constraint({idx_x(i, t): 1.0 for i in range(N)}, lb=float(roster_size), ub=float(roster_size))

        # B. Salary Cap: sum_i price_i * x_{i, t} <= budget_cap
        add_constraint({idx_x(i, t): float(players[i].get('price') or players[i].get('salary') or 5.0) for i in range(N)}, lb=0.0, ub=float(budget_cap))

        # C. Exact 1 IGL: sum_i y_{i, t} = 1
        add_constraint({idx_y(i, t): 1.0 for i in range(N)}, lb=1.0, ub=1.0)

        # D. IGL Roster Inclusion: y_{i, t} - x_{i, t} <= 0 for each i
        for i in range(N):
            add_constraint({idx_y(i, t): 1.0, idx_x(i, t): -1.0}, lb=-np.inf, ub=0.0)

        # E. Positional Role Bounds (for canonical roles)
        for role in CANONICAL_ROLES:
            role_indices = [i for i, p in enumerate(players) if _normalize_role(p.get('role', '')) == role]
            if role_indices:
                add_constraint({idx_x(i, t): 1.0 for i in role_indices}, lb=float(min_role_count), ub=np.inf)

        # F. Max Team Limit: sum_{i in team} x_{i, t} <= max_team_count
        for team_name in teams:
            team_indices = [i for i, p in enumerate(players) if p.get('team', '') == team_name]
            if team_indices:
                add_constraint({idx_x(i, t): 1.0 for i in team_indices}, lb=0.0, ub=float(max_team_count))

        # G. Max Transfers Cap per week: sum_i u_{i, t} <= max_transfers_per_week
        # Note: If t == 0 and x_0 is empty/unassigned, the initial draft is not constrained by weekly transfer cap.
        is_initial_draft = (t == 0) and (np.sum(x_0) < roster_size)
        if not is_initial_draft:
            add_constraint({idx_u(i, t): 1.0 for i in range(N)}, lb=0.0, ub=float(max_transfers_per_week))

    # Stack constraints
    A_mat = np.vstack(constraints_A)
    linear_constraints = LinearConstraint(A_mat, lb=lb_list, ub=ub_list)

    # Solve MILP
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=linear_constraints)

    if not res.success:
        return HorizonOptimizationResult(
            success=False,
            status_message=f"MILP Solver Failed: {res.status}",
            total_horizon_ev=0.0,
            weekly_evs=[],
            weekly_rosters=[],
            weekly_igls=[],
            weekly_transfers_in=[],
            weekly_transfers_out=[],
            core_anchors=[],
            swing_slots=[]
        )

    # Extract solution vector x_sol
    x_sol = np.round(res.x).astype(int)
    total_horizon_ev = float(-res.fun)

    weekly_evs = []
    weekly_rosters = []
    weekly_igls = []
    weekly_transfers_in = []
    weekly_transfers_out = []
    player_hold_counts: Dict[str, int] = {}

    for t in range(K):
        roster_t = []
        igl_t = None
        t_in = []
        t_out = []
        w_ev = 0.0

        for i in range(N):
            p = dict(players[i])
            is_drafted = bool(x_sol[idx_x(i, t)])
            is_igl = bool(x_sol[idx_y(i, t)])
            is_in = bool(x_sol[idx_u(i, t)])
            is_out = bool(x_sol[idx_v(i, t)])

            if is_drafted:
                p['is_igl'] = is_igl
                p['weekly_ev'] = float(ev_matrix[i, t])
                roster_t.append(p)
                w_ev += float(ev_matrix[i, t]) * (2.0 if is_igl else 1.0)
                if is_igl:
                    igl_t = p
                
                p_name = p.get('name', p.get('player_name', ''))
                player_hold_counts[p_name] = player_hold_counts.get(p_name, 0) + 1

            if is_in:
                t_in.append(p)
            if is_out:
                t_out.append(p)

        weekly_evs.append(round(w_ev, 2))
        weekly_rosters.append(roster_t)
        weekly_igls.append(igl_t)
        weekly_transfers_in.append(t_in)
        weekly_transfers_out.append(t_out)

    # Classify Core Anchors (held >= 3 weeks) vs Tactical Swing Slots
    core_anchors = []
    swing_slots = []
    seen = set()
    for t_roster in weekly_rosters:
        for p in t_roster:
            p_name = p.get('name', p.get('player_name', ''))
            if p_name not in seen:
                seen.add(p_name)
                holds = player_hold_counts.get(p_name, 0)
                p['weeks_held'] = holds
                if holds >= 3:
                    p['slot_type'] = "Core Anchor ⚓"
                    core_anchors.append(p)
                else:
                    p['slot_type'] = "Tactical Swing 🔄"
                    swing_slots.append(p)

    return HorizonOptimizationResult(
        success=True,
        status_message="Multi-Period Horizon Roster Optimization Solved Successfully.",
        total_horizon_ev=round(total_horizon_ev, 2),
        weekly_evs=weekly_evs,
        weekly_rosters=weekly_rosters,
        weekly_igls=weekly_igls,
        weekly_transfers_in=weekly_transfers_in,
        weekly_transfers_out=weekly_transfers_out,
        core_anchors=core_anchors,
        swing_slots=swing_slots
    )
