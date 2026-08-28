"""
v9_h2h_and_calibration.py
-------------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 3: Head-to-Head Proxies & Post-Gameweek Calibration.

This module resolves cross-regional matchup sparsity using dynamic scaled H2H blending,
team-level Elo proxy multipliers for sparse Head-to-Head samples, and post-gameweek
momentum calibration learning loops.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class H2HEloBlendResult:
    """Result of combining H2H match history, baseline prior EV, and Team Elo proxy."""
    ev_final: float                  # Final combined EV projection
    ev_blended: float                # Blended EV before Elo proxy multiplier
    ev_h2h: float                    # Player's historical H2H EV vs opponent
    ev_prior: float                  # Player's baseline prior EV
    w_h2h: float                     # Dynamic H2H weight applied
    m_proxy: float                   # Applied Team Elo proxy multiplier
    n_maps: float                    # Number of H2H maps available


@dataclass
class CalibrationPassResult:
    """Result of running a post-gameweek calibration pass."""
    updated_priors: Dict[str, float]                  # Player ID -> updated prior mu for t+1
    error_deltas: Dict[str, float]                    # Player ID -> (Actual - Predicted)
    mean_absolute_error: float                        # Mean Absolute Error across evaluated players
    learning_rate: float                              # Base momentum learning rate alpha applied
    applied_alphas: Optional[Dict[str, float]] = None # Player ID -> applied alpha
    innovating_players: Optional[List[str]] = None    # Players triggering adaptive jump-diffusion


def calculate_scaled_h2h_weight(
    n_maps: float,
    n_threshold: float = 2.0,
    k: float = 1.5,
    max_weight: float = 0.70
) -> float:
    """
    Calculates the dynamic sigmoid Head-to-Head (H2H) weight.
    
    Formula:
        w_h2h(N) = max_weight / (1 + exp(-k * (N - N_threshold)))
        
    Behavior:
        - N = 0: w_h2h = 0.0
        - N = 2: w_h2h = 0.35 (35% H2H, 65% prior)
        - N >= 5: w_h2h approaches 0.70 ceiling
    """
    n = float(n_maps)
    if n <= 0:
        return 0.0
        
    w = max_weight / (1.0 + np.exp(-k * (n - n_threshold)))
    return float(w)


def compute_team_elo_proxy_multiplier(
    elo_team_a: float,
    elo_team_b: float,
    gamma: float = 0.15
) -> float:
    """
    Calculates the team-level cross-regional proxy multiplier when player H2H data is sparse.
    
    Formula:
        M_proxy = 1.0 + gamma * tanh((R_teamA - R_teamB) / 400)
        
    Bounds:
        Bounded strictly between (1.0 - gamma) and (1.0 + gamma), i.e., [0.85, 1.15].
    """
    delta_elo = float(elo_team_a - elo_team_b)
    multiplier = 1.0 + gamma * np.tanh(delta_elo / 400.0)
    return float(multiplier)


def combine_h2h_prior_and_elo_proxy(
    ev_h2h: float,
    ev_prior: float,
    n_maps: float,
    elo_team_a: float,
    elo_team_b: float,
    gamma: float = 0.15,
    n_threshold: float = 2.0,
    k: float = 1.5,
    max_weight: float = 0.70,
    sparse_h2h_threshold: float = 2.0
) -> H2HEloBlendResult:
    """
    Combines H2H EV, baseline prior EV, and Team Elo proxy.
    
    Mathematical Formulation:
        EV_blended = w_h2h * EV_h2h + (1 - w_h2h) * EV_prior
        If N < 2: EV_final = EV_blended * M_proxy
        If N >= 2: EV_final = EV_blended
    """
    w_h2h = calculate_scaled_h2h_weight(n_maps, n_threshold=n_threshold, k=k, max_weight=max_weight)
    ev_blended = float(w_h2h * ev_h2h + (1.0 - w_h2h) * ev_prior)
    
    m_proxy = compute_team_elo_proxy_multiplier(elo_team_a, elo_team_b, gamma=gamma)
    
    if n_maps < sparse_h2h_threshold:
        ev_final = float(ev_blended * m_proxy)
    else:
        ev_final = float(ev_blended)
        
    return H2HEloBlendResult(
        ev_final=ev_final,
        ev_blended=ev_blended,
        ev_h2h=float(ev_h2h),
        ev_prior=float(ev_prior),
        w_h2h=w_h2h,
        m_proxy=m_proxy,
        n_maps=float(n_maps)
    )


def execute_post_gw_calibration_pass(
    predicted_evs: Dict[str, float],
    actual_points: Dict[str, float],
    prior_mus: Dict[str, float],
    alpha: float = 0.20,
    previous_errors: Optional[Dict[str, float]] = None,
    innovation_threshold: float = 10.0,
    sustain_threshold: float = 5.0,
    adaptive_alpha: float = 0.60
) -> CalibrationPassResult:
    """
    Executes a Post-Gameweek Calibration Pass (Adaptive Momentum Learning Loop).
    
    Phase 4: Incorporates a Jump-Diffusion / Change-Point Detector.
    If a player experiences a massive breakout (error_t > 10.0) following prior over-performance
    (error_t_minus_1 > 5.0), the learning rate jumps from alpha=0.20 to adaptive_alpha=0.60
    to immediately absorb the structural shift into the player's prior baseline.
    
    Mathematical Formulation:
        epsilon_{i, t} = ActualPoints_{i, t} - PredictedEV_{i, t}
        is_innovating = (epsilon_{i, t} > 10.0) and (epsilon_{i, t-1} > 5.0)
        alpha_{i} = 0.60 if is_innovating else 0.20
        mu_{prior, t+1} = mu_{prior, t} + alpha_{i} * epsilon_{i, t}
    """
    updated_priors: Dict[str, float] = {}
    error_deltas: Dict[str, float] = {}
    applied_alphas: Dict[str, float] = {}
    innovating_players: List[str] = []
    abs_errors: List[float] = []
    
    prev_errs = previous_errors or {}
    
    # Process all players in predicted set or prior set
    all_player_ids = set(predicted_evs.keys()).union(set(prior_mus.keys()))
    
    for pid in sorted(all_player_ids):
        pred_ev = float(predicted_evs.get(pid, prior_mus.get(pid, 20.0)))
        prior_mu = float(prior_mus.get(pid, pred_ev))
        
        if pid in actual_points:
            act_pt = float(actual_points[pid])
            delta = float(act_pt - pred_ev)
            error_deltas[pid] = delta
            abs_errors.append(abs(delta))
            
            # Phase 4: Jump-Diffusion Adaptive Momentum
            error_t_minus_1 = float(prev_errs.get(pid, 0.0))
            is_innovating = (delta > innovation_threshold) and (error_t_minus_1 > sustain_threshold)
            
            player_alpha = float(adaptive_alpha if is_innovating else alpha)
            applied_alphas[pid] = player_alpha
            if is_innovating:
                innovating_players.append(pid)
                
            # Momentum prior update
            updated_priors[pid] = float(prior_mu + player_alpha * delta)
        else:
            # Unchanged prior if player did not participate in gameweek
            updated_priors[pid] = prior_mu
            
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    
    return CalibrationPassResult(
        updated_priors=updated_priors,
        error_deltas=error_deltas,
        mean_absolute_error=mae,
        learning_rate=float(alpha),
        applied_alphas=applied_alphas,
        innovating_players=innovating_players
    )
