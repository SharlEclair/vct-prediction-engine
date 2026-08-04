"""
v9_map_scenario_simulation.py
------------------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 2: Discrete Scenario Simulation for Map Expected Value (EV).

This module implements discrete scenario mapping logic to evaluate VFL map and BO3 series
scoring payouts based on VetoPredictor win probabilities and MapScoreRegressor expected
round margins. Integrates the VFL Map Cap rule (top 2 map scores kept in a 3-map series).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm


@dataclass
class MapMarginProbabilities:
    """Probabilities for discrete round margin outcome buckets for a single map."""
    p_win: float                      # Total win probability
    p_margin_5_to_9: float           # Win by 5 to 9 rounds
    p_margin_10_plus: float          # Non-sweep win by 10+ rounds (13-1, 13-2, 13-3)
    p_sweep_13_0: float              # 13-0 sweep win
    p_loss_10_plus: float            # Non-sweep loss by 10+ rounds (1-13, 2-13, 3-13)
    p_sweep_0_13: float              # 0-13 sweep loss
    ev_margin_bonus: float           # Net expected margin bonus/penalty points


@dataclass
class MapEVResult:
    """Expected value breakdown for a single played map."""
    map_name: str                     # Name of the map
    win_probability: float           # Win probability p_m
    expected_margin: float           # Continuous expected round differential (mu_margin)
    base_win_ev: float               # +1.0 * p_win
    margin_ev: float                 # Net expected margin bonus/penalty points
    total_map_ev: float              # base_win_ev + margin_ev
    margin_probs: MapMarginProbabilities


@dataclass
class SeriesEVResult:
    """Expected value breakdown for a BO3 series incorporating VFL Map Cap rules."""
    team_name: str                    # Evaluated team name
    opponent_name: str                # Opponent team name
    p_2_0_sweep: float               # Probability of winning 2-0 (p_m1 * p_m2)
    p_map3_played: float             # Probability of series reaching Map 3 (p1(1-p2) + (1-p1)p2)
    ev_map1: float                   # Expected EV for Map 1
    ev_map2: float                   # Expected EV for Map 2
    ev_map3_played: float            # Conditional Expected EV for Map 3 (p_map3_played * EV_m3)
    ev_sweep_bonus: float            # Expected 2-0 sweep bonus (+2.0 * p_2_0_sweep)
    map_cap_discount: float          # Expected discount for discarding lowest map in 3-map series
    total_series_ev: float           # Net total expected VFL series score
    maps: List[MapEVResult]          # Breakdown for each map in the veto pool


def compute_map_margin_probabilities(
    mu_margin: float,
    p_win: float,
    sigma: float = 3.0,
    sigma_extreme: float = 2.0
) -> MapMarginProbabilities:
    """
    Computes discrete VFL margin bucket probabilities using Gaussian CDF mapping.
    
    Standard margin buckets (sigma = 3.0):
        P(Margin 5 to 9) = Phi((9.5 - mu_margin)/3) - Phi((4.5 - mu_margin)/3)
        P(Margin 10+ Total) = 1.0 - Phi((9.5 - mu_margin)/3)
        P(Loss 10+ Total) = Phi((-9.5 - mu_margin)/3)
        
    Extreme outlier buckets (sigma = 2.0):
        P(13-0 Sweep) = 1.0 - Phi((12.5 - mu_margin)/2)
        P(0-13 Sweep) = Phi((-12.5 - mu_margin)/2)
    """
    mu = float(mu_margin)
    
    # Standard margin CDF probabilities (sigma = 3.0)
    p_margin_5_to_9 = float(norm.cdf((9.5 - mu) / sigma) - norm.cdf((4.5 - mu) / sigma))
    p_margin_10_plus_total = float(1.0 - norm.cdf((9.5 - mu) / sigma))
    p_loss_10_plus_total = float(norm.cdf((-9.5 - mu) / sigma))
    
    # Extreme outlier CDF probabilities (sigma = 2.0)
    p_sweep_13_0 = float(1.0 - norm.cdf((12.5 - mu) / sigma_extreme))
    p_sweep_0_13 = float(norm.cdf((-12.5 - mu) / sigma_extreme))
    
    # Non-sweep margin probabilities (ensure non-negative)
    p_margin_10_plus = float(max(0.0, p_margin_10_plus_total - p_sweep_13_0))
    p_loss_10_plus = float(max(0.0, p_loss_10_plus_total - p_sweep_0_13))
    
    # Net Margin Bonus/Penalty calculation:
    # Margin 5-9 win: +1 bonus pt
    # Margin 10+ non-sweep win: +2 bonus pts
    # 13-0 Sweep win: +5 total pts (+1 win + 4 bonus) -> bonus = +4.0
    # Loss 10+ non-sweep: -1 penalty pt
    # 0-13 Sweep loss: -5 total pts (0 loss - 5 penalty) -> penalty = -5.0
    ev_margin_bonus = (
        p_margin_5_to_9 * 1.0 +
        p_margin_10_plus * 2.0 +
        p_sweep_13_0 * 4.0 +
        p_loss_10_plus * (-1.0) +
        p_sweep_0_13 * (-5.0)
    )
    
    return MapMarginProbabilities(
        p_win=float(p_win),
        p_margin_5_to_9=p_margin_5_to_9,
        p_margin_10_plus=p_margin_10_plus,
        p_sweep_13_0=p_sweep_13_0,
        p_loss_10_plus=p_loss_10_plus,
        p_sweep_0_13=p_sweep_0_13,
        ev_margin_bonus=ev_margin_bonus
    )


def compute_single_map_ev(
    p_win: float,
    mu_margin: float,
    map_name: str = "Map 1",
    sigma: float = 3.0,
    sigma_extreme: float = 2.0
) -> MapEVResult:
    """
    Computes total expected VFL points for a single played map.
    
    EV_map = p_win * 1.0 + EV_margin_bonus
    """
    margin_probs = compute_map_margin_probabilities(
        mu_margin=mu_margin,
        p_win=p_win,
        sigma=sigma,
        sigma_extreme=sigma_extreme
    )
    
    base_win_ev = float(p_win * 1.0)
    margin_ev = margin_probs.ev_margin_bonus
    total_map_ev = float(base_win_ev + margin_ev)
    
    return MapEVResult(
        map_name=map_name,
        win_probability=float(p_win),
        expected_margin=float(mu_margin),
        base_win_ev=base_win_ev,
        margin_ev=margin_ev,
        total_map_ev=total_map_ev,
        margin_probs=margin_probs
    )


def compute_bo3_series_ev(
    maps_p_win: List[float],
    maps_mu_margin: List[float],
    map_names: Optional[List[str]] = None,
    team_name: str = "Team",
    opponent_name: str = "Opponent",
    sigma: float = 3.0,
    sigma_extreme: float = 2.0
) -> SeriesEVResult:
    """
    Computes the total Expected Value for a BO3 series incorporating VFL Map Cap rules.
    
    Mathematical Formulation:
        EV_m1 = p_m1 + EV_margin1
        EV_m2 = p_m2 + EV_margin2
        EV_m3 = p_m3 + EV_margin3
        
        P(2-0) = p_m1 * p_m2
        EV_sweep = P(2-0) * 2.0
        
        P(Map 3 Played) = p_m1 * (1 - p_m2) + (1 - p_m1) * p_m2
        EV_m3_played = P(Map 3 Played) * EV_m3
        
        Map Cap Discount = P(Map 3 Played) * min(EV_m1, EV_m2, EV_m3)
        Total Series EV = EV_m1 + EV_m2 + EV_m3_played + EV_sweep - Map Cap Discount
    """
    if len(maps_p_win) < 2 or len(maps_mu_margin) < 2:
        raise ValueError("At least 2 maps are required for a BO3 series EV calculation.")
        
    # If 3rd map parameters not provided, default to neutral map 3 (p=0.5, mu_margin=0.0)
    p_wins = list(maps_p_win)
    mu_margins = list(maps_mu_margin)
    names = map_names or [f"Map {i+1}" for i in range(len(p_wins))]
    
    if len(p_wins) == 2:
        p_wins.append(0.5)
        mu_margins.append(0.0)
        names.append("Map 3 (Decider)")
        
    p_m1, p_m2, p_m3 = p_wins[0], p_wins[1], p_wins[2]
    mu_m1, mu_m2, mu_m3 = mu_margins[0], mu_margins[1], mu_margins[2]
    
    map1_res = compute_single_map_ev(p_m1, mu_m1, map_name=names[0], sigma=sigma, sigma_extreme=sigma_extreme)
    map2_res = compute_single_map_ev(p_m2, mu_m2, map_name=names[1], sigma=sigma, sigma_extreme=sigma_extreme)
    map3_res = compute_single_map_ev(p_m3, mu_m3, map_name=names[2], sigma=sigma, sigma_extreme=sigma_extreme)
    
    ev_m1 = map1_res.total_map_ev
    ev_m2 = map2_res.total_map_ev
    ev_m3 = map3_res.total_map_ev
    
    # 2-0 Sweep Probability & Bonus
    p_2_0_sweep = float(p_m1 * p_m2)
    ev_sweep_bonus = float(p_2_0_sweep * 2.0)
    
    # Map 3 Played Probability (Split first 2 maps)
    p_map3_played = float(p_m1 * (1.0 - p_m2) + (1.0 - p_m1) * p_m2)
    ev_m3_played = float(p_map3_played * ev_m3)
    
    # VFL Map Cap Approximation Rule:
    # Lowest map score in a 3-map series is discarded.
    min_map_ev = float(min(ev_m1, ev_m2, ev_m3))
    map_cap_discount = float(p_map3_played * min_map_ev)
    
    total_series_ev = float(ev_m1 + ev_m2 + ev_m3_played + ev_sweep_bonus - map_cap_discount)
    
    return SeriesEVResult(
        team_name=team_name,
        opponent_name=opponent_name,
        p_2_0_sweep=p_2_0_sweep,
        p_map3_played=p_map3_played,
        ev_map1=ev_m1,
        ev_map2=ev_m2,
        ev_map3_played=ev_m3_played,
        ev_sweep_bonus=ev_sweep_bonus,
        map_cap_discount=map_cap_discount,
        total_series_ev=total_series_ev,
        maps=[map1_res, map2_res, map3_res]
    )
