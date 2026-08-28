"""
v9_historical_stats.py
----------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 1: Historical Stats & Telemetry Fusion.

This module replaces legacy unweighted PPG averages with time-decaying baseline Expected Value (EV)
projections (Exponential, Logistic, and Bayesian Shrinkage with Kish's Effective Sample Size)
and integrates role-normalized telemetry metrics (ADR, KAST%, First Deaths) to dynamically
adjust Conditional Value at Risk (CVaR_10 floor and CVaR_90 ceiling).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.stats import norm


@dataclass
class DecayStatsResult:
    """Container for baseline decay statistical outputs."""
    ev: float                         # Expected Value (Weighted mean / Posterior mean)
    std: float                        # Weighted std / Posterior std
    cvar_10: float                    # Base Floor (10% CVaR)
    cvar_90: float                    # Base Ceiling (90% CVaR)
    effective_sample_size: float      # Kish's Effective Sample Size (n_eff)
    weights: np.ndarray               # Applied temporal weights array
    raw_mean: float                   # Unweighted mean score
    raw_variance: float               # Unweighted variance score


@dataclass
class ModifiedStatsResult:
    """Container for telemetry-modified EV and CVaR outputs."""
    ev_modified: float                # Telemetry-adjusted EV
    cvar_10_modified: float           # Telemetry-adjusted Floor (CVaR_10)
    cvar_90_modified: float           # Telemetry-adjusted Ceiling (CVaR_90)
    z_kast: float                     # Normalized KAST% z-score
    z_adr: float                      # Normalized ADR z-score
    z_fd: float                       # Normalized First Deaths z-score
    base_stats: DecayStatsResult       # Original baseline decay result


# Default role-level population benchmarks for telemetry z-score normalization
DEFAULT_ROLE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Duelist": {
        "adr_mean": 150.0, "adr_std": 20.0,
        "kast_mean": 0.70, "kast_std": 0.08,
        "fd_mean": 0.15,  "fd_std": 0.05
    },
    "Initiator": {
        "adr_mean": 130.0, "adr_std": 18.0,
        "kast_mean": 0.75, "kast_std": 0.07,
        "fd_mean": 0.08,  "fd_std": 0.04
    },
    "Controller": {
        "adr_mean": 125.0, "adr_std": 16.0,
        "kast_mean": 0.76, "kast_std": 0.06,
        "fd_mean": 0.07,  "fd_std": 0.03
    },
    "Sentinel": {
        "adr_mean": 128.0, "adr_std": 17.0,
        "kast_mean": 0.74, "kast_std": 0.07,
        "fd_mean": 0.08,  "fd_std": 0.04
    },
    "Global": {
        "adr_mean": 133.0, "adr_std": 20.0,
        "kast_mean": 0.74, "kast_std": 0.07,
        "fd_mean": 0.10,  "fd_std": 0.05
    }
}


def _compute_cvar_bounds(mu: float, sigma: float) -> Tuple[float, float]:
    """
    Computes base floor (CVaR_10) and ceiling (CVaR_90) given mean and standard deviation.
    
    Formula:
        CVaR_10 = mu - sigma * ( phi(z_0.10) / 0.10 )
        CVaR_90 = mu + sigma * ( phi(z_0.90) / 0.10 )
    """
    z_10 = norm.ppf(0.10)
    z_90 = norm.ppf(0.90)
    
    phi_10 = norm.pdf(z_10)
    phi_90 = norm.pdf(z_90)
    
    cvar_10_factor = phi_10 / 0.10  # Approx 1.75498
    cvar_90_factor = phi_90 / 0.10  # Approx 1.75498
    
    cvar_10 = mu - sigma * cvar_10_factor
    cvar_90 = mu + sigma * cvar_90_factor
    
    return cvar_10, cvar_90


def compute_exponential_decay_stats(
    fantasy_points: Union[List[float], np.ndarray],
    days_elapsed: Union[List[float], np.ndarray],
    lam: float = 0.005,
    eps: float = 1e-8
) -> DecayStatsResult:
    """
    Candidate 1: Exponential Decay Stats.
    
    Temporal weights:
        w_i = exp(-lambda * delta_t_i)
        
    Calculates weighted mean, Kish's effective sample size (n_eff), sample weighted variance,
    and base CVaR_10 / CVaR_90 bounds.
    """
    x = np.asarray(fantasy_points, dtype=np.float64)
    dt = np.asarray(days_elapsed, dtype=np.float64)
    
    if len(x) == 0 or len(dt) == 0 or len(x) != len(dt):
        raise ValueError("fantasy_points and days_elapsed must be non-empty arrays of equal length.")
        
    weights = np.exp(-lam * dt)
    w_sum = np.sum(weights)
    w2_sum = np.sum(weights ** 2)
    
    if w_sum < eps:
        weights = np.ones_like(x, dtype=np.float64) / len(x)
        w_sum = np.sum(weights)
        w2_sum = np.sum(weights ** 2)
        
    # Weighted mean
    mu_w = float(np.sum(weights * x) / w_sum)
    
    # Kish's Effective Sample Size
    n_eff = float((w_sum ** 2) / (w2_sum + eps))
    
    # Unweighted mean and variance
    raw_mean = float(np.mean(x))
    raw_variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
    
    # Sample Weighted Variance
    denom = w_sum - (w2_sum / w_sum)
    if denom > eps:
        s_w_sq = float(np.sum(weights * ((x - mu_w) ** 2)) / denom)
    else:
        s_w_sq = raw_variance
        
    sigma_w = float(np.sqrt(max(s_w_sq, eps)))
    cvar_10, cvar_90 = _compute_cvar_bounds(mu_w, sigma_w)
    
    return DecayStatsResult(
        ev=mu_w,
        std=sigma_w,
        cvar_10=cvar_10,
        cvar_90=cvar_90,
        effective_sample_size=n_eff,
        weights=weights,
        raw_mean=raw_mean,
        raw_variance=raw_variance
    )


def compute_logistic_decay_stats(
    fantasy_points: Union[List[float], np.ndarray],
    days_elapsed: Union[List[float], np.ndarray],
    t_half: float = 45.0,
    k: float = 0.1,
    eps: float = 1e-8
) -> DecayStatsResult:
    """
    Candidate 2: Logistic (Half-Life) Decay Stats.
    
    Temporal weights:
        w_i = 1 / (1 + exp(k * (delta_t_i - t_half)))
        
    Calculates weighted mean, Kish's effective sample size (n_eff), sample weighted variance,
    and base CVaR_10 / CVaR_90 bounds.
    """
    x = np.asarray(fantasy_points, dtype=np.float64)
    dt = np.asarray(days_elapsed, dtype=np.float64)
    
    if len(x) == 0 or len(dt) == 0 or len(x) != len(dt):
        raise ValueError("fantasy_points and days_elapsed must be non-empty arrays of equal length.")
        
    # Logistic decay weighting
    weights = 1.0 / (1.0 + np.exp(k * (dt - t_half)))
    w_sum = np.sum(weights)
    w2_sum = np.sum(weights ** 2)
    
    if w_sum < eps:
        weights = np.ones_like(x, dtype=np.float64) / len(x)
        w_sum = np.sum(weights)
        w2_sum = np.sum(weights ** 2)
        
    mu_w = float(np.sum(weights * x) / w_sum)
    n_eff = float((w_sum ** 2) / (w2_sum + eps))
    
    raw_mean = float(np.mean(x))
    raw_variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
    
    denom = w_sum - (w2_sum / w_sum)
    if denom > eps:
        s_w_sq = float(np.sum(weights * ((x - mu_w) ** 2)) / denom)
    else:
        s_w_sq = raw_variance
        
    sigma_w = float(np.sqrt(max(s_w_sq, eps)))
    cvar_10, cvar_90 = _compute_cvar_bounds(mu_w, sigma_w)
    
    return DecayStatsResult(
        ev=mu_w,
        std=sigma_w,
        cvar_10=cvar_10,
        cvar_90=cvar_90,
        effective_sample_size=n_eff,
        weights=weights,
        raw_mean=raw_mean,
        raw_variance=raw_variance
    )


def calculate_mastery_index(
    player_name: str,
    agent_name: str,
    telemetry_db: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    maps_played: Optional[float] = None
) -> float:
    """
    Phase 1: Calculates normalized Agent Mastery Index in [0.0, 1.0].
    
    Formula:
        mastery_index = min(maps_played / 50.0, 1.0)
    """
    if maps_played is not None:
        count = float(maps_played)
    elif telemetry_db is not None:
        count = 0.0
        if isinstance(telemetry_db, dict):
            # Check direct agent_maps or nested player entry
            if agent_name in telemetry_db and isinstance(telemetry_db[agent_name], (int, float)):
                count = float(telemetry_db[agent_name])
            elif "agent_maps" in telemetry_db and isinstance(telemetry_db["agent_maps"], dict):
                count = float(telemetry_db["agent_maps"].get(agent_name, 0.0))
            elif player_name in telemetry_db:
                p_data = telemetry_db[player_name]
                if isinstance(p_data, dict):
                    if agent_name in p_data and isinstance(p_data[agent_name], (int, float)):
                        count = float(p_data[agent_name])
                    elif "agent_maps" in p_data and isinstance(p_data["agent_maps"], dict):
                        count = float(p_data["agent_maps"].get(agent_name, 0.0))
                    elif "agents" in p_data and isinstance(p_data["agents"], dict):
                        ag_info = p_data["agents"].get(agent_name, 0.0)
                        if isinstance(ag_info, dict):
                            count = float(ag_info.get("maps", ag_info.get("maps_played", 0.0)))
                        else:
                            count = float(ag_info)
        elif isinstance(telemetry_db, list):
            for item in telemetry_db:
                p_handle = str(item.get("player_name") or item.get("name") or "").strip().lower()
                if p_handle == player_name.strip().lower():
                    ag_maps = item.get("agent_maps", item.get("agents", {}))
                    if isinstance(ag_maps, dict):
                        count = float(ag_maps.get(agent_name, 0.0))
                    break
    else:
        count = 0.0
        
    mastery_index = min(max(count / 50.0, 0.0), 1.0)
    return float(mastery_index)


def compute_mastery_inertia_buffer(mastery_index: float) -> float:
    """
    Phase 1: Computes the inertia buffer absorbing up to 60% of nerf shock for veteran players.
    
    Formula:
        inertia_buffer = 1.0 - (0.60 * mastery_index)
    """
    m = min(max(float(mastery_index), 0.0), 1.0)
    return float(1.0 - 0.60 * m)


def apply_patch_shock_with_mastery(
    base_ev: float,
    patch_shock: float,
    mastery_index: float = 0.0
) -> Tuple[float, float, float]:
    """
    Phase 1: Applies patch shock to base EV while conditionally absorbing negative shocks (nerfs)
    based on the player's Agent Mastery Index.
    
    Returns:
        Tuple[float, float, float]: (adjusted_ev, effective_shock, inertia_buffer)
    """
    inertia_buf = compute_mastery_inertia_buffer(mastery_index)
    
    # Apply buffer conditionally ONLY to nerfs (negative shocks)
    if patch_shock < 0:
        effective_shock = float(patch_shock * inertia_buf)
    else:
        effective_shock = float(patch_shock)
        
    # Scale base_ev by (1.0 + effective_shock)
    adjusted_ev = float(base_ev * (1.0 + effective_shock))
    return adjusted_ev, effective_shock, inertia_buf


def compute_bayesian_shrinkage_stats(
    fantasy_points: Union[List[float], np.ndarray],
    days_elapsed: Union[List[float], np.ndarray],
    mu_prior: float = 20.0,
    sigma_prior_sq: float = 25.0,
    decay_type: str = "exponential",
    lam: float = 0.005,
    t_half: float = 45.0,
    k: float = 0.1,
    eps: float = 1e-8,
    patch_shock: float = 0.0,
    mastery_index: float = 0.0
) -> DecayStatsResult:
    """
    Candidate 3: Bayesian Updating Framework with Prior Shrinkage & Agent Mastery Inertia.
    
    Effective sample size (Kish's formula):
        n_eff = (sum(w_i))^2 / sum(w_i^2)
        
    Sample Weighted Variance:
        s_w^2 = [ sum(w_i) / (sum(w_i) - sum(w_i^2)/sum(w_i)) ] * sum(w_i * (x_i - mu_w)^2)
        
    Bayesian Posterior Mean (Precision-weighted):
        mu_post = [ (n_eff / s_w^2) * mu_w + (1 / sigma_prior^2) * effective_mu_prior ] / [ (n_eff / s_w^2) + (1 / sigma_prior^2) ]
        
    Bayesian Posterior Variance:
        sigma_post^2 = [ 1 / ( (n_eff / s_w^2) + (1 / sigma_prior^2) ) ] + s_w^2
    """
    x = np.asarray(fantasy_points, dtype=np.float64)
    dt = np.asarray(days_elapsed, dtype=np.float64)
    
    if len(x) == 0 or len(dt) == 0 or len(x) != len(dt):
        raise ValueError("fantasy_points and days_elapsed must be non-empty arrays of equal length.")
        
    if decay_type.lower() == "exponential":
        weights = np.exp(-lam * dt)
    elif decay_type.lower() == "logistic":
        weights = 1.0 / (1.0 + np.exp(k * (dt - t_half)))
    else:
        raise ValueError(f"Unsupported decay_type: '{decay_type}'. Choose 'exponential' or 'logistic'.")
        
    w_sum = np.sum(weights)
    w2_sum = np.sum(weights ** 2)
    
    if w_sum < eps:
        weights = np.ones_like(x, dtype=np.float64) / len(x)
        w_sum = np.sum(weights)
        w2_sum = np.sum(weights ** 2)
        
    mu_w = float(np.sum(weights * x) / w_sum)
    n_eff = float((w_sum ** 2) / (w2_sum + eps))
    
    raw_mean = float(np.mean(x))
    raw_variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
    
    denom = w_sum - (w2_sum / w_sum)
    if denom > eps and len(x) > 1:
        s_w_sq = float(np.sum(weights * ((x - mu_w) ** 2)) / denom)
    else:
        s_w_sq = float(raw_variance) if (len(x) > 1 and raw_variance > eps) else float(sigma_prior_sq)
        
    # Apply variance floor when effective sample size is small to prevent zero-variance precision explosion
    if n_eff < 2.0:
        effective_s_w_sq = max(s_w_sq, sigma_prior_sq)
    else:
        effective_s_w_sq = max(s_w_sq, eps)
        
    # Apply Agent Mastery Inertia to mu_prior if patch shock is active
    if abs(patch_shock) > 1e-6:
        effective_mu_prior, _, _ = apply_patch_shock_with_mastery(mu_prior, patch_shock, mastery_index)
    else:
        effective_mu_prior = mu_prior

    sample_precision = n_eff / effective_s_w_sq
    prior_precision = 1.0 / max(sigma_prior_sq, eps)
    
    total_precision = sample_precision + prior_precision
    mu_post = float((sample_precision * mu_w + prior_precision * effective_mu_prior) / total_precision)
    
    sigma_post_sq = float((1.0 / total_precision) + s_w_sq)
    sigma_post = float(np.sqrt(max(sigma_post_sq, eps)))
    
    cvar_10, cvar_90 = _compute_cvar_bounds(mu_post, sigma_post)
    
    return DecayStatsResult(
        ev=mu_post,
        std=sigma_post,
        cvar_10=cvar_10,
        cvar_90=cvar_90,
        effective_sample_size=n_eff,
        weights=weights,
        raw_mean=raw_mean,
        raw_variance=raw_variance
    )


def compute_vlr_rating_2_zscore(rating: float) -> float:
    """
    Computes the standardized z-score for a VLR Rating 2.0 metric.
    
    Analytical Distribution of VLR Rating 2.0:
        Mean mu = 1.0, Standard Deviation sigma = 1/3 (0.33333...)
        Z_rating = (Rating - 1.0) / (1/3) = 3.0 * (Rating - 1.0)
    """
    return float(3.0 * (float(rating) - 1.0))


def compute_adjusted_adr(adr: float, kpr: float, avg_damage_per_kill: float = 140.0) -> float:
    """
    Computes Adjusted ADR (ADRa) by decoupling KPR from raw ADR.
    
    Formula (matching VLR Rating 2.0 methodology):
        ADRa = max(0.0, ADR - (KPR * avg_damage_per_kill))
    """
    return float(max(0.0, float(adr) - (float(kpr) * avg_damage_per_kill)))


def compute_telemetry_zscores(
    adr: float,
    kast: float,
    fd: float,
    role: str = "Global",
    custom_benchmarks: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[float, float, float]:
    """
    Normalizes raw ADR, KAST%, and First Deaths (FD) into role-adjusted z-scores.
    
    Returns:
        Tuple[float, float, float]: (z_adr, z_kast, z_fd)
    """
    benchmarks = custom_benchmarks or DEFAULT_ROLE_BENCHMARKS
    role_bench = benchmarks.get(role, benchmarks.get("Global", DEFAULT_ROLE_BENCHMARKS["Global"]))
    
    # Handle KAST percentage scale (if passed as 0..100 instead of 0..1)
    if kast > 1.0 and role_bench["kast_mean"] <= 1.0:
        kast = kast / 100.0
        
    z_adr = (adr - role_bench["adr_mean"]) / role_bench["adr_std"]
    z_kast = (kast - role_bench["kast_mean"]) / role_bench["kast_std"]
    z_fd = (fd - role_bench["fd_mean"]) / role_bench["fd_std"]
    
    return float(z_adr), float(z_kast), float(z_fd)


def apply_telemetry_modifiers(
    base_ev: Union[float, DecayStatsResult],
    base_cvar10: Optional[float] = None,
    base_cvar90: Optional[float] = None,
    z_kast: float = 0.0,
    z_adr: float = 0.0,
    z_fd: float = 0.0,
    beta_fd: float = 0.5,
    beta_kast: float = 1.0,
    beta_adr: float = 1.0,
    role: str = "Global"
) -> ModifiedStatsResult:
    """
    Applies telemetry z-score modifiers to baseline EV, CVaR_10 (floor), and CVaR_90 (ceiling).
    
    Phase 3: Implements Skill-Ceiling Elasticity with a bifurcated exponential kicker
    for high-ADR Specialists (Z_ADR > 1.0) and role-dynamic beta for Duelists.
    
    Formulas:
        EV_modified = mu_post * (1.0 - beta_FD * Z_FD)
        CVaR_10_modified = CVaR_10 + (beta_KAST * Z_KAST)
        Z_ADR_kicker = Z_ADR ** 1.5 if Z_ADR > 1.0 else Z_ADR
        beta_ADR_dynamic = 1.5 if role == "Duelist" else beta_ADR
        CVaR_90_modified = CVaR_90 + (beta_ADR_dynamic * Z_ADR_kicker)
    """
    if isinstance(base_ev, DecayStatsResult):
        stats_obj = base_ev
        ev_val = stats_obj.ev
        cvar10_val = stats_obj.cvar_10
        cvar90_val = stats_obj.cvar_90
    else:
        if base_cvar10 is None or base_cvar90 is None:
            raise ValueError("base_cvar10 and base_cvar90 must be provided when base_ev is a float.")
        ev_val = float(base_ev)
        cvar10_val = float(base_cvar10)
        cvar90_val = float(base_cvar90)
        stats_obj = DecayStatsResult(
            ev=ev_val,
            std=0.0,
            cvar_10=cvar10_val,
            cvar_90=cvar90_val,
            effective_sample_size=0.0,
            weights=np.array([]),
            raw_mean=ev_val,
            raw_variance=0.0
        )
        
    # Phase 3: Bifurcated Exponential Kicker for Specialists (Z_ADR > 1.0)
    z_adr_val = float(z_adr)
    z_adr_kicker = float(z_adr_val ** 1.5) if z_adr_val > 1.0 else z_adr_val
    
    # Phase 3: Dynamic Beta based on role multi-kill capability
    role_norm = str(role).strip().title() if role else "Global"
    beta_adr_dynamic = 1.5 if role_norm == "Duelist" else float(beta_adr)

    ev_modified = float(ev_val * (1.0 - beta_fd * z_fd))
    cvar_10_modified = float(cvar10_val + (beta_kast * z_kast))
    cvar_90_modified = float(cvar90_val + (beta_adr_dynamic * z_adr_kicker))
    
    return ModifiedStatsResult(
        ev_modified=ev_modified,
        cvar_10_modified=cvar_10_modified,
        cvar_90_modified=cvar_90_modified,
        z_kast=float(z_kast),
        z_adr=float(z_adr),
        z_fd=float(z_fd),
        base_stats=stats_obj
    )

