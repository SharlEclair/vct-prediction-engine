"""
test_v9_historical_stats.py
----------------------------
Comprehensive test suite for v9_historical_stats.py.

Verifies:
1. Recency Decay: Older matches decay in weight relative to recent matches.
2. Bayesian Shrinkage: Low match counts shrink toward global prior, while large match counts converge to sample mean.
3. Telemetry Modifiers:
   - High KAST% elevates CVaR_10 floor.
   - High ADR elevates CVaR_90 ceiling.
   - High First Deaths (FD) decreases EV due to margin liability.
4. Kish's Effective Sample Size and weighted variance properties.
"""

import pytest
import numpy as np
from v9_historical_stats import (
    compute_exponential_decay_stats,
    compute_logistic_decay_stats,
    compute_bayesian_shrinkage_stats,
    compute_telemetry_zscores,
    apply_telemetry_modifiers,
    DecayStatsResult,
    ModifiedStatsResult
)


def test_exponential_decay_older_matches_decayed():
    """
    Verifies that older matches receive lower weights than recent matches,
    and recent performance disproportionately impacts EV.
    """
    days = np.array([100.0, 50.0, 0.0])
    
    # Case A: Recent match is high score (30.0), old match is low score (10.0)
    scores_a = np.array([10.0, 20.0, 30.0])
    stats_a = compute_exponential_decay_stats(scores_a, days, lam=0.005)
    
    # Case B: Recent match is low score (10.0), old match is high score (30.0)
    scores_b = np.array([30.0, 20.0, 10.0])
    stats_b = compute_exponential_decay_stats(scores_b, days, lam=0.005)
    
    # Unweighted average for both is 20.0
    assert stats_a.raw_mean == pytest.approx(20.0)
    assert stats_b.raw_mean == pytest.approx(20.0)
    
    # Recent high score must yield higher EV than recent low score
    assert stats_a.ev > 20.0
    assert stats_b.ev < 20.0
    assert stats_a.weights[2] > stats_a.weights[0]  # w(delta_t=0) > w(delta_t=100)


def test_logistic_decay_cliff_behavior():
    """Verifies logistic decay retains high weight before cliff (t_half=45) and decays past cliff."""
    days = np.array([5.0, 40.0, 90.0])
    scores = np.array([25.0, 25.0, 10.0])
    
    stats = compute_logistic_decay_stats(scores, days, t_half=45.0, k=0.1)
    
    # Weights for t=5 and t=40 should be close to 1.0, while t=90 should be near 0.0
    assert stats.weights[0] > 0.95
    assert stats.weights[1] > 0.60
    assert stats.weights[2] < 0.05
    assert stats.ev > 20.0


def test_bayesian_shrinkage_low_match_counts():
    """
    Verifies that low match counts (e.g. N=1) safely shrink toward the global prior,
    whereas high match counts (e.g. N=50) converge toward the sample mean.
    """
    mu_prior = 20.0
    sigma_prior_sq = 25.0
    
    # Low match count (N=1), outlier performance 40.0
    days_small = np.array([0.0])
    scores_small = np.array([40.0])
    
    stats_small = compute_bayesian_shrinkage_stats(
        scores_small, days_small,
        mu_prior=mu_prior, sigma_prior_sq=sigma_prior_sq
    )
    
    # EV should shrink significantly from 40.0 toward 20.0
    assert stats_small.ev < 30.0
    assert stats_small.ev > 20.0
    assert stats_small.effective_sample_size == pytest.approx(1.0)
    
    # Large match count (N=50), consistent performance 40.0
    days_large = np.zeros(50)
    scores_large = np.full(50, 40.0)
    
    stats_large = compute_bayesian_shrinkage_stats(
        scores_large, days_large,
        mu_prior=mu_prior, sigma_prior_sq=sigma_prior_sq
    )
    
    # With N=50 identical scores, EV converges close to 40.0
    assert stats_large.ev > 39.0


def test_telemetry_kast_elevates_cvar10_floor():
    """Verifies that high KAST% (positive Z_KAST) elevates the CVaR_10 floor."""
    base_ev = 20.0
    base_cvar10 = 12.0
    base_cvar90 = 28.0
    
    # Normal/Average KAST% (Z=0)
    result_neutral = apply_telemetry_modifiers(
        base_ev, base_cvar10, base_cvar90,
        z_kast=0.0, z_adr=0.0, z_fd=0.0
    )
    assert result_neutral.cvar_10_modified == pytest.approx(base_cvar10)
    
    # High KAST% (Z = +1.5)
    result_high_kast = apply_telemetry_modifiers(
        base_ev, base_cvar10, base_cvar90,
        z_kast=1.5, z_adr=0.0, z_fd=0.0, beta_kast=1.0
    )
    
    # Floor should elevate by beta_kast * z_kast = 1.0 * 1.5 = 1.5
    assert result_high_kast.cvar_10_modified == pytest.approx(base_cvar10 + 1.5)
    assert result_high_kast.cvar_10_modified > result_neutral.cvar_10_modified


def test_telemetry_adr_elevates_cvar90_ceiling():
    """Verifies that high ADR (positive Z_ADR) elevates the CVaR_90 ceiling."""
    base_ev = 20.0
    base_cvar10 = 12.0
    base_cvar90 = 28.0
    
    # High ADR (Z = +2.0)
    result_high_adr = apply_telemetry_modifiers(
        base_ev, base_cvar10, base_cvar90,
        z_kast=0.0, z_adr=2.0, z_fd=0.0, beta_adr=1.0
    )
    
    # Ceiling should elevate by beta_adr * z_adr = 1.0 * 2.0 = 2.0
    assert result_high_adr.cvar_90_modified == pytest.approx(base_cvar90 + 2.0)
    assert result_high_adr.cvar_90_modified > base_cvar90


def test_telemetry_first_deaths_reduces_ev():
    """Verifies that high First Deaths (positive Z_FD) reduces EV due to margin liability."""
    base_ev = 20.0
    base_cvar10 = 12.0
    base_cvar90 = 28.0
    
    # High First Deaths (Z = +1.0)
    result_high_fd = apply_telemetry_modifiers(
        base_ev, base_cvar10, base_cvar90,
        z_kast=0.0, z_adr=0.0, z_fd=1.0, beta_fd=0.5
    )
    
    # EV_modified = 20.0 * (1.0 - 0.5 * 1.0) = 10.0
    assert result_high_fd.ev_modified == pytest.approx(10.0)
    assert result_high_fd.ev_modified < base_ev


def test_telemetry_zscore_computation():
    """Verifies normalization of raw ADR, KAST%, and FD against role benchmarks."""
    z_adr, z_kast, z_fd = compute_telemetry_zscores(
        adr=170.0,   # Mean=150, Std=20 -> Z = +1.0
        kast=0.78,   # Mean=0.70, Std=0.08 -> Z = +1.0
        fd=0.20,     # Mean=0.15, Std=0.05 -> Z = +1.0
        role="Duelist"
    )
    
    assert z_adr == pytest.approx(1.0)
    assert z_kast == pytest.approx(1.0)
    assert z_fd == pytest.approx(1.0)


def test_decay_stats_result_integration():
    """Verifies seamless integration between DecayStatsResult object and apply_telemetry_modifiers."""
    days = np.array([20.0, 10.0, 0.0])
    scores = np.array([18.0, 22.0, 25.0])
    
    base_stats = compute_bayesian_shrinkage_stats(scores, days)
    modified = apply_telemetry_modifiers(
        base_stats,
        z_kast=1.0,
        z_adr=1.5,
        z_fd=-0.5
    )
    
    assert isinstance(modified, ModifiedStatsResult)
    assert modified.base_stats == base_stats
    assert modified.ev_modified > base_stats.ev  # Negative Z_FD increases EV
    assert modified.cvar_10_modified > base_stats.cvar_10
    assert modified.cvar_90_modified > base_stats.cvar_90


if __name__ == "__main__":
    pytest.main([__file__])
