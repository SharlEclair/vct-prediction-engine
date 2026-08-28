"""
Covariance Profiler Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 2).

Handles Task 2.2 (Simulation Matrix Extraction) and Task 2.3 (Correlation Profiling).
Ingests the 10,000 x 10 simulation matrix from dag_simulation.py, calculates the 10 x 10 Spearman
Rank Correlation matrix (Sigma_MC), and validates structural tactical shooter domain constraints.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .dag_simulation import MockDAGSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_simulation_matrix(num_iterations: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Task 2.2: Simulation Matrix Extraction.
    Runs Monte Carlo simulation and structures outputs into a 10,000 x 10 unified matrix.
    
    Args:
        num_iterations (int): Number of Monte Carlo iterations (default 10,000).
        seed (int): Random seed.
        
    Returns:
        pd.DataFrame: Formatted 10,000 x 10 matrix representing raw DFS points for all 10 players.
    """
    simulator = MockDAGSimulator(seed=seed)
    df_sim = simulator.simulate_iterations(num_iterations=num_iterations)
    
    # Verify dimensions
    assert df_sim.shape == (num_iterations, 10), f"Expected shape ({num_iterations}, 10), got {df_sim.shape}"
    logger.info("Task 2.2 Complete: Extracted unified simulation matrix with shape %s.", df_sim.shape)
    return df_sim


def compute_spearman_covariance(df_sim: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Task 2.3: Correlation Profiling.
    Calculates the 10 x 10 Spearman Rank Correlation matrix (Sigma_MC) and validates domain constraints.
    
    Args:
        df_sim (pd.DataFrame): 10,000 x 10 simulation matrix.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: Spearman correlation matrix dataframe and summary diagnostics.
    """
    logger.info("Computing 10x10 Spearman Rank Correlation matrix (Sigma_MC)...")
    
    # Calculate Spearman correlation matrix
    spearman_corr, _ = spearmanr(df_sim.values, axis=0)
    df_corr = pd.DataFrame(spearman_corr, index=df_sim.columns, columns=df_sim.columns)
    
    # Segment sub-blocks for validation
    team_a_cols = [c for c in df_sim.columns if "TeamA" in c]
    team_b_cols = [c for c in df_sim.columns if "TeamB" in c]
    
    # Intra-team correlations (off-diagonal elements within Team A and Team B)
    corr_team_a = df_corr.loc[team_a_cols, team_a_cols].values
    corr_team_b = df_corr.loc[team_b_cols, team_b_cols].values
    
    mask_offdiag = ~np.eye(5, dtype=bool)
    avg_intra_team_a = float(corr_team_a[mask_offdiag].mean())
    avg_intra_team_b = float(corr_team_b[mask_offdiag].mean())
    avg_intra_team = (avg_intra_team_a + avg_intra_team_b) / 2.0
    
    # Inter-team correlations (Team A vs Team B)
    corr_inter = df_corr.loc[team_a_cols, team_b_cols].values
    avg_inter_team = float(corr_inter.mean())
    
    diagnostics = {
        "avg_intra_team_a_corr": avg_intra_team_a,
        "avg_intra_team_b_corr": avg_intra_team_b,
        "avg_intra_team_corr": avg_intra_team,
        "avg_inter_team_corr": avg_inter_team,
        "valid_tactical_physics": bool(avg_inter_team < 0 and avg_intra_team > 0)
    }
    
    logger.info("=== TASK 2.3 SPEARMAN CORRELATION DIAGNOSTICS ===")
    logger.info("Average Intra-Team Correlation (Teammates) : %+.4f", avg_intra_team)
    logger.info("Average Inter-Team Correlation (Opponents) : %+.4f", avg_inter_team)
    logger.info("Tactical Shooter Physics Validated?         : %s", "YES" if diagnostics["valid_tactical_physics"] else "NO")
    
    return df_corr, diagnostics


def print_correlation_matrix_summary(df_corr: pd.DataFrame) -> None:
    """
    Utility to format and output clean visual view of the 10x10 correlation matrix.
    """
    logger.info("\n=== 10x10 SPEARMAN RANK CORRELATION MATRIX (Sigma_MC) ===\n" + df_corr.to_string(float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    df_matrix = extract_simulation_matrix(num_iterations=10000, seed=42)
    df_spearman, diag = compute_spearman_covariance(df_matrix)
    print_correlation_matrix_summary(df_spearman)
