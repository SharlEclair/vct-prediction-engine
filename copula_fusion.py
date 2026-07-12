"""
Copula Fusion Engine Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 3).

Executes the Iman-Conover transformation to mathematically synthesize Top-Down XGBoost predictions
(mu_TD from Phase 1) with Bottom-Up Monte Carlo structural correlations (Sigma_MC from Phase 2).
Preserves exact marginal means while inducing in-game covariance.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import norm

from archive.covariance_profiler import extract_simulation_matrix, compute_spearman_covariance
from utils.utils import load_slate_payload

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_top_down_predictions() -> Dict[str, float]:
    """
    Retrieve Phase 1 XGBoost expected value predictions (mu_TD) dynamically for players in active slate.
    Loads predictions from xgb_predictions.json and raises ValueError if missing.
    
    Returns:
        Dict[str, float]: Player ID to predicted mean DFS points.
    """
    from pathlib import Path
    import json
    
    root_dir = Path(__file__).resolve().parent
    slate_path = root_dir / "data" / "processed" / "current_slate.json"
    pred_path = root_dir / "data" / "processed" / "xgb_predictions.json"
    
    if not pred_path.exists():
        raise ValueError(f"XGBoost predictions file not found at {pred_path}. Please run model_training.py first.")
        
    with open(pred_path, "r", encoding="utf-8") as f:
        xgb_preds = json.load(f)
        
    if not slate_path.exists():
        raise ValueError(f"Slate payload file not found at {slate_path}.")
        
    with open(slate_path, "r", encoding="utf-8") as f:
        slate = json.load(f)
        
    predictions = {}
    for item in slate:
        pid = item["player_id"]
        name = item["name"]
        
        # Look up by player_id or player name in the predictions
        ev = None
        if pid in xgb_preds:
            ev = xgb_preds[pid]
        elif name in xgb_preds:
            ev = xgb_preds[name]
            
        if ev is None:
            raise ValueError(f"XGBoost expected value prediction is missing for player {name} (ID: {pid}) in predictions file.")
            
        predictions[pid] = float(ev)
        
    logger.info("Dynamically loaded Top-Down XGBoost predictions (mu_TD) for %d players from %s.", len(predictions), pred_path)
    return predictions


def generate_independent_marginals(
    predictions: Dict[str, float], 
    num_iterations: int = 10000, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Task 3.1 & 3.2: Marginal Generation and Independent Sampling.
    Draws N independent samples from parametric marginal distributions (Gamma) parameterized
    so expected value exactly equals Phase 1 XGBoost mu_TD.
    
    Args:
        predictions (Dict[str, float]): Map of player to target mean.
        num_iterations (int): Number of independent samples (N=10,000).
        seed (int): Random seed.
        
    Returns:
        pd.DataFrame: Uncorrelated N x 10 matrix M.
    """
    np.random.seed(seed)
    player_names = list(predictions.keys())
    matrix_m = np.zeros((num_iterations, len(player_names)))
    
    for idx, name in enumerate(player_names):
        mu = predictions[name]
        # Use Gamma distribution for realistic right-skewed fantasy point distributions
        # shape (k) * scale (theta) = mu. Let shape = 16.0 for realistic CV ~ 0.25
        shape = 16.0
        scale = mu / shape
        matrix_m[:, idx] = np.random.gamma(shape=shape, scale=scale, size=num_iterations)
        
    df_m = pd.DataFrame(matrix_m, columns=player_names)
    logger.info("Task 3.1 & 3.2 Complete: Generated uncorrelated matrix M of shape %s.", df_m.shape)
    return df_m


def run_iman_conover_fusion(
    df_m: pd.DataFrame, 
    target_corr_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Task 3.3 & 3.4: Cholesky Decomposition, Normal Scores, and Rank Reordering.
    Executes the Iman-Conover transformation to induce target rank correlation matrix.
    
    Args:
        df_m (pd.DataFrame): Uncorrelated N x 10 marginal sample matrix M.
        target_corr_matrix (pd.DataFrame): Target 10 x 10 Spearman correlation matrix Sigma_MC.
        
    Returns:
        pd.DataFrame: Fused N x 10 sample matrix matching target correlation while preserving exact marginals.
    """
    logger.info("Executing Iman-Conover copula fusion transformation...")
    M = df_m.values
    N, k = M.shape
    C_target = target_corr_matrix.values
    
    # Step 1: Compute Van der Waerden normal scores matrix S
    S = np.zeros((N, k))
    for j in range(k):
        # Calculate fractional ranks in (0, 1)
        ranks = pd.Series(M[:, j]).rank(method="average").values
        u = ranks / (N + 1)
        S[:, j] = norm.ppf(u)
        
    # Step 2: Uncorrelate S to orthogonal matrix Z
    C_S = np.corrcoef(S, rowvar=False)
    L_S = np.linalg.cholesky(C_S)
    # Z = S * (L_S^-1)^T => Z * L_S^T = S
    Z = np.linalg.solve(L_S, S.T).T
    
    # Step 3: Compute Cholesky factor of target matrix and induce correlation
    try:
        L_C = np.linalg.cholesky(C_target + 1e-8 * np.eye(k))
    except np.linalg.LinAlgError:
        logger.warning("Target correlation matrix is not positive definite. Projecting to nearest positive definite matrix...")
        eigenvalues, eigenvectors = np.linalg.eigh(C_target)
        eigenvalues = np.maximum(eigenvalues, 1e-4)
        C_target_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        d = np.sqrt(np.diag(C_target_fixed))
        C_target_fixed = C_target_fixed / np.outer(d, d)
        L_C = np.linalg.cholesky(C_target_fixed + 1e-8 * np.eye(k))
        
    Y = Z @ L_C.T
    
    # Step 4: Rank Reordering (Iman-Conover final step)
    M_fused = np.zeros((N, k))
    for j in range(k):
        # Get rank order of target correlated column Y[:, j]
        rank_order_y = np.argsort(np.argsort(Y[:, j]))
        # Sort original marginal column M[:, j]
        sorted_m = np.sort(M[:, j])
        # Reorder sorted marginal values to match rank order of Y
        M_fused[:, j] = sorted_m[rank_order_y]
        
    df_fused = pd.DataFrame(M_fused, columns=df_m.columns)
    logger.info("Task 3.3 & 3.4 Complete: Successfully executed Iman-Conover rank reordering.")
    return df_fused


def validate_and_extract_metrics(
    df_m: pd.DataFrame, 
    df_fused: pd.DataFrame, 
    target_corr_matrix: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Task 3.5: Metric Extraction and Validation Check.
    Validates mean preservation and correlation alignment, then extracts EV, Floor (p15), and Ceiling (p85).
    
    Args:
        df_m (pd.DataFrame): Original uncorrelated marginal samples.
        df_fused (pd.DataFrame): Fused correlated samples.
        target_corr_matrix (pd.DataFrame): Target Spearman matrix.
        
    Returns:
        Dict[str, Dict[str, float]]: Player projection metrics (EV, Floor, Ceiling).
    """
    logger.info("=== TASK 3.5 VALIDATION CHECKS ===")
    
    # 1. Validate exact mean preservation
    mean_diffs = np.abs(df_fused.mean() - df_m.mean())
    max_mean_diff = mean_diffs.max()
    logger.info("Max Player Mean Absolute Difference (Fused vs Uncorrelated): %.6f", max_mean_diff)
    assert max_mean_diff < 1e-4, f"Mean preservation check failed! Max diff: {max_mean_diff}"
    
    # 2. Validate correlation alignment
    fused_corr = df_fused.corr(method="spearman").values
    corr_err = np.abs(fused_corr - target_corr_matrix.values).mean()
    logger.info("Mean Absolute Error vs Target Spearman Matrix: %.4f", corr_err)
    
    # Extract metrics for all 10 players
    projections = {}
    for col in df_fused.columns:
        series = df_fused[col]
        ev = float(series.mean())
        floor_p15 = float(np.percentile(series, 15))
        ceiling_p85 = float(np.percentile(series, 85))
        
        # Calculate CVaR 90 (expected value in top 10% GPP scenarios)
        var_90 = float(np.percentile(series, 90))
        cvar_90 = float(np.mean(series[series >= var_90]))
        
        # Calculate CVaR 10 (expected value in bottom 10% cash scenarios)
        var_10 = float(np.percentile(series, 10))
        cvar_10 = float(np.mean(series[series <= var_10]))
        
        projections[col] = {
            "EV": round(ev, 2),
            "Floor_p15": round(floor_p15, 2),
            "Ceiling_p85": round(ceiling_p85, 2),
            "cvar_90": round(cvar_90, 2),
            "cvar_10": round(cvar_10, 2)
        }
        
    logger.info("Successfully extracted EV, Floor (p15), Ceiling (p85), and CVaR metrics for all players.")
    return projections


def print_projections_table(projections: Dict[str, Dict[str, float]]) -> None:
    """
    Format and output clean table of player projections.
    """
    df_proj = pd.DataFrame.from_dict(projections, orient="index")
    logger.info("\n=== FINAL COPULA FUSED PLAYER PROJECTIONS ===\n" + df_proj.to_string())


if __name__ == "__main__":
    # 1. Fetch Phase 1 Top-Down predictions
    predictions_td = get_top_down_predictions()
    
    # 2. Fetch Phase 2 Target Spearman Correlation matrix
    df_sim_matrix = extract_simulation_matrix(num_iterations=10000, seed=42)
    df_target_corr, _ = compute_spearman_covariance(df_sim_matrix)
    
    # 3. Task 3.1 & 3.2: Generate independent marginals
    df_marginal = generate_independent_marginals(predictions_td, num_iterations=10000, seed=42)
    
    # 4. Task 3.3 & 3.4: Run Iman-Conover fusion
    df_fused_result = run_iman_conover_fusion(df_marginal, df_target_corr)
    
    # 5. Task 3.5: Validate and extract metrics
    final_projections = validate_and_extract_metrics(df_marginal, df_fused_result, df_target_corr)
    print_projections_table(final_projections)
