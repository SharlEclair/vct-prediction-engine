"""
Feature Engineering Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 1).

Handles Task 1.3 (EMA Construction with dynamic alphas) and Task 1.4 (ODR Matrix Generation via Ridge Regression).
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_player_ema(
    df: pd.DataFrame, 
    target_col: str = "clipped_kpr", 
    alphas: Tuple[float, ...] = (0.1, 0.4)
) -> pd.DataFrame:
    """
    Task 1.3: Exponential Moving Average (EMA) Construction.
    Calculates dynamic temporal form windows for players based on chronological match order.
    Formula: EMA_t = alpha * X_t + (1 - alpha) * EMA_{t-1}
    
    Args:
        df (pd.DataFrame): Processed match telemetry dataframe sorted chronologically.
        target_col (str): Target column to compute EMA over (default 'clipped_kpr').
        alphas (Tuple[float, ...]): Dynamic decay factors (e.g., 0.1 for slow decay, 0.4 for rapid form).
        
    Returns:
        pd.DataFrame: DataFrame populated with new EMA columns (e.g., 'ema_kpr_alpha_0.1').
    """
    df = df.copy()
    
    # Ensure dataframe is sorted by timestamp and player
    if "match_timestamp" in df.columns:
        df.sort_values(by=["player_id", "match_timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    for alpha in alphas:
        col_name = f"ema_kpr_alpha_{alpha}"
        
        # Calculate expanding/exponentially weighted moving average per player
        # Note: pandas ewm with adjust=False matches the exact recursive recursive formula:
        # y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        df[col_name] = (
            df.groupby("player_id")[target_col]
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        )
        logger.info("Task 1.3 complete: Computed EMA feature '%s'.", col_name)
        
    return df


def generate_odr_matrix(
    df: pd.DataFrame, 
    target_col: str = "kpr", 
    alpha_ridge: float = 1.0
) -> Dict[str, float]:
    """
    Task 1.4: Opponent Defensive Rating (ODR) Matrix Generation.
    Formulates a Ridge-penalized regression solver to isolate schedule-adjusted defensive suppression capabilities.
    System of equations: KPR_{ij} = mu_{league} + Offense_i - Defense_j + epsilon_{ij}
    
    Args:
        df (pd.DataFrame): Match telemetry containing 'team_name', 'opponent_team_name', and target metric.
        target_col (str): Rate metric column (default 'kpr').
        alpha_ridge (float): Ridge regularization hyperparameter.
        
    Returns:
        Dict[str, float]: Mapping of team name to continuous Defense_j scalar (expected kills suppressed per round).
    """
    all_teams = sorted(list(set(df["team_name"]).union(set(df["opponent_team_name"]))))
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    num_teams = len(all_teams)
    
    num_samples = len(df)
    # Feature matrix X: Offense_i columns followed by Defense_j columns
    # Offense_i gets +1, Defense_j gets -1 so coefficient directly represents Defense_j suppression.
    X = np.zeros((num_samples, 2 * num_teams))
    y = df[target_col].values
    
    for row_idx, (_, row) in enumerate(df.iterrows()):
        off_idx = team_to_idx[row["team_name"]]
        def_idx = team_to_idx[row["opponent_team_name"]]
        
        X[row_idx, off_idx] = 1.0                # Offense_i
        X[row_idx, num_teams + def_idx] = -1.0     # -Defense_j
        
    ridge = Ridge(alpha=alpha_ridge, fit_intercept=True)
    ridge.fit(X, y)
    
    mu_league = ridge.intercept_
    def_coefs = ridge.coef_[num_teams:] # Coefficients corresponding to -Defense_j
    
    odr_matrix = {team: float(def_coefs[idx]) for team, idx in team_to_idx.items()}
    
    logger.info("Task 1.4 complete: ODR Matrix solved for %d teams across %d observations. Baseline mu_league = %.4f", 
                num_teams, num_samples, mu_league)
    for team, odr in odr_matrix.items():
        logger.debug("Team: %-15s ODR (Defensive Suppression): %+.4f KPR", team, odr)
        
    return odr_matrix


def attach_odr_features(df: pd.DataFrame, odr_matrix: Dict[str, float]) -> pd.DataFrame:
    """
    Attach opponent defensive rating (ODR) to each row based on opponent_team_name.
    
    Args:
        df (pd.DataFrame): Match telemetry.
        odr_matrix (Dict[str, float]): Map of team -> Defense_j scalar.
        
    Returns:
        pd.DataFrame: DataFrame with added 'opponent_odr' column.
    """
    df = df.copy()
    df["opponent_odr"] = df["opponent_team_name"].map(odr_matrix).fillna(0.0)
    return df
