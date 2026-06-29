"""
DAG Simulation Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 2 & 5).

Repurposes the legacy V5 Directed Acyclic Graph (DAG) pipeline structure into a high-variance
Monte Carlo simulator. Simulates 10,000 iterations across four sequential stages:
1. Contextual Bandit Map Veto
2. Hungarian Agent Draft
3. Side-Conditioned Markov Round Simulator
4. Dirichlet Regression Kill Share & DFS Scoring

All static constants (roles, weights, map pools) are dynamically loaded from config.yaml.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Load dynamic configuration
_config = load_config()
ROLE_ALPHA_WEIGHTS = _config.get("ROLE_ALPHA_WEIGHTS", {})
ROLES = list(ROLE_ALPHA_WEIGHTS.keys())
COMPETITIVE_MAP_POOL = _config.get("COMPETITIVE_MAP_POOL", ["Ascent", "Bind", "Haven"])


class MockDAGSimulator:
    """
    Monte Carlo DAG Simulator approximating the V5 sequential pipeline:
    Map Veto -> Draft -> Markov Rounds -> Dirichlet Kill Share.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.config = load_config()
        self.role_alpha_weights = self.config.get("ROLE_ALPHA_WEIGHTS", {})
        self.roles = list(self.role_alpha_weights.keys())
        self.map_pool = self.config.get("COMPETITIVE_MAP_POOL", ["Ascent", "Bind", "Haven"])
        
    def _stage_1_map_veto(self) -> Dict[str, float]:
        """
        Stage 1: Contextual Bandit Map Veto.
        Determines map difficulty, momentum bias, and volatility modifiers using config map pool.
        """
        selected_map = np.random.choice(self.map_pool)
        volatility = np.random.uniform(0.8, 1.2)
        team_a_side_bias = np.random.normal(0.52, 0.05)
        
        return {
            "map": selected_map,
            "volatility": volatility,
            "team_a_side_bias": float(np.clip(team_a_side_bias, 0.40, 0.65))
        }

    def _stage_2_agent_draft(self) -> Tuple[List[str], List[str]]:
        """
        Stage 2: Hungarian Agent Draft.
        Assigns tactical agent roles to 5 players on Team A and 5 players on Team B.
        """
        team_a_roles = list(np.random.permutation(self.roles))
        team_b_roles = list(np.random.permutation(self.roles))
        return team_a_roles, team_b_roles

    def _stage_3_markov_round_simulator(self, map_context: Dict[str, float]) -> Tuple[int, int]:
        """
        Stage 3: Side-Conditioned Markov Round Simulator.
        Simulates round-by-round momentum and win conditions, capturing sweeps and overtime variance.
        """
        p_win_a = map_context["team_a_side_bias"]
        volatility = map_context["volatility"]
        
        rounds_a, rounds_b = 0, 0
        
        while True:
            lead_diff = (rounds_a - rounds_b) * 0.015 * volatility
            current_p_a = np.clip(p_win_a + lead_diff, 0.20, 0.80)
            
            if np.random.rand() < current_p_a:
                rounds_a += 1
            else:
                rounds_b += 1
                
            if rounds_a >= 13 and rounds_a - rounds_b >= 2:
                break
            if rounds_b >= 13 and rounds_b - rounds_a >= 2:
                break
            if rounds_a + rounds_b >= 40:
                break
                
        return rounds_a, rounds_b

    def _stage_4_dirichlet_kill_share(
        self, 
        rounds_a: int, 
        rounds_b: int, 
        roles_a: List[str], 
        roles_b: List[str]
    ) -> np.ndarray:
        """
        Stage 4: Dirichlet Regression Kill Share & DFS Fantasy Point Scoring.
        Distributes team kills using dynamic role weights from configuration.
        """
        total_rounds = rounds_a + rounds_b
        round_diff = rounds_a - rounds_b
        
        team_a_performance = np.random.normal(1.0 + round_diff * 0.03, 0.15)
        team_b_performance = np.random.normal(1.0 - round_diff * 0.03, 0.15)
        
        kills_a_total = int(np.round((rounds_a * 2.6 + rounds_b * 1.2) * team_a_performance))
        kills_b_total = int(np.round((rounds_b * 2.6 + rounds_a * 1.2) * team_b_performance))
        
        kills_a_total = max(5, kills_a_total)
        kills_b_total = max(5, kills_b_total)
        
        rate_a = np.array([self.role_alpha_weights.get(r, 2.0) for r in roles_a])
        rate_a = rate_a / sum(rate_a)
        rate_b = np.array([self.role_alpha_weights.get(r, 2.0) for r in roles_b])
        rate_b = rate_b / sum(rate_b)
        
        exp_kills_a = kills_a_total * rate_a * np.random.normal(1.0, 0.1, 5)
        exp_kills_b = kills_b_total * rate_b * np.random.normal(1.0, 0.1, 5)
        
        player_kills_a = np.random.poisson(np.maximum(0.5, exp_kills_a))
        player_kills_b = np.random.poisson(np.maximum(0.5, exp_kills_b))
        
        deaths_a = np.random.multinomial(kills_b_total, np.full(5, 0.2))
        deaths_b = np.random.multinomial(kills_a_total, np.full(5, 0.2))
        
        assists_a = np.round(player_kills_a * np.random.uniform(0.35, 0.55) + np.random.poisson(1.5, 5)).astype(int)
        assists_b = np.round(player_kills_b * np.random.uniform(0.35, 0.55) + np.random.poisson(1.5, 5)).astype(int)
        
        fb_share_a = np.random.dirichlet([self.role_alpha_weights.get(r, 2.0) for r in roles_a])
        fb_share_b = np.random.dirichlet([self.role_alpha_weights.get(r, 2.0) for r in roles_b])
        fb_a = np.random.multinomial(min(rounds_a, 13), fb_share_a)
        fb_b = np.random.multinomial(min(rounds_b, 13), fb_share_b)
        
        win_bonus_a = 3.0 if rounds_a > rounds_b else 0.0
        win_bonus_b = 3.0 if rounds_b > rounds_a else 0.0
        
        dfs_pts_a = player_kills_a * 3.0 - deaths_a * 1.0 + assists_a * 1.5 + fb_a * 2.0 + win_bonus_a
        dfs_pts_b = player_kills_b * 3.0 - deaths_b * 1.0 + assists_b * 1.5 + fb_b * 2.0 + win_bonus_b
        
        return np.concatenate([dfs_pts_a, dfs_pts_b])

    def simulate_iterations(self, num_iterations: int = 10000) -> pd.DataFrame:
        """
        Run Task 2.1: Monte Carlo simulation iterations.
        """
        logger.info("Executing Monte Carlo DAG simulation for %d iterations (Config driven)...", num_iterations)
        
        results_matrix = np.zeros((num_iterations, 10))
        
        for i in range(num_iterations):
            map_ctx = self._stage_1_map_veto()
            roles_a, roles_b = self._stage_2_agent_draft()
            ra, rb = self._stage_3_markov_round_simulator(map_ctx)
            iteration_dfs_points = self._stage_4_dirichlet_kill_share(ra, rb, roles_a, roles_b)
            results_matrix[i, :] = iteration_dfs_points
            
        columns = [f"P{j}_TeamA" for j in range(5)] + [f"P{j}_TeamB" for j in range(5, 10)]
        df_sim = pd.DataFrame(results_matrix, columns=columns)
        
        logger.info("Task 2.1 Complete: Generated simulation matrix of shape %s.", df_sim.shape)
        return df_sim
