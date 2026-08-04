"""
v9_bracket_monte_carlo.py
-------------------------
Valorant Fantasy League (VFL) DFS Prediction Engine - v9 Architecture.
Phase 6: Tournament Stage & Team Elimination Lives Stochastic Simulator.

This module models bracket survival probabilities S_{i, t}(L_k) and expected match
volumes for draw-dependent future gameweeks across Swiss, Double Elim, GSL, and Single Elim formats.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class TeamLivesState:
    """State container for a team's elimination lives and bracket status."""
    team_name: str
    lives_remaining: int = 2        # Default 2 lives (Upper Bracket / 1-1 Swiss)
    stage_preset: str = "Double Elimination Playoffs"
    is_lower_bracket: bool = False
    custom_survival_override: Optional[float] = None


STAGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Double Elimination Playoffs": {
        "upper_bracket_lives": 2,
        "lower_bracket_lives": 1,
        "default_survival_upper": 0.90,
        "default_survival_lower": 0.50,
    },
    "Swiss Stage (3-Loss Elimination)": {
        "undefeated_lives": 3,
        "mid_record_lives": 2,
        "knockout_lives": 1,
        "default_survival_3_lives": 1.00,
        "default_survival_2_lives": 0.85,
        "default_survival_1_life": 0.45,
    },
    "GSL Groups (4-Team Double Elim)": {
        "winners_bracket_lives": 2,
        "decider_bracket_lives": 1,
        "default_survival_2_lives": 0.88,
        "default_survival_1_life": 0.50,
    },
    "Single Elimination Playoffs": {
        "default_lives": 1,
        "default_survival": 0.50,
    }
}


def compute_survival_probability(
    lives_remaining: int,
    stage_preset: str = "Double Elimination Playoffs",
    win_probability_override: Optional[float] = None
) -> float:
    """
    Computes survival probability S_{i, t}(L_k) for a team with L_k lives remaining.
    """
    if win_probability_override is not None:
        p_win = max(0.05, min(0.95, win_probability_override))
    else:
        p_win = 0.50  # Default 50/50 match win probability

    if lives_remaining >= 3:
        return 1.00
    elif lives_remaining == 2:
        # One cushion loss available: Survival = 1 - (P_loss)^2 = 1 - (1 - p_win)^2
        p_loss = 1.0 - p_win
        return float(1.0 - (p_loss ** 2))
    elif lives_remaining == 1:
        # Knockout match: Must win next match to survive
        return float(p_win)
    else:
        # Already eliminated
        return 0.00


class StochasticBracketSimulator:
    """
    Simulates multi-period player expected values discounted by tournament stage
    survival probabilities S_{i, t}(L_k) across K gameweeks.
    """

    def __init__(self, stage_preset: str = "Double Elimination Playoffs"):
        self.stage_preset = stage_preset
        self.team_states: Dict[str, TeamLivesState] = {}

    def set_team_lives(self, team_name: str, lives: int, is_lower_bracket: bool = False, custom_override: Optional[float] = None):
        """Sets or updates the elimination lives state for a specific VCT team."""
        self.team_states[team_name] = TeamLivesState(
            team_name=team_name,
            lives_remaining=lives,
            stage_preset=self.stage_preset,
            is_lower_bracket=is_lower_bracket,
            custom_survival_override=custom_override
        )

    def configure_tier1_presets(
        self,
        all_teams: List[str],
        lower_bracket_teams: Optional[List[str]] = None,
        knockout_teams: Optional[List[str]] = None
    ):
        """
        Tier-1 Quick Configurator: Automatically sets default lives for all teams,
        moving specified lower bracket or knockout teams to 1 life.
        """
        lower_bracket_teams = lower_bracket_teams or []
        knockout_teams = knockout_teams or []

        for team in all_teams:
            if team in lower_bracket_teams or team in knockout_teams:
                self.set_team_lives(team, lives=1, is_lower_bracket=True)
            else:
                if self.stage_preset == "Swiss Stage (3-Loss Elimination)":
                    self.set_team_lives(team, lives=3)
                else:
                    self.set_team_lives(team, lives=2)

    def calculate_stochastic_player_ev_matrix(
        self,
        players: List[Dict[str, Any]],
        horizon_weeks: int = 4,
        known_schedule_weeks: int = 2,
        risk_bias_mode: str = "Balanced"
    ) -> np.ndarray:
        """
        Calculates an N x K matrix of stochastic player Expected Values across K gameweeks.
        
        Parameters:
            players: List of N player dicts with 'name', 'team', 'ppg' (or 'computed_ppg')
            horizon_weeks: Total gameweeks K (e.g. 4)
            known_schedule_weeks: Number of initial weeks with exact fixture schedule
            risk_bias_mode: 'Risk-Averse', 'Balanced', or 'Aggressive'
            
        Returns:
            N x K numpy matrix EV_matrix[i, t]
        """
        N = len(players)
        K = horizon_weeks
        EV_matrix = np.zeros((N, K), dtype=np.float64)

        # Risk bias scaling exponent gamma
        if risk_bias_mode == "Risk-Averse":
            gamma_exp = 1.8  # Severely penalizes low survival probability
        elif risk_bias_mode == "Aggressive":
            gamma_exp = 0.5  # Softens elimination penalty for upside chasing
        else:
            gamma_exp = 1.0  # Balanced linear probability

        for i, player in enumerate(players):
            base_ev = float(player.get('computed_ppg') or player.get('ppg') or 10.0)
            team = player.get('team', '')

            # Get team lives state
            state = self.team_states.get(team, TeamLivesState(team_name=team, lives_remaining=2))
            survival_p1 = compute_survival_probability(
                state.lives_remaining,
                self.stage_preset,
                state.custom_survival_override
            )

            for t in range(K):
                if t < known_schedule_weeks:
                    # Deterministic week: full EV
                    EV_matrix[i, t] = base_ev
                else:
                    # Stochastic week: decay survival probability compounding over unassigned weeks
                    unassigned_index = (t - known_schedule_weeks) + 1
                    compounded_survival = float(survival_p1 ** unassigned_index)
                    scaled_survival = float(compounded_survival ** gamma_exp)
                    EV_matrix[i, t] = base_ev * scaled_survival

        return EV_matrix
