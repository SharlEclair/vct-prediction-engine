"""
Bracket Matchup Predictor Module — Phase 21
============================================
Provides fallback match pairing inference for tournament formats (Group Stage, Swiss, Double Elimination)
when the live VFL Schedule API returns no confirmed matches for a gameweek.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import itertools
import logging

logger = logging.getLogger("bracket_matchup_predictor")


@dataclass
class MatchupPrediction:
    team_a: str
    team_b: str
    probability: float = 1.0
    source: str = "schedule_api"  # "schedule_api" | "group_stage_inferred" | "swiss_inferred" | "double_elim_inferred"


class BracketMatchupPredictor:
    """
    Infers likely team pairings for tournament stages when exact schedule API data is missing.
    """

    def predict_group_stage(
        self,
        groups: Dict[str, List[str]],
        team_win_rates: Optional[Dict[str, float]] = None
    ) -> List[MatchupPrediction]:
        """
        Generates all valid within-group pairings for group stage matches.
        In a group stage, teams ONLY play opponents in their own group.
        Assigns probability = 1.0 / (num_teams_in_group - 1) as a uniform prior.
        """
        predictions: List[MatchupPrediction] = []

        for group_name, team_list in groups.items():
            clean_teams = [t.strip() for t in team_list if t and t.strip()]
            n_teams = len(clean_teams)

            if n_teams < 2:
                logger.warning(f"Group '{group_name}' has fewer than 2 teams: {clean_teams}")
                continue

            # Prior probability of any pair meeting in round-robin group stage
            prob = round(1.0 / max(n_teams - 1, 1), 4)

            # Generate unique pairings within this group
            for t1, t2 in itertools.combinations(clean_teams, 2):
                predictions.append(
                    MatchupPrediction(
                        team_a=t1,
                        team_b=t2,
                        probability=prob,
                        source=f"group_stage_inferred ({group_name})"
                    )
                )

        logger.info(f"Group Stage Inference generated {len(predictions)} pairings across {len(groups)} groups.")
        return predictions

    def predict_swiss(
        self,
        team_pool: List[str],
        standings_order: Optional[List[str]] = None
    ) -> List[MatchupPrediction]:
        """
        Stub for Swiss-format pairing inference (reserved for future Swiss stage events).
        """
        raise NotImplementedError("Swiss-format inference will be enabled for Swiss Stage events.")

    def predict_double_elim(
        self,
        upper_teams: List[str],
        lower_teams: Optional[List[str]] = None
    ) -> List[MatchupPrediction]:
        """
        Stub for Double Elimination bracket pairing inference (reserved for Playoff events).
        """
        raise NotImplementedError("Double-elimination bracket inference will be enabled for Playoff events.")
