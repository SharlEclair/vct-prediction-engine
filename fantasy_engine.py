"""
VFL (Valorant Fantasy League) Optimization & Scoring Engine
============================================================
Implements fantasy scoring, player metrics calculation, and roster optimization
using scipy.optimize.milp (Mixed-Integer Linear Programming).
"""

import os
import json
import glob
import logging
import re
from collections import defaultdict
import numpy as np
from datetime import datetime
from scipy.optimize import milp, LinearConstraint, Bounds

logger = logging.getLogger("fantasy_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

def parse_match_date(date_str: str) -> datetime:
    # Basic date parsing fallback
    clean_str = date_str.split(" Patch ")[0]
    clean_str = re.sub(r'\s+[A-Z]{3,4}$', '', clean_str).strip()
    clean_str = re.sub(r'^[A-Za-z]+,\s*', '', clean_str).strip()
    
    # Check simple year match
    year_match = re.search(r'\b(20\d{2})\b', date_str)
    year = int(year_match.group(1)) if year_match else 2026
    
    month_day_match = re.search(r'^([A-Za-z]+)\s+(\d+)', clean_str)
    if not month_day_match:
        return datetime(2026, 6, 22)
    month = month_day_match.group(1)
    day = int(month_day_match.group(2))
    
    time_match = re.search(r'(\d+:\d+)\s+([AP]M)', clean_str)
    time_str = time_match.group(1) if time_match else "12:00"
    ampm = time_match.group(2) if time_match else "PM"
    
    try:
        normalized_date_str = f"{month} {day}, {year} {time_str} {ampm}"
        return datetime.strptime(normalized_date_str, "%B %d, %Y %I:%M %p")
    except Exception:
        return datetime(2026, 6, 22)

# --- Helper functions to retrieve metadata ---
def get_upcoming_matchups(raw_dir: str = RAW_DIR) -> list[tuple[int, int]]:
    """Gets list of head-to-head matchups in upcoming gameweek window."""
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    matchups = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                seg = content["data"]["segments"][0]
                ts = parse_match_date(seg["date"])
                # Active gameweek window (June 20 to June 30, 2026)
                if datetime(2026, 6, 20) <= ts <= datetime(2026, 6, 30):
                    t1_id = int(seg["teams"][0]["id"])
                    t2_id = int(seg["teams"][1]["id"])
                    matchups.append((t1_id, t2_id))
        except Exception:
            pass
    if not matchups:
        # Fallback to seed matchups
        matchups = [
            (624, 2359),   # Paper Rex vs LEVIATÁN
            (2, 1001)      # Sentinels vs Team Heretics
        ]
    return list(set(matchups))

def get_team_win_rates_by_id(raw_dir: str = RAW_DIR) -> tuple[dict[int, float], dict[str, int]]:
    """Computes win rates for teams by ID and generates name mapping."""
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    team_matches = defaultdict(int)
    team_wins = defaultdict(int)
    team_name_to_id = {}
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                seg = content["data"]["segments"][0]
                t1 = seg["teams"][0]
                t2 = seg["teams"][1]
                t1_name = t1["name"]
                t2_name = t2["name"]
                t1_id = int(t1["id"])
                t2_id = int(t2["id"])
                team_name_to_id[t1_name] = t1_id
                team_name_to_id[t2_name] = t2_id
                
                team_matches[t1_id] += 1
                team_matches[t2_id] += 1
                
                score_1 = int(t1.get("score") or 0)
                score_2 = int(t2.get("score") or 0)
                if t1.get("is_winner") is True:
                    team_wins[t1_id] += 1
                elif t2.get("is_winner") is True:
                    team_wins[t2_id] += 1
                elif score_1 > score_2:
                    team_wins[t1_id] += 1
                elif score_2 > score_1:
                    team_wins[t2_id] += 1
        except Exception:
            pass
            
    win_rates = {}
    for tid, matches in team_matches.items():
        if matches > 0:
            win_rates[tid] = team_wins[tid] / matches
            
    return win_rates, team_name_to_id


class VCTFantasyEngine:
    """Calculates player map, series, and match points based on precise VFL rules."""
    
    def calculate_kills_points(self, kills: int) -> int:
        """Kills Metrics: Piecewise penalties/rewards."""
        if kills == 0:
            return -3
        elif 1 <= kills <= 4:
            return -1
        elif 5 <= kills <= 9:
            return 0
        else:  # kills >= 10
            return 1 + (kills - 10) // 5

    def calculate_multikill_points(self, k4: int, k5: int, k6: int, k7: int) -> float:
        """Multi-Kills Vector: 4K (+1), 5K (+3), 6K (+5), and 7K (+10) points."""
        return k4 * 1.0 + k5 * 3.0 + k6 * 5.0 + k7 * 10.0

    def calculate_round_margin_points(self, team_score: int, opp_score: int) -> int:
        """Map Out-of-Bounds Metrics: Round difference points."""
        if team_score == 13 and opp_score == 0:
            return 5
        elif team_score == 0 and opp_score == 13:
            return -5
        
        if team_score > opp_score:
            diff = team_score - opp_score
            pts = 1  # Map win
            if 5 <= diff <= 9:
                pts += 1  # 5-9 margin
            elif diff >= 10:
                pts += 2  # 10+ sweep margin
            return pts
        else:
            diff = opp_score - team_score
            if diff >= 10:
                return -1  # 10+ round loss penalty
            return 0

    def calculate_series_bonus(self, player_team: str, team_a: str, team_b: str, score_a: int, score_b: int) -> int:
        """Series Scale Modifiers: 2-0 (+2), 3-0 (+4), and 3-1 (+1) series bonuses."""
        pt = player_team.lower().strip()
        ta = team_a.lower().strip()
        tb = team_b.lower().strip()
        
        is_team_a = pt in ta or ta in pt
        is_team_b = pt in tb or tb in pt
        
        if score_a > score_b:
            if not is_team_a:
                return 0
            if score_a == 2 and score_b == 0:
                return 2
            elif score_a == 3 and score_b == 0:
                return 4
            elif score_a == 3 and score_b == 1:
                return 1
        elif score_b > score_a:
            if not is_team_b:
                return 0
            if score_b == 2 and score_a == 0:
                return 2
            elif score_b == 3 and score_a == 0:
                return 4
            elif score_b == 3 and score_a == 1:
                return 1
        return 0

    def get_rating_scaling_bonus(self, avg_rating: float) -> int:
        """VLR Rating absolute scaling modifiers: +1 for 1.5+, +2 for 1.75+, +3 for 2.0+."""
        if avg_rating >= 2.0:
            return 3
        elif avg_rating >= 1.75:
            return 2
        elif avg_rating >= 1.5:
            return 1
        return 0

    def score_match_json(self, match_filepath: str) -> list[dict]:
        """Loads a raw match JSON and computes fantasy leaderboard scores for all players."""
        with open(match_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "data" not in data or "segments" not in data["data"] or not data["data"]["segments"]:
            return []
            
        segment = data["data"]["segments"][0]
        team_a_name = segment["teams"][0]["name"]
        team_b_name = segment["teams"][1]["name"]
        
        try:
            score_a = int(segment["teams"][0]["score"] or 0)
            score_b = int(segment["teams"][1]["score"] or 0)
        except ValueError:
            score_a, score_b = 0, 0
            
        players_data = defaultdict(lambda: {"team": "", "map_scores": {}, "ratings": [], "kills": []})
        
        for map_data in segment.get("maps", []):
            map_name = map_data.get("map_name")
            if not map_name or map_name.lower() in ["all maps", "none"]:
                continue
                
            m_score = map_data.get("score", {})
            m_score_t1 = m_score.get("team1", 0)
            m_score_t2 = m_score.get("team2", 0)
            
            delta_pts_team1 = self.calculate_round_margin_points(m_score_t1, m_score_t2)
            delta_pts_team2 = self.calculate_round_margin_points(m_score_t2, m_score_t1)
            
            perf = map_data.get("performance", {})
            adv_stats = perf.get("advanced_stats", [])
            
            adv_lookup = {}
            for row in adv_stats:
                p_label = row.get("player", "")
                adv_lookup[p_label.lower().strip()] = row
                
            for p in map_data.get("players", {}).get("team1", []):
                p_name = p["name"]
                players_data[p_name]["team"] = team_a_name
                
                kills = int(p.get("kills") or 0)
                players_data[p_name]["kills"].append(kills)
                rating = float(p.get("rating") or 0.0)
                players_data[p_name]["ratings"].append(rating)
                
                kills_pts = self.calculate_kills_points(kills)
                
                k4, k5, k6, k7 = 0, 0, 0, 0
                matched_row = None
                for key_label, row in adv_lookup.items():
                    if p_name.lower().strip() in key_label:
                        matched_row = row
                        break
                        
                if matched_row:
                    k4 = int(matched_row.get("4") or 0)
                    k5 = int(matched_row.get("10") or 0)
                    
                multikill_pts = self.calculate_multikill_points(k4, k5, k6, k7)
                
                map_score = kills_pts + multikill_pts + delta_pts_team1
                players_data[p_name]["map_scores"][map_name] = map_score
                
            for p in map_data.get("players", {}).get("team2", []):
                p_name = p["name"]
                players_data[p_name]["team"] = team_b_name
                
                kills = int(p.get("kills") or 0)
                players_data[p_name]["kills"].append(kills)
                rating = float(p.get("rating") or 0.0)
                players_data[p_name]["ratings"].append(rating)
                
                kills_pts = self.calculate_kills_points(kills)
                
                k4, k5, k6, k7 = 0, 0, 0, 0
                matched_row = None
                for key_label, row in adv_lookup.items():
                    if p_name.lower().strip() in key_label:
                        matched_row = row
                        break
                        
                if matched_row:
                    k4 = int(matched_row.get("4") or 0)
                    k5 = int(matched_row.get("10") or 0)
                    
                multikill_pts = self.calculate_multikill_points(k4, k5, k6, k7)
                
                map_score = kills_pts + multikill_pts + delta_pts_team2
                players_data[p_name]["map_scores"][map_name] = map_score
                
        leaderboard = []
        all_players_ratings = []
        for name, p_data in players_data.items():
            if p_data["ratings"]:
                avg_rating = sum(p_data["ratings"]) / len(p_data["ratings"])
            else:
                avg_rating = 0.0
            all_players_ratings.append((name, avg_rating))
            
        all_players_ratings.sort(key=lambda x: x[1], reverse=True)
        placements = {}
        for rank, (name, _) in enumerate(all_players_ratings):
            if rank == 0:
                placements[name] = 3
            elif rank == 1:
                placements[name] = 2
            elif rank == 2:
                placements[name] = 1
            else:
                placements[name] = 0
                
        for name, p_data in players_data.items():
            map_scores_list = list(p_data["map_scores"].values())
            sorted_map_scores = sorted(map_scores_list)
            # Capping points at top 2 maps per gameweek
            top_2_scores = sorted_map_scores[-2:] if len(sorted_map_scores) >= 2 else sorted_map_scores
            map_score_agg = sum(top_2_scores)
            
            series_bonus = self.calculate_series_bonus(p_data["team"], team_a_name, team_b_name, score_a, score_b)
            rating_placement_bonus = placements.get(name, 0)
            avg_rating = sum(p_data["ratings"]) / len(p_data["ratings"]) if p_data["ratings"] else 0.0
            rating_scaling_bonus = self.get_rating_scaling_bonus(avg_rating)
            
            total_fantasy_score = map_score_agg + series_bonus + rating_placement_bonus + rating_scaling_bonus
            
            leaderboard.append({
                "player": name,
                "team": p_data["team"],
                "avg_rating": round(avg_rating, 2),
                "map_scores": p_data["map_scores"],
                "map_score_agg": map_score_agg,
                "series_bonus": series_bonus,
                "rating_placement_bonus": rating_placement_bonus,
                "rating_scaling_bonus": rating_scaling_bonus,
                "total_score": total_fantasy_score
            })
            
        leaderboard.sort(key=lambda x: x["total_score"], reverse=True)
        return leaderboard


def compute_all_players_historical_stats(raw_dir: str = RAW_DIR) -> dict[str, dict]:
    """Computes PPG and std_dev (sigma) for all players based on database matches."""
    logger.info("Computing all player statistics from historical matches...")
    engine = VCTFantasyEngine()
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    player_scores = defaultdict(list)
    
    for f in files:
        try:
            leaderboard = engine.score_match_json(f)
            for entry in leaderboard:
                p_name = entry["player"]
                player_scores[p_name].append(entry["total_score"])
        except Exception:
            pass
            
    stats = {}
    for p_name, scores in player_scores.items():
        if scores:
            stats[p_name] = {
                "ppg": float(np.mean(scores)),
                "sigma": float(np.std(scores)) if len(scores) > 1 else 3.0,
                "matches_played": len(scores)
            }
    return stats

# --- MILP Roster Optimizer ---
def optimize_roster(
    vfl_players: list[dict],
    salary_cap: int = 50,
    roster_size: int = 6,
    max_per_team: int = 2,
    survival_threshold: float = 0.35,
    upcoming_matchups: list[tuple[int, int]] = None,
    team_win_rates: dict[int, float] = None,
    player_stats: dict[str, dict] = None,
    transfer_constraint: dict = None  # Contains 'current_roster' and 'max_transfers'
) -> dict:
    """
    Solves VFL roster selection as a Mixed-Integer Linear Program (MILP).
    Enforces constraints:
      - Budget <= salary_cap (50 VP). Soft penalty for cost > 48 VP.
      - Roster length = exactly roster_size (6).
      - Max max_per_team (2) from same team.
      - Exactly 1 Duelist, 1 Initiator, 1 Controller, 1 Sentinel, and 2 Wildcard slots.
      - Head-to-head match penalty.
      - Survival threshold filtering.
    Dynamically identifies IGL (highest floor = PPG - 1.0 * sigma) and attaches 2x multiplier.
    """
    if upcoming_matchups is None:
        upcoming_matchups = get_upcoming_matchups(RAW_DIR)
    if team_win_rates is None:
        team_win_rates, _ = get_team_win_rates_by_id(RAW_DIR)
    if player_stats is None:
        player_stats = compute_all_players_historical_stats(RAW_DIR)

    # 1. Filter out players below the survival win rate threshold
    filtered_players = []
    for p in vfl_players:
        tid = p.get("vlr_team_id")
        wr = team_win_rates.get(tid, 0.50) if tid is not None else 0.50
        if wr >= survival_threshold:
            # Enrich player data with stats database lookup
            pname = p.get("player_name", "")
            stats = player_stats.get(pname, {"ppg": p.get("ppg", 10.0), "sigma": 3.0})
            p["computed_ppg"] = stats.get("ppg", p.get("ppg", 10.0))
            p["computed_sigma"] = stats.get("sigma", 3.0)
            p["floor"] = p["computed_ppg"] - 1.0 * p["computed_sigma"]
            filtered_players.append(p)
            
    n = len(filtered_players)
    if n == 0:
        return {"optimal_roster": [], "total_cost": 0, "projected_points": 0.0, "solver_status": "no_feasible_players"}

    # Formulate head-to-head matchup pairs
    h2h_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            tid_a = filtered_players[i].get("vlr_team_id")
            tid_b = filtered_players[j].get("vlr_team_id")
            if tid_a is not None and tid_b is not None:
                # Check if they face each other in upcoming week
                for match in upcoming_matchups:
                    if (tid_a == match[0] and tid_b == match[1]) or (tid_a == match[1] and tid_b == match[0]):
                        h2h_pairs.append((i, j))
                        break
                        
    m = len(h2h_pairs)
    
    # Variables definition for optimization:
    # Index [0, N-1]: x_i_nat (selected to fill natural role)
    # Index [N, 2N-1]: x_i_wild (selected to fill wildcard role)
    # Index [2N]: u (auxiliary continuous variable for soft budget cap penalty: u >= sum(x_i * c_i) - 48)
    # Index [2N+1, 2N+M]: w_k (binary representing each head-to-head penalty variable)
    
    num_vars = 2 * n + 1 + m
    
    # We iterate over each candidate player as the potential In-Game Leader (IGL).
    # Since IGL must have the highest floor among selected players, we force that candidate's
    # selection, set its points multiplier to 2.0, and force all players with a floor strictly
    # higher than the candidate to not be selected (i.e. x_i_nat = x_i_wild = 0).
    best_overall_score = -float('inf')
    best_result = None
    best_igl_idx = -1
    
    # Set role mapping indicators
    is_duelist = np.array([1.0 if p["role"] == "Duelist" else 0.0 for p in filtered_players])
    is_initiator = np.array([1.0 if p["role"] == "Initiator" else 0.0 for p in filtered_players])
    is_controller = np.array([1.0 if p["role"] == "Controller" else 0.0 for p in filtered_players])
    is_sentinel = np.array([1.0 if p["role"] == "Sentinel" else 0.0 for p in filtered_players])
    
    costs = np.array([p["price"] for p in filtered_players], dtype=float)
    
    # Loop over all players as potential IGL candidates
    for k in range(n):
        # Candidate IGL floor
        igl_floor = filtered_players[k]["floor"]
        
        # Objective coefficient vector (we minimize, so we negate points)
        # points = ppg for normal selections. Player k points are doubled.
        pts = np.array([p["computed_ppg"] for p in filtered_players], dtype=float)
        
        c = np.zeros(num_vars)
        c[:n] = -pts          # x_nat coefficients
        c[n:2*n] = -pts       # x_wild coefficients
        c[k] -= pts[k]        # Add extra -pts[k] for natural selection IGL (since x_k_nat + x_k_wild = 1)
        c[n + k] -= pts[k]    # Add extra -pts[k] for wildcard selection IGL
        
        c[2*n] = 0.5          # soft penalty coefficient for cost > 48 VP (add 0.5 per VP above 48)
        c[2*n + 1:] = 20.0    # head-to-head matchup penalty of 20 points
        
        # Build constraints
        A_ub_rows = []
        b_ub_lower = []
        b_ub_upper = []
        
        A_eq_rows = []
        b_eq = []
        
        # 1. x_i = x_i_nat + x_i_wild <= 1
        for i in range(n):
            row = np.zeros(num_vars)
            row[i] = 1.0
            row[n + i] = 1.0
            A_ub_rows.append(row)
            b_ub_lower.append(0.0)
            b_ub_upper.append(1.0)
            
        # 2. Hard selection check: IGL candidate k must be drafted
        row = np.zeros(num_vars)
        row[k] = 1.0
        row[n + k] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)
        
        # 3. IGL floor constraint: any player i with floor > igl_floor cannot be selected
        for i in range(n):
            if filtered_players[i]["floor"] > igl_floor:
                row = np.zeros(num_vars)
                row[i] = 1.0
                row[n + i] = 1.0
                A_eq_rows.append(row)
                b_eq.append(0.0)
                
        # 4. Role natural counts
        # sum_{i in Duelist} x_i_nat = 1
        row = np.zeros(num_vars)
        row[:n] = is_duelist
        A_eq_rows.append(row)
        b_eq.append(1.0)
        
        # sum_{i in Initiator} x_i_nat = 1
        row = np.zeros(num_vars)
        row[:n] = is_initiator
        A_eq_rows.append(row)
        b_eq.append(1.0)
        
        # sum_{i in Controller} x_i_nat = 1
        row = np.zeros(num_vars)
        row[:n] = is_controller
        A_eq_rows.append(row)
        b_eq.append(1.0)
        
        # sum_{i in Sentinel} x_i_nat = 1
        row = np.zeros(num_vars)
        row[:n] = is_sentinel
        A_eq_rows.append(row)
        b_eq.append(1.0)
        
        # sum_i x_i_wild = 2
        row = np.zeros(num_vars)
        row[n:2*n] = 1.0
        A_eq_rows.append(row)
        b_eq.append(2.0)
        
        # 5. Hard Budget Constraint: sum(x_i * cost_i) <= 50
        row = np.zeros(num_vars)
        row[:n] = costs
        row[n:2*n] = costs
        A_ub_rows.append(row)
        b_ub_lower.append(0.0)
        b_ub_upper.append(float(salary_cap))
        
        # 6. Soft budget cap constraint: u >= sum(x_i * cost_i) - 48
        # => sum(x_i * cost_i) - u <= 48
        row = np.zeros(num_vars)
        row[:n] = costs
        row[n:2*n] = costs
        row[2*n] = -1.0
        A_ub_rows.append(row)
        b_ub_lower.append(-np.inf)
        b_ub_upper.append(48.0)
        
        # 7. Team limit constraint: max 2 players from any real VCT team ID
        all_team_ids = set(p["vlr_team_id"] for p in filtered_players if p.get("vlr_team_id") is not None)
        for tid in all_team_ids:
            is_team = np.array([1.0 if p.get("vlr_team_id") == tid else 0.0 for p in filtered_players])
            row = np.zeros(num_vars)
            row[:n] = is_team
            row[n:2*n] = is_team
            A_ub_rows.append(row)
            b_ub_lower.append(0.0)
            b_ub_upper.append(float(max_per_team))
            
        # 8. Head-to-head matchup penalty constraints:
        # w_k >= x_i + x_j - 1
        # => x_i + x_j - w_k <= 1
        for idx, (idx_i, idx_j) in enumerate(h2h_pairs):
            row = np.zeros(num_vars)
            row[idx_i] = 1.0
            row[n + idx_i] = 1.0
            row[idx_j] = 1.0
            row[n + idx_j] = 1.0
            row[2*n + 1 + idx] = -1.0
            A_ub_rows.append(row)
            b_ub_lower.append(-np.inf)
            b_ub_upper.append(1.0)
            
        # 9. Transfer constraints (for suggestion component)
        if transfer_constraint is not None:
            curr_names = set(p["player_name"] for p in transfer_constraint["current_roster"])
            max_tr = transfer_constraint["max_transfers"]
            # Number of selected players NOT in current roster <= max_transfers
            non_curr_indicator = np.array([1.0 if p["player_name"] not in curr_names else 0.0 for p in filtered_players])
            row = np.zeros(num_vars)
            row[:n] = non_curr_indicator
            row[n:2*n] = non_curr_indicator
            A_ub_rows.append(row)
            b_ub_lower.append(0.0)
            b_ub_upper.append(float(max_tr))
            
        # Define bounds and integrality
        bounds_lower = np.zeros(num_vars)
        bounds_upper = np.ones(num_vars)
        bounds_upper[2*n] = np.inf  # u can grow to infinity
        bounds = Bounds(lb=bounds_lower, ub=bounds_upper)
        
        integrality = np.ones(num_vars)
        integrality[2*n] = 0.0  # u is continuous
        
        # Combine constraints into lists of LinearConstraint
        A_ub = np.vstack(A_ub_rows)
        A_eq = np.vstack(A_eq_rows)
        
        b_ub_lower = np.array(b_ub_lower)
        b_ub_upper = np.array(b_ub_upper)
        b_eq = np.array(b_eq)
        
        constraints = [
            LinearConstraint(A_ub, b_ub_lower, b_ub_upper),
            LinearConstraint(A_eq, b_eq, b_eq)
        ]
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            score = -res.fun  # Negate because we minimized the negated points
            if score > best_overall_score:
                best_overall_score = score
                best_result = res
                best_igl_idx = k
                
    if best_result is None:
        logger.warning("Roster Optimization MILP was unable to find a feasible solution.")
        return {"optimal_roster": [], "total_cost": 0, "projected_points": 0.0, "solver_status": "infeasible"}
        
    # Extract selected player indices
    selected_indices = []
    selected_as_wildcard = {}
    
    for i in range(n):
        nat_val = best_result.x[i]
        wild_val = best_result.x[n + i]
        if nat_val > 0.5 or wild_val > 0.5:
            selected_indices.append(i)
            selected_as_wildcard[i] = (wild_val > 0.5)
            
    optimal_roster = []
    for idx in selected_indices:
        p = filtered_players[idx]
        is_wild = selected_as_wildcard[idx]
        optimal_roster.append({
            "player_name": p["player_name"],
            "vlr_team_id": p["vlr_team_id"],
            "role": p["role"],
            "price": p["price"],
            "ppg": p["computed_ppg"],
            "sigma": p["computed_sigma"],
            "floor": p["floor"],
            "is_wildcard": is_wild,
            "is_igl": (idx == best_igl_idx)
        })
        
    total_cost = sum(p["price"] for p in optimal_roster)
    
    # Calculate exact reward sum (applying 2x multiplier for IGL)
    projected_points = 0.0
    for p in optimal_roster:
        pts = p["ppg"]
        if p["is_igl"]:
            pts *= 2.0
        projected_points += pts
        
    # Apply soft cost penalty to final score
    if total_cost > 48:
        projected_points -= 0.5 * (total_cost - 48)
        
    # Check if head-to-head occurred
    h2h_occurred = []
    for (i, j) in h2h_pairs:
        if i in selected_indices and j in selected_indices:
            h2h_occurred.append((filtered_players[i]["player_name"], filtered_players[j]["player_name"]))
            projected_points -= 20.0
            
    optimal_roster.sort(key=lambda p: (not p["is_igl"], p["price"]))
    
    return {
        "optimal_roster": optimal_roster,
        "total_cost": total_cost,
        "projected_points": round(projected_points, 2),
        "igl_player": optimal_roster[0]["player_name"] if optimal_roster else "None",
        "h2h_penalties": h2h_occurred,
        "solver_status": "optimal"
    }


def generate_stage_2_baseline(vfl_players: list[dict]) -> dict:
    """Calculates the optimal baseline starting roster for VCT 2026 Stage 2 Kickoff."""
    logger.info("Generating optimal baseline roster for Stage 2 kickoff...")
    return optimize_roster(
        vfl_players=vfl_players,
        salary_cap=50,
        survival_threshold=0.35
    )


def suggest_transfers(current_roster: list[dict], vfl_players: list[dict]) -> dict:
    """
    Transfer Advisor component. Capped at exactly 3 swaps.
    Runs optimization with transfer limit <= 3 and computes best transactions.
    """
    logger.info("Running Transfer Advisor with a hard cap of 3 trades...")
    
    # Solve optimization with transfer constraint
    transfer_constraint = {
        "current_roster": current_roster,
        "max_transfers": 3
    }
    
    result = optimize_roster(
        vfl_players=vfl_players,
        salary_cap=50,
        survival_threshold=0.35,
        transfer_constraint=transfer_constraint
    )
    
    if result["solver_status"] != "optimal":
        return {
            "transfers_in": [],
            "transfers_out": [],
            "projected_gain": 0.0,
            "new_roster": current_roster,
            "solver_status": result["solver_status"]
        }
        
    new_roster = result["optimal_roster"]
    
    curr_names = set(p["player_name"] for p in current_roster)
    new_names = set(p["player_name"] for p in new_roster)
    
    transfers_out = [p for p in current_roster if p["player_name"] not in new_names]
    transfers_in = [p for p in new_roster if p["player_name"] not in curr_names]
    
    # Calculate projected gain (difference in points)
    # Re-calculate current points including IGL multiplier
    # Identify current IGL (player in current roster with highest floor)
    current_roster_enriched = []
    player_stats = compute_all_players_historical_stats(RAW_DIR)
    
    for p in current_roster:
        stats = player_stats.get(p["player_name"], {"ppg": p.get("ppg", 10.0), "sigma": 3.0})
        p_enriched = p.copy()
        p_enriched["ppg"] = stats.get("ppg", p.get("ppg", 10.0))
        p_enriched["sigma"] = stats.get("sigma", 3.0)
        p_enriched["floor"] = p_enriched["ppg"] - 1.0 * p_enriched["sigma"]
        current_roster_enriched.append(p_enriched)
        
    current_roster_enriched.sort(key=lambda x: x["floor"], reverse=True)
    
    current_points = 0.0
    for i, p in enumerate(current_roster_enriched):
        pts = p["ppg"]
        if i == 0:  # Highest floor is IGL
            pts *= 2.0
        current_points += pts
        
    current_cost = sum(p["price"] for p in current_roster_enriched)
    if current_cost > 48:
        current_points -= 0.5 * (current_cost - 48)
        
    projected_gain = result["projected_points"] - current_points
    
    return {
        "transfers_in": transfers_in,
        "transfers_out": transfers_out,
        "projected_gain": round(max(projected_gain, 0.0), 2),
        "new_roster": new_roster,
        "new_total_cost": result["total_cost"],
        "new_projected_points": result["projected_points"],
        "solver_status": "optimal"
    }


if __name__ == "__main__":
    from vfl_scraper import VFLScraper
    scraper = VFLScraper()
    players = scraper.get_players()
    
    # Test Stage 2 baseline optimal roster
    print("\n" + "=" * 60)
    print("ROSTER OPTIMIZER TEST: VCT 2026 STAGE 2 BASELINE")
    print("=" * 60)
    
    res = generate_stage_2_baseline(players)
    print(f"Solver Status: {res['solver_status']}")
    print(f"Total Cost: {res['total_cost']} / 50 VP")
    print(f"Projected Points: {res['projected_points']} pts")
    print(f"IGL Player: {res['igl_player']}")
    print("\nOptimal Roster:")
    for idx, p in enumerate(res["optimal_roster"]):
        igl_tag = " (IGL - 2x Multiplier)" if p["is_igl"] else ""
        wc_tag = " (Wildcard)" if p["is_wildcard"] else f" ({p['role']})"
        print(f"  #{idx+1} {p['player_name']:<12} | Team ID: {p['vlr_team_id']:<5} | Cost: {p['price']:>2} VP | PPG: {p['ppg']:>4.1f} | Floor: {p['floor']:>4.1f}{wc_tag}{igl_tag}")
    
    # Test suggest transfers
    print("\n" + "=" * 60)
    print("TRANSFER ADVISOR TEST")
    print("=" * 60)
    
    # Sample current roster from the optimal lineup with 3 swaps made manually
    curr = [
        {"player_name": "something", "price": 10},
        {"player_name": "mindfreak", "price": 8},
        {"player_name": "C0M", "price": 8},
        {"player_name": "zekken", "price": 10},
        {"player_name": "Boo", "price": 8},
        {"player_name": "Boaster", "price": 8}
    ]
    
    transfer_res = suggest_transfers(curr, players)
    print(f"Projected Gain: +{transfer_res['projected_gain']} points")
    print("\nTransfers Out:")
    for p in transfer_res["transfers_out"]:
        print(f"  - {p['player_name']}")
    print("\nTransfers In:")
    for p in transfer_res["transfers_in"]:
        print(f"  - {p['player_name']}")
