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
        """Loads a raw match JSON and computes fantasy leaderboard scores for all players. Supports both legacy nested schema and flat segment schema."""
        with open(match_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "data" not in data or "segments" not in data["data"] or not data["data"]["segments"]:
            return []
            
        first_seg = data["data"]["segments"][0]
        
        # Check if legacy nested schema (has "teams" list inside first segment)
        if "teams" in first_seg and isinstance(first_seg["teams"], list) and len(first_seg["teams"]) >= 2:
            team_a_name = first_seg["teams"][0]["name"]
            team_b_name = first_seg["teams"][1]["name"]
            
            try:
                score_a = int(first_seg["teams"][0]["score"] or 0)
                score_b = int(first_seg["teams"][1]["score"] or 0)
            except ValueError:
                score_a, score_b = 0, 0
                
            players_data = defaultdict(lambda: {"team": "", "opponent": "", "map_scores": {}, "ratings": [], "kills": []})
            
            for map_data in first_seg.get("maps", []):
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
                    players_data[p_name]["opponent"] = team_b_name
                    
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
                    players_data[p_name]["opponent"] = team_a_name
                    
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

        else:
            # Current flat schema: each segment in data["data"]["segments"] represents one map
            segments = data["data"]["segments"]
            team_a_name = first_seg.get("team1", "Team 1")
            team_b_name = first_seg.get("team2", "Team 2")
            
            # Count maps won per team
            score_a, score_b = 0, 0
            for seg in segments:
                winner = seg.get("winner")
                if winner == team_a_name:
                    score_a += 1
                elif winner == team_b_name:
                    score_b += 1
                    
            players_data = defaultdict(lambda: {"team": "", "opponent": "", "map_scores": {}, "ratings": [], "kills": []})
            
            for map_idx, seg in enumerate(segments):
                map_name = seg.get("map", f"Map {map_idx + 1}")
                if not map_name or map_name.lower() in ["all maps", "none"]:
                    continue
                    
                t1 = seg.get("team1", team_a_name)
                t2 = seg.get("team2", team_b_name)
                
                # Count round scores
                round_hist = seg.get("round_history", [])
                rounds_t1 = sum(1 for r in round_hist if r.get("winner") == t1)
                rounds_t2 = sum(1 for r in round_hist if r.get("winner") == t2)
                
                delta_pts_t1 = self.calculate_round_margin_points(rounds_t1, rounds_t2)
                delta_pts_t2 = self.calculate_round_margin_points(rounds_t2, rounds_t1)
                
                adv_stats = seg.get("performance", {}).get("advanced_stats", [])
                adv_lookup = {}
                for row in adv_stats:
                    p_label = row.get("player", "")
                    adv_lookup[p_label.lower().strip()] = row
                    
                raw_players = seg.get("players", [])
                for p in raw_players:
                    p_name = p.get("name")
                    if not p_name:
                        continue
                    p_team = p.get("team", t1)
                    opp_team = t2 if p_team == t1 else t1
                    
                    players_data[p_name]["team"] = p_team
                    players_data[p_name]["opponent"] = opp_team
                    
                    kills = int(p.get("kills") or 0)
                    players_data[p_name]["kills"].append(kills)
                    rating = float(p.get("rating") or 0.0)
                    players_data[p_name]["ratings"].append(rating)
                    
                    kills_pts = self.calculate_kills_points(kills)
                    
                    k4, k5 = 0, 0
                    matched_row = None
                    for key_label, row in adv_lookup.items():
                        if p_name.lower().strip() in key_label:
                            matched_row = row
                            break
                            
                    if matched_row:
                        k4 = int(matched_row.get("4") or 0)
                        k5 = int(matched_row.get("10") or 0)
                        
                    multikill_pts = self.calculate_multikill_points(k4, k5, 0, 0)
                    delta_pts = delta_pts_t1 if p_team == t1 else delta_pts_t2
                    
                    map_score = kills_pts + multikill_pts + delta_pts
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
                "opponent_team": p_data.get("opponent", ""),
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
            scores_arr = np.array(scores)
            var_90 = float(np.percentile(scores_arr, 90))
            cvar_90 = float(np.mean(scores_arr[scores_arr >= var_90]))
            var_10 = float(np.percentile(scores_arr, 10))
            cvar_10 = float(np.mean(scores_arr[scores_arr <= var_10]))
            
            stats[p_name] = {
                "ppg": float(np.mean(scores)),
                "sigma": float(np.std(scores)) if len(scores) > 1 else 3.0,
                "cvar_90": cvar_90,
                "cvar_10": cvar_10,
                "matches_played": len(scores)
            }
    return stats


def compute_all_players_opponent_stats(raw_dir: str = RAW_DIR) -> dict[str, dict[str, dict]]:
    """
    Computes per-opponent historical performance for each player across database matches.
    Returns: Dict[player_name, Dict[opponent_team_name, {ppg, sigma, cvar_90, cvar_10, n_maps}]]
    """
    logger.info("Computing opponent-conditioned player statistics...")
    engine = VCTFantasyEngine()
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    
    # Key: (player_name, opponent_team_name) -> list of scores
    h2h_scores = defaultdict(list)
    
    for f in files:
        try:
            leaderboard = engine.score_match_json(f)
            for entry in leaderboard:
                p_name = entry["player"]
                opp_team = entry.get("opponent_team")
                score = entry.get("total_score")
                if p_name and opp_team and score is not None:
                    h2h_scores[(p_name, opp_team)].append(score)
        except Exception:
            pass
            
    opponent_stats = defaultdict(dict)
    for (p_name, opp_team), scores in h2h_scores.items():
        if scores:
            scores_arr = np.array(scores)
            var_90 = float(np.percentile(scores_arr, 90))
            cvar_90 = float(np.mean(scores_arr[scores_arr >= var_90]))
            var_10 = float(np.percentile(scores_arr, 10))
            cvar_10 = float(np.mean(scores_arr[scores_arr <= var_10]))
            
            opponent_stats[p_name][opp_team] = {
                "ppg": float(np.mean(scores)),
                "sigma": float(np.std(scores)) if len(scores) > 1 else 3.0,
                "cvar_90": cvar_90,
                "cvar_10": cvar_10,
                "n_maps": len(scores)
            }
            
    return dict(opponent_stats)


def blend_ev(global_stats: dict, h2h_stats: dict, opponent: str, player_name: str) -> dict:
    """
    Blends a player's opponent-conditioned H2H stats with their global average stats.
    Minimum threshold: 3 maps against the opponent to activate H2H weighting.
    Weighting formula: ramps from 30% at 3 maps to max 70% at 10+ maps.
    """
    g_player = global_stats.get(player_name, {}) if global_stats else {}
    global_ppg = float(g_player.get("ppg", 10.0))
    global_sigma = float(g_player.get("sigma", 3.0))
    
    if not h2h_stats or player_name not in h2h_stats:
        return {
            "ppg": global_ppg,
            "sigma": global_sigma,
            "h2h_used": False,
            "n_maps": 0,
            "global_ppg": global_ppg,
            "opponent": opponent
        }
        
    p_h2h = h2h_stats[player_name]
    
    # Matching opponent team name (exact or case-insensitive)
    matched_opp = None
    if opponent in p_h2h:
        matched_opp = opponent
    else:
        opp_clean = opponent.lower().strip()
        for k in p_h2h.keys():
            if k.lower().strip() == opp_clean:
                matched_opp = k
                break
                
    if not matched_opp:
        return {
            "ppg": global_ppg,
            "sigma": global_sigma,
            "h2h_used": False,
            "n_maps": 0,
            "global_ppg": global_ppg,
            "opponent": opponent
        }
        
    opp_data = p_h2h[matched_opp]
    n_maps = int(opp_data.get("n_maps", 0))
    
    if n_maps < 3:
        return {
            "ppg": global_ppg,
            "sigma": global_sigma,
            "h2h_used": False,
            "n_maps": n_maps,
            "global_ppg": global_ppg,
            "opponent": opponent
        }
        
    h2h_ppg = float(opp_data.get("ppg", global_ppg))
    h2h_sigma = float(opp_data.get("sigma", global_sigma))
    
    # Blend: min 0.3 at n=3, max 0.7 at n=10+
    weight_h2h = min(n_maps / 10.0, 0.7)
    weight_global = 1.0 - weight_h2h
    
    blended_ppg = weight_h2h * h2h_ppg + weight_global * global_ppg
    blended_sigma = weight_h2h * h2h_sigma + weight_global * global_sigma
    
    return {
        "ppg": round(blended_ppg, 2),
        "sigma": round(blended_sigma, 2),
        "h2h_used": True,
        "n_maps": n_maps,
        "h2h_ppg": round(h2h_ppg, 2),
        "global_ppg": round(global_ppg, 2),
        "opponent": opponent
    }

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
    transfer_constraint: dict = None,  # Contains 'current_roster' and 'max_transfers'
    forced_igl_name: str = None,
    excluded_rosters: list[list[str]] = None,
    active_team_pool: set = None,
    matchup_pairs: list[tuple[str, str]] = None,
    h2h_stats: dict = None
) -> dict:
    """
    Solves VFL roster selection as a Mixed-Integer Linear Program (MILP).
    Enforces Phase 18 strict constraints:
      - Budget <= salary_cap (50 VP). Soft penalty for cost > 48 VP.
      - Roster length = exactly roster_size (6).
      - Max max_per_team (2) from same team.
      - Exactly 1 Duelist, 1 Initiator, 1 Controller, 1 Sentinel, and 2 Wildcard slots.
      - Head-to-head match penalty.
      - Survival threshold filtering.
    Dynamically identifies IGL (highest floor = PPG - 1.0 * sigma) and attaches 2x multiplier.
    Supports Phase 21 Opponent-Conditioned H2H Blended EV.
    """
    if upcoming_matchups is None:
        upcoming_matchups = get_upcoming_matchups(RAW_DIR)
    if team_win_rates is None:
        team_win_rates, _ = get_team_win_rates_by_id(RAW_DIR)
    if player_stats is None:
        player_stats = compute_all_players_historical_stats(RAW_DIR)

    # 1. Filter out players below the survival win rate threshold, unless in current roster
    curr_names_set = set()
    if transfer_constraint is not None:
        for p in transfer_constraint["current_roster"]:
            name_val = p.get("player_name") or p.get("name")
            if name_val:
                curr_names_set.add(name_val.lower().strip())

    filtered_players = []
    for p in vfl_players:
        p_norm = dict(p)
        pname = p_norm.get("player_name") or p_norm.get("name") or ""
        p_norm["player_name"] = pname
        p_norm["name"] = pname
        if "price" not in p_norm:
            p_norm["price"] = p_norm.get("salary") or p_norm.get("cost") or 8.0
            
        team_val = p_norm.get("vlr_team_id") or p_norm.get("team_name") or p_norm.get("team")
        p_norm["vlr_team_id"] = team_val
        p_norm["team_name"] = team_val
        p_norm["team"] = team_val
        
        is_in_curr = pname.lower().strip() in curr_names_set
        
        tid = p_norm.get("vlr_team_id")
        wr = team_win_rates.get(tid, 0.50) if isinstance(tid, int) else 0.50
        if wr >= survival_threshold or is_in_curr:
            # Check if team is inactive in upcoming team pool
            p_team_name = str(p_norm.get("team_name") or p_norm.get("team") or "").strip()
            if p_team_name.isdigit() and p_norm.get("team_name"):
                p_team_name = str(p_norm.get("team_name")).strip()
            p_team_short = str(p_norm.get("team_short") or "").strip()
            p_vlr_id = str(p_norm.get("vlr_team_id") or "")

            is_active = True
            if active_team_pool is not None and len(active_team_pool) > 0:
                is_active = False
                for at in active_team_pool:
                    at_clean = str(at).lower().strip()
                    if not at_clean:
                        continue
                    if (p_team_name and (p_team_name.lower() in at_clean or at_clean in p_team_name.lower())) or \
                       (p_team_short and p_team_short.lower() == at_clean) or \
                       (p_vlr_id and p_vlr_id == str(at)):
                        is_active = True
                        break

            # Preserve caller's precomputed points if provided (e.g. historical actuals in gw_actual_pool)
            has_precomputed = ("computed_ppg" in p or "ppg" in p) and not h2h_stats

            if not is_active and not has_precomputed:
                p_norm["computed_ppg"] = 0.0
                p_norm["computed_sigma"] = 0.0
                p_norm["cvar_90"] = 0.0
                p_norm["cvar_10"] = 0.0
                p_norm["floor"] = 0.0
            else:
                stats = player_stats.get(pname, {})
                if has_precomputed:
                    pre_pts = float(p_norm.get("computed_ppg") if p_norm.get("computed_ppg") is not None else p_norm.get("ppg", 0.0))
                    p_norm["computed_ppg"] = pre_pts
                    p_norm["global_ppg"] = pre_pts
                    p_norm["computed_sigma"] = 0.0
                    p_norm["h2h_used"] = False
                    p_norm["opponent"] = None
                else:
                    global_ppg = float(p_norm.get("computed_ppg") or p_norm.get("EV") or p_norm.get("ppg") or stats.get("ppg", 10.0))
                    global_sigma = float(stats.get("sigma", 3.0))
                    
                    # Resolve opponent from matchup_pairs
                    opponent = None
                    if matchup_pairs:
                        for (t_a, t_b) in matchup_pairs:
                            t_a_c = str(t_a).lower().strip()
                            t_b_c = str(t_b).lower().strip()
                            p_t_c = p_team_name.lower().strip()
                            if p_t_c and (p_t_c in t_a_c or t_a_c in p_t_c):
                                opponent = t_b
                                break
                            elif p_t_c and (p_t_c in t_b_c or t_b_c in p_t_c):
                                opponent = t_a
                                break
                                
                    if opponent and h2h_stats:
                        blended = blend_ev(player_stats, h2h_stats, opponent, pname)
                        p_norm["computed_ppg"] = blended["ppg"]
                        p_norm["computed_sigma"] = blended["sigma"]
                        p_norm["h2h_used"] = blended["h2h_used"]
                        p_norm["opponent"] = opponent
                        p_norm["global_ppg"] = blended.get("global_ppg", global_ppg)
                    else:
                        p_norm["computed_ppg"] = global_ppg
                        p_norm["computed_sigma"] = global_sigma
                        p_norm["h2h_used"] = False
                        p_norm["opponent"] = opponent
                        p_norm["global_ppg"] = global_ppg
                
                # Use continuous CVaR 90 (expected ceiling) and CVaR 10 (expected worst-case floor)
                p_norm["cvar_90"] = stats.get("cvar_90", p_norm["computed_ppg"] + 1.5 * p_norm["computed_sigma"])
                p_norm["cvar_10"] = stats.get("cvar_10", p_norm["computed_ppg"] - 1.5 * p_norm["computed_sigma"])
                p_norm["floor"] = p_norm["cvar_10"]
            filtered_players.append(p_norm)
            
    n = len(filtered_players)
    if n == 0:
        return {"optimal_roster": [], "total_cost": 0, "projected_points": 0.0, "solver_status": "no_feasible_players"}

    # Formulate head-to-head matchup pairs
    h2h_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            team_a = str(filtered_players[i].get("team") or filtered_players[i].get("team_name") or "").lower().strip()
            team_b = str(filtered_players[j].get("team") or filtered_players[j].get("team_name") or "").lower().strip()
            tid_a = filtered_players[i].get("vlr_team_id")
            tid_b = filtered_players[j].get("vlr_team_id")
            
            is_opponents = False
            # Check string matchup_pairs (from live schedule API / bracket predictor)
            if matchup_pairs and team_a and team_b and team_a != team_b:
                for (m1, m2) in matchup_pairs:
                    m1_clean = str(m1).lower().strip()
                    m2_clean = str(m2).lower().strip()
                    if (team_a in m1_clean and team_b in m2_clean) or (team_a in m2_clean and team_b in m1_clean) or (m1_clean in team_a and m2_clean in team_b) or (m2_clean in team_a and m1_clean in team_b):
                        is_opponents = True
                        break
                        
            # Check integer upcoming_matchups if not matched
            if not is_opponents and tid_a is not None and tid_b is not None and upcoming_matchups:
                for match in upcoming_matchups:
                    if (tid_a == match[0] and tid_b == match[1]) or (tid_a == match[1] and tid_b == match[0]):
                        is_opponents = True
                        break
                        
            if is_opponents:
                h2h_pairs.append((i, j))
                        
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
    
    # Identify forced IGL index if any
    forced_idx = None
    if forced_igl_name:
        matched = [i for i, p in enumerate(filtered_players) if p["player_name"].lower().strip() == forced_igl_name.lower().strip()]
        if matched:
            forced_idx = matched[0]

    # Loop over all players as potential IGL candidates
    for k in range(n):
        if forced_idx is not None and k != forced_idx:
            continue
            
        is_forced_igl_run = (forced_idx is not None and k == forced_idx)
        is_other_igl_run = False
        
        # Candidate IGL floor
        igl_floor = filtered_players[k]["floor"]
        
        # Objective coefficient vector (we minimize, so we negate points)
        # points = computed_ppg (Expected VFL Points given weekly matchups). Player k points are doubled (IGL).
        pts = np.array([p.get("computed_ppg", p.get("EV", p.get("ppg", 10.0))) for p in filtered_players], dtype=float)
        
        c = np.zeros(num_vars)
        c[:n] = -pts          # x_nat coefficients
        c[n:2*n] = -pts       # x_wild coefficients
        c[k] -= pts[k]        # Add extra -pts[k] for natural selection IGL (since x_k_nat + x_k_wild = 1)
        c[n + k] -= pts[k]    # Add extra -pts[k] for wildcard selection IGL
        
        # No soft penalty - VFL ruleset strictly capped at 50 VP
        c[2*n] = 0.0          
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
        if is_forced_igl_run:
            # Candidate k (the forced IGL) must be selected
            row = np.zeros(num_vars)
            row[k] = 1.0
            row[n + k] = 1.0
            A_eq_rows.append(row)
            b_eq.append(1.0)
        elif is_other_igl_run:
            # Candidate k is the selected IGL (must be in the roster)
            row = np.zeros(num_vars)
            row[k] = 1.0
            row[n + k] = 1.0
            A_eq_rows.append(row)
            b_eq.append(1.0)
            
            # The forced IGL player (forced_idx) must be NOT selected (swapped out)
            row = np.zeros(num_vars)
            row[forced_idx] = 1.0
            row[n + forced_idx] = 1.0
            A_eq_rows.append(row)
            b_eq.append(0.0)
        else:
            # Normal run: candidate k must be selected
            row = np.zeros(num_vars)
            row[k] = 1.0
            row[n + k] = 1.0
            A_eq_rows.append(row)
            b_eq.append(1.0)
        
        # 3. Role natural counts based on active VFL ruleset
        min_role_count = 2.0 if roster_size == 11 else 1.0
        wildcard_count = 3.0 if roster_size == 11 else 2.0
        
        # sum_{i in Duelist} x_i_nat = min_role_count
        row = np.zeros(num_vars)
        row[:n] = is_duelist
        A_eq_rows.append(row)
        b_eq.append(min_role_count)
        
        # sum_{i in Initiator} x_i_nat = min_role_count
        row = np.zeros(num_vars)
        row[:n] = is_initiator
        A_eq_rows.append(row)
        b_eq.append(min_role_count)
        
        # sum_{i in Controller} x_i_nat = min_role_count
        row = np.zeros(num_vars)
        row[:n] = is_controller
        A_eq_rows.append(row)
        b_eq.append(min_role_count)
        
        # sum_{i in Sentinel} x_i_nat = min_role_count
        row = np.zeros(num_vars)
        row[:n] = is_sentinel
        A_eq_rows.append(row)
        b_eq.append(min_role_count)
        
        # sum_i x_i_wild = wildcard_count
        row = np.zeros(num_vars)
        row[n:2*n] = 1.0
        A_eq_rows.append(row)
        b_eq.append(wildcard_count)
        
        # 5. Hard Budget Constraint: sum(x_i * cost_i) <= 50
        row = np.zeros(num_vars)
        row[:n] = costs
        row[n:2*n] = costs
        A_ub_rows.append(row)
        b_ub_lower.append(0.0)
        b_ub_upper.append(float(salary_cap))
        
        # 6. Soft budget cap constraint: u >= sum(x_i * cost_i) - (salary_cap - 2)
        # => sum(x_i * cost_i) - u <= salary_cap - 2
        row = np.zeros(num_vars)
        row[:n] = costs
        row[n:2*n] = costs
        row[2*n] = -1.0
        A_ub_rows.append(row)
        b_ub_lower.append(-np.inf)
        b_ub_upper.append(float(salary_cap - 2.0))
        
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
            curr_roster = transfer_constraint["current_roster"]
            curr_names = set()
            curr_core_names_by_role = {"Duelist": set(), "Initiator": set(), "Controller": set(), "Sentinel": set()}
            curr_wildcard_names = set()
            
            min_role_c = 2 if roster_size == 11 else 1
            role_assigned_counts = {"Duelist": 0, "Initiator": 0, "Controller": 0, "Sentinel": 0}
            
            for p in curr_roster:
                name_val = p.get("player_name") or p.get("name")
                if name_val:
                    p_name = name_val.lower().strip()
                    curr_names.add(p_name)
                    p_slot = p.get("roster_slot")
                    p_role = p.get("role")
                    
                    if p_slot in curr_core_names_by_role:
                        curr_core_names_by_role[p_slot].add(p_name)
                    elif p_slot == "Wildcard":
                        curr_wildcard_names.add(p_name)
                    else:
                        if p_role in curr_core_names_by_role and role_assigned_counts[p_role] < min_role_c:
                            curr_core_names_by_role[p_role].add(p_name)
                            role_assigned_counts[p_role] += 1
                        else:
                            curr_wildcard_names.add(p_name)

            max_tr = transfer_constraint["max_transfers"]
            exact_tr = transfer_constraint.get("exact", False)
            
            # Number of selected players NOT in current roster
            non_curr_indicator = np.array([1.0 if p["player_name"].lower().strip() not in curr_names else 0.0 for p in filtered_players])
            row = np.zeros(num_vars)
            row[:n] = non_curr_indicator
            row[n:2*n] = non_curr_indicator
            if exact_tr:
                A_eq_rows.append(row)
                b_eq.append(float(max_tr))
            else:
                A_ub_rows.append(row)
                b_ub_lower.append(0.0)
                b_ub_upper.append(float(max_tr))

            # Role-for-role slot consistency constraints for current roster members
            for i, p in enumerate(filtered_players):
                p_name = p["player_name"].lower().strip()
                p_role = p["role"]
                
                # If player is in a current core role slot, prevent them from taking a wildcard slot
                if p_name in curr_core_names_by_role.get(p_role, set()):
                    row = np.zeros(num_vars)
                    row[n + i] = 1.0  # x_i_wild
                    A_eq_rows.append(row)
                    b_eq.append(0.0)
                    
                # If player is in a current wildcard slot, prevent them from taking a natural core role slot
                elif p_name in curr_wildcard_names:
                    row = np.zeros(num_vars)
                    row[i] = 1.0  # x_i_nat
                    A_eq_rows.append(row)
                    b_eq.append(0.0)
            
        # 10. Roster exclusion constraints (to find alternative suggestions)
        if excluded_rosters is not None:
            for excl in excluded_rosters:
                excl_set = set(name.lower().strip() for name in excl)
                idxs = [i for i, p in enumerate(filtered_players) if p["player_name"].lower().strip() in excl_set]
                if len(idxs) == 6:
                    row = np.zeros(num_vars)
                    for idx in idxs:
                        row[idx] = 1.0
                        row[n + idx] = 1.0
                    A_ub_rows.append(row)
                    b_ub_lower.append(-np.inf)
                    b_ub_upper.append(5.0)
            
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
        p_dict = dict(p)
        p_dict.update({
            "player_name": p["player_name"],
            "name": p["player_name"],
            "team": p.get("team") or p.get("team_name") or p.get("vlr_team_id"),
            "vlr_team_id": p.get("vlr_team_id"),
            "role": p["role"],
            "price": p["price"],
            "ppg": p.get("computed_ppg", p.get("ppg", 0.0)),
            "computed_ppg": p.get("computed_ppg", p.get("ppg", 0.0)),
            "sigma": p.get("computed_sigma", 3.0),
            "floor": p.get("floor", 0.0),
            "is_wildcard": is_wild,
            "is_igl": (idx == best_igl_idx)
        })
        optimal_roster.append(p_dict)
        
    total_cost = sum(p["price"] for p in optimal_roster)
    
    # Calculate exact reward sum (applying 2x multiplier for IGL)
    projected_points = 0.0
    for p in optimal_roster:
        pts = p["ppg"]
        if p["is_igl"]:
            pts *= 2.0
        projected_points += pts
        
    # Soft cost penalty removed to enforce strict budget VFL limits
        
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


def suggest_transfers(
    current_roster: list[dict], 
    vfl_players: list[dict], 
    remaining_bank_balance: float = 0.0, 
    forced_igl_name: str = None,
    salary_cap: float = 50.0,
    roster_size: int = 6,
    active_team_pool: set = None,
    matchup_pairs: list[tuple[str, str]] = None,
    h2h_stats: dict = None
) -> dict:
    """
    Transfer Advisor component. Enforces VFL ruleset.
    Finds up to 3 distinct optimal transfer recommendations.
    Supports Phase 21 Opponent-Conditioned H2H Blended EV.
    """
    current_roster_costs = [p.get("price", p.get("cost", 0)) for p in current_roster]
    floating_bank = salary_cap - sum(current_roster_costs)
    
    logger.info(f"Running VFL Transfer Advisor: Floating bank: {floating_bank:.2f} VP | Cap: {salary_cap} VP | Size: {roster_size}")
    
    recommendations = []
    excluded_rosters = []
    
    # Generate up to 3 distinct recommendations
    for opt_idx in range(3):
        result = None
        
        # Try exact 3 transfers
        transfer_constraint = {
            "current_roster": current_roster,
            "max_transfers": 3,
            "exact": True
        }
        res = optimize_roster(
            vfl_players=vfl_players,
            salary_cap=salary_cap,
            roster_size=roster_size,
            survival_threshold=0.35,
            transfer_constraint=transfer_constraint,
            forced_igl_name=forced_igl_name,
            excluded_rosters=excluded_rosters,
            active_team_pool=active_team_pool,
            matchup_pairs=matchup_pairs,
            h2h_stats=h2h_stats
        )
        if res["solver_status"] == "optimal":
            result = res
            
        # Fallback to exact 2 transfers
        if result is None or result["solver_status"] != "optimal":
            logger.info(f"Option {opt_idx+1}: Exact 3 transfers infeasible or already found. Trying exact 2 transfers...")
            transfer_constraint["max_transfers"] = 2
            res = optimize_roster(
                vfl_players=vfl_players,
                salary_cap=salary_cap,
                roster_size=roster_size,
                survival_threshold=0.35,
                transfer_constraint=transfer_constraint,
                forced_igl_name=forced_igl_name,
                excluded_rosters=excluded_rosters,
                active_team_pool=active_team_pool,
                matchup_pairs=matchup_pairs,
                h2h_stats=h2h_stats
            )
            if res["solver_status"] == "optimal":
                result = res
                
        # Fallback to exact 1 transfer
        if result is None or result["solver_status"] != "optimal":
            logger.info(f"Option {opt_idx+1}: Exact 2 transfers infeasible. Trying exact 1 transfer...")
            transfer_constraint["max_transfers"] = 1
            res = optimize_roster(
                vfl_players=vfl_players,
                salary_cap=salary_cap,
                roster_size=roster_size,
                survival_threshold=0.35,
                transfer_constraint=transfer_constraint,
                forced_igl_name=forced_igl_name,
                excluded_rosters=excluded_rosters,
                active_team_pool=active_team_pool,
                matchup_pairs=matchup_pairs,
                h2h_stats=h2h_stats
            )
            if res["solver_status"] == "optimal":
                result = res
                
        # Fallback to <= 3 transfers
        if result is None or result["solver_status"] != "optimal":
            logger.info(f"Option {opt_idx+1}: Exact transfers infeasible. Falling back to <= 3 transfers...")
            transfer_constraint["exact"] = False
            transfer_constraint["max_transfers"] = 3
            res = optimize_roster(
                vfl_players=vfl_players,
                salary_cap=salary_cap,
                roster_size=roster_size,
                survival_threshold=0.35,
                transfer_constraint=transfer_constraint,
                forced_igl_name=forced_igl_name,
                excluded_rosters=excluded_rosters,
                active_team_pool=active_team_pool,
                matchup_pairs=matchup_pairs,
                h2h_stats=h2h_stats
            )
            if res["solver_status"] == "optimal":
                result = res
                
        if result is not None and result["solver_status"] == "optimal":
            new_roster = result["optimal_roster"]
            
            curr_names = set(p.get("player_name") or p.get("name") for p in current_roster)
            new_names = set(p.get("player_name") or p.get("name") for p in new_roster)
            
            transfers_out = [p for p in current_roster if (p.get("player_name") or p.get("name")) not in new_names]
            transfers_in = [p for p in new_roster if (p.get("player_name") or p.get("name")) not in curr_names]
            
            incoming_cost = sum(p.get("price", p.get("cost", 0)) for p in transfers_in)
            outgoing_cost = sum(p.get("price", p.get("cost", 0)) for p in transfers_out)
            
            # Strict liquidity condition check
            if incoming_cost > outgoing_cost + floating_bank:
                roster_names = [p.get("player_name") or p.get("name") for p in new_roster]
                excluded_rosters.append(roster_names)
                continue

            # Exclude this roster combination from subsequent option searches
            roster_names = [p.get("player_name") or p.get("name") for p in new_roster]
            excluded_rosters.append(roster_names)
            
            # Calculate projected gain (difference in points)
            current_roster_enriched = []
            player_stats = compute_all_players_historical_stats(RAW_DIR)
            
            for p in current_roster:
                p_name = p.get("player_name") or p.get("name")
                p_team = p.get("team") or p.get("team_name")
                stats = player_stats.get(p_name, {"ppg": p.get("ppg", 10.0), "sigma": 3.0})
                p_enriched = p.copy()
                p_enriched["player_name"] = p_name
                p_enriched["name"] = p_name
                
                # Check for opponent in matchup_pairs
                opponent = None
                if matchup_pairs:
                    for (t_a, t_b) in matchup_pairs:
                        if p_team and t_a and p_team.lower().strip() in t_a.lower().strip():
                            opponent = t_b
                            break
                        elif p_team and t_b and p_team.lower().strip() in t_b.lower().strip():
                            opponent = t_a
                            break
                            
                if opponent and h2h_stats:
                    blended = blend_ev(player_stats, h2h_stats, opponent, p_name)
                    p_enriched["ppg"] = blended["ppg"]
                    p_enriched["sigma"] = blended["sigma"]
                else:
                    p_enriched["ppg"] = stats.get("ppg", p.get("ppg", 10.0))
                    p_enriched["sigma"] = stats.get("sigma", 3.0)
                    
                p_enriched["floor"] = p_enriched["ppg"] - 1.0 * p_enriched["sigma"]
                current_roster_enriched.append(p_enriched)
                
            # Determine IGL index in current roster
            igl_index = 0
            if forced_igl_name:
                matched_curr = [idx for idx, p in enumerate(current_roster_enriched) if p["player_name"].lower().strip() == forced_igl_name.lower().strip()]
                if matched_curr:
                    igl_index = matched_curr[0]
                else:
                    current_roster_enriched.sort(key=lambda x: x["floor"], reverse=True)
                    igl_index = 0
            else:
                current_roster_enriched.sort(key=lambda x: x["floor"], reverse=True)
                igl_index = 0
            
            current_points = 0.0
            for idx, p in enumerate(current_roster_enriched):
                p_team = p.get("team") or p.get("team_name")
                if active_team_pool is not None and p_team not in active_team_pool:
                    pts = 0.0
                else:
                    pts = p.get("ppg", p.get("EV", 10.0))
                    if idx == igl_index:  # Active IGL is doubled
                        pts *= 2.0
                current_points += pts
                
            projected_gain = result["projected_points"] - current_points
            
            recommendations.append({
                "transfers_in": transfers_in,
                "transfers_out": transfers_out,
                "projected_gain": round(max(projected_gain, 0.0), 2),
                "new_roster": new_roster,
                "new_total_cost": result["total_cost"],
                "old_projected_points": round(current_points, 2),
                "new_projected_points": result["projected_points"]
            })
        else:
            break
            
    return {
        "recommendations": recommendations,
        "solver_status": "optimal" if recommendations else "infeasible"
    }


if __name__ == "__main__":
    from scrapers.vfl_scraper import VFLScraper
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
    
    # Sample current roster from the database
    curr = [
        {"player_name": "something", "price": 10, "role": "Duelist"},
        {"player_name": "Boo", "price": 8, "role": "Controller"},
        {"player_name": "Keiko", "price": 10, "role": "Controller"},
        {"player_name": "CHICHOO", "price": 8, "role": "Sentinel"},
        {"player_name": "invy", "price": 9, "role": "Initiator"},
        {"player_name": "d4v41", "price": 9, "role": "Initiator"}
    ]
    
    transfer_res = suggest_transfers(curr, players)
    print(f"Solver Status: {transfer_res['solver_status']}")
    if transfer_res["solver_status"] == "optimal":
        for idx, rec in enumerate(transfer_res["recommendations"]):
            print(f"\n--- Recommendation {idx+1} (+{rec['projected_gain']} pts) ---")
            print("Transfers Out:")
            for p in rec["transfers_out"]:
                print(f"  - {p['player_name']} (${p['price']} VP)")
            print("Transfers In:")
            for p in rec["transfers_in"]:
                print(f"  - {p['player_name']} (${p['price']} VP)")
