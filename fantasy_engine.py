import os
import json
import glob
import logging
from collections import defaultdict

logger = logging.getLogger("fantasy_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class VCTFantasyEngine:
    def __init__(self):
        pass
        
    def calculate_kills_points(self, kills: int) -> int:
        """Kills Metrics: Piecewise penalties/rewards."""
        if kills == 0:
            return -3
        elif 1 <= kills <= 4:
            return -1
        elif 5 <= kills <= 9:
            return 0
        else: # kills >= 10
            return 1 + (kills - 10) // 5

    def calculate_multikill_points(self, k4: int, k5: int, k6: int, k7: int) -> float:
        """Multi-Kills Vector: 4K (+1), 5K (+3), 6K (+5), and 7K (+10) points."""
        return k4 * 1.0 + k5 * 3.0 + k6 * 5.0 + k7 * 10.0

    def calculate_round_delta_points(self, team_score: int, opp_score: int) -> int:
        """Map Out-of-Bounds Metrics: Round difference points."""
        if team_score == 13 and opp_score == 0:
            return 5
        elif team_score == 0 and opp_score == 13:
            return -5
        
        diff = team_score - opp_score
        if team_score > opp_score and diff >= 10:
            return 2
        elif team_score > opp_score and 5 <= diff <= 9:
            return 1
        elif team_score < opp_score and (opp_score - team_score) >= 10:
            return -1
            
        return 0

    def calculate_series_bonus(self, player_team: str, team_a: str, team_b: str, score_a: int, score_b: int) -> int:
        """Series Scale Modifiers: 2-0 (+2), 3-0 (+4), and 3-1 (+1) series bonuses."""
        # Normalize team names to match lower
        pt = player_team.lower().strip()
        ta = team_a.lower().strip()
        tb = team_b.lower().strip()
        
        is_team_a = pt in ta or ta in pt
        is_team_b = pt in tb or tb in pt
        
        if score_a > score_b:
            # Team A won
            if not is_team_a:
                return 0
            if score_a == 2 and score_b == 0:
                return 2
            elif score_a == 3 and score_b == 0:
                return 4
            elif score_a == 3 and score_b == 1:
                return 1
        elif score_b > score_a:
            # Team B won
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
            logger.error(f"Invalid match JSON format in {match_filepath}")
            return []
            
        segment = data["data"]["segments"][0]
        team_a_name = segment["teams"][0]["name"]
        team_b_name = segment["teams"][1]["name"]
        
        try:
            score_a = int(segment["teams"][0]["score"] or 0)
            score_b = int(segment["teams"][1]["score"] or 0)
        except ValueError:
            score_a, score_b = 0, 0
            
        # Compile player statistics per map
        # Structure: player_name -> { "team": team_name, "map_scores": {map_name: float}, "ratings": list[float] }
        players_data = defaultdict(lambda: {"team": "", "map_scores": {}, "ratings": [], "kills": []})
        
        for map_data in segment.get("maps", []):
            map_name = map_data.get("map_name")
            if not map_name or map_name.lower() in ["all maps", "none"]:
                continue
                
            # Get map score deltas
            m_score = map_data.get("score", {})
            m_score_t1 = m_score.get("team1", 0)
            m_score_t2 = m_score.get("team2", 0)
            
            # Map out-of-bounds metrics for both sides
            delta_pts_team1 = self.calculate_round_delta_points(m_score_t1, m_score_t2)
            delta_pts_team2 = self.calculate_round_delta_points(m_score_t2, m_score_t1)
            
            # Parse advanced stats for multi-kills on this map
            perf = map_data.get("performance", {})
            adv_stats = perf.get("advanced_stats", [])
            
            # Create a lookup for player performance row
            adv_lookup = {}
            for row in adv_stats:
                p_label = row.get("player", "")
                adv_lookup[p_label.lower().strip()] = row
                
            # Process team1 players
            for p in map_data.get("players", {}).get("team1", []):
                p_name = p["name"]
                players_data[p_name]["team"] = team_a_name
                
                kills = int(p.get("kills") or 0)
                players_data[p_name]["kills"].append(kills)
                rating = float(p.get("rating") or 0.0)
                players_data[p_name]["ratings"].append(rating)
                
                # Kills points
                kills_pts = self.calculate_kills_points(kills)
                
                # Multi-kill points
                k4, k5, k6, k7 = 0, 0, 0, 0
                # Match advanced stats row
                matched_row = None
                for key_label, row in adv_lookup.items():
                    if p_name.lower().strip() in key_label:
                        matched_row = row
                        break
                        
                if matched_row:
                    k4 = int(matched_row.get("4") or 0)
                    k5 = int(matched_row.get("10") or 0) # Key '10' is 5K
                    # 6K/7K default to 0 as they aren't tracked
                    
                multikill_pts = self.calculate_multikill_points(k4, k5, k6, k7)
                
                # Map-level fantasy points
                map_score = kills_pts + multikill_pts + delta_pts_team1
                players_data[p_name]["map_scores"][map_name] = map_score
                
            # Process team2 players
            for p in map_data.get("players", {}).get("team2", []):
                p_name = p["name"]
                players_data[p_name]["team"] = team_b_name
                
                kills = int(p.get("kills") or 0)
                players_data[p_name]["kills"].append(kills)
                rating = float(p.get("rating") or 0.0)
                players_data[p_name]["ratings"].append(rating)
                
                # Kills points
                kills_pts = self.calculate_kills_points(kills)
                
                # Multi-kill points
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
                
                # Map-level fantasy points
                map_score = kills_pts + multikill_pts + delta_pts_team2
                players_data[p_name]["map_scores"][map_name] = map_score
                
        # Perform aggregate calculations for each player
        leaderboard = []
        
        # We need to sort players' average ratings to assign top 3 placements
        all_players_ratings = []
        for name, p_data in players_data.items():
            if p_data["ratings"]:
                avg_rating = sum(p_data["ratings"]) / len(p_data["ratings"])
            else:
                avg_rating = 0.0
            all_players_ratings.append((name, avg_rating))
            
        # Sort by rating descending
        all_players_ratings.sort(key=lambda x: x[1], reverse=True)
        placements = {}
        for rank, (name, _) in enumerate(all_players_ratings):
            if rank == 0:
                placements[name] = 3 # Top 1 gets +3
            elif rank == 1:
                placements[name] = 2 # Top 2 gets +2
            elif rank == 2:
                placements[name] = 1 # Top 3 gets +1
            else:
                placements[name] = 0
                
        for name, p_data in players_data.items():
            map_scores_list = list(p_data["map_scores"].values())
            # Cap score at top 2 maps
            sorted_map_scores = sorted(map_scores_list)
            top_2_scores = sorted_map_scores[-2:] if len(sorted_map_scores) >= 2 else sorted_map_scores
            map_score_agg = sum(top_2_scores)
            
            # Series Win Bonus
            series_bonus = self.calculate_series_bonus(p_data["team"], team_a_name, team_b_name, score_a, score_b)
            
            # VLR Rating placement bonus
            rating_placement_bonus = placements.get(name, 0)
            
            # VLR Rating absolute scaling modifier
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
            
        # Sort leaderboard by total fantasy score descending
        leaderboard.sort(key=lambda x: x["total_score"], reverse=True)
        return leaderboard

if __name__ == "__main__":
    engine = VCTFantasyEngine()
    
    # Test scoring on Grand Final match
    match_file = r"c:\Users\91704\Desktop\vct-prediction-model\data\raw\match_670471.json"
    if os.path.exists(match_file):
        print("Scoring Grand Final Match (PRX vs LEV, Match ID 670471)...")
        leaderboard = engine.score_match_json(match_file)
        
        print(f"\nVCT Fantasy League Leaderboard (Total Players scored: {len(leaderboard)}):")
        print(f"{'Rank':<5} | {'Player':<15} | {'Team':<12} | {'Rating':<6} | {'Map Agg':<7} | {'Series':<6} | {'Rating Pl':<9} | {'Rating Sc':<9} | {'Total':<5}")
        print("-" * 90)
        for rank, p in enumerate(leaderboard):
            print(f"{rank+1:<5} | {p['player']:<15} | {p['team']:<12} | {p['avg_rating']:<6} | {p['map_score_agg']:<7} | {p['series_bonus']:<6} | {p['rating_placement_bonus']:<9} | {p['rating_scaling_bonus']:<9} | {p['total_score']:<5}")
    else:
        print(f"Match file {match_file} not found for testing.")
