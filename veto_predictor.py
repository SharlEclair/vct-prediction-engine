import os
import json
import glob
import re
import logging
from collections import defaultdict

logger = logging.getLogger("veto_predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class VCTMapVetoPredictor:
    def __init__(self, raw_dir="./data/raw"):
        self.raw_dir = raw_dir
        # Structure: team_name -> map_name -> count
        self.bans = defaultdict(lambda: defaultdict(int))
        self.picks = defaultdict(lambda: defaultdict(int))
        self.plays = defaultdict(lambda: defaultdict(int))
        self.wins = defaultdict(lambda: defaultdict(int))
        self.total_matches = defaultdict(int)
        self.map_pool = set()
        self.team_names = set()
        
    def match_team(self, token_team: str, team_a: str, team_b: str) -> str:
        """Helper to match a token team name to team_a or team_b."""
        token_team = token_team.lower().strip()
        ta = team_a.lower().strip()
        tb = team_b.lower().strip()
        
        if token_team in ta or ta in token_team:
            return team_a
        if token_team in tb or tb in token_team:
            return team_b
            
        if len(token_team) >= 3:
            prefix = token_team[:3]
            if ta.startswith(prefix):
                return team_a
            if tb.startswith(prefix):
                return team_b
                
        # Initials matching e.g. "PRX" -> "Paper Rex"
        def get_initials(name: str) -> str:
            return "".join(word[0] for word in name.split() if word)
            
        ta_init = get_initials(ta).lower()
        tb_init = get_initials(tb).lower()
        if token_team == ta_init:
            return team_a
        if token_team == tb_init:
            return team_b
            
        return ""

    def fit(self):
        """Loads raw match files and compiles veto/map statistics."""
        files = glob.glob(os.path.join(self.raw_dir, "match_*.json"))
        logger.info(f"VetoPredictor: Loading {len(files)} matches to compile veto stats...")
        
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    content = json.load(file)
                if "data" not in content or "segments" not in content["data"] or not content["data"]["segments"]:
                    continue
                segment = content["data"]["segments"][0]
                team_a = segment["teams"][0]["name"]
                team_b = segment["teams"][1]["name"]
                self.team_names.add(team_a)
                self.team_names.add(team_b)
                self.total_matches[team_a] += 1
                self.total_matches[team_b] += 1
                
                # 1. Parse Map Vetoes
                map_vetos_str = segment.get("map_vetos", "")
                if map_vetos_str:
                    tokens = [t.strip() for t in map_vetos_str.split(";") if t.strip()]
                    for token in tokens:
                        # Parse Bans
                        if "ban" in token:
                            parts = token.split("ban")
                            if len(parts) == 2:
                                team_part, map_part = parts
                                matched_team = self.match_team(team_part, team_a, team_b)
                                if matched_team:
                                    map_name = map_part.strip()
                                    self.bans[matched_team][map_name] += 1
                                    self.map_pool.add(map_name)
                        # Parse Picks
                        elif "pick" in token:
                            parts = token.split("pick")
                            if len(parts) == 2:
                                team_part, map_part = parts
                                matched_team = self.match_team(team_part, team_a, team_b)
                                if matched_team:
                                    map_name = map_part.strip()
                                    self.picks[matched_team][map_name] += 1
                                    self.map_pool.add(map_name)
                
                # 2. Parse Played Maps and Wins
                for m_idx, map_data in enumerate(segment.get("maps", [])):
                    map_name = map_data.get("map_name")
                    if not map_name or map_name.lower() == "all maps" or map_name.lower() == "none":
                        continue
                    self.map_pool.add(map_name)
                    self.plays[team_a][map_name] += 1
                    self.plays[team_b][map_name] += 1
                    
                    score = map_data.get("score", {})
                    t1_score = score.get("team1")
                    t2_score = score.get("team2")
                    if t1_score is not None and t2_score is not None:
                        if t1_score > t2_score:
                            self.wins[team_a][map_name] += 1
                        elif t2_score > t1_score:
                            self.wins[team_b][map_name] += 1
                            
            except Exception as e:
                logger.warning(f"Error parsing match veto file {f}: {e}")
                
        # Clean map pool from any invalid values
        self.map_pool = {m for m in self.map_pool if m and m.lower() not in ["all maps", "none"]}
        logger.info(f"VetoPredictor compiled: {len(self.team_names)} teams, Map pool: {list(self.map_pool)}")

    def get_map_scores(self, team: str):
        """Computes pick, ban, play, and win rate scores per map for a team."""
        scores = {}
        total_matches = self.total_matches.get(team, 1)
        for m in self.map_pool:
            ban_count = self.bans[team].get(m, 0)
            pick_count = self.picks[team].get(m, 0)
            play_count = self.plays[team].get(m, 0)
            win_count = self.wins[team].get(m, 0)
            
            ban_score = ban_count / total_matches
            pick_score = pick_count / total_matches
            win_rate = win_count / play_count if play_count > 0 else 0.5
            
            # Prefer maps picked often and won often
            pick_preference = pick_score * 2.0 + win_rate
            # Prefer banning maps banned often or where win rate is lowest
            ban_preference = ban_score * 2.0 + (1.0 - win_rate)
            
            scores[m] = {
                "ban_pref": ban_preference,
                "pick_pref": pick_preference,
                "win_rate": win_rate,
                "plays": play_count
            }
        return scores

    def predict_veto(self, team_a: str, team_b: str, series_type: str = "Bo3") -> dict:
        """Simulates the pick/ban process for team_a and team_b."""
        scores_a = self.get_map_scores(team_a)
        scores_b = self.get_map_scores(team_b)
        
        # Sort maps by overall occurrences or use active map pool
        # Standard active map pool contains Ascent, Bind, Breeze, Icebox, Lotus, Split, Sunset, Fracture, Haven, Pearl
        # We will filter self.map_pool to standard map pool to avoid stale/custom maps if possible
        # Or we can just use all maps in our map pool
        current_pool = list(self.map_pool)
        
        # We want to simulate the veto sequence.
        # Track active maps, bans, picks, decider
        available_maps = list(current_pool)
        banned_maps = []
        picked_maps = []
        veto_weights = {}
        veto_steps = []
        
        # Step-by-step veto simulation
        if series_type == "Bo5":
            # Bo5 Veto:
            # 1. Team A bans
            # 2. Team B bans
            # 3. Team A picks (Map 1) -> weight = 1
            # 4. Team B picks (Map 2) -> weight = -1
            # 5. Team A picks (Map 3) -> weight = 1
            # 6. Team B picks (Map 4) -> weight = -1
            # 7. Remaining map is Decider (Map 5) -> weight = 0
            
            # Ban 1: Team A
            map_to_ban_a = max(available_maps, key=lambda m: scores_a.get(m, {}).get("ban_pref", 0))
            available_maps.remove(map_to_ban_a)
            banned_maps.append(map_to_ban_a)
            veto_steps.append(f"{team_a} ban {map_to_ban_a}")
            
            # Ban 2: Team B
            map_to_ban_b = max(available_maps, key=lambda m: scores_b.get(m, {}).get("ban_pref", 0))
            available_maps.remove(map_to_ban_b)
            banned_maps.append(map_to_ban_b)
            veto_steps.append(f"{team_b} ban {map_to_ban_b}")
            
            # Pick 1: Team A
            map_to_pick_a1 = max(available_maps, key=lambda m: scores_a.get(m, {}).get("pick_pref", 0))
            available_maps.remove(map_to_pick_a1)
            picked_maps.append(map_to_pick_a1)
            veto_weights[map_to_pick_a1] = 1
            veto_steps.append(f"{team_a} pick {map_to_pick_a1}")
            
            # Pick 2: Team B
            map_to_pick_b1 = max(available_maps, key=lambda m: scores_b.get(m, {}).get("pick_pref", 0))
            available_maps.remove(map_to_pick_b1)
            picked_maps.append(map_to_pick_b1)
            veto_weights[map_to_pick_b1] = -1
            veto_steps.append(f"{team_b} pick {map_to_pick_b1}")
            
            # Pick 3: Team A
            map_to_pick_a2 = max(available_maps, key=lambda m: scores_a.get(m, {}).get("pick_pref", 0))
            available_maps.remove(map_to_pick_a2)
            picked_maps.append(map_to_pick_a2)
            veto_weights[map_to_pick_a2] = 1
            veto_steps.append(f"{team_a} pick {map_to_pick_a2}")
            
            # Pick 4: Team B
            map_to_pick_b2 = max(available_maps, key=lambda m: scores_b.get(m, {}).get("pick_pref", 0))
            available_maps.remove(map_to_pick_b2)
            picked_maps.append(map_to_pick_b2)
            veto_weights[map_to_pick_b2] = -1
            veto_steps.append(f"{team_b} pick {map_to_pick_b2}")
            
            # Decider: 1 or more maps remaining. We pick the one both teams have played/prefer most or left as decider
            if len(available_maps) > 0:
                # Select the remaining map with highest play count or decider affinity
                # As a heuristic, pick the map with highest total play count in dataset
                decider_map = max(available_maps, key=lambda m: (scores_a.get(m, {}).get("plays", 0) + scores_b.get(m, {}).get("plays", 0)))
                veto_weights[decider_map] = 0
                picked_maps.append(decider_map)
                veto_steps.append(f"{decider_map} remains")
            
        else:
            # Bo3 Veto:
            # 1. Team A bans
            # 2. Team B bans
            # 3. Team A picks (Map 1) -> weight = 1
            # 4. Team B picks (Map 2) -> weight = -1
            # 5. Team A bans
            # 6. Team B bans
            # 7. Remaining map is Decider (Map 3) -> weight = 0
            
            # Ban 1: Team A
            map_to_ban_a1 = max(available_maps, key=lambda m: scores_a.get(m, {}).get("ban_pref", 0))
            available_maps.remove(map_to_ban_a1)
            banned_maps.append(map_to_ban_a1)
            veto_steps.append(f"{team_a} ban {map_to_ban_a1}")
            
            # Ban 2: Team B
            map_to_ban_b1 = max(available_maps, key=lambda m: scores_b.get(m, {}).get("ban_pref", 0))
            available_maps.remove(map_to_ban_b1)
            banned_maps.append(map_to_ban_b1)
            veto_steps.append(f"{team_b} ban {map_to_ban_b1}")
            
            # Pick 1: Team A
            map_to_pick_a = max(available_maps, key=lambda m: scores_a.get(m, {}).get("pick_pref", 0))
            available_maps.remove(map_to_pick_a)
            picked_maps.append(map_to_pick_a)
            veto_weights[map_to_pick_a] = 1
            veto_steps.append(f"{team_a} pick {map_to_pick_a}")
            
            # Pick 2: Team B
            map_to_pick_b = max(available_maps, key=lambda m: scores_b.get(m, {}).get("pick_pref", 0))
            available_maps.remove(map_to_pick_b)
            picked_maps.append(map_to_pick_b)
            veto_weights[map_to_pick_b] = -1
            veto_steps.append(f"{team_b} pick {map_to_pick_b}")
            
            # Ban 3: Team A
            map_to_ban_a2 = max(available_maps, key=lambda m: scores_a.get(m, {}).get("ban_pref", 0))
            available_maps.remove(map_to_ban_a2)
            banned_maps.append(map_to_ban_a2)
            veto_steps.append(f"{team_a} ban {map_to_ban_a2}")
            
            # Ban 4: Team B
            map_to_ban_b2 = max(available_maps, key=lambda m: scores_b.get(m, {}).get("ban_pref", 0))
            available_maps.remove(map_to_ban_b2)
            banned_maps.append(map_to_ban_b2)
            veto_steps.append(f"{team_b} ban {map_to_ban_b2}")
            
            # Decider
            if len(available_maps) > 0:
                decider_map = max(available_maps, key=lambda m: (scores_a.get(m, {}).get("plays", 0) + scores_b.get(m, {}).get("plays", 0)))
                veto_weights[decider_map] = 0
                picked_maps.append(decider_map)
                veto_steps.append(f"{decider_map} remains")
                
        veto_str = "; ".join(veto_steps)
        return {
            "maps": picked_maps,
            "veto_weights": veto_weights,
            "veto_str": veto_str
        }

if __name__ == "__main__":
    predictor = VCTMapVetoPredictor()
    predictor.fit()
    # Test prediction
    result_bo3 = predictor.predict_veto("Paper Rex", "LEVIATÁN", "Bo3")
    print("Bo3 Veto Prediction:")
    print(" Maps:", result_bo3["maps"])
    print(" Weights:", result_bo3["veto_weights"])
    print(" Veto Str:", result_bo3["veto_str"])
    
    result_bo5 = predictor.predict_veto("Paper Rex", "LEVIATÁN", "Bo5")
    print("\nBo5 Veto Prediction:")
    print(" Maps:", result_bo5["maps"])
    print(" Weights:", result_bo5["veto_weights"])
    print(" Veto Str:", result_bo5["veto_str"])
