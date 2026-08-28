import os
import json
import glob
import logging
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from collections import defaultdict
from utils.match_adapter import normalize_match

logger = logging.getLogger("generative_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class MapScoreRegressor:
    def __init__(self, model_dir="./data/processed"):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "score_regressor.cbm")
        self.model = None
        self.is_trained = False
        
    def fit(self, X_features_path="./data/processed/X_features.csv", raw_dir="./data/raw"):
        """Compiles map-level score dataset and trains the CatBoostRegressor."""
        logger.info("MapScoreRegressor: Loading feature store...")
        if not os.path.exists(X_features_path):
            raise FileNotFoundError(f"Feature store not found at {X_features_path}")
            
        df_features = pd.read_csv(X_features_path)
        
        # Load raw match files to get map-level scores
        files = glob.glob(os.path.join(raw_dir, "match_*.json"))
        match_scores = {}
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    content = json.load(file)
                norm = normalize_match(content)
                match_id = norm.get("match_id")
                if not match_id:
                    continue
                
                maps_data = []
                for map_info in norm.get("maps", []):
                    m_name = map_info.get("map_name")
                    score = map_info.get("score") or {}
                    t1_score = score.get("team1_score") if score.get("team1_score") is not None else score.get("team1")
                    t2_score = score.get("team2_score") if score.get("team2_score") is not None else score.get("team2")
                    if m_name and t1_score is not None and t2_score is not None:
                        maps_data.append({
                            "map_name": m_name,
                            "score_a": t1_score,
                            "score_b": t2_score,
                            "score_diff": t1_score - t2_score
                        })
                match_scores[match_id] = maps_data
            except Exception as e:
                logger.warning(f"Error reading match {f} for score extraction: {e}")
                
        # Build map-level training dataset
        for _, row in X_df.iterrows():
            match_id = str(row["match_id"])
            if match_id not in match_scores:
                continue
                
            # For each map played in this match
            for map_info in match_scores[match_id]:
                map_name = map_info["map_name"]
                score_diff = map_info["score_diff"]
                
                # Determine map veto weight for this specific map in this match
                # Check map_1_name, map_2_name, map_3_name, map_4_name, map_5_name in row
                veto_weight = 0
                for i in range(1, 6):
                    col_name = f"map_{i}_name"
                    col_veto = f"map_{i}_veto_weight"
                    if col_name in row and row[col_name] == map_name:
                        veto_weight = row[col_veto]
                        break
                
                map_row = {
                    "map_name": map_name,
                    "veto_weight": veto_weight,
                    "team_a_historical_acs_ema": row.get("team_a_historical_acs_ema", 200.0),
                    "team_a_historical_avg_loadout": row.get("team_a_historical_avg_loadout", 20000.0),
                    "team_a_comfort_pick_differential": row.get("team_a_comfort_pick_differential", 0.0),
                    "team_b_historical_acs_ema": row.get("team_b_historical_acs_ema", 200.0),
                    "team_b_historical_avg_loadout": row.get("team_b_historical_avg_loadout", 20000.0),
                    "team_b_comfort_pick_differential": row.get("team_b_comfort_pick_differential", 0.0),
                    "y_diff": score_diff
                }
                map_rows.append(map_row)
                
        df_map_level = pd.DataFrame(map_rows)
        logger.info(f"Formed map-level training dataset with {len(df_map_level)} rows.")
        
        if len(df_map_level) == 0:
            logger.error("No map-level rows to train on!")
            return
            
        # Features and target
        X = df_map_level.drop(columns=["y_diff"])
        y = df_map_level["y_diff"]
        
        # Identify categorical features
        cat_features = ["map_name"]
        
        # Train CatBoostRegressor
        logger.info("Training CatBoostRegressor...")
        self.model = CatBoostRegressor(
            iterations=150,
            learning_rate=0.05,
            depth=4,
            cat_features=cat_features,
            random_seed=42,
            verbose=0
        )
        self.model.fit(X, y)
        self.is_trained = True
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_model(self.model_path)
        logger.info(f"MapScoreRegressor model saved to {self.model_path}")
        
    def load_model(self):
        """Loads the trained regressor model."""
        if os.path.exists(self.model_path):
            self.model = CatBoostRegressor()
            self.model.load_model(self.model_path)
            self.is_trained = True
            logger.info("Loaded MapScoreRegressor model successfully.")
        else:
            logger.warning("MapScoreRegressor model file not found. Need to fit model first.")
            self.is_trained = False
            
    def predict_score(self, team_a_features: dict, team_b_features: dict, map_name: str, veto_weight: float) -> tuple[int, int]:
        """Predicts the map score (team_a_rounds, team_b_rounds) based on features."""
        if not self.is_trained:
            self.load_model()
            if not self.is_trained:
                # Simple fallback heuristic if model not trained/loaded
                logger.warning("Model not trained. Using heuristic fallback score prediction.")
                return (13, 9)
                
        input_data = pd.DataFrame([{
            "map_name": map_name,
            "veto_weight": veto_weight,
            "team_a_historical_acs_ema": team_a_features.get("acs_ema", 200.0),
            "team_a_historical_avg_loadout": team_a_features.get("avg_loadout", 20000.0),
            "team_a_comfort_pick_differential": team_a_features.get("comfort_diff", 0.0),
            "team_b_historical_acs_ema": team_b_features.get("acs_ema", 200.0),
            "team_b_historical_avg_loadout": team_b_features.get("avg_loadout", 20000.0),
            "team_b_comfort_pick_differential": team_b_features.get("comfort_diff", 0.0),
        }])
        
        pred_diff = self.model.predict(input_data)[0]
        
        # Round and determine rounds won by each team
        if pred_diff >= 0:
            score_a = 13
            # Limit the score of loser to [0, 11] (close game is 13-11)
            score_b = int(np.clip(round(13 - pred_diff), 0, 11))
        else:
            score_b = 13
            score_a = int(np.clip(round(13 + pred_diff), 0, 11))
            
        return (score_a, score_b)


class AgentCompositionGenerator:
    def __init__(self, raw_dir="./data/raw"):
        self.raw_dir = raw_dir
        # Structure: player -> map -> agent -> play_count / sum_acs
        self.player_map_agent_plays = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.player_map_agent_acs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.player_agent_plays = defaultdict(lambda: defaultdict(int))
        self.player_agent_acs = defaultdict(lambda: defaultdict(float))
        # Structure: team_name -> list of players in most recent match
        self.team_recent_roster = {}
        # Structure: team_name -> timestamp of most recent match
        self.team_recent_ts = {}
        
        self.agent_roles = {}
        self.load_roles()
        
    def load_roles(self):
        """Loads agent roles configuration."""
        roles_path = os.path.join(self.raw_dir, "agent_roles.json")
        if os.path.exists(roles_path):
            with open(roles_path, "r", encoding="utf-8") as f:
                self.agent_roles = json.load(f)
        else:
            logger.warning("agent_roles.json not found. Roles fallback will be used.")

    def fit(self):
        """Loads matches to compile player comfort levels and team rosters."""
        files = glob.glob(os.path.join(self.raw_dir, "match_*.json"))
        logger.info(f"AgentCompositionGenerator: Ingesting {len(files)} matches for roster & comfort metrics...")
        
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    content = json.load(file)
                norm = normalize_match(content)
                team_a = norm.get("team_a", "")
                team_b = norm.get("team_b", "")

                if not team_a or not team_b:
                    continue

                date_str = norm.get("date", "")
                from feature_engineering import parse_match_date
                try:
                    ts = parse_match_date(date_str)
                except Exception:
                    ts = None
                
                # Check recent rosters
                roster_a = set()
                roster_b = set()
                
                maps_list = norm.get("maps", [])

                for map_data in maps_list:
                    map_name = map_data.get("map_name") or map_data.get("map")
                    if not map_name or str(map_name).lower() in ["all maps", "none"]:
                        continue
                    
                    raw_players = map_data.get("players", [])
                    if isinstance(raw_players, dict):
                        # Players team 1 (Team A)
                        for p in raw_players.get("team1", []):
                            p_name = p.get("name")
                            agent = p.get("agent")
                            acs = float(p.get("acs", 0.0) or 0.0)
                            if p_name:
                                roster_a.add(p_name)
                                if agent:
                                    self.player_map_agent_plays[p_name][map_name][agent] += 1
                                    self.player_map_agent_acs[p_name][map_name][agent] += acs
                                    self.player_agent_plays[p_name][agent] += 1
                                    self.player_agent_acs[p_name][agent] += acs
                                    
                        # Players team 2 (Team B)
                        for p in raw_players.get("team2", []):
                            p_name = p.get("name")
                            agent = p.get("agent")
                            acs = float(p.get("acs", 0.0) or 0.0)
                            if p_name:
                                roster_b.add(p_name)
                                if agent:
                                    self.player_map_agent_plays[p_name][map_name][agent] += 1
                                    self.player_map_agent_acs[p_name][map_name][agent] += acs
                                    self.player_agent_plays[p_name][agent] += 1
                                    self.player_agent_acs[p_name][agent] += acs
                    elif isinstance(raw_players, list):
                        for p in raw_players:
                            p_name = p.get("name")
                            p_team = p.get("team", team_a)
                            agent = p.get("agent")
                            acs = float(p.get("acs", 0.0) or 0.0)
                            if p_name:
                                if p_team == team_a:
                                    roster_a.add(p_name)
                                else:
                                    roster_b.add(p_name)
                                if agent:
                                    self.player_map_agent_plays[p_name][map_name][agent] += 1
                                    self.player_map_agent_acs[p_name][map_name][agent] += acs
                                    self.player_agent_plays[p_name][agent] += 1
                                    self.player_agent_acs[p_name][agent] += acs
                            
                # Update team recent rosters based on match timestamp
                if roster_a and len(roster_a) >= 5:
                    if ts is not None and (team_a not in self.team_recent_ts or self.team_recent_ts[team_a] is None or ts > self.team_recent_ts[team_a]):
                        self.team_recent_roster[team_a] = list(roster_a)[:5]
                        self.team_recent_ts[team_a] = ts
                if roster_b and len(roster_b) >= 5:
                    if ts is not None and (team_b not in self.team_recent_ts or self.team_recent_ts[team_b] is None or ts > self.team_recent_ts[team_b]):
                        self.team_recent_roster[team_b] = list(roster_b)[:5]
                        self.team_recent_ts[team_b] = ts
                        
            except Exception as e:
                logger.warning(f"Error parsing match {f} in AgentComp fit: {e}")

    def predict_composition(self, team_name: str, map_name: str) -> dict:
        """Predicts the most probable 5-agent composition for a team on a map (no duplicates)."""
        roster = self.team_recent_roster.get(team_name)
        if not roster:
            # Fallback default roster
            logger.warning(f"No roster found for team {team_name}. Using generic fallback roster.")
            roster = [f"Player{i+1}" for i in range(5)]
            
        # For each player, score all available agents
        # We collect comfort scores for all player-agent pairs
        candidates = []
        global_agents = list(self.agent_roles.keys())
        if not global_agents:
            global_agents = ["Jett", "Omen", "Sova", "Killjoy", "Breach", "Phoenix", "Reyna", "Viper", "Cypher"]
            
        for player in roster:
            for agent in global_agents:
                # Score formula:
                # If played on this map: 5 * plays + avg_acs / 10
                # Else if played overall: 1 * plays + avg_acs / 10
                # Else: 0
                plays_map = self.player_map_agent_plays[player][map_name].get(agent, 0)
                acs_map = self.player_map_agent_acs[player][map_name].get(agent, 0.0)
                avg_acs_map = acs_map / plays_map if plays_map > 0 else 0.0
                
                plays_all = self.player_agent_plays[player].get(agent, 0)
                acs_all = self.player_agent_acs[player].get(agent, 0.0)
                avg_acs_all = acs_all / plays_all if plays_all > 0 else 0.0
                
                if plays_map > 0:
                    score = plays_map * 10.0 + avg_acs_map / 10.0
                elif plays_all > 0:
                    score = plays_all * 2.0 + avg_acs_all / 10.0
                else:
                    score = 0.0
                    
                candidates.append((player, agent, score))
                
        # Sort candidates by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy assignment to ensure no duplicate agents and exactly 1 agent per player
        assigned_agents = {} # player -> agent
        used_agents = set()
        
        for player, agent, score in candidates:
            if player not in assigned_agents and agent not in used_agents:
                assigned_agents[player] = agent
                used_agents.add(agent)
                
        # Assign fallbacks for players who didn't get an agent greedily
        for player in roster:
            if player not in assigned_agents:
                # Find any unused agent in global pool
                for agent in global_agents:
                    if agent not in used_agents:
                        assigned_agents[player] = agent
                        used_agents.add(agent)
                        break
                # If still none, add a fallback unique name
                if player not in assigned_agents:
                    fallback_agent = f"Agent_{player}"
                    assigned_agents[player] = fallback_agent
                    used_agents.add(fallback_agent)
                    
        # Return results mapping player to their agent and role
        result = {}
        for player, agent in assigned_agents.items():
            role = self.agent_roles.get(agent, "Duelist") # default to Duelist if role unknown
            result[player] = {
                "agent": agent,
                "role": role
            }
        return result


if __name__ == "__main__":
    # Test execution
    regressor = MapScoreRegressor()
    try:
        regressor.fit()
    except Exception as e:
        print("Fit failed (expected if CSV features aren't built yet):", e)
        
    comp_gen = AgentCompositionGenerator()
    comp_gen.fit()
    
    # Test dynamic roster retrieval
    print("Recent Rosters keys:", list(comp_gen.team_recent_roster.keys())[:5])
    
    if "Paper Rex" in comp_gen.team_recent_roster:
        comp = comp_gen.predict_composition("Paper Rex", "Fracture")
        print("\nPredicted Paper Rex Fracture Roster & Agent Compositions:")
        for player, details in comp.items():
            print(f"  {player} -> Agent: {details['agent']} ({details['role']})")
