import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, ".")

from utils.match_adapter import normalize_match
from utils.team_registry import resolve_team_info

def save_dataframe_parquet(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

def canonicalize_team(name: str) -> str:
    return resolve_team_info(name).get("name", name)

logger = logging.getLogger("ml.feature_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

RAW_DATA_DIR = "data/raw"
FEATURES_DIR = "data/features"

def load_and_normalize_all_matches(raw_dir: str = RAW_DATA_DIR) -> List[Dict[str, Any]]:
    """Loads all raw match JSON files and normalizes them chronologically."""
    json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    matches = []
    
    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            # Exclude non-match JSONs (e.g., registries)
            if not isinstance(content, dict):
                continue
            if "match_id" not in content and "data" not in content and "overview" not in content:
                continue
            norm = normalize_match(content)
            matches.append(norm)
        except Exception as e:
            # Skip invalid non-match files
            continue
            
    # Sort strictly chronologically by date
    matches.sort(key=lambda x: x.get("date") or datetime(2025, 1, 1))
    logger.info(f"Successfully normalized and sorted {len(matches)} matches.")
    return matches


def build_team_features(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes chronological team-level features up to each match.
    Features:
    - matches_played, wins, losses, win_rate
    - map-specific: map_pick_rate, map_win_rate, recent_map_form
    - economy: full_buy_win_rate, semi_buy_win_rate, eco_conversion, pistol_win_rate
    - round: attack_round_win_rate, defense_round_win_rate, first_half_win_rate, comeback_rate
    """
    records = []
    
    # State tracking dictionary for cumulative stats
    team_history: Dict[str, Dict[str, Any]] = {}

    def get_team_state(t: str) -> Dict[str, Any]:
        c_team = canonicalize_team(t)
        if c_team not in team_history:
            team_history[c_team] = {
                "matches": 0, "wins": 0, "losses": 0,
                "maps": {},  # map -> {"played": 0, "wins": 0, "recent": []}
                "econ": {"full_buy_w": 0, "full_buy_tot": 0, "semi_buy_w": 0, "semi_buy_tot": 0, "eco_w": 0, "eco_tot": 0, "pistol_w": 0, "pistol_tot": 0},
                "rounds": {"atk_w": 0, "atk_tot": 0, "def_w": 0, "def_tot": 0, "fh_w": 0, "fh_tot": 0, "comebacks": 0}
            }
        return team_history[c_team]

    for match in matches:
        match_id = match["match_id"]
        date = match["date"]
        t1 = canonicalize_team(match["teams"]["team1"])
        t2 = canonicalize_team(match["teams"]["team2"])
        winner = canonicalize_team(match.get("winner", ""))

        st1 = get_team_state(t1)
        st2 = get_team_state(t2)

        # Snapshot pre-match features for team 1
        t1_win_rate = st1["wins"] / st1["matches"] if st1["matches"] > 0 else 0.5
        t2_win_rate = st2["wins"] / st2["matches"] if st2["matches"] > 0 else 0.5

        # Record pre-match features
        for team, st, opp_st in [(t1, st1, st2), (t2, st2, st1)]:
            rec = {
                "match_id": match_id,
                "date": date,
                "team": team,
                "matches_played": st["matches"],
                "wins": st["wins"],
                "losses": st["losses"],
                "win_rate": st["wins"] / st["matches"] if st["matches"] > 0 else 0.5,
                "full_buy_win_rate": st["econ"]["full_buy_w"] / max(1, st["econ"]["full_buy_tot"]),
                "semi_buy_win_rate": st["econ"]["semi_buy_w"] / max(1, st["econ"]["semi_buy_tot"]),
                "eco_conversion": st["econ"]["eco_w"] / max(1, st["econ"]["eco_tot"]),
                "pistol_win_rate": st["econ"]["pistol_w"] / max(1, st["econ"]["pistol_tot"]),
                "attack_round_win_rate": st["rounds"]["atk_w"] / max(1, st["rounds"]["atk_tot"]),
                "defense_round_win_rate": st["rounds"]["def_w"] / max(1, st["rounds"]["def_tot"]),
                "first_half_win_rate": st["rounds"]["fh_w"] / max(1, st["rounds"]["fh_tot"]),
                "comeback_rate": st["rounds"]["comebacks"] / max(1, st["matches"])
            }
            records.append(rec)

        # Update state post-match
        if winner == t1:
            st1["wins"] += 1
            st2["losses"] += 1
        elif winner == t2:
            st2["wins"] += 1
            st1["losses"] += 1

        st1["matches"] += 1
        st2["matches"] += 1

        # Process maps & economy & round history
        for m in match.get("maps", []):
            m_name = m.get("map_name", "Unknown")
            m_winner = canonicalize_team(m.get("winner", ""))

            for t_curr, st_curr in [(t1, st1), (t2, st2)]:
                if m_name not in st_curr["maps"]:
                    st_curr["maps"][m_name] = {"played": 0, "wins": 0, "recent": []}
                st_curr["maps"][m_name]["played"] += 1
                is_win = 1 if m_winner == t_curr else 0
                st_curr["maps"][m_name]["wins"] += is_win
                st_curr["maps"][m_name]["recent"].append(is_win)
                if len(st_curr["maps"][m_name]["recent"]) > 5:
                    st_curr["maps"][m_name]["recent"].pop(0)

            # Round history updates
            for r in m.get("round_history", []):
                r_win = canonicalize_team(r.get("winner", ""))
                r_num = int(r.get("round_num", 1))
                if r_win == t1:
                    st1["rounds"]["atk_w" if r_num <= 12 else "def_w"] += 1
                    if r_num <= 12: st1["rounds"]["fh_w"] += 1
                elif r_win == t2:
                    st2["rounds"]["atk_w" if r_num <= 12 else "def_w"] += 1
                    if r_num <= 12: st2["rounds"]["fh_w"] += 1
                st1["rounds"]["atk_tot" if r_num <= 12 else "def_tot"] += 1
                st2["rounds"]["atk_tot" if r_num <= 12 else "def_tot"] += 1
                if r_num <= 12:
                    st1["rounds"]["fh_tot"] += 1
                    st2["rounds"]["fh_tot"] += 1

    df = pd.DataFrame(records)
    logger.info(f"Built team features dataframe: {df.shape}")
    return df


def build_player_features(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes player performance EMA and aggregate features up to each match.
    Features: ACS_EMA, rating_EMA, KAST_EMA, ADR_EMA, FK_rate, FD_rate, clutch stats, ECON_EMA
    """
    records = []
    player_history: Dict[str, Dict[str, Any]] = {}

    def get_p_state(p_name: str) -> Dict[str, Any]:
        if p_name not in player_history:
            player_history[p_name] = {
                "obs": 0,
                "acs_hist": [], "rating_hist": [], "kast_hist": [], "adr_hist": [], "econ_hist": [],
                "fk": 0, "fd": 0, "first_duels": 0,
                "clutches": {"1v1": 0, "1v2": 0, "1v3": 0, "1v4": 0, "1v5": 0},
                "multikills": {"2k": 0, "3k": 0, "4k": 0, "5k": 0}
            }
        return player_history[p_name]

    def calc_ema(vals: List[float], alpha: float = 0.2) -> float:
        if not vals:
            return 0.0
        ema = vals[0]
        for v in vals[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)

    for match in matches:
        match_id = match["match_id"]
        date = match["date"]

        for m in match.get("maps", []):
            for player in m.get("players", []):
                if isinstance(player, str):
                    p_name = player
                    player_dict = {}
                elif isinstance(player, dict):
                    p_name = player.get("name") or player.get("player_name") or player.get("id")
                    player_dict = player
                else:
                    continue

                if not p_name:
                    continue

                st = get_p_state(p_name)

                # Record snapshot pre-observation
                rec = {
                    "match_id": match_id,
                    "date": date,
                    "player": p_name,
                    "observations": st["obs"],
                    "ACS_EMA": calc_ema(st["acs_hist"]),
                    "rating_EMA": calc_ema(st["rating_hist"]),
                    "KAST_EMA": calc_ema(st["kast_hist"]),
                    "ADR_EMA": calc_ema(st["adr_hist"]),
                    "ECON_EMA": calc_ema(st["econ_hist"]),
                    "FK_rate": st["fk"] / max(1, st["obs"]),
                    "FD_rate": st["fd"] / max(1, st["obs"]),
                    "first_duel_success": st["fk"] / max(1, st["first_duels"]),
                    "2K_rate": st["multikills"]["2k"] / max(1, st["obs"]),
                    "3K_rate": st["multikills"]["3k"] / max(1, st["obs"]),
                    "4K_rate": st["multikills"]["4k"] / max(1, st["obs"]),
                    "5K_rate": st["multikills"]["5k"] / max(1, st["obs"]),
                    "1v1_success": st["clutches"]["1v1"],
                    "1v2_success": st["clutches"]["1v2"]
                }
                records.append(rec)

                # Update history
                st["obs"] += 1
                try:
                    acs = float(player_dict.get("acs") or player_dict.get("average_combat_score") or 0)
                    rating = float(player_dict.get("rating") or 1.0)
                    kast_str = str(player_dict.get("kast") or "70%").replace("%", "")
                    kast = float(kast_str) if kast_str else 70.0
                    adr = float(player_dict.get("adr") or player_dict.get("average_damage_per_round") or 0)
                    econ = float(player_dict.get("econ") or 100)

                    st["acs_hist"].append(acs)
                    st["rating_hist"].append(rating)
                    st["kast_hist"].append(kast)
                    st["adr_hist"].append(adr)
                    st["econ_hist"].append(econ)

                    fk = int(player_dict.get("fk") or player_dict.get("first_kills") or 0)
                    fd = int(player_dict.get("fd") or player_dict.get("first_deaths") or 0)
                    st["fk"] += fk
                    st["fd"] += fd
                    st["first_duels"] += (fk + fd)

                    st["multikills"]["2k"] += int(player_dict.get("2k") or 0)
                    st["multikills"]["3k"] += int(player_dict.get("3k") or 0)
                    st["multikills"]["4k"] += int(player_dict.get("4k") or 0)
                    st["multikills"]["5k"] += int(player_dict.get("5k") or 0)
                except Exception:
                    pass

    df = pd.DataFrame(records)
    logger.info(f"Built player features dataframe: {df.shape}")
    return df


def build_map_features(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """Computes global map performance statistics."""
    map_stats: Dict[str, Dict[str, Any]] = {}

    for match in matches:
        for m in match.get("maps", []):
            m_name = m.get("map_name", "Unknown")
            if m_name not in map_stats:
                map_stats[m_name] = {"total_played": 0, "t1_wins": 0, "t2_wins": 0, "rounds_played": 0}

            map_stats[m_name]["total_played"] += 1
            s = m.get("score", {})
            s1 = s.get("team1_score", 0)
            s2 = s.get("team2_score", 0)
            map_stats[m_name]["rounds_played"] += (s1 + s2)
            if s1 > s2:
                map_stats[m_name]["t1_wins"] += 1
            elif s2 > s1:
                map_stats[m_name]["t2_wins"] += 1

    records = []
    for m_name, st in map_stats.items():
        records.append({
            "map_name": m_name,
            "total_played": st["total_played"],
            "avg_rounds": st["rounds_played"] / max(1, st["total_played"]),
            "t1_win_ratio": st["t1_wins"] / max(1, st["total_played"])
        })

    df = pd.DataFrame(records)
    logger.info(f"Built map features dataframe: {df.shape}")
    return df


def build_feature_manifest(features_dir: str = FEATURES_DIR) -> Dict[str, Any]:
    """Generates feature_manifest.json describing schema version and available features."""
    manifest = {
        "version": "10.1",
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat(),
        "features": {
            "team": [
                "matches_played", "win_rate", "full_buy_win_rate", "semi_buy_win_rate",
                "eco_conversion", "pistol_win_rate", "attack_round_win_rate",
                "defense_round_win_rate", "first_half_win_rate", "comeback_rate"
            ],
            "player": [
                "ACS_EMA", "rating_EMA", "KAST_EMA", "ADR_EMA", "ECON_EMA",
                "FK_rate", "FD_rate", "first_duel_success", "1v1_success", "1v2_success",
                "2K_rate", "3K_rate", "4K_rate", "5K_rate"
            ],
            "map": [
                "map_name", "total_played", "avg_rounds", "t1_win_ratio"
            ]
        }
    }
    
    os.makedirs(features_dir, exist_ok=True)
    manifest_path = os.path.join(features_dir, "feature_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Saved feature manifest to {manifest_path}")
    return manifest


def generate_feature_store(raw_dir: str = RAW_DATA_DIR, out_dir: str = FEATURES_DIR):
    """Main pipeline execution for Feature Store."""
    os.makedirs(out_dir, exist_ok=True)
    matches = load_and_normalize_all_matches(raw_dir)

    team_df = build_team_features(matches)
    save_dataframe_parquet(team_df, os.path.join(out_dir, "team_features.parquet"))

    player_df = build_player_features(matches)
    save_dataframe_parquet(player_df, os.path.join(out_dir, "player_features.parquet"))

    map_df = build_map_features(matches)
    save_dataframe_parquet(map_df, os.path.join(out_dir, "map_features.parquet"))

    build_feature_manifest(out_dir)
    logger.info("Feature Store generation complete!")


if __name__ == "__main__":
    generate_feature_store()
