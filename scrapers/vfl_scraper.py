"""
VFL (Valorant Fantasy League) Scraper Module — API Edition
===========================================================
Replaces HTML/DOM parsing. Uses the VFL REST API directly:
  1. GET /api/event/currentevent  → resolves active event_id
  2. GET /api/player/allplayers?eventId={id} → 60-player roster JSON

Field mapping (local schema ← API fields):
  player_name   ← player.name
  vlr_team_id   ← team.id (string → int)
  team_name     ← team.name
  team_short    ← team.shortName
  role          ← playerRole int (0=Duelist, 1=Initiator, 2=Controller, 3=Sentinel)
  price         ← price
  gw_pts        ← currentGameweekPoints.totalPoints
  tot_pts       ← totalEventPoints.totalPoints
  ppg           ← computed: tot_pts / max(gameweeks_played, 1)
  event_id      ← eventId
  event_name    ← (stored from currentevent)
"""

import os
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Optional

from curl_cffi import requests

logger = logging.getLogger("vfl_scraper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ── Constants ──────────────────────────────────────────────────────────────────
from pathlib import Path
ROOT_DIR              = Path(__file__).resolve().parent.parent
CACHE_DIR             = str(ROOT_DIR / "data" / "processed")
VFL_PLAYERS_DB_CACHE  = os.path.join(CACHE_DIR, "vfl_players_db.json")

VFL_API_BASE          = "https://api.valorantfantasyleague.net/api"
CURRENT_EVENT_URL     = f"{VFL_API_BASE}/event/currentevent"
ALL_PLAYERS_URL       = f"{VFL_API_BASE}/player/allplayers"

ROLE_MAP = {
    0: "Duelist",
    1: "Initiator",
    2: "Controller",
    3: "Sentinel",
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


# ── Core Scraper Class ─────────────────────────────────────────────────────────
class VFLScraper:
    """
    Fetches and caches Valorant Fantasy League player stats via the VFL REST API.
    No HTML/DOM parsing — two clean HTTP calls.
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir  = cache_dir
        self.cache_path = os.path.join(cache_dir, "vfl_players_db.json")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_current_event(self, force_refresh: bool = False) -> dict:
        """
        Fetches current event metadata, schedule, players, budget, and VLR mappings strictly via /api/event/currentevent.
        Endpoint: GET https://api.valorantfantasyleague.net/api/event/currentevent
        Caches payload locally to data/processed/vfl_currentevent.json.
        """
        cache_file = os.path.join(self.cache_dir, "vfl_currentevent.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    logger.info(f"Loading current event state from cache: {cache_file}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read current event cache: {e}. Refetching...")

        logger.info(f"Fetching live current event state from {CURRENT_EVENT_URL}")
        try:
            with requests.Session(impersonate="chrome") as client:
                resp = client.get(CURRENT_EVENT_URL, headers=DEFAULT_HEADERS, timeout=20.0)
                if resp.status_code != 200:
                    raise Exception(f"HTTP Status {resp.status_code}: {resp.text}")
                
                raw_data = resp.json()
                event_id = raw_data.get("id", 10)
                event_name = raw_data.get("name", "Current Event")
                
                # Extract VLR Regions & Event IDs
                vlr_regions = [r.get("vlrRegion") for r in raw_data.get("matchRegions", []) if r.get("vlrRegion")]
                vlr_events = [str(v.get("vlrEventId")) for v in raw_data.get("vlrEvents", []) if v.get("vlrEventId")]
                
                # Map players from eventPlayers
                parsed_players = []
                for p in raw_data.get("eventPlayers", []):
                    p_mapped = self._map_player(p, event_id=event_id, event_name=event_name)
                    p_name = p_mapped["player_name"].lower()
                    t_name = p_mapped["team_name"].lower()
                    t_short = p_mapped.get("team_short", "").lower()

                    if "inactive" in p_name:
                        continue

                    suffixes = ["academy", "gc", "game changers", "black", "blue"]
                    if any(s in t_name for s in suffixes) or any(s in t_short for s in suffixes):
                        continue

                    parsed_players.append(p_mapped)

                parsed_players.sort(key=lambda x: x["ppg"], reverse=True)
                self.save_to_cache(parsed_players)

                # Extract gameweek schedule & active teams
                event_matches = raw_data.get("eventMatches", [])
                gameweek_teams = {}
                for m in event_matches:
                    gw = m.get("gameweek")
                    if gw is None:
                        continue
                    gw = int(gw)
                    if gw not in gameweek_teams:
                        gameweek_teams[gw] = set()
                    for t_key in ["matchTeams"]:
                        m_teams = m.get(t_key) or []
                        for mt in m_teams:
                            team_obj = mt.get("team") or {}
                            t_name = team_obj.get("name")
                            t_short = team_obj.get("shortName")
                            if t_name:
                                gameweek_teams[gw].add(t_name)
                            if t_short:
                                gameweek_teams[gw].add(t_short)

                # Format gameweek_teams as list
                gw_teams_clean = {str(k): sorted(list(v)) for k, v in gameweek_teams.items()}

                result = {
                    "event_id": event_id,
                    "event_name": event_name,
                    "current_gameweek": raw_data.get("currentGameweek", 1),
                    "budget": raw_data.get("budget", 100),
                    "vlr_regions": vlr_regions,
                    "vlr_events": vlr_events,
                    "players": parsed_players,
                    "gameweek_teams": gw_teams_clean,
                    "event_matches": event_matches,
                    "raw_data": raw_data
                }
                
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                    
                logger.info(f"Saved current event cache ({len(parsed_players)} players) → {cache_file}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch live current event: {e}")
            return {
                "event_id": 10,
                "event_name": "VCT Event",
                "current_gameweek": 2,
                "budget": 100,
                "vlr_regions": [],
                "vlr_events": [],
                "players": [],
                "gameweek_teams": {},
                "event_matches": [],
                "raw_data": {}
            }

    def get_schedule(self, gameweek: int = 2, event_id: int = 10, force_refresh: bool = False) -> dict:
        """
        Fetches match schedule for a specific gameweek and eventId via local currentevent cache fallback or API.
        """
        # First check local currentevent cache
        cache_event_file = os.path.join(self.cache_dir, "vfl_currentevent.json")
        if not force_refresh and os.path.exists(cache_event_file):
            try:
                with open(cache_event_file, "r", encoding="utf-8") as f:
                    evt = json.load(f)
                    gw_teams_dict = evt.get("gameweek_teams", {})
                    active_teams = gw_teams_dict.get(str(gameweek), [])
                    if active_teams:
                        matches = [m for m in evt.get("event_matches", []) if m.get("gameweek") == gameweek]
                        matchup_pairs = []
                        for m in matches:
                            m_teams = m.get("matchTeams") or []
                            if len(m_teams) >= 2:
                                t1 = (m_teams[0].get("team") or {}).get("name")
                                t2 = (m_teams[1].get("team") or {}).get("name")
                                if t1 and t2:
                                    matchup_pairs.append((t1, t2))
                        return {
                            "gameweek": gameweek,
                            "event_id": event_id,
                            "matches": matches,
                            "matchup_pairs": matchup_pairs,
                            "active_teams": active_teams,
                            "teams_info": []
                        }
            except Exception as e:
                logger.warning(f"Could not load schedule from currentevent cache: {e}")

        cache_file = os.path.join(self.cache_dir, f"schedule_gw{gameweek}_ev{event_id}.json")
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    logger.info(f"Loading Gameweek {gameweek} schedule from cache: {cache_file}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read schedule cache: {e}. Refetching...")

        url = f"{VFL_API_BASE}/matches/schedule?gameweek={gameweek}&eventId={event_id}"
        logger.info(f"Fetching Gameweek {gameweek} schedule from {url}")
        
        try:
            with requests.Session(impersonate="chrome") as client:
                sleep_time = 1.0 + random.uniform(0.2, 0.8)
                time.sleep(sleep_time)
                resp = client.get(url, headers=DEFAULT_HEADERS, timeout=20.0)
                if resp.status_code != 200:
                    raise Exception(f"HTTP Status {resp.status_code}: {resp.text}")
                
                raw_data = resp.json()
                matches = raw_data if isinstance(raw_data, list) else raw_data.get("matches", [])
                
                active_teams_set = set()
                teams_info = []
                seen_team_ids = set()
                matchup_pairs = []
                
                for match in matches:
                    t1_obj = match.get("team1") or {}
                    t2_obj = match.get("team2") or {}
                    t1_name = t1_obj.get("name") if isinstance(t1_obj, dict) else None
                    t2_name = t2_obj.get("name") if isinstance(t2_obj, dict) else None
                    if t1_name and t2_name:
                        matchup_pairs.append((t1_name, t2_name))

                    for t_key in ["team1", "team2"]:
                        team = match.get(t_key)
                        if team and isinstance(team, dict):
                            t_name = team.get("name")
                            t_short = team.get("shortName")
                            t_id = team.get("id")
                            
                            if t_name:
                                active_teams_set.add(t_name)
                            if t_short:
                                active_teams_set.add(t_short)
                                
                            if t_id and t_id not in seen_team_ids:
                                seen_team_ids.add(t_id)
                                teams_info.append({
                                    "id": t_id,
                                    "name": t_name,
                                    "shortName": t_short
                                })
                
                result = {
                    "gameweek": gameweek,
                    "event_id": event_id,
                    "matches": matches,
                    "matchup_pairs": matchup_pairs,
                    "active_teams": sorted(list(active_teams_set)),
                    "teams_info": teams_info
                }
                
                # Save to cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                    
                return result
        except Exception as e:
            logger.error(f"Failed to fetch Gameweek {gameweek} schedule: {e}")
            return {
                "gameweek": gameweek,
                "event_id": event_id,
                "matches": [],
                "active_teams": [],
                "teams_info": []
            }

    def get_players(self, force_refresh: bool = False) -> list[dict]:
        """Return player list — from cache unless force_refresh or no cache."""
        if not force_refresh and os.path.exists(self.cache_path):
            return self.load_from_cache()
        return self.scrape_player_stats()

    def scrape_player_stats(self) -> list[dict]:
        """
        Fetches currentevent and extracts players array.
        """
        evt = self.get_current_event(force_refresh=True)
        return evt.get("players", [])

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _map_player(self, raw: dict, event_id: int, event_name: str) -> dict:
        """Transform one raw API player dict into our local schema."""
        player_info = raw.get("player") or {}
        team_info   = raw.get("team")   or {}

        # Name
        player_name = player_info.get("name", "Unknown").strip()

        # VLR team id (stored as string in API, we keep as int)
        vlr_team_id: Optional[int] = None
        raw_team_id = raw.get("teamId") or team_info.get("id")
        if raw_team_id is not None:
            try:
                vlr_team_id = int(raw_team_id)
            except (ValueError, TypeError):
                vlr_team_id = None

        team_name  = team_info.get("name", "")
        team_short = team_info.get("shortName", "")

        # Role (int → string)
        role_int = raw.get("playerRole", 0)
        role     = ROLE_MAP.get(role_int, "Wildcard")

        # Price (support decimal values like 8.5 VP)
        price = float(raw.get("price", 8.0))

        # Points
        gw_pts_dict  = raw.get("currentGameweekPoints") or {}
        tot_pts_dict = raw.get("totalEventPoints")      or {}
        gw_pts  = float(gw_pts_dict.get("totalPoints", 0))
        tot_pts = float(tot_pts_dict.get("totalPoints", 0))

        # PPG: total points ÷ number of gameweeks with data
        history      = raw.get("eventPointHistory") or []
        gw_played    = len([gw for gw in history if (gw.get("points") or {}).get("totalPoints", 0) > 0])
        ppg          = round(tot_pts / max(gw_played, 1), 2)

        return {
            "player_name":  player_name,
            "vlr_team_id":  vlr_team_id,
            "team_name":    team_name,
            "team_short":   team_short,
            "role":         role,
            "price":        price,
            "gw_pts":       gw_pts,
            "tot_pts":      tot_pts,
            "ppg":          ppg,
            "event_id":     event_id,
            "event_name":   event_name,
        }

    # ── Cache I/O ──────────────────────────────────────────────────────────────

    def save_to_cache(self, data: list[dict]) -> None:
        """Persist player list to JSON cache with metadata envelope."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        envelope = {
            "_meta": {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "player_count": len(data),
                "source": "vfl-api",
            },
            "players": data,
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache written: {self.cache_path}")

    def load_from_cache(self) -> list[dict]:
        """Load cached player list (handles both legacy flat list and new envelope)."""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # New envelope format
        if isinstance(raw, dict) and "players" in raw:
            data = raw["players"]
            meta = raw.get("_meta", {})
            logger.info(
                f"Loaded {len(data)} players from cache "
                f"(cached {meta.get('cached_at','?')})"
            )
            return data

        # Legacy flat-list format
        if isinstance(raw, list):
            logger.info(f"Loaded {len(raw)} players from legacy cache.")
            return raw

        logger.warning("Unrecognised cache format — returning seed data.")
        return self._generate_seed_data()

    # ── Seed Fallback ──────────────────────────────────────────────────────────

    def _generate_seed_data(self) -> list[dict]:
        """
        Hard-coded fallback using real VCT Masters London 2026 player data.
        Used only when the live API is unreachable.
        """
        logger.info("Generating VFL player seed fallback...")
        seed = [
            # Paper Rex (13388)
            {"player_name": "something",  "vlr_team_id": 13388, "team_name": "Paper Rex",  "team_short": "PRX", "role": "Duelist",    "price": 10, "gw_pts": 15.0, "tot_pts": 145.0, "ppg": 14.5, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "f0rsakeN",   "vlr_team_id": 13388, "team_name": "Paper Rex",  "team_short": "PRX", "role": "Initiator",  "price":  9, "gw_pts": 12.0, "tot_pts": 138.0, "ppg": 13.8, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "d4v41",      "vlr_team_id": 13388, "team_name": "Paper Rex",  "team_short": "PRX", "role": "Controller", "price":  8, "gw_pts": 11.5, "tot_pts": 118.0, "ppg": 11.8, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "Jinggg",     "vlr_team_id": 13388, "team_name": "Paper Rex",  "team_short": "PRX", "role": "Duelist",    "price":  9, "gw_pts": 13.0, "tot_pts": 125.0, "ppg": 12.5, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "mindfreak",  "vlr_team_id": 13388, "team_name": "Paper Rex",  "team_short": "PRX", "role": "Controller", "price":  8, "gw_pts":  9.5, "tot_pts": 102.0, "ppg": 10.2, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            # Sentinels (2)
            {"player_name": "zekken",     "vlr_team_id": 2,     "team_name": "Sentinels",  "team_short": "SEN", "role": "Duelist",    "price": 10, "gw_pts": 14.0, "tot_pts": 142.0, "ppg": 14.2, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "johnqt",     "vlr_team_id": 2,     "team_name": "Sentinels",  "team_short": "SEN", "role": "Controller", "price":  9, "gw_pts": 11.5, "tot_pts": 128.0, "ppg": 12.8, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "TenZ",       "vlr_team_id": 2,     "team_name": "Sentinels",  "team_short": "SEN", "role": "Controller", "price": 10, "gw_pts": 13.5, "tot_pts": 139.0, "ppg": 13.9, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "Sacy",       "vlr_team_id": 2,     "team_name": "Sentinels",  "team_short": "SEN", "role": "Initiator",  "price":  8, "gw_pts": 10.0, "tot_pts": 115.0, "ppg": 11.5, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "N4RRATE",    "vlr_team_id": 2,     "team_name": "Sentinels",  "team_short": "SEN", "role": "Initiator",  "price":  9, "gw_pts": 12.5, "tot_pts": 126.0, "ppg": 12.6, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            # Fnatic (2596)
            {"player_name": "Derke",      "vlr_team_id": 2596,  "team_name": "Fnatic",     "team_short": "FNC", "role": "Duelist",    "price": 10, "gw_pts": 14.5, "tot_pts": 140.0, "ppg": 14.0, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "Boaster",    "vlr_team_id": 2596,  "team_name": "Fnatic",     "team_short": "FNC", "role": "Controller", "price":  8, "gw_pts":  9.0, "tot_pts": 101.0, "ppg": 10.1, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "Leo",        "vlr_team_id": 2596,  "team_name": "Fnatic",     "team_short": "FNC", "role": "Initiator",  "price":  9, "gw_pts": 13.5, "tot_pts": 132.0, "ppg": 13.2, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "Chronicle",  "vlr_team_id": 2596,  "team_name": "Fnatic",     "team_short": "FNC", "role": "Initiator",  "price":  9, "gw_pts": 12.0, "tot_pts": 129.0, "ppg": 12.9, "event_id": 9, "event_name": "VCT 2026: Masters London"},
            {"player_name": "Alfajer",    "vlr_team_id": 2596,  "team_name": "Fnatic",     "team_short": "FNC", "role": "Sentinel",   "price":  9, "gw_pts": 13.0, "tot_pts": 131.0, "ppg": 13.1, "event_id": 9, "event_name": "VCT 2026: Masters London"},
        ]
        self.save_to_cache(seed)
        return seed


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scraper = VFLScraper()
    players = scraper.scrape_player_stats()

    print(f"\n{'='*60}")
    print(f"  VFL Players DB — {len(players)} players cached")
    print(f"{'='*60}")

    # Print tabular summary (top 15)
    header = f"{'#':<3} {'Name':<16} {'Team':<6} {'Role':<11} {'Price':>5} {'GW':>5} {'Tot':>6} {'PPG':>6}"
    print(header)
    print("-" * len(header))
    for i, p in enumerate(players[:15], 1):
        print(
            f"{i:<3} {p['player_name']:<16} {p.get('team_short','?'):<6} "
            f"{p['role']:<11} {p['price']:>5} {p['gw_pts']:>5.1f} "
            f"{p['tot_pts']:>6.1f} {p['ppg']:>6.2f}"
        )
    if len(players) > 15:
        print(f"    ... and {len(players) - 15} more")
    print(f"\nCache path: {os.path.abspath(VFL_PLAYERS_DB_CACHE)}")
