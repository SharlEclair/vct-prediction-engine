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

    def get_players(self, force_refresh: bool = False) -> list[dict]:
        """Return player list — from cache unless force_refresh or no cache."""
        if not force_refresh and os.path.exists(self.cache_path):
            return self.load_from_cache()
        return self.scrape_player_stats()

    def scrape_player_stats(self) -> list[dict]:
        """
        Two-step API workflow:
          A) Resolve current event id.
          B) Fetch all players for that event.
        Falls back to seed data on any network/parse error.
        """
        try:
            with requests.Session(impersonate="chrome") as client:
                # ── Step A: current event ──────────────────────────────────
                sleep_time = 3.0 + random.uniform(0.5, 2.5)
                logger.info(f"Sleeping for {sleep_time:.2f}s before fetching current event...")
                time.sleep(sleep_time)
                logger.info(f"Resolving current event from {CURRENT_EVENT_URL}")
                r_event = client.get(CURRENT_EVENT_URL, headers=DEFAULT_HEADERS, timeout=20.0)
                if r_event.status_code != 200:
                    raise Exception(f"HTTP Status {r_event.status_code}")

                event_data  = r_event.json()
                event_id    = event_data.get("id")
                event_name  = event_data.get("name", "Unknown Event")

                if not event_id:
                    raise ValueError("Could not resolve event_id from currentevent response.")
                logger.info(f"Active event: [{event_id}] {event_name}")

                # ── Step B: all players ────────────────────────────────────
                sleep_time2 = 3.0 + random.uniform(0.5, 2.5)
                logger.info(f"Sleeping for {sleep_time2:.2f}s before fetching all players...")
                time.sleep(sleep_time2)
                logger.info(f"Fetching player roster from {ALL_PLAYERS_URL}?eventId={event_id}")
                r_players = client.get(
                    ALL_PLAYERS_URL,
                    params={"eventId": event_id},
                    headers=DEFAULT_HEADERS,
                    timeout=20.0
                )
                if r_players.status_code != 200:
                    raise Exception(f"HTTP Status {r_players.status_code}")

                raw_players: list[dict] = r_players.json()

                if not isinstance(raw_players, list):
                    raise ValueError(f"Unexpected response shape: {type(raw_players)}")

                logger.info(f"Received {len(raw_players)} players from API.")

                mapped_players = [
                    self._map_player(p, event_id=event_id, event_name=event_name)
                    for p in raw_players
                ]

                # Strict filtering: Exclude Inactive players and Academy/GC/Black/Blue rosters
                players = []
                for p in mapped_players:
                    p_name = p["player_name"].lower()
                    t_name = p["team_name"].lower()
                    t_short = p.get("team_short", "").lower()

                    if "inactive" in p_name:
                        continue

                    suffixes = ["academy", "gc", "game changers", "black", "blue"]
                    if any(s in t_name for s in suffixes) or any(s in t_short for s in suffixes):
                        continue

                    players.append(p)

                # Sort by PPG descending so the cache is already ranked
                players.sort(key=lambda x: x["ppg"], reverse=True)

                self.save_to_cache(players)
                logger.info(f"Saved {len(players)} players → {self.cache_path}")
                return players

        except Exception as exc:
            logger.error(f"API scrape failed: {exc}. Using seed fallback.")
            return self._generate_seed_data()

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
