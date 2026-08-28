"""
build_global_player_ledger.py
─────────────────────────────
Reads all match JSON files in data/raw/, sorts them chronologically, and
produces data/processed/global_player_ledger.json — the Global Player Entity
Ledger that decouples player performance history from team affiliation.

Design decisions (locked in v5_architecture_proposal.md):
  - Transfer date inferred from first appearance under a new team banner
  - EMA alpha = 0.10 (≈10 effective lookback maps)
  - Cohesion saturation M_sat = 25 maps
  - Maps validated against temporal_map_registry.json (observations on retired
    maps are skipped so they don't pollute active-pool comfort scores)
  - Idempotent: safe to re-run; overwrites the ledger on each run

Run from repo root:
    python build_global_player_ledger.py
"""

import os
import glob
import json
import logging
from datetime import datetime
from utils.match_adapter import normalize_match, parse_match_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ledger_builder")

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
REGISTRY_PATH = os.path.join(PROCESSED_DIR, "temporal_map_registry.json")
LEDGER_PATH = os.path.join(PROCESSED_DIR, "global_player_ledger.json")

EMA_ALPHA = 0.10           # Exponential moving average decay factor
COHESION_SAT_MAPS = 25     # Maps to reach full cohesion score (CF=1.0)


# ---------------------------------------------------------------------------
# Date parser (mirrors v5_simulation_engine.parse_simulation_match_date)
# ---------------------------------------------------------------------------

def parse_match_date(date_str: str) -> datetime:
    if not date_str:
        return datetime(2026, 1, 1)
    clean = date_str.split(" Patch ")[0]
    clean = re.sub(r'\s+[A-Z]{3,4}$', '', clean).strip()
    clean = re.sub(r'^[A-Za-z]+,\s*', '', clean).strip()
    year_match = re.search(r'\b(20\d{2})\b', date_str)
    year = int(year_match.group(1)) if year_match else 2026
    month_day = re.search(r'^([A-Za-z]+)\s+(\d+)', clean)
    if not month_day:
        return datetime(year, 6, 22)
    try:
        return datetime.strptime(f"{month_day.group(1)} {month_day.group(2)}, {year}", "%B %d, %Y")
    except Exception:
        return datetime(year, 6, 22)


# ---------------------------------------------------------------------------
# Registry loader & active-pool resolver
# ---------------------------------------------------------------------------

def load_registry(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_active_pool(match_date: datetime, registry: dict) -> tuple[str, list[str]]:
    """
    Returns (window_id, active_maps) for the given match date.
    Iterates windows in reverse so the most-recent match for late matches is
    correct even when end_date_approx is null.
    """
    windows = registry["patch_windows"]
    for window in reversed(windows):
        start = datetime.fromisoformat(window["start_date_approx"])
        end_str = window.get("end_date_approx")
        end = datetime.fromisoformat(end_str) if end_str else datetime.max
        if start <= match_date < end:
            return window["window_id"], window["active_maps"]
    # Fallback: most recent window
    last = windows[-1]
    return last["window_id"], last["active_maps"]


# ---------------------------------------------------------------------------
# EMA updater
# ---------------------------------------------------------------------------

def update_ema(current_ema: float, new_value: float, count: int, alpha: float = EMA_ALPHA) -> float:
    """
    Warm-start EMA: use simple average for first 5 observations to avoid
    cold-start bias, then switch to EMA.
    """
    if count <= 5:
        return current_ema  # will be recalculated from running_sum / count
    return alpha * new_value + (1 - alpha) * current_ema


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_ledger() -> dict:
    # 1. Load temporal registry
    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(f"Temporal map registry not found at {REGISTRY_PATH}. "
                                 "Run build_temporal_map_registry.py first.")
    registry = load_registry(REGISTRY_PATH)
    logger.info(f"Loaded registry with {len(registry['patch_windows'])} patch windows.")

    # 2. Load and sort all match files chronologically using match_adapter
    files = sorted(glob.glob(os.path.join(RAW_DIR, "match_*.json")))
    matches = []
    skipped = 0
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            norm = normalize_match(content)
            matches.append({
                "date": norm["date"],
                "teams": [
                    {"name": norm["teams"]["team1"]},
                    {"name": norm["teams"]["team2"]}
                ],
                "maps": norm["maps"],
                "performance": norm.get("performance")
            })
        except Exception:
            skipped += 1

    matches.sort(key=lambda x: x["date"])
    logger.info(f"Loaded {len(matches)} matches ({skipped} skipped). Processing chronologically...")

    # 3. Ledger state — keyed by canonical player name (lowercased display name)
    ledger: dict[str, dict] = {}

    def get_or_create_player(name: str) -> dict:
        if name not in ledger:
            ledger[name] = {
                "display_name": name,
                "canonical_id": name,
                "career_stats": {
                    "global_acs_ema": 0.0,
                    "global_kast_ema": 0.72,      # bootstrap prior until match data provides kast
                    "global_duel_diff_ema": 0.0,
                    "global_adr_ema": 0.0,
                    "total_maps_played": 0,
                    "running_acs_sum": 0.0,        # for warm-start EMA
                    "career_start_date": None,
                    "last_updated": None,
                },
                "team_history": [],               # list of {team_name, joined_date, departed_date, maps_played_with_team, acs_with_team}
                "agent_comfort": {},              # {agent: {global_maps, global_acs_avg, running_sum, per_map_comfort: {map: {maps, running_sum, acs_avg}}}}
            }
        return ledger[name]

    def get_current_team(player_entry: dict) -> str | None:
        if not player_entry["team_history"]:
            return None
        return player_entry["team_history"][-1]["team_name"]

    def record_team_appearance(player_entry: dict, team_name: str, match_date: datetime):
        """Infer team transfer from first match appearance under new banner."""
        current = get_current_team(player_entry)
        if current is None:
            # First ever appearance
            player_entry["team_history"].append({
                "team_name": team_name,
                "joined_date": match_date.isoformat(),
                "departed_date": None,
                "maps_played_with_team": 0,
                "acs_with_team_sum": 0.0,
                "acs_with_team_avg": 0.0,
            })
        elif current.lower().strip() != team_name.lower().strip():
            # Transfer detected: close out previous entry
            player_entry["team_history"][-1]["departed_date"] = match_date.isoformat()
            # Open new entry
            player_entry["team_history"].append({
                "team_name": team_name,
                "joined_date": match_date.isoformat(),
                "departed_date": None,
                "maps_played_with_team": 0,
                "acs_with_team_sum": 0.0,
                "acs_with_team_avg": 0.0,
            })
        # Return current (now-active) team entry
        return player_entry["team_history"][-1]

    # 4. Iterate matches chronologically
    total_observations = 0
    skipped_retired_map = 0

    for match in matches:
        match_date = match["date"]
        window_id, active_maps = resolve_active_pool(match_date, registry)

        # Build team name lookup: team1 / team2 → display name
        team_names = {
            "team1": match["teams"][0]["name"],
            "team2": match["teams"][1]["name"],
        }

        for map_data in match["maps"]:
            map_name = map_data.get("map_name", "")

            # Skip observations on maps not active during this window
            if map_name and map_name not in active_maps:
                skipped_retired_map += 1
                continue

            raw_players = map_data.get("players", [])
            all_players = []
            if isinstance(raw_players, list):
                all_players = raw_players
            elif isinstance(raw_players, dict):
                for tk in ["team1", "team2"]:
                    all_players.extend(raw_players.get(tk, []))

            for player_data in all_players:
                p_name = player_data.get("name") or player_data.get("player") or ""
                if not p_name:
                    continue
                team_name = player_data.get("team") or "Unknown"

                agent = player_data.get("agent", "")
                acs_val = player_data.get("acs")
                try:
                    acs_val = float(acs_val) if acs_val is not None else 0.0
                except (ValueError, TypeError):
                    acs_val = 0.0

                if acs_val <= 0:
                    continue  # skip maps with no recorded ACS

                total_observations += 1
                pentry = get_or_create_player(p_name)
                cs = pentry["career_stats"]

                # ── Career stats update ──────────────────────────
                n = cs["total_maps_played"]
                if n == 0:
                    cs["career_start_date"] = match_date.isoformat()
                    cs["global_acs_ema"] = acs_val
                elif n < 5:
                    cs["running_acs_sum"] = cs.get("running_acs_sum", 0.0) + acs_val
                    cs["global_acs_ema"] = cs["running_acs_sum"] / n
                else:
                    cs["global_acs_ema"] = update_ema(cs["global_acs_ema"], acs_val, n)

                cs["running_acs_sum"] = cs.get("running_acs_sum", 0.0) + acs_val
                cs["total_maps_played"] += 1
                cs["last_updated"] = match_date.isoformat()

                # ── Team history / cohesion tracking ─────────────
                team_entry = record_team_appearance(pentry, team_name, match_date)
                team_entry["maps_played_with_team"] += 1
                team_entry["acs_with_team_sum"] = team_entry.get("acs_with_team_sum", 0.0) + acs_val
                maps_w = team_entry["maps_played_with_team"]
                team_entry["acs_with_team_avg"] = team_entry["acs_with_team_sum"] / maps_w

                # ── Agent comfort update ──────────────────────────
                if agent:
                    ac = pentry["agent_comfort"]
                    if agent not in ac:
                        ac[agent] = {
                            "global_maps": 0,
                            "global_acs_avg": 0.0,
                            "_running_sum": 0.0,
                            "per_map_comfort": {},
                        }
                    ag = ac[agent]
                    ag["global_maps"] += 1
                    ag["_running_sum"] += acs_val
                    ag["global_acs_avg"] = ag["_running_sum"] / ag["global_maps"]

                # Per-map comfort
                if map_name:
                    pmc = ag["per_map_comfort"]
                    if map_name not in pmc:
                        pmc[map_name] = {"maps": 0, "_running_sum": 0.0, "acs_avg": 0.0}
                    pmc[map_name]["maps"] += 1
                    pmc[map_name]["_running_sum"] += acs_val
                    pmc[map_name]["acs_avg"] = pmc[map_name]["_running_sum"] / pmc[map_name]["maps"]

    logger.info(f"Processed {total_observations} player-map observations.")
    logger.info(f"Skipped {skipped_retired_map} observations on retired/inactive maps.")
    logger.info(f"Built ledger for {len(ledger)} unique players.")

    # 5. Clean up internal running sums from output (keep schema clean)
    for p_name, pentry in ledger.items():
        cs = pentry["career_stats"]
        cs.pop("running_acs_sum", None)
        for agent, ag in pentry["agent_comfort"].items():
            ag.pop("_running_sum", None)
            for map_name, pmc in ag["per_map_comfort"].items():
                pmc.pop("_running_sum", None)

    # 6. Compute cohesion scores for current team window
    for p_name, pentry in ledger.items():
        for th in pentry["team_history"]:
            maps_w = th.get("maps_played_with_team", 0)
            th["cohesion_score"] = round(min(maps_w, COHESION_SAT_MAPS) / COHESION_SAT_MAPS, 4)
            # Remove internal sum from output
            th.pop("acs_with_team_sum", None)

    return {
        "_schema_version": "1.0",
        "_generated_at": datetime.utcnow().isoformat() + "Z",
        "_total_players": len(ledger),
        "_total_observations": total_observations,
        "_skipped_retired_map_observations": skipped_retired_map,
        "_ema_alpha": EMA_ALPHA,
        "_cohesion_saturation_maps": COHESION_SAT_MAPS,
        "players": ledger,
    }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Global Player Entity Ledger Builder — V5 Architecture")
    logger.info("=" * 60)
    ledger_output = build_ledger()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Ledger written to {LEDGER_PATH}")
    logger.info(f"  Players indexed: {ledger_output['_total_players']}")
    logger.info(f"  Total observations: {ledger_output['_total_observations']}")
    logger.info(f"  Retired-map observations skipped: {ledger_output['_skipped_retired_map_observations']}")

    # Quick spot-check: print top 5 players by maps played
    players = ledger_output["players"]
    top5 = sorted(players.items(), key=lambda x: x[1]["career_stats"]["total_maps_played"], reverse=True)[:5]
    logger.info("\nTop 5 players by maps played:")
    for name, p in top5:
        cs = p["career_stats"]
        current_team = p["team_history"][-1]["team_name"] if p["team_history"] else "Unknown"
        cf = p["team_history"][-1].get("cohesion_score", 0.0) if p["team_history"] else 0.0
        logger.info(
            f"  {name:20s} | maps={cs['total_maps_played']:4d} | "
            f"acs_ema={cs['global_acs_ema']:6.1f} | team={current_team} | CF={cf:.2f}"
        )
