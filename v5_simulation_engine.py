import os
import json
import glob
import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger("v5_simulation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"


# ---------------------------------------------------------------------------
# Temporal Map Pool Registry
# ---------------------------------------------------------------------------

TEMPORAL_MAP_POOLS = [
    {
        "end_date": "2024-10-21",
        "pool": ["Abyss", "Ascent", "Bind", "Haven", "Icebox", "Lotus", "Sunset"]
    },
    {
        "end_date": "2025-01-06",
        "pool": ["Abyss", "Ascent", "Bind", "Haven", "Pearl", "Split", "Sunset"]
    },
    {
        "end_date": "2025-03-03",
        "pool": ["Abyss", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
    },
    {
        "end_date": "2025-06-23",
        "pool": ["Ascent", "Fracture", "Haven", "Icebox", "Lotus", "Pearl", "Split"]
    },
    {
        "end_date": "2025-08-18",
        "pool": ["Ascent", "Bind", "Corrode", "Fracture", "Haven", "Icebox", "Lotus"]
    },
    {
        "end_date": "2025-10-13",
        "pool": ["Abyss", "Ascent", "Bind", "Corrode", "Fracture", "Haven", "Lotus"]
    },
    {
        "end_date": "2026-01-05",
        "pool": ["Abyss", "Bind", "Corrode", "Fracture", "Haven", "Pearl", "Split"]
    },
    {
        "end_date": "2026-03-16",
        "pool": ["Abyss", "Bind", "Breeze", "Corrode", "Haven", "Pearl", "Split"]
    },
    {
        "end_date": "2026-04-27",
        "pool": ["Bind", "Breeze", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
    },
    {
        "end_date": "2026-06-22",
        "pool": ["Ascent", "Breeze", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
    },
    {
        "end_date": "2026-12-31",
        "pool": ["Ascent", "Breeze", "Haven", "Lotus", "Split", "Summit", "Sunset"]
    }
]

class TemporalMapRegistry:
    """
    Resolves the 7-map VCT competitive pool that was active on any given date.
    Loaded from data/processed/temporal_map_registry.json or TEMPORAL_MAP_POOLS fallback.

    Design: stateless lookup — no mutable state after __init__.
    """
    _FALLBACK_POOL = ["Ascent", "Bind", "Haven", "Icebox", "Lotus", "Abyss", "Sunset"]

    def __init__(self, processed_dir: str = PROCESSED_DIR):
        self.windows = []
        self.re_entry_decay_rho = 0.65
        registry_path = os.path.join(processed_dir, "temporal_map_registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.windows = data.get("patch_windows", [])
                self.re_entry_decay_rho = data.get("re_entry_decay_rho", 0.65)
                logger.info(f"TemporalMapRegistry loaded: {len(self.windows)} patch windows.")
            except Exception as e:
                logger.warning(f"TemporalMapRegistry: failed to load registry ({e}). Using fallback pool.")
        else:
            logger.warning("TemporalMapRegistry: registry file not found. Using static fallback pool.")

    def resolve_pool(self, match_date: datetime) -> tuple[str, list[str]]:
        """
        Returns (window_id, active_maps_list) for the given match_date.
        Cross-references match_date against TEMPORAL_MAP_POOLS and patch windows.
        """
        date_str = match_date.strftime("%Y-%m-%d") if isinstance(match_date, datetime) else str(match_date)[:10]
        for entry in TEMPORAL_MAP_POOLS:
            if date_str <= entry["end_date"]:
                return f"temporal_{entry['end_date']}", entry["pool"]

        if not self.windows:
            return "fallback", self._FALLBACK_POOL
        for window in reversed(self.windows):
            start = datetime.fromisoformat(window["start_date_approx"])
            end_str = window.get("end_date_approx")
            end = datetime.fromisoformat(end_str) if end_str else datetime.max
            if start <= match_date < end:
                return window["window_id"], window["active_maps"]
        earliest = self.windows[0]
        return earliest["window_id"], earliest["active_maps"]

    def is_map_active(self, map_name: str, match_date: datetime) -> bool:
        _, pool = self.resolve_pool(match_date)
        return map_name in pool

    def resolve_current_pool(self) -> list[str]:
        """Convenience: resolve pool for today (used by predict_veto default)."""
        return self.resolve_pool(datetime.now())[1]

    def get_window_id(self, match_date: datetime) -> str:
        return self.resolve_pool(match_date)[0]


# Module-level singleton — shared by MapVetoBandit and VCTv5SimulationEngine
_temporal_registry: TemporalMapRegistry | None = None

def _get_registry(processed_dir: str = PROCESSED_DIR) -> TemporalMapRegistry:
    global _temporal_registry
    if _temporal_registry is None:
        _temporal_registry = TemporalMapRegistry(processed_dir)
    return _temporal_registry

# --- Math Sub-models ---

class MapVetoBandit:
    """
    Sub-Model 1: Multi-armed Contextual Bandit for Map Vetoes.
    Uses Inverse Propensity Score (IPS) off-policy evaluation to estimate unbiased map win-rates
    and simulates the pick/ban sequence.

    V5 upgrade: map pool is resolved from TemporalMapRegistry, not hardcoded.
    IPS propensity is computed per-window (not globally) to avoid retired-map
    frequency pollution.
    """
    # All known VCT maps across all time — used only as the superset when
    # accumulating raw data; active arms are pruned per-date at predict-time.
    _ALL_MAPS = [
        "Abyss", "Ascent", "Bind", "Breeze", "Fracture",
        "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset",
    ]

    def __init__(self, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.registry = _get_registry(processed_dir)
        # Legacy flat pool kept for callers that inspect self.map_pool
        self.map_pool = self.registry.resolve_current_pool()
        # Per-window IPS accumulators: {window_id: {map: {plays, wins}}}
        self.window_plays: dict[str, dict[str, dict]] = {}
        # Cross-window aggregated team stats (date-aware build)
        self.team_plays: dict[str, dict[str, int]] = {}
        self.team_wins:  dict[str, dict[str, int]] = {}
        # Per-window propensity (frequency of play within window)
        self.window_propensity: dict[str, dict[str, float]] = {}
        # Global fallback propensity over the current active pool
        self.map_frequency: dict[str, float] = {}
        self.fit()

    def fit(self):
        """
        Build per-window IPS accumulators and cross-window team stats.
        Each match observation is tagged with its temporal window so that
        propensity scores are not polluted by retired-map frequencies.
        """
        files = glob.glob(os.path.join(self.raw_dir, "match_*.json"))

        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    content = json.load(file)
                if "data" not in content or "segments" not in content["data"] or not content["data"]["segments"]:
                    continue
                seg = content["data"]["segments"][0]
                if len(seg.get("teams", [])) < 2:
                    continue

                team_a = seg["teams"][0]["name"]
                team_b = seg["teams"][1]["name"]
                match_date = parse_simulation_match_date(seg.get("date", ""))
                window_id, active_maps = self.registry.resolve_pool(match_date)

                # Initialise per-window bucket
                if window_id not in self.window_plays:
                    self.window_plays[window_id] = {m: {"plays": 0, "wins_a": 0} for m in self._ALL_MAPS}

                # Initialise team trackers (over ALL maps — pruned at predict-time)
                for team in (team_a, team_b):
                    if team not in self.team_plays:
                        self.team_plays[team] = {m: 0 for m in self._ALL_MAPS}
                        self.team_wins[team] = {m: 0 for m in self._ALL_MAPS}

                for map_data in seg.get("maps", []):
                    m_name = map_data.get("map_name")
                    if not m_name or m_name not in active_maps:
                        # Retired/inactive map for this match's window — skip
                        continue

                    # Per-window play count (for propensity)
                    self.window_plays[window_id][m_name]["plays"] += 1

                    # Team-level accumulation
                    self.team_plays[team_a][m_name] = self.team_plays[team_a].get(m_name, 0) + 1
                    self.team_plays[team_b][m_name] = self.team_plays[team_b].get(m_name, 0) + 1

                    score = map_data.get("score", {})
                    t1_score = score.get("team1")
                    t2_score = score.get("team2")
                    if t1_score is not None and t2_score is not None:
                        if t1_score > t2_score:
                            self.team_wins[team_a][m_name] = self.team_wins[team_a].get(m_name, 0) + 1
                        elif t2_score > t1_score:
                            self.team_wins[team_b][m_name] = self.team_wins[team_b].get(m_name, 0) + 1

            except Exception:
                pass

        # Build per-window propensity scores
        for wid, map_buckets in self.window_plays.items():
            total_w = sum(b["plays"] for b in map_buckets.values()) + 1
            self.window_propensity[wid] = {
                m: (b["plays"] + 1) / (total_w + len(self._ALL_MAPS))
                for m, b in map_buckets.items()
            }

        # Global fallback propensity for the CURRENT active pool only
        current_pool = self.registry.resolve_current_pool()
        _, latest_window_id = self.registry.get_window_id(datetime.now()), None
        current_wid = self.registry.get_window_id(datetime.now())
        if current_wid in self.window_propensity:
            self.map_frequency = {
                m: self.window_propensity[current_wid].get(m, 0.05)
                for m in current_pool
            }
        else:
            self.map_frequency = {m: 1.0 / len(current_pool) for m in current_pool}

        # Update self.map_pool to always reflect current active pool
        self.map_pool = current_pool
        logger.info(f"MapVetoBandit fitted: {len(self.window_plays)} temporal windows, "
                    f"current pool = {self.map_pool}")

    def predict_map_win_rate_dr(self, team: str, opponent: str, map_name: str,
                                 target_date: datetime | None = None) -> float:
        """
        Estimates win rate on a map using Doubly Robust (DR) estimation.

        Propensity is sourced from the window matching target_date so that
        retired maps never inflate the denominator for active maps.
        If target_date is None, uses the current active window propensity.
        """
        if team not in self.team_plays:
            return 0.5

        plays = self.team_plays[team].get(map_name, 0)
        wins  = self.team_wins[team].get(map_name, 0)

        # Resolve propensity from the correct temporal window
        if target_date is not None:
            wid = self.registry.get_window_id(target_date)
            propensity = self.window_propensity.get(wid, {}).get(map_name, 0.1)
        else:
            propensity = self.map_frequency.get(map_name, 0.1)

        empirical_win_rate = wins / plays if plays > 0 else 0.5
        baseline_mu = (wins + 1.0) / (plays + 2.0) if plays > 0 else 0.5
        
        epsilon = 1e-5
        raw_dr = baseline_mu + (empirical_win_rate - baseline_mu) / (propensity + epsilon)
        return float(np.clip(raw_dr * 0.8 + 0.1, 0.1, 0.9))

    def predict_veto(self, team_a: str, team_b: str, series_type: str = "Bo3",
                     stochastic: bool = False,
                     target_date: datetime | None = None,
                     ub_advantage: bool | str = False,
                     veto_priority: str = "team_a") -> dict:
        """
        Simulates veto picks/bans using official strict VCT sequences.
        Supports target_date, ub_advantage, and tactical seat selection for the priority team.
        """
        if target_date is None:
            target_date = datetime.now()
            
        # Determine who acts as Team A and Team B in the veto sequence.
        # "team_a" priority means team_a chooses. "team_b" means team_b chooses.
        # "random" means 50/50 skirmish.
        acting_team_a = team_a
        acting_team_b = team_b
        
        if veto_priority == "team_b" or (ub_advantage and str(ub_advantage).lower() == team_b.lower()):
            priority_team = team_b
            other_team = team_a
            has_priority = True
        elif veto_priority == "random":
            import random
            if random.random() < 0.5:
                priority_team = team_a
                other_team = team_b
            else:
                priority_team = team_b
                other_team = team_a
            has_priority = True
        else:
            priority_team = team_a
            other_team = team_b
            has_priority = (veto_priority == "team_a" or ub_advantage is True or str(ub_advantage).lower() == team_a.lower())

        if has_priority:
            # Tactical seat choice: compare expected map win rate if priority_team is Team A vs Team B.
            # Case 1: priority_team acts as Team A, other_team acts as Team B
            res_a = self._simulate_strict_veto_sequence(priority_team, other_team, series_type, stochastic, target_date, ub_advantage)
            avg_wr_a = np.mean([self.predict_map_win_rate_dr(priority_team, other_team, m, target_date) for m in res_a["maps"]])
            
            # Case 2: other_team acts as Team A, priority_team acts as Team B
            res_b = self._simulate_strict_veto_sequence(other_team, priority_team, series_type, stochastic, target_date, ub_advantage)
            avg_wr_b = np.mean([self.predict_map_win_rate_dr(priority_team, other_team, m, target_date) for m in res_b["maps"]])
            
            if avg_wr_a >= avg_wr_b:
                acting_team_a = priority_team
                acting_team_b = other_team
                veto_res = res_a
            else:
                acting_team_a = other_team
                acting_team_b = priority_team
                veto_res = res_b
        else:
            veto_res = self._simulate_strict_veto_sequence(acting_team_a, acting_team_b, series_type, stochastic, target_date, ub_advantage)

        # Map side choice to starting_side_a ("DEF" or "ATK" for team_a)
        # veto_res["side_choices"] has the team name choosing the starting side on each map.
        starting_sides_a = []
        for idx, m_name in enumerate(veto_res["maps"]):
            chooser = veto_res["side_choices"][idx]
            bias_info = MAP_SIDE_BIAS.get(m_name, {"type": "NEUTRAL", "bias": 0.0})
            preferred_side = bias_info["type"]
            if preferred_side == "NEUTRAL":
                preferred_side = "DEF"
                
            if chooser.lower().strip() == team_a.lower().strip():
                # Team A chooses side, so Team A starts on their preferred side
                starting_sides_a.append(preferred_side)
            else:
                # Team B chooses side, so Team B starts on their preferred side, Team A gets opposite
                starting_sides_a.append("ATK" if preferred_side == "DEF" else "DEF")

        return {
            "maps": veto_res["maps"],
            "veto_weights": veto_res["veto_weights"],
            "veto_str": f"Seat Selection: {acting_team_a} acts as Team A, {acting_team_b} acts as Team B; Veto: " + veto_res["veto_str"],
            "starting_sides_a": starting_sides_a
        }

    def _simulate_strict_veto_sequence(self, team_a: str, team_b: str, series_type: str,
                                       stochastic: bool, target_date: datetime, ub_advantage: bool | str = False) -> dict:
        _, active_pool = self.registry.resolve_pool(target_date)
        available_maps = list(active_pool)
        
        scores_a = {m: self.predict_map_win_rate_dr(team_a, team_b, m, target_date) for m in available_maps}
        scores_b = {m: self.predict_map_win_rate_dr(team_b, team_a, m, target_date) for m in available_maps}
        
        if stochastic:
            scores_a = {m: val + np.random.normal(0, 0.05) for m, val in scores_a.items()}
            scores_b = {m: val + np.random.normal(0, 0.05) for m, val in scores_b.items()}
            
        banned_maps = []
        picked_maps = []
        side_choices = []
        veto_weights = {}
        veto_steps = []
        
        if series_type == "Bo1":
            # BO1 Veto: A Ban 1 -> B Ban 1 -> A Ban 2 -> B Ban 2 -> A Ban 3 -> B Picks Map 1 -> A Picks Side
            # Ban 1: Team A
            m_ban_a1 = min(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_ban_a1)
            banned_maps.append(m_ban_a1)
            veto_steps.append(f"{team_a} ban {m_ban_a1}")
            
            # Ban 2: Team B
            m_ban_b1 = min(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_ban_b1)
            banned_maps.append(m_ban_b1)
            veto_steps.append(f"{team_b} ban {m_ban_b1}")
            
            # Ban 3: Team A
            m_ban_a2 = min(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_ban_a2)
            banned_maps.append(m_ban_a2)
            veto_steps.append(f"{team_a} ban {m_ban_a2}")
            
            # Ban 4: Team B
            m_ban_b2 = min(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_ban_b2)
            banned_maps.append(m_ban_b2)
            veto_steps.append(f"{team_b} ban {m_ban_b2}")
            
            # Ban 5: Team A
            m_ban_a3 = min(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_ban_a3)
            banned_maps.append(m_ban_a3)
            veto_steps.append(f"{team_a} ban {m_ban_a3}")
            
            # Pick 1: Team B
            m_pick_b = max(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_pick_b)
            picked_maps.append(m_pick_b)
            veto_weights[m_pick_b] = -1
            side_choices.append(team_a) # Team A chooses side
            veto_steps.append(f"{team_b} pick {m_pick_b}")
            
        elif series_type == "Bo5":
            # BO5 Veto: A Ban 1 -> B Ban 1 -> A Picks Map 1 -> B Picks Map 2 -> A Picks Map 3 -> B Picks Map 4 -> Map 5 Remains
            # Ban 1: Team A
            m_ban_a = min(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_ban_a)
            banned_maps.append(m_ban_a)
            veto_steps.append(f"{team_a} ban {m_ban_a}")
            
            # Ban 2: Team B
            m_ban_b = min(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_ban_b)
            banned_maps.append(m_ban_b)
            veto_steps.append(f"{team_b} ban {m_ban_b}")
            
            # Pick 1: Team A
            m_pick_a1 = max(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_pick_a1)
            picked_maps.append(m_pick_a1)
            veto_weights[m_pick_a1] = 1
            side_choices.append(team_b) # Team B chooses side
            veto_steps.append(f"{team_a} pick {m_pick_a1}")
            
            # Pick 2: Team B
            m_pick_b1 = max(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_pick_b1)
            picked_maps.append(m_pick_b1)
            veto_weights[m_pick_b1] = -1
            side_choices.append(team_a) # Team A chooses side
            veto_steps.append(f"{team_b} pick {m_pick_b1}")
            
            # Pick 3: Team A
            m_pick_a2 = max(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_pick_a2)
            picked_maps.append(m_pick_a2)
            veto_weights[m_pick_a2] = 1
            side_choices.append(team_b) # Team B chooses side
            veto_steps.append(f"{team_a} pick {m_pick_a2}")
            
            # Pick 4: Team B
            m_pick_b2 = max(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_pick_b2)
            picked_maps.append(m_pick_b2)
            veto_weights[m_pick_b2] = -1
            side_choices.append(team_a) # Team A chooses side
            veto_steps.append(f"{team_b} pick {m_pick_b2}")
            
            # Decider: Map 5 remains
            if available_maps:
                m_decider = available_maps[0]
                picked_maps.append(m_decider)
                veto_weights[m_decider] = 0
                side_choices.append(team_b) # Team B chooses side
                veto_steps.append(f"{m_decider} remains")
                
        else: # Default: Bo3
            # BO3 Veto: A Ban 1 -> B Ban 1 -> A Picks Map 1 -> B Picks Map 2 -> A Ban 2 -> B Ban 2 -> Map 3 Remains
            # Ban 1: Team A
            m_ban_a1 = min(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_ban_a1)
            banned_maps.append(m_ban_a1)
            veto_steps.append(f"{team_a} ban {m_ban_a1}")
            
            # Ban 2: Team B
            m_ban_b1 = min(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_ban_b1)
            banned_maps.append(m_ban_b1)
            veto_steps.append(f"{team_b} ban {m_ban_b1}")
            
            # Pick 1: Team A
            m_pick_a = max(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_pick_a)
            picked_maps.append(m_pick_a)
            veto_weights[m_pick_a] = 1
            side_choices.append(team_b) # Team B chooses side
            veto_steps.append(f"{team_a} pick {m_pick_a}")
            
            # Pick 2: Team B
            m_pick_b = max(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_pick_b)
            picked_maps.append(m_pick_b)
            veto_weights[m_pick_b] = -1
            side_choices.append(team_a) # Team A chooses side
            veto_steps.append(f"{team_b} pick {m_pick_b}")
            
            # Ban 3: Team A
            m_ban_a2 = min(available_maps, key=lambda m: scores_a[m])
            available_maps.remove(m_ban_a2)
            banned_maps.append(m_ban_a2)
            veto_steps.append(f"{team_a} ban {m_ban_a2}")
            
            # Ban 4: Team B
            m_ban_b2 = min(available_maps, key=lambda m: scores_b[m])
            available_maps.remove(m_ban_b2)
            banned_maps.append(m_ban_b2)
            veto_steps.append(f"{team_b} ban {m_ban_b2}")
            
            # Decider: Map 3 remains
            if available_maps:
                m_decider = available_maps[0]
                picked_maps.append(m_decider)
                veto_weights[m_decider] = 0
                side_choices.append(team_a) # Team A chooses side
                veto_steps.append(f"{m_decider} remains")
                
        return {
            "maps": picked_maps,
            "veto_weights": veto_weights,
            "veto_str": "; ".join(veto_steps),
            "side_choices": side_choices
        }


class SynergisticDraftEngine:
    """
    Sub-Model 2: Synergistic Combinatorial Draft Engine.
    Maximizes multi-factor utility (base comfort + historical pick rates) combined with
    composition-wide synergy modifiers (e.g. Duelist+Initiator bonus, missing role penalties)
    using a fast, exact search over players' top 5 comfortable agents.
    """
    def __init__(self, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.agent_roles = {}
        self.jsd_matrix = {}
        self.nerf_registry = {}
        self.agent_comfort_matrix = {}   # legacy fallback
        self.player_global_stats = {}    # legacy fallback
        self.player_ledger: dict = {}    # V5: global player entity ledger

        self.load_configurations()

    def load_configurations(self):
        roles_path = os.path.join(self.raw_dir, "agent_roles.json")
        if os.path.exists(roles_path):
            with open(roles_path, "r", encoding="utf-8") as f:
                self.agent_roles = json.load(f)

        jsd_path = os.path.join(self.processed_dir, "patch_distance_matrix.json")
        if os.path.exists(jsd_path):
            with open(jsd_path, "r", encoding="utf-8") as f:
                self.jsd_matrix = json.load(f)

        nerfs_path = os.path.join(self.processed_dir, "automated_patch_nerf_registry.json")
        if os.path.exists(nerfs_path):
            with open(nerfs_path, "r", encoding="utf-8") as f:
                self.nerf_registry = json.load(f)

        # V5: load global player ledger (team-decoupled career stats)
        ledger_path = os.path.join(self.processed_dir, "global_player_ledger.json")
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.player_ledger = data.get("players", {})
                logger.info(f"SynergisticDraftEngine: loaded ledger with "
                            f"{len(self.player_ledger)} players.")
            except Exception as e:
                logger.warning(f"SynergisticDraftEngine: ledger load failed ({e}). "
                               f"Falling back to legacy comfort matrix.")
        else:
            logger.warning("SynergisticDraftEngine: global_player_ledger.json not found. "
                           "Run build_global_player_ledger.py to generate it.")

    def fit_comfort(self, player_agent_stats, player_global_stats=None):
        """Legacy fallback path — kept for backward compatibility with predict_match.py."""
        self.agent_comfort_matrix = player_agent_stats
        self.player_global_stats = player_global_stats or {}

    def resolve_comfort(self, player: str, map_name: str, agent: str) -> tuple[float, int]:
        """
        Resolve Bayesian-smoothed ACS comfort from the global player ledger.
        Returns (bayesian_acs, observation_count).
        Falls back to legacy agent_comfort_matrix if ledger has no data for this player.
        """
        pdata = self.player_ledger.get(player)
        if pdata:
            global_acs = pdata["career_stats"].get("global_acs_ema", 200.0)
            agent_data = pdata.get("agent_comfort", {}).get(agent, {})
            map_specific = agent_data.get("per_map_comfort", {}).get(map_name)

            if map_specific and map_specific["maps"] >= 3:
                map_acs = map_specific["acs_avg"]
                count = map_specific["maps"]
            elif agent_data.get("global_maps", 0) > 0:
                map_acs = agent_data["global_acs_avg"]
                count = agent_data["global_maps"]
            else:
                # Agent never played by this player in ledger
                return global_acs * 0.9, 0  # mild penalty for untested agent

            alpha_smooth = 3.0
            bayesian_acs = (count * map_acs + alpha_smooth * global_acs) / (count + alpha_smooth)
            return bayesian_acs, count

        # Fallback: legacy in-memory dict
        comfort_stat = self.agent_comfort_matrix.get((player, map_name, agent), {"sum_acs": 0.0, "count": 0})
        count = comfort_stat.get("count", 0)
        acs = comfort_stat["sum_acs"] / count if count > 0 else 0.0
        global_stat = self.player_global_stats.get(player, {"sum_acs": 0.0, "count": 0})
        global_acs = global_stat["sum_acs"] / global_stat["count"] if global_stat.get("count", 0) > 0 else 200.0
        if count > 0:
            alpha_smooth = 3.0
            bayesian_acs = (count * acs + alpha_smooth * global_acs) / (count + alpha_smooth)
        else:
            bayesian_acs = global_acs * 0.9
        return bayesian_acs, count

    def get_player_global_acs(self, player: str) -> float:
        """Returns global career ACS EMA from ledger, or legacy fallback."""
        pdata = self.player_ledger.get(player)
        if pdata:
            return pdata["career_stats"].get("global_acs_ema", 200.0)
        global_stat = self.player_global_stats.get(player, {"sum_acs": 0.0, "count": 0})
        if global_stat.get("count", 0) > 0:
            return global_stat["sum_acs"] / global_stat["count"]
        return 200.0

    def predict_composition(self, team_name: str, map_name: str, roster: list[str],
                             target_patch: str = "9.02", temperature: float = 25.0) -> list[str]:
        """
        Optimal draft selection maximizing utility and synergistic compositions.
        """
        agents_pool = list(self.agent_roles.keys())
        if not agents_pool:
            agents_pool = ["Jett", "Raze", "Omen", "Breach", "Killjoy", "Sova", "Cypher", "Sage", "Viper", "Phoenix"]

        n_players = min(5, len(roster))
        
        # Precompute individual player agent utility vectors
        player_choices = []
        for i in range(n_players):
            player = roster[i]
            player_global_acs = self.get_player_global_acs(player)
            if player_global_acs <= 0:
                player_global_acs = 200.0

            # Total map matches for pick-rate denominator (from ledger or fallback)
            pdata = self.player_ledger.get(player)
            if pdata:
                total_map_matches = sum(
                    pdata.get("agent_comfort", {}).get(a, {}).get("per_map_comfort", {}).get(map_name, {}).get("maps", 0)
                    for a in agents_pool
                )
            else:
                total_map_matches = sum(
                    self.agent_comfort_matrix.get((player, map_name, a), {}).get("count", 0)
                    for a in agents_pool
                )

            agent_utils = []
            for agent in agents_pool:
                base_comfort, count = self.resolve_comfort(player, map_name, agent)
                
                # Nerf penalty
                nerfs = self.nerf_registry.get(target_patch, {})
                nerf_penalty = nerfs.get(agent, 0.0)
                comfort_score = base_comfort - 100.0 * nerf_penalty
                normalized_comfort = comfort_score / player_global_acs

                # Historical pick rate
                historical_pick_rate = count / total_map_matches if total_map_matches > 0 else 0.0

                # Base multi-factor utility
                utility = 0.3 * normalized_comfort + 0.7 * historical_pick_rate
                
                # Explorable noise injection
                utility += float(np.random.normal(0, 0.05))
                
                agent_utils.append((agent, utility))
                
            # Keep only the top 5 comfortable agents for search space reduction & performance stability
            agent_utils.sort(key=lambda x: x[1], reverse=True)
            player_choices.append(agent_utils[:5])

        # Combinatorial search over player choice combinations to optimize Synergy Multipliers
        best_composition = None
        best_utility = -99999.0

        def search(player_idx, current_agents, current_utility):
            nonlocal best_composition, best_utility
            if player_idx == n_players:
                # Calculate composition synergy modifiers
                roles = [self.agent_roles.get(a, 'Sentinel') for a in current_agents]
                has_duelist = 'Duelist' in roles
                has_initiator = 'Initiator' in roles
                has_controller = 'Controller' in roles
                has_sentinel = 'Sentinel' in roles

                synergy_mult = 1.0
                if has_duelist and has_initiator:
                    synergy_mult *= 1.10
                if not has_controller:
                    synergy_mult *= 0.85
                if not has_sentinel:
                    synergy_mult *= 0.85

                final_val = current_utility * synergy_mult
                if final_val > best_utility:
                    best_utility = final_val
                    best_composition = list(current_agents)
                return

            for agent, util in player_choices[player_idx]:
                if agent not in current_agents:
                    current_agents.append(agent)
                    search(player_idx + 1, current_agents, current_utility + util)
                    current_agents.pop()

        search(0, [], 0.0)
        
        # Fallback to simple unique assignment if combinatorial search fails
        if best_composition is None or len(best_composition) < n_players:
            best_composition = []
            for choices in player_choices:
                for agent, _ in choices:
                    if agent not in best_composition:
                        best_composition.append(agent)
                        break
            while len(best_composition) < n_players:
                best_composition.append(agents_pool[len(best_composition) % len(agents_pool)])

        return best_composition


MAP_SIDE_BIAS = {
    "Ascent": {"type": "DEF", "bias": 0.044},
    "Split": {"type": "DEF", "bias": 0.030},
    "Summit": {"type": "DEF", "bias": 0.022},
    "Lotus": {"type": "ATK", "bias": 0.028},
    "Abyss": {"type": "ATK", "bias": 0.043},
}

class StatefulEconomySimulator:
    """
    Sub-Model 3: Stateful Economy Simulator.
    Tracks scoring, team-level loss-streaks, dynamic credit injections (loss bonuses),
    halftime/OT resets, and weapon saving survival penalties.
    """
    def __init__(self, team_a_stats: dict | float = 200.0, team_b_stats: dict | float = 200.0, map_name: str = "Ascent", starting_side_a: str = "DEF"):
        if isinstance(team_a_stats, (int, float)):
            self.acs_a = float(team_a_stats)
        else:
            self.acs_a = float(team_a_stats.get("acs", 200.0))
            
        if isinstance(team_b_stats, (int, float)):
            self.acs_b = float(team_b_stats)
        else:
            self.acs_b = float(team_b_stats.get("acs", 200.0))
            
        self.map_name = map_name
        self.starting_side_a = starting_side_a
        
    def get_side_advantage_a(self, current_side_a: str) -> float:
        """Computes side advantage multiplier for Team A based on current side and map bias."""
        info = MAP_SIDE_BIAS.get(self.map_name, {"type": "NEUTRAL", "bias": 0.0})
        if info["type"] == "NEUTRAL" or info["bias"] == 0.0:
            return 0.0
            
        if current_side_a == info["type"]:
            return info["bias"]
        else:
            return -info["bias"]

    def simulate_rounds(self, rate_a: float = None, rate_b: float = None) -> tuple[int, int]:
        """
        Runs discrete-time round simulation from Round 1 up to terminal state (13 wins or OT).
        Tracks scores, loss streaks, and stateful credit balances.
        """
        score_a = 0
        score_b = 0
        loss_streak_a = 0
        loss_streak_b = 0
        
        # Initial pistol round economy
        econ_power_a = 800.0
        econ_power_b = 800.0
        
        # Initial side assignment
        current_side_a = self.starting_side_a
        round_number = 1
        
        while True:
            # Check Round 13 Side Swap
            if round_number == 13:
                current_side_a = "ATK" if current_side_a == "DEF" else "DEF"
                # Halftime Reset: reset economy and streaks
                econ_power_a = 800.0
                econ_power_b = 800.0
                loss_streak_a = 0
                loss_streak_b = 0
                
            # Check Overtime (12-12)
            is_ot = (score_a >= 12 and score_b >= 12)
            if is_ot:
                side_adv_a = 0.0
                # Overtime Reset: standard OT buy credits
                econ_power_a = 5000.0
                econ_power_b = 5000.0
                loss_streak_a = 0
                loss_streak_b = 0
            else:
                side_adv_a = self.get_side_advantage_a(current_side_a)
                
            # Scale team economy to drive loadout delta
            loadout_a = econ_power_a * 4.0
            loadout_b = econ_power_b * 4.0
            
            # Compute log-odds Z based on true economy difference
            acs_diff = self.acs_a - self.acs_b
            eco_diff = loadout_a - loadout_b
            z = 0.003 * acs_diff + 0.00004 * eco_diff + 2.0 * side_adv_a
            prob_win_a = 1.0 / (1.0 + np.exp(-z))
            prob_win_a = float(np.clip(prob_win_a, 0.20, 0.80))
            
            # Sample round winner
            if np.random.rand() < prob_win_a:
                score_a += 1
                
                # Winner updates
                loss_streak_a = 0
                econ_power_a = min(9000.0, max(4500.0, econ_power_a + 3000.0))
                
                # Loser updates
                loss_streak_b = min(3, loss_streak_b + 1)
                bonus_b = 1900.0 if loss_streak_b == 1 else (2400.0 if loss_streak_b == 2 else 2900.0)
                
                # Saving check: 15% probability of saving weapon
                if np.random.rand() < 0.15:
                    econ_power_b = min(9000.0, 1000.0 + 0.70 * econ_power_b + bonus_b)
                else:
                    econ_power_b = min(9000.0, econ_power_b + bonus_b)
            else:
                score_b += 1
                
                # Winner updates
                loss_streak_b = 0
                econ_power_b = min(9000.0, max(4500.0, econ_power_b + 3000.0))
                
                # Loser updates
                loss_streak_a = min(3, loss_streak_a + 1)
                bonus_a = 1900.0 if loss_streak_a == 1 else (2400.0 if loss_streak_a == 2 else 2900.0)
                
                # Saving check
                if np.random.rand() < 0.15:
                    econ_power_a = min(9000.0, 1000.0 + 0.70 * econ_power_a + bonus_a)
                else:
                    econ_power_a = min(9000.0, econ_power_a + bonus_a)
                
            round_number += 1
            
            # Check terminal states
            if not is_ot:
                if score_a >= 13 or score_b >= 13:
                     break
            else:
                if abs(score_a - score_b) >= 2:
                     break
                     
        return score_a, score_b


class KDACopulaEngine:
    """
    Sub-Model 4: Copula covariance to enforce player K/D/A summation constraint and retain positive correlation.
    """
    COHESION_SAT_MAPS = 25   # M_sat: maps to reach full cohesion (CF=1.0)
    CF_MIN_SCALE = 0.60       # At CF=0, alpha is scaled down to 60% (40% wider variance)

    def __init__(self, agent_roles, player_ledger: dict | None = None):
        self.agent_roles = agent_roles
        self.player_ledger = player_ledger or {}

    def get_cohesion_coefficient(self, player: str) -> float:
        """
        CF(player) = min(maps_with_current_team, M_sat) / M_sat.
        Returns 1.0 if ledger has no data (assume full cohesion for known veterans).
        """
        pdata = self.player_ledger.get(player)
        if not pdata or not pdata.get("team_history"):
            return 1.0  # no transfer history data — assume veteran
        current_team_entry = pdata["team_history"][-1]
        maps_with_team = current_team_entry.get("maps_played_with_team", 0)
        return min(maps_with_team, self.COHESION_SAT_MAPS) / self.COHESION_SAT_MAPS

    def sample_kda(self, roster: list[str], agents: list[str], total_kills: int,
                   total_deaths: int, total_assists: int, player_emas: dict,
                   baseline_lookup: dict) -> tuple[dict, dict, dict]:
        """
        Samples individual kills, deaths, and assists using a Copula-based 
        Shared Latent Momentum approach, ensuring K/D/A sum to their totals exactly.
        """
        import scipy.stats as stats
        
        # 1. Generate shared team momentum
        # Draw from standard normal latent space first
        z_team = np.random.normal(0.0, 1.0)
        u_team = stats.norm.cdf(z_team)
        
        # Map to Gumbel distribution for the right-tail skewed momentum factor
        team_momentum = stats.gumbel_r.ppf(u_team, loc=1.0, scale=0.3)
        team_momentum = float(np.clip(team_momentum, 0.15, 8.0))
        
        # Correlation coefficient for Gaussian Copula
        rho = 0.65
        
        raw_kills = []
        raw_assists = []
        raw_deaths = []
        
        for idx, player in enumerate(roster):
            agent = agents[idx]
            role = self.agent_roles.get(agent, "Sentinel")
            
            # --- KILLS ---
            alpha_k_0 = {"Duelist": 3.8, "Initiator": 2.3, "Controller": 1.6, "Sentinel": 1.2}.get(role, 1.5)
            feat = player_emas.get(player, baseline_lookup.get(player, {"acs": 200.0, "duel_diff": 0.0}))
            acs = feat.get("acs", 200.0)
            duel_diff = feat.get("duel_diff", 0.0)
            alpha_k_scaled = alpha_k_0 * np.exp(0.004 * (acs - 200.0) + 0.3 * duel_diff)
            
            cf = self.get_cohesion_coefficient(player)
            cohesion_gate = self.CF_MIN_SCALE + (1.0 - self.CF_MIN_SCALE) * cf
            shape_k = max(alpha_k_scaled * cohesion_gate, 0.1)
            
            # --- DEATHS ---
            alpha_d_0 = {"Duelist": 2.8, "Initiator": 2.2, "Controller": 1.9, "Sentinel": 1.6}.get(role, 2.0)
            alpha_d_scaled = alpha_d_0 * np.exp(-0.2 * duel_diff)
            shape_d = max(alpha_d_scaled, 0.1)
            
            # --- ASSISTS ---
            alpha_a_0 = {"Initiator": 3.2, "Controller": 2.8, "Sentinel": 1.6, "Duelist": 1.2}.get(role, 2.0)
            shape_a = max(alpha_a_0, 0.1)
            
            # --- Latent variables for Gaussian Copula ---
            eps_k = np.random.normal(0.0, 1.0)
            eps_a = np.random.normal(0.0, 1.0)
            eps_d = np.random.normal(0.0, 1.0)
            
            z_k = rho * z_team + np.sqrt(1.0 - rho**2) * eps_k
            z_a = rho * z_team + np.sqrt(1.0 - rho**2) * eps_a
            z_d = -rho * z_team + np.sqrt(1.0 - rho**2) * eps_d
            
            u_k = float(np.clip(stats.norm.cdf(z_k), 0.001, 0.999))
            u_a = float(np.clip(stats.norm.cdf(z_a), 0.001, 0.999))
            u_d = float(np.clip(stats.norm.cdf(z_d), 0.001, 0.999))
            
            # Modulate scale parameters: positive correlation with momentum for kills/assists, negative for deaths
            scale_k = team_momentum
            scale_a = team_momentum
            scale_d = 1.0 / team_momentum
            
            # Generate raw Gamma values
            raw_k_val = stats.gamma.ppf(u_k, a=shape_k, scale=scale_k)
            raw_a_val = stats.gamma.ppf(u_a, a=shape_a, scale=scale_a)
            raw_d_val = stats.gamma.ppf(u_d, a=shape_d, scale=scale_d)
            
            raw_kills.append(max(raw_k_val, 1e-5))
            raw_assists.append(max(raw_a_val, 1e-5))
            raw_deaths.append(max(raw_d_val, 1e-5))
            
        def distribute_totals(raw_vals, total_target, roster_list):
            if total_target <= 0:
                return {p: 0 for p in roster_list}
            sum_raw = sum(raw_vals)
            proportions = np.array(raw_vals) / sum_raw
            
            floored = np.floor(proportions * total_target).astype(int)
            remainder = total_target - np.sum(floored)
            
            fractional_parts = (proportions * total_target) - floored
            indices = np.argsort(fractional_parts)[::-1]
            for i in range(int(remainder)):
                floored[indices[i]] += 1
                
            return {roster_list[i]: int(floored[i]) for i in range(len(roster_list))}
            
        kills_dict = distribute_totals(raw_kills, total_kills, roster)
        deaths_dict = distribute_totals(raw_deaths, total_deaths, roster)
        assists_dict = distribute_totals(raw_assists, total_assists, roster)
        
        return kills_dict, deaths_dict, assists_dict

    def sample_kills(self, roster: list[str], agents: list[str], total_kills: int,
                      player_emas: dict, baseline_lookup: dict) -> dict:
        """
        Backward compatible wrapper for legacy calls.
        """
        kills, _, _ = self.sample_kda(roster, agents, total_kills, total_kills, total_kills, player_emas, baseline_lookup)
        return kills


# --- V5 Simulation Wrapper ---

class VCTv5SimulationEngine:
    def __init__(self, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        # V5: MapVetoBandit now receives processed_dir for TemporalMapRegistry
        self.veto_bandit = MapVetoBandit(self.raw_dir, self.processed_dir)
        self.agent_assigner = SynergisticDraftEngine(self.raw_dir, self.processed_dir)
        self.agent_transformer = self.agent_assigner  # Alias for backward compatibility

        # Load datasets (legacy path: populates player_emas, baseline_lookup)
        self.player_emas, self.baseline_lookup, self.team_stats, self.player_global_stats, self.player_agent_stats = get_simulation_historical_stats(self.raw_dir)
        self.agent_assigner.fit_comfort(self.player_agent_stats, self.player_global_stats)

        # V7.1: pass global player ledger to KDACopulaEngine
        self.kda_copula = KDACopulaEngine(
            self.agent_assigner.agent_roles,
            player_ledger=self.agent_assigner.player_ledger
        )
        self.kill_dirichlet = self.kda_copula # For backward compatibility
        
    def sample_deaths(self, roster: list[str], agents: list[str], total_deaths: int) -> dict:
        """
        Samples individual player deaths matching total_deaths constraint exactly.
        Prior alpha parameters set from agent role and historical comfort.
        """
        _, deaths, _ = self.kda_copula.sample_kda(
            roster, agents, total_deaths, total_deaths, total_deaths,
            self.player_emas, self.baseline_lookup
        )
        return deaths

    def sample_assists(self, roster: list[str], agents: list[str], total_assists: int) -> dict:
        """
        Samples individual player assists matching total_assists constraint exactly.
        Priors favor Initiators and Controllers.
        """
        _, _, assists = self.kda_copula.sample_kda(
            roster, agents, total_assists, total_assists, total_assists,
            self.player_emas, self.baseline_lookup
        )
        return assists

    def calculate_acs(self, player: str, kills: int, assists: int, rounds: int) -> int:
        """
        Estimates map-level ACS based on round performance and historical EMA baseline.
        """
        feat = self.player_emas.get(player, self.baseline_lookup.get(player, {"acs": 200.0}))
        base_acs = feat.get("acs", 200.0)
        kpr = kills / rounds if rounds > 0 else 0.0
        apr = assists / rounds if rounds > 0 else 0.0
        estimated_acs = 170.0 * kpr + 45.0 * apr + base_acs * 0.35 + np.random.normal(0, 12.0)
        return int(max(estimated_acs, 30.0))

    def simulate_match(self, team_a: str, team_b: str, series_type: str = "Bo3", target_patch: str = "9.02", num_iterations: int = 10000, override_maps: list[str] = None, target_date: datetime | None = None, ub_advantage: bool | str = False, veto_priority: str = "team_a") -> dict:
        """
        Runs Monte Carlo pipeline (10,000 iterations) with Side-Conditioned Markov Simulator
        and Synergistic Agent Assignment to generate player EV fantasy projections. Supports target_date, ub_advantage, and veto_priority.
        """
        logger.info(f"V5 Engine: Starting {num_iterations} Monte Carlo iterations for {team_a} vs {team_b}...")
        if target_date is None:
            target_date = datetime.now()
            
        # 1. Identify rosters from history
        roster_a = get_simulation_roster(team_a, self.raw_dir)
        roster_b = get_simulation_roster(team_b, self.raw_dir)
        
        if not roster_a or not roster_b:
            logger.warning("Empty rosters identified. Falling back to default baseline.")
            roster_a = roster_a or ["something", "aspas", "zekken", "wo0t", "Derke"]
            roster_b = roster_b or ["Leo", "trent", "chronicle", "Sacy", "Boaster"]
            
        # 2. Predict map veto or use override
        from collections import defaultdict
        if override_maps:
            series_maps = override_maps
            veto_res = {
                "maps": override_maps,
                "veto_weights": {m: 0 for m in override_maps},
                "veto_str": "Manual Override: " + ", ".join(override_maps),
                "starting_sides_a": ["DEF"] * len(override_maps)
            }
            veto_confidences = [(f"Force Play {m}", 1.0) for m in override_maps]
        else:
            # Deterministic base veto prediction
            veto_res = self.veto_bandit.predict_veto(team_a, team_b, series_type, stochastic=False, target_date=target_date, ub_advantage=ub_advantage, veto_priority=veto_priority)
            series_maps = veto_res["maps"]
            
            # Stochastic veto simulations to calculate veto confidences
            veto_step_counts = defaultdict(lambda: defaultdict(int))
            num_veto_sims = 1000
            for _ in range(num_veto_sims):
                v_res = self.veto_bandit.predict_veto(team_a, team_b, series_type, stochastic=True, target_date=target_date, ub_advantage=ub_advantage, veto_priority=veto_priority)
                steps = v_res["veto_str"].split("; ")
                for step_idx, step in enumerate(steps):
                    veto_step_counts[step_idx][step] += 1
            
            veto_steps = veto_res["veto_str"].split("; ")
            veto_confidences = []
            for step_idx, step in enumerate(veto_steps):
                total_count = sum(veto_step_counts[step_idx].values())
                step_count = veto_step_counts[step_idx].get(step, 0)
                conf = (step_count / total_count) if total_count > 0 else 1.0
                veto_confidences.append((step, conf))
        
        # Expected value accumulators
        player_points_sum = {p: 0.0 for p in roster_a + roster_b}
        player_sim_counts = {p: 0 for p in roster_a + roster_b}
        
        # Track map-by-map statistics
        map_raw_stats = {}
        for map_name in series_maps:
            map_raw_stats[map_name] = {
                "scorelines": [],
                "agent_picks_a": {p: [] for p in roster_a},
                "agent_picks_b": {p: [] for p in roster_b},
                "player_perf": {p: {"kills": [], "deaths": [], "assists": [], "acs": [], "points": []} for p in roster_a + roster_b}
            }
            
        # Team wins tracker
        team_a_wins = 0
        team_b_wins = 0
        
        # Compute team ACS stats for Markov simulator
        def get_team_avg_acs(roster):
            acs_list = []
            for p in roster:
                feat = self.player_emas.get(p, self.baseline_lookup.get(p, {"acs": 200.0}))
                acs_list.append(feat["acs"])
            return sum(acs_list) / len(acs_list) if acs_list else 200.0
            
        acs_team_a = get_team_avg_acs(roster_a)
        acs_team_b = get_team_avg_acs(roster_b)
        
        # Run MC Loop
        for it in range(num_iterations):
            # Track series wins in this MC iteration
            iter_wins_a = 0
            iter_wins_b = 0
            
            # Series map results
            series_map_scores = []
            
            # Predict agent comps for this iteration
            map_compositions = {}
            for map_name in series_maps:
                comp_a = self.agent_assigner.predict_composition(team_a, map_name, roster_a, target_patch)
                comp_b = self.agent_assigner.predict_composition(team_b, map_name, roster_b, target_patch)
                map_compositions[map_name] = (comp_a, comp_b)
                
            # Simulate each map
            for map_idx, map_name in enumerate(series_maps):
                starting_side_a = veto_res.get("starting_sides_a", ["DEF"] * len(series_maps))[map_idx]
                # Run Stateful Economy round simulation
                markov_sim = StatefulEconomySimulator(acs_team_a, acs_team_b, map_name, starting_side_a=starting_side_a)
                score_a, score_b = markov_sim.simulate_rounds()
                series_map_scores.append((score_a, score_b))
                
                # Check map winner
                if score_a > score_b:
                    iter_wins_a += 1
                else:
                    iter_wins_b += 1
                    
                # Total kills/deaths/assists simulation
                total_kills_a = int(4.7 * score_b + 2.1 * score_a)
                total_kills_b = int(4.7 * score_a + 2.1 * score_b)
                
                total_deaths_a = total_kills_b
                total_deaths_b = total_kills_a
                
                total_assists_a = int(round(np.clip(np.random.normal(0.40, 0.08) * total_kills_a, 0, total_kills_a)))
                total_assists_b = int(round(np.clip(np.random.normal(0.40, 0.08) * total_kills_b, 0, total_kills_b)))
                
                comp_a, comp_b = map_compositions[map_name]
                
                # Sample statistics matching constraints using Copula engine
                kills_a, deaths_a, assists_a = self.kda_copula.sample_kda(
                    roster_a, comp_a, total_kills_a, total_deaths_a, total_assists_a,
                    self.player_emas, self.baseline_lookup
                )
                kills_b, deaths_b, assists_b = self.kda_copula.sample_kda(
                    roster_b, comp_b, total_kills_b, total_deaths_b, total_assists_b,
                    self.player_emas, self.baseline_lookup
                )
                
                rounds_played = score_a + score_b
                
                # Calculate map fantasy points according to VFL rules
                margin_pts_a = calculate_vfl_margin_points(score_a, score_b)
                margin_pts_b = calculate_vfl_margin_points(score_b, score_a)
                
                # Record iteration stats for Team A
                for idx_p, p in enumerate(roster_a):
                    k = kills_a[p]
                    d = deaths_a[p]
                    a = assists_a[p]
                    acs = self.calculate_acs(p, k, a, rounds_played)
                    
                    k_pts = calculate_vfl_kill_points(k)
                    pts = k_pts + margin_pts_a
                    player_points_sum[p] += pts
                    player_sim_counts[p] += 1
                    
                    # Store raw map performance
                    map_raw_stats[map_name]["player_perf"][p]["kills"].append(k)
                    map_raw_stats[map_name]["player_perf"][p]["deaths"].append(d)
                    map_raw_stats[map_name]["player_perf"][p]["assists"].append(a)
                    map_raw_stats[map_name]["player_perf"][p]["acs"].append(acs)
                    map_raw_stats[map_name]["player_perf"][p]["points"].append(pts)
                    map_raw_stats[map_name]["agent_picks_a"][p].append(comp_a[idx_p])
                    
                # Record iteration stats for Team B
                for idx_p, p in enumerate(roster_b):
                    k = kills_b[p]
                    d = deaths_b[p]
                    a = assists_b[p]
                    acs = self.calculate_acs(p, k, a, rounds_played)
                    
                    k_pts = calculate_vfl_kill_points(k)
                    pts = k_pts + margin_pts_b
                    player_points_sum[p] += pts
                    player_sim_counts[p] += 1
                    
                    # Store raw map performance
                    map_raw_stats[map_name]["player_perf"][p]["kills"].append(k)
                    map_raw_stats[map_name]["player_perf"][p]["deaths"].append(d)
                    map_raw_stats[map_name]["player_perf"][p]["assists"].append(a)
                    map_raw_stats[map_name]["player_perf"][p]["acs"].append(acs)
                    map_raw_stats[map_name]["player_perf"][p]["points"].append(pts)
                    map_raw_stats[map_name]["agent_picks_b"][p].append(comp_b[idx_p])
                    
                # Record scoreline
                map_raw_stats[map_name]["scorelines"].append((score_a, score_b))
                
                # Break if Bo3/Bo5 has decider already settled
                req_wins = 2 if series_type == "Bo3" else 3
                if iter_wins_a == req_wins or iter_wins_b == req_wins:
                    break
                    
            if iter_wins_a > iter_wins_b:
                team_a_wins += 1
            else:
                team_b_wins += 1
                
            # Series scale modifiers/bonuses (2-0, 3-0, 3-1 bonuses)
            for p in roster_a:
                bonus = calculate_vfl_series_bonus(team_a, team_a, team_b, iter_wins_a, iter_wins_b, series_type)
                player_points_sum[p] += bonus
            for p in roster_b:
                bonus = calculate_vfl_series_bonus(team_b, team_a, team_b, iter_wins_a, iter_wins_b, series_type)
                player_points_sum[p] += bonus
                
        # Projections expected values
        projections = {}
        for p in roster_a + roster_b:
            sum_pts = player_points_sum[p]
            count = player_sim_counts[p]
            ev_points = sum_pts / count if count > 0 else 0.0
            feat = self.player_emas.get(p, self.baseline_lookup.get(p, {"acs": 200.0}))
            acs = feat.get("acs", 200.0)
            rating_bonus = 1.0 if acs > 220.0 else (0.5 if acs > 200.0 else 0.0)
            projections[p] = round(ev_points + rating_bonus, 2)
            
        win_prob_a = team_a_wins / num_iterations
        win_prob_b = team_b_wins / num_iterations
        
        # Compile map details dictionary
        map_details = {}
        for map_name in series_maps:
            raw = map_raw_stats[map_name]
            map_play_count = len(raw["scorelines"])
            
            if map_play_count == 0:
                map_details[map_name] = {
                    "played": False,
                    "play_probability": 0.0,
                    "most_probable_score": "N/A",
                    "score_confidence": 0.0,
                    "score_distribution": {},
                    "player_agents": {},
                    "player_stats": []
                }
                continue
                
            # 1. Most probable scoreline
            scoreline_counts = defaultdict(int)
            for sc in raw["scorelines"]:
                scoreline_counts[sc] += 1
            sorted_scores = sorted(scoreline_counts.items(), key=lambda x: x[1], reverse=True)
            best_sc, best_count = sorted_scores[0]
            score_confidence = best_count / map_play_count
            
            # Format score distribution
            score_distribution = {f"{sc[0]} - {sc[1]}": count for sc, count in sorted_scores[:10]}
            
            # 2. Player agents pick probability
            player_agents_info = {}
            for p in roster_a:
                picks = raw["agent_picks_a"][p]
                pick_counts = defaultdict(int)
                for a in picks:
                    pick_counts[a] += 1
                sorted_picks = sorted(pick_counts.items(), key=lambda x: x[1], reverse=True)
                best_agent, best_agent_count = sorted_picks[0]
                player_agents_info[p] = {
                    "agent": best_agent,
                    "pick_probability": round((best_agent_count / map_play_count) * 100, 1)
                }
            for p in roster_b:
                picks = raw["agent_picks_b"][p]
                pick_counts = defaultdict(int)
                for a in picks:
                    pick_counts[a] += 1
                sorted_picks = sorted(pick_counts.items(), key=lambda x: x[1], reverse=True)
                best_agent, best_agent_count = sorted_picks[0]
                player_agents_info[p] = {
                    "agent": best_agent,
                    "pick_probability": round((best_agent_count / map_play_count) * 100, 1)
                }
                
            # 3. Player performance stats table
            player_stats_table = []
            for p in roster_a + roster_b:
                perf = raw["player_perf"][p]
                if not perf["kills"]:
                    continue
                kills_mean = np.mean(perf["kills"])
                kills_p10 = np.percentile(perf["kills"], 10)
                kills_p90 = np.percentile(perf["kills"], 90)
                
                deaths_mean = np.mean(perf["deaths"])
                deaths_p10 = np.percentile(perf["deaths"], 10)
                deaths_p90 = np.percentile(perf["deaths"], 90)
                
                assists_mean = np.mean(perf["assists"])
                assists_p10 = np.percentile(perf["assists"], 10)
                assists_p90 = np.percentile(perf["assists"], 90)
                
                acs_mean = np.mean(perf["acs"])
                acs_p10 = np.percentile(perf["acs"], 10)
                acs_p90 = np.percentile(perf["acs"], 90)
                
                ev_points = np.mean(perf["points"])
                role = self.agent_transformer.agent_roles.get(player_agents_info[p]["agent"], "Sentinel")
                
                player_stats_table.append({
                    "Player": p,
                    "Team": team_a if p in roster_a else team_b,
                    "Role": role,
                    "Kills": f"{kills_mean:.1f} ({kills_p10:.0f} - {kills_p90:.0f})",
                    "kills_mean": round(float(kills_mean), 2),
                    "Deaths": f"{deaths_mean:.1f} ({deaths_p10:.0f} - {deaths_p90:.0f})",
                    "Assists": f"{assists_mean:.1f} ({assists_p10:.0f} - {assists_p90:.0f})",
                    "ACS": f"{acs_mean:.1f} ({acs_p10:.0f} - {acs_p90:.0f})",
                    "Expected VFL Points": round(ev_points, 2)
                })
                
            map_details[map_name] = {
                "played": True,
                "play_probability": round((map_play_count / num_iterations) * 100, 1),
                "most_probable_score": f"{best_sc[0]} - {best_sc[1]}",
                "score_confidence": round(score_confidence * 100, 1),
                "score_distribution": score_distribution,
                "player_agents": player_agents_info,
                "player_stats": player_stats_table
            }
            
        return {
            "team_a": team_a,
            "team_b": team_b,
            "win_prob_a": win_prob_a,
            "win_prob_b": win_prob_b,
            "predicted_maps": series_maps,
            "veto_str": veto_res["veto_str"],
            "veto_confidences": veto_confidences,
            "projections": projections,
            "roster_a": roster_a,
            "roster_b": roster_b,
            "map_details": map_details
        }


# --- VFL Scoring Constants & Functions ---

def calculate_vfl_kill_points(kills: int) -> int:
    if kills == 0:
        return -3
    elif 1 <= kills <= 4:
        return -1
    elif 5 <= kills <= 9:
        return 0
    else:  # kills >= 10
        return 1 + (kills - 10) // 5

def calculate_vfl_margin_points(team_score: int, opp_score: int) -> int:
    if team_score == 13 and opp_score == 0:
        return 5
    elif team_score == 0 and opp_score == 13:
        return -5
    if team_score > opp_score:
        diff = team_score - opp_score
        pts = 1
        if 5 <= diff <= 9:
            pts += 1
        elif diff >= 10:
            pts += 2
        return pts
    else:
        diff = opp_score - team_score
        if diff >= 10:
            return -1
        return 0

def calculate_vfl_series_bonus(player_team: str, team_a: str, team_b: str, score_a: int, score_b: int, series_type: str) -> int:
    pt = player_team.lower().strip()
    ta = team_a.lower().strip()
    tb = team_b.lower().strip()
    
    is_team_a = pt in ta or ta in pt
    is_team_b = pt in tb or tb in pt
    
    if score_a > score_b:
        if not is_team_a:
            return 0
        if score_a == 2 and score_b == 0 and series_type == "Bo3":
            return 2
        elif score_a == 3 and score_b == 0 and series_type == "Bo5":
            return 4
        elif score_a == 3 and score_b == 1 and series_type == "Bo5":
            return 1
    elif score_b > score_a:
        if not is_team_b:
            return 0
        if score_b == 2 and score_a == 0 and series_type == "Bo3":
            return 2
        elif score_b == 3 and score_a == 0 and series_type == "Bo5":
            return 4
        elif score_b == 3 and score_a == 1 and series_type == "Bo5":
            return 1
    return 0


# --- Data loading helpers specifically for engine ---

def parse_simulation_match_date(date_str: str) -> datetime:
    clean_str = date_str.split(" Patch ")[0]
    clean_str = re.sub(r'\s+[A-Z]{3,4}$', '', clean_str).strip()
    clean_str = re.sub(r'^[A-Za-z]+,\s*', '', clean_str).strip()
    
    year_match = re.search(r'\b(20\d{2})\b', date_str)
    year = int(year_match.group(1)) if year_match else 2026
    
    month_day_match = re.search(r'^([A-Za-z]+)\s+(\d+)', clean_str)
    if not month_day_match:
        return datetime(2026, 6, 22)
    month = month_day_match.group(1)
    day = int(month_day_match.group(2))
    
    try:
        normalized_date_str = f"{month} {day}, {year} 12:00 PM"
        return datetime.strptime(normalized_date_str, "%B %d, %Y %I:%M %p")
    except Exception:
        return datetime(2026, 6, 22)

def get_simulation_historical_stats(raw_dir: str):
    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    matches = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                seg = content["data"]["segments"][0]
                seg["timestamp"] = parse_simulation_match_date(seg["date"])
                matches.append(seg)
        except Exception:
            pass
            
    matches.sort(key=lambda x: x["timestamp"])
    
    # Load player stats baseline lookup
    player_stats_path = os.path.join(raw_dir, "player_stats.json")
    baseline_lookup = {}
    if os.path.exists(player_stats_path):
        with open(player_stats_path, "r", encoding="utf-8") as f:
            player_stats_baseline = json.load(f)["data"]["segments"]
            for ps in player_stats_baseline:
                p_name = ps["player"]
                acs_b = float(ps.get("average_combat_score", 200.0))
                kast_val = ps.get("kill_assists_survived_traded")
                if isinstance(kast_val, str) and "%" in kast_val:
                    kast_b = float(kast_val.replace("%", "")) / 100.0
                elif kast_val is not None and kast_val != "":
                    try:
                        kast_b = float(kast_val)
                        if kast_b > 1.0:
                            kast_b = kast_b / 100.0
                    except (ValueError, TypeError):
                        kast_b = 0.70
                else:
                    kast_b = 0.70
                fk_per_r = float(ps.get("first_kills_per_round", 0.0))
                fd_per_r = float(ps.get("first_deaths_per_round", 0.0))
                baseline_lookup[p_name] = {"acs": acs_b, "kast": kast_b, "duel_diff": fk_per_r - fd_per_r}

    player_emas = {}
    player_global_stats = {}
    player_agent_stats = {}
    
    for m in matches:
        for map_data in m.get("maps", []):
            map_name = map_data.get('map_name', '')
            for team_key in ['team1', 'team2']:
                for p in map_data.get('players', {}).get(team_key, []):
                    p_name = p['name']
                    agent = p.get('agent', '')
                    acs_val = float(p.get('acs') or 0.0)
                    
                    if p_name not in player_global_stats:
                        player_global_stats[p_name] = {'sum_acs': 0.0, 'count': 0, 'acs_history': []}
                    if acs_val > 0:
                        player_global_stats[p_name]['sum_acs'] += acs_val
                        player_global_stats[p_name]['count'] += 1
                        player_global_stats[p_name].setdefault('acs_history', []).append(acs_val)
                        
                        if agent:
                            # Global key
                            if (p_name, agent) not in player_agent_stats:
                                player_agent_stats[(p_name, agent)] = {'sum_acs': 0.0, 'count': 0, 'acs_history': []}
                            player_agent_stats[(p_name, agent)]['sum_acs'] += acs_val
                            player_agent_stats[(p_name, agent)]['count'] += 1
                            player_agent_stats[(p_name, agent)].setdefault('acs_history', []).append(acs_val)
                            
                            # Map-specific key
                            if (p_name, map_name, agent) not in player_agent_stats:
                                player_agent_stats[(p_name, map_name, agent)] = {'sum_acs': 0.0, 'count': 0, 'acs_history': []}
                            player_agent_stats[(p_name, map_name, agent)]['sum_acs'] += acs_val
                            player_agent_stats[(p_name, map_name, agent)]['count'] += 1
                            player_agent_stats[(p_name, map_name, agent)].setdefault('acs_history', []).append(acs_val)
                    
                    # V5 fix: accumulate kast and duel_diff from actual match fields
                    kast_raw = p.get('kast')
                    fk_raw   = p.get('fk')
                    fd_raw   = p.get('fd')

                    if kast_raw is not None:
                        try:
                            kast_str = str(kast_raw).replace('%', '')
                            kast_val = float(kast_str) / 100.0 if float(kast_str) > 1.0 else float(kast_str)
                            gs = player_global_stats.setdefault(p_name, {'sum_acs': 0.0, 'count': 0, 'acs_history': []})
                            gs['sum_kast'] = gs.get('sum_kast', 0.0) + kast_val
                            gs['kast_count'] = gs.get('kast_count', 0) + 1
                        except (ValueError, TypeError):
                            pass

                    if fk_raw is not None and fd_raw is not None:
                        try:
                            fk_val = float(fk_raw)
                            fd_val = float(fd_raw)
                            gs = player_global_stats.setdefault(p_name, {'sum_acs': 0.0, 'count': 0, 'acs_history': []})
                            gs['sum_duel_diff'] = gs.get('sum_duel_diff', 0.0) + (fk_val - fd_val)
                            gs['duel_count']    = gs.get('duel_count', 0) + 1
                        except (ValueError, TypeError):
                            pass

    # Helper function for 5th/95th percentile outlier clipping
    def compute_clipped_acs(stat_dict: dict) -> float:
        history = stat_dict.get('acs_history', [])
        if len(history) > 3:
            p5 = np.percentile(history, 5)
            p95 = np.percentile(history, 95)
            clipped = np.clip(history, p5, p95)
            return float(np.mean(clipped))
        elif len(history) > 0:
            return float(np.mean(history))
        elif stat_dict.get('count', 0) > 0:
            return stat_dict['sum_acs'] / stat_dict['count']
        return 200.0

    # Apply outlier clipping to all player_agent_stats sum_acs entries for legacy fallback accuracy
    for k, stat_dict in player_agent_stats.items():
        stat_dict['sum_acs'] = compute_clipped_acs(stat_dict) * stat_dict['count']

    # Fill EMAs using historical global averages with outlier clipping applied
    for p_name, stats in player_global_stats.items():
        if stats['count'] > 0:
            if stats.get('kast_count', 0) > 0:
                kast_ema = stats['sum_kast'] / stats['kast_count']
            elif p_name in baseline_lookup:
                kast_ema = baseline_lookup[p_name]['kast']
            else:
                kast_ema = 0.72

            if stats.get('duel_count', 0) > 0:
                duel_ema = stats['sum_duel_diff'] / stats['duel_count']
            elif p_name in baseline_lookup:
                duel_ema = baseline_lookup[p_name]['duel_diff']
            else:
                duel_ema = 0.0

            player_emas[p_name] = {
                "acs": compute_clipped_acs(stats),
                "kast": float(np.clip(kast_ema, 0.0, 1.0)),
                "duel_diff": float(np.clip(duel_ema, -0.5, 0.5))
            }
            
    # Default fallback for any unseen player
    for p_name in baseline_lookup:
        if p_name not in player_emas:
            player_emas[p_name] = {
                "acs": baseline_lookup[p_name]["acs"],
                "kast": baseline_lookup[p_name]["kast"],
                "duel_diff": baseline_lookup[p_name]["duel_diff"]
            }
            
    return player_emas, baseline_lookup, {}, player_global_stats, player_agent_stats

def is_strict_simulation_team_match(target: str, candidate: str) -> bool:
    target = target.lower().strip()
    candidate = candidate.lower().strip()
    if target == candidate:
        return True
    suffixes = ["academy", "gc", "game changers", "black", "blue"]
    target_has_suffix = any(s in target for s in suffixes)
    candidate_has_suffix = any(s in candidate for s in suffixes)
    if target_has_suffix != candidate_has_suffix:
        return False
    if target in candidate or candidate in target:
        return True
    def get_initials(name: str) -> str:
        return "".join(word[0] for word in name.split() if word)
    t_init = get_initials(target)
    c_init = get_initials(candidate)
    if t_init == candidate or c_init == target:
        return True
    return False

def get_simulation_roster(team_name: str, raw_dir: str) -> list[str]:
    # 1. Check for manual roster override
    try:
        processed_dir = os.path.join(os.path.dirname(raw_dir), "processed")
        override_path = os.path.join(processed_dir, "roster_overrides.json")
        if os.path.exists(override_path):
            with open(override_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
                for k, v in overrides.items():
                    if k.lower().strip() == team_name.lower().strip():
                        logger.info(f"V5 Engine: Using manual roster override for {team_name}: {v}")
                        return v
    except Exception as e:
        logger.warning(f"V5 Engine: Failed to load overrides: {e}")

    files = glob.glob(os.path.join(raw_dir, "match_*.json"))
    matches_with_team = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                seg = content["data"]["segments"][0]
                ta = seg["teams"][0]["name"]
                tb = seg["teams"][1]["name"]
                
                # Use strict team match
                is_ta_match = is_strict_simulation_team_match(team_name, ta)
                is_tb_match = is_strict_simulation_team_match(team_name, tb)
                
                if is_ta_match or is_tb_match:
                    ts = parse_simulation_match_date(seg["date"])
                    matches_with_team.append((ts, seg, ta, tb))
        except Exception:
            pass
            
    if not matches_with_team:
        return []
        
    matches_with_team.sort(key=lambda x: x[0], reverse=True)
    latest_entry = matches_with_team[0]
    latest_seg = latest_entry[1]
    ta_name = latest_entry[2]
    
    # Determine team_key strictly
    team_key = 'team1' if is_strict_simulation_team_match(team_name, ta_name) else 'team2'
    
    roster = set()
    for map_data in latest_seg.get('maps', []):
        for p in map_data.get('players', {}).get(team_key, []):
            p_name = p['name']
            if "inactive" in p_name.lower():
                continue
            roster.add(p_name)
            
    return list(roster)

if __name__ == "__main__":
    logger.info("VCT V5 Bottom-Up Simulation Engine unit test...")
    engine = VCTv5SimulationEngine()
    
    # Simulate Paper Rex vs LEVIATÁN Bo3 (runs fast 1000 iterations for unit test check)
    res = engine.simulate_match("Paper Rex", "LEVIATÁN", "Bo3", num_iterations=1000)
    print("\n" + "="*60)
    print("V5 SIMULATION ENGINE UNIT TEST COMPLETE")
    print("="*60)
    print(f"Match: {res['team_a']} ({res['win_prob_a']:.1%}) vs {res['team_b']} ({res['win_prob_b']:.1%})")
    print("Maps Veto:", res["predicted_maps"])
    print("Top Projections (EV Points):")
    sorted_proj = sorted(res["projections"].items(), key=lambda x: x[1], reverse=True)
    for p, pts in sorted_proj[:6]:
        print(f"  {p}: {pts} pts")
    print("="*60 + "\n")
