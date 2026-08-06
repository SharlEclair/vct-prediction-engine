import os
import re
import json
import logging
import random
import time
import datetime
from curl_cffi import requests
from selectolax.parser import HTMLParser

logger = logging.getLogger("vlr_scraper")
logging.basicConfig(level=logging.INFO)

VLR_BASE_URL = "https://www.vlr.gg"

_session = None

BUY_TYPE_MAP = {
    "": "eco",
    "eco": "eco",
    "$": "semi_eco",
    "semi_eco": "semi_eco",
    "$$": "semi_buy",
    "semi_buy": "semi_buy",
    "$$$": "full_buy",
    "full_buy": "full_buy"
}

def clean_numeric(val):
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "null" or val_str == "\xa0" or val_str == "&nbsp;":
        return None
    if "%" in val_str:
        try:
            return round(float(val_str.replace("%", "")) / 100.0, 4)
        except ValueError:
            return None
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def parse_played_won(val_str: str) -> dict:
    """
    Parses 'played (won)' strings like '14 (7)' into:
    {'played': 14, 'won': 7, 'lost': 7, 'win_rate': 0.5000}
    Preserves raw inputs (played, won) alongside derived metrics (lost, win_rate).
    """
    if not val_str:
        return {"played": 0, "won": 0, "lost": 0, "win_rate": 0.0}
    val_str = val_str.strip()
    m = re.search(r'(\d+)\s*\(\s*(\d+)\s*\)', val_str)
    if m:
        p = int(m.group(1))
        w = int(m.group(2))
        l = max(0, p - w)
        wr = round(w / p, 4) if p > 0 else 0.0
        return {"played": p, "won": w, "lost": l, "win_rate": wr}
    try:
        p = int(val_str)
        return {"played": p, "won": 0, "lost": p, "win_rate": 0.0}
    except ValueError:
        return {"played": 0, "won": 0, "lost": 0, "win_rate": 0.0}

def extract_player_name(cell) -> str:
    """Extracts player name from a cell by stripping team tag elements."""
    if not cell:
        return ""
    tag_el = cell.css_first(".team-tag") or cell.css_first(".ge-text-faded")
    sub_txt = tag_el.text() if tag_el else ""
    raw = cell.text()
    return clean_text(raw.replace(sub_txt, ""))

def get_curl_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session(impersonate="chrome")
    return _session

def fetch_url_with_curl(url: str, session: requests.Session = None) -> str:
    """Fetch URL using curl_cffi Session to bypass Cloudflare WAF, with randomized jitter."""
    sleep_time = 3.0 + random.uniform(0.5, 2.5)
    logger.info(f"Sleeping for {sleep_time:.2f}s to prevent IP throttling/shadow-bans...")
    time.sleep(sleep_time)
    
    if session is None:
        session = get_curl_session()
        
    try:
        response = session.get(url, timeout=20.0)
        if response.status_code != 200:
            logger.error(f"HTTP request returned status {response.status_code} for URL {url}")
            return ""
        return response.text
    except Exception as e:
        logger.error(f"curl_cffi fetch failed for URL {url}: {e}")
        return ""

def is_tier1_event(event_name: str) -> bool:
    """Tier 1 Strict Filtering: Blacklist strictly overrides whitelist globally."""
    if not event_name:
        return False
    name_lower = event_name.lower()
    blacklist_keywords = [
        'challengers', 'ascension', 'game changers', 'gc', 'premier', 'grassroots',
        'fortress', 'collegiate', 'university', 'showmatch', 'community', 'trial',
        'open qualifier', 'cup', 'weekly', 'monthly', 'amateur'
    ]
    whitelist_keywords = ['masters', 'champions', 'vct', 'champions tour']
    if any(ex in name_lower for ex in blacklist_keywords):
        return False
    return any(kw in name_lower for kw in whitelist_keywords)


def parse_overview_tab(html_text: str) -> tuple[dict, list[dict], dict]:
    """
    Parses the Overview tab HTML.
    Returns:
      - metadata: {date, patch, event, teams, score, maps_played}
      - segments: list of map segment dicts
      - map_info_by_gid: dict mapping data-game-id to map info dict {map_id, map_name}
    """
    parser = HTMLParser(html_text)

    # --- Event ---
    event_elem = parser.css_first(".match-header-super a")
    event = clean_text(event_elem.text()) if event_elem else ""
    if not event:
        event_div = parser.css_first(".match-header-super")
        if event_div:
            event = clean_text(event_div.text())

    # --- Date: prefer ISO UTC from data-utc-ts attribute ---
    date_str = ""
    date_elem = parser.css_first(".match-header-date")
    if date_elem:
        utc_el = date_elem.css_first("[data-utc-ts]")
        if utc_el:
            raw_utc = utc_el.attributes.get("data-utc-ts", "").strip()
            if raw_utc:
                date_str = raw_utc.replace(" ", "T") + "Z"
        if not date_str:
            date_str = re.sub(
                r'\s*Patch\s+[0-9.]+', '',
                clean_text(date_elem.text()),
                flags=re.IGNORECASE
            ).strip()

    # --- Patch ---
    patch = None
    if date_elem:
        italic_div = date_elem.css_first("div[style*='italic']")
        if italic_div:
            m = re.search(r'Patch\s+([0-9.]+)', clean_text(italic_div.text()), re.IGNORECASE)
            if m:
                patch = m.group(1).strip()
    if not patch:
        note_elem = parser.css_first(".match-header-note")
        if note_elem:
            m = re.search(r'Patch\s+([0-9.]+)', clean_text(note_elem.text()), re.IGNORECASE)
            if m:
                patch = m.group(1).strip()

    # --- Teams ---
    team1_elem = parser.css_first(".match-header-link-name.mod-1")
    team1 = "Team 1"
    if team1_elem:
        elo_el = team1_elem.css_first(".match-header-link-name-elo")
        if elo_el:
            team1 = clean_text(team1_elem.text().replace(elo_el.text(), ""))
        else:
            team1 = clean_text(team1_elem.text())

    team2_elem = parser.css_first(".match-header-link-name.mod-2")
    team2 = "Team 2"
    if team2_elem:
        elo_el = team2_elem.css_first(".match-header-link-name-elo")
        if elo_el:
            team2 = clean_text(team2_elem.text().replace(elo_el.text(), ""))
        else:
            team2 = clean_text(team2_elem.text())

    # --- Match-level winner & overall series score ---
    winner = ""
    team1_score = 0
    team2_score = 0
    scores = parser.css(".match-header-vs-score span")
    scored_spans = [s for s in scores if s.text(strip=True).isdigit()]
    if len(scored_spans) >= 2:
        team1_score = int(scored_spans[0].text(strip=True))
        team2_score = int(scored_spans[1].text(strip=True))
        cls0 = scored_spans[0].attributes.get("class", "")
        cls1 = scored_spans[1].attributes.get("class", "")
        if "winner" in cls0:
            winner = team1
        elif "winner" in cls1:
            winner = team2

    match_score = {
        "team1_score": team1_score,
        "team2_score": team2_score,
        team1: team1_score,
        team2: team2_score
    }

    # --- Vetoes ---
    vetoes = []
    note_elems = parser.css(".match-header-note")
    for el in note_elems:
        txt = clean_text(el.text())
        parts = re.split(r'[;\n\r]', txt)
        for part in parts:
            part = part.strip()
            if any(w in part.lower() for w in ["ban", "pick", "remains", "decider"]):
                vetoes.append(part)

    # --- Per-map game blocks & tab mapping ---
    game_blocks = parser.css("div.vm-stats-game")
    valid_game_blocks = [
        gb for gb in game_blocks
        if gb.attributes.get("data-game-id", "") not in ("", "all")
    ]

    map_segments = []
    maps_played = []
    map_info_by_gid = {"all": {"map_id": "all_maps", "map_name": "All Maps"}}

    for idx, gb in enumerate(valid_game_blocks):
        gid = gb.attributes.get("data-game-id", "")

        # -- Map name, picked_by, duration --
        map_name = "Unknown"
        picked_by = ""
        duration = ""
        map_elem = gb.css_first(".map")
        if map_elem:
            name_div = map_elem.css_first("div[style*='font-weight']")
            raw_text = clean_text(name_div.text()) if name_div else clean_text(map_elem.text())
            map_name = re.sub(r'\s+(PICK|DECIDER|BAN)\b.*', '', raw_text, flags=re.IGNORECASE).strip()
            map_name = re.sub(r'\s+\d{1,2}:\d{2}(?::\d{2})?.*$', '', map_name).strip()

            pick_span = map_elem.css_first(".picked, .pick")
            if pick_span:
                picked_by = clean_text(pick_span.text())
            dur_el = map_elem.css_first(".map-duration")
            if dur_el:
                duration = clean_text(dur_el.text())

        map_id = re.sub(r'[^a-z0-9_]', '', map_name.lower().replace(' ', '_'))
        if map_name != "Unknown":
            maps_played.append({"map_id": map_id, "map_name": map_name})
        if gid:
            map_info_by_gid[gid] = {"map_id": map_id, "map_name": map_name}

        # -- Map-level winner & map score --
        map_winner = ""
        s1 = 0
        s2 = 0
        score_els = gb.css(".vm-stats-game-header .score")
        if len(score_els) >= 2:
            s1_txt = clean_text(score_els[0].text())
            s2_txt = clean_text(score_els[1].text())
            s1 = int(s1_txt) if s1_txt.isdigit() else 0
            s2 = int(s2_txt) if s2_txt.isdigit() else 0
        else:
            team_blocks = gb.css(".team")
            s1_txt = clean_text(team_blocks[0].css_first(".score").text()) if len(team_blocks) > 0 and team_blocks[0].css_first(".score") else ""
            s2_txt = clean_text(team_blocks[1].css_first(".score").text()) if len(team_blocks) > 1 and team_blocks[1].css_first(".score") else ""
            s1 = int(s1_txt) if s1_txt.isdigit() else 0
            s2 = int(s2_txt) if s2_txt.isdigit() else 0

        if s1 > s2:
            map_winner = team1
        elif s2 > s1:
            map_winner = team2

        map_score = {
            "team1_score": s1,
            "team2_score": s2,
            team1: s1,
            team2: s2
        }

        # -- Composition & Players --
        composition = {team1: [], team2: []}
        players_list = []
        map_ovw_tables = gb.css("div.ovw-table")

        for t_idx, div_table in enumerate(map_ovw_tables[:2]):
            current_team = team1 if t_idx == 0 else team2

            for row in div_table.css("div.ovw-row"):
                if "mod-head" in row.attributes.get("class", ""):
                    continue

                p_name_el = row.css_first(".ovw-player-name.text-of")
                p_name = clean_text(p_name_el.text()) if p_name_el else ""
                if not p_name:
                    continue

                agent = ""
                img = row.css_first(".ovw-agents img")
                if img:
                    agent = img.attributes.get("title", "") or img.attributes.get("alt", "")

                if agent:
                    composition[current_team].append(agent)

                def get_stat(selector: str) -> str | None:
                    el = row.css_first(selector)
                    if el is None:
                        return None
                    txt = el.text(strip=True)
                    return None if (not txt or txt == "\xa0") else txt

                rating  = get_stat("[data-col='rating2'] .side.mod-both")
                acs     = get_stat("[data-col='acs'] .side.mod-both")
                kills   = get_stat(".ovw-cell.mod-kda [data-col='kills'] .side.mod-both")
                deaths  = get_stat(".ovw-cell.mod-kda [data-col='deaths'] .side.mod-both")
                assists = get_stat(".ovw-cell.mod-kda [data-col='assists'] .side.mod-both")
                kast    = get_stat("[data-col='kast'] .side.mod-both")
                adr     = get_stat("[data-col='adr'] .side.mod-both")
                hs_pct  = get_stat("[data-col='hsp'] .side.mod-both")
                fk      = get_stat("[data-col='fb'] .side.mod-both")
                fd      = get_stat("[data-col='fd'] .side.mod-both")

                players_list.append({
                    "name":    p_name,
                    "team":    current_team,
                    "agent":   agent,
                    "rating":  clean_numeric(rating),
                    "acs":     clean_numeric(acs),
                    "kills":   clean_numeric(kills),
                    "deaths":  clean_numeric(deaths),
                    "assists": clean_numeric(assists),
                    "kast":    clean_numeric(kast),
                    "adr":     clean_numeric(adr),
                    "hs_pct":  clean_numeric(hs_pct),
                    "fk":      clean_numeric(fk),
                    "fd":      clean_numeric(fd),
                })

        # -- Round History --
        round_history = []
        vlr_rnd = gb.css_first(".vlr-rounds")
        if vlr_rnd:
            for col in vlr_rnd.css(".vlr-rounds-row-col"):
                rnd_num_el = col.css_first(".rnd-num")
                if not rnd_num_el:
                    continue
                try:
                    r_num = int(clean_text(rnd_num_el.text()))
                except ValueError:
                    continue

                sqs = col.css(".rnd-sq")
                if len(sqs) < 2:
                    continue

                sq0_cls = sqs[0].attributes.get("class", "")
                sq1_cls = sqs[1].attributes.get("class", "")

                if "mod-win" in sq0_cls:
                    r_winner = team1
                    side = "ct" if "mod-ct" in sq0_cls else "t"
                elif "mod-win" in sq1_cls:
                    r_winner = team2
                    side = "ct" if "mod-ct" in sq1_cls else "t"
                else:
                    continue

                round_history.append({
                    "round_num": r_num,
                    "winner":    r_winner,
                    "side":      side,
                })

        map_segments.append({
            "map_id":        map_id,
            "map_name":      map_name,
            "team1":         team1,
            "team2":         team2,
            "winner":        map_winner,
            "score":         map_score,
            "picked_by":     picked_by,
            "duration":      duration,
            "composition":   composition,
            "players":       players_list,
            "round_history": round_history,
            "vetoes":        vetoes,
        })

    # Post-processing map winners
    for raw_map in map_segments:
        team_a = raw_map.get("team1")
        team_b = raw_map.get("team2")
        clean_rounds = raw_map.get("round_history", [])
        t1_wins = sum(1 for r in clean_rounds if r.get("winner") == team_a)
        t2_wins = sum(1 for r in clean_rounds if r.get("winner") == team_b)
        if max(t1_wins, t2_wins) >= 13:
            expected_winner = team_a if t1_wins > t2_wins else team_b
            if raw_map.get("winner") != expected_winner:
                raw_map["winner"] = expected_winner

    metadata = {
        "date": date_str,
        "patch": patch,
        "event": event,
        "teams": {
            "team1": team1,
            "team2": team2
        },
        "score": match_score,
        "maps_played": maps_played
    }

    return metadata, map_segments, map_info_by_gid


def parse_performance_tab(html_text: str, map_info_by_gid: dict) -> dict | None:
    if not html_text or "Stats from this map are not available yet" in html_text or "not available yet" in html_text.lower():
        return None

    parser = HTMLParser(html_text)
    game_blocks = parser.css("div.vm-stats-game")
    if not game_blocks:
        return None

    perf_maps = {}

    for gb in game_blocks:
        gid = gb.attributes.get("data-game-id", "")
        if not gid:
            continue

        minfo = map_info_by_gid.get(gid, {"map_id": "all_maps" if gid == "all" else gid, "map_name": "All Maps" if gid == "all" else gid})
        map_id = minfo["map_id"]
        map_name = minfo["map_name"]

        player_stats = {}
        adv_table = gb.css_first("table.mod-adv-stats")
        if adv_table:
            adv_rows = adv_table.css("tbody tr") if adv_table.css("tbody tr") else adv_table.css("tr")
            for r in adv_rows:
                cells = r.css("td")
                if not cells:
                    continue
                p_name = extract_player_name(cells[0])
                if not p_name:
                    continue

                def get_int_cell(idx: int) -> int:
                    if idx < len(cells):
                        txt = cells[idx].text(strip=True)
                        m = re.match(r'^(\d+)', txt)
                        if m:
                            return int(m.group(1))
                    return 0

                player_stats[p_name] = {
                    "2K": get_int_cell(2),
                    "3K": get_int_cell(3),
                    "4K": get_int_cell(4),
                    "5K": get_int_cell(5),
                    "1v1": get_int_cell(6),
                    "1v2": get_int_cell(7),
                    "1v3": get_int_cell(8),
                    "1v4": get_int_cell(9),
                    "1v5": get_int_cell(10),
                    "ECON": get_int_cell(11),
                    "PL": get_int_cell(12),
                    "DE": get_int_cell(13)
                }

        matrix_configs = [
            ("all_kills", "table.mod-matrix.mod-normal"),
            ("first_kills", "table.mod-matrix.mod-fkfd"),
            ("op_kills", "table.mod-matrix.mod-op")
        ]

        duels_dict = {}

        for subtab_key, css_sel in matrix_configs:
            m_table = gb.css_first(css_sel)
            duels_list = []
            players = []
            if m_table:
                rows = m_table.css("tr")
                if len(rows) > 1:
                    header_cells = rows[0].css("th, td")
                    victims = [extract_player_name(c) for c in header_cells[1:]]

                    for r in rows[1:]:
                        r_cells = r.css("th, td")
                        if len(r_cells) < 2:
                            continue
                        attacker = extract_player_name(r_cells[0])
                        if attacker and attacker not in players:
                            players.append(attacker)

                        for c_idx, cell in enumerate(r_cells[1:]):
                            if c_idx < len(victims):
                                victim = victims[c_idx]
                                if victim and victim not in players:
                                    players.append(victim)

                                if attacker == victim:
                                    continue

                                sqs = cell.css(".stats-sq")
                                if len(sqs) >= 1:
                                    try:
                                        k_val = int(sqs[0].text(strip=True))
                                        # sqs[1] is victim's eliminations against attacker (attacker's deaths)
                                        d_val = int(sqs[1].text(strip=True)) if len(sqs) >= 2 and sqs[1].text(strip=True).isdigit() else 0
                                        if k_val > 0 or d_val > 0:
                                            duels_list.append({
                                                "attacker": attacker,
                                                "victim": victim,
                                                "kills": k_val,  # attacker -> victim eliminations
                                                "deaths": d_val  # victim -> attacker eliminations
                                            })
                                    except ValueError:
                                        pass
            duels_dict[subtab_key] = {
                "players": players,
                "duels": duels_list
            }

        perf_maps[map_id] = {
            "map_name": map_name,
            "player_stats": player_stats,
            "duels": duels_dict
        }

    return {"maps": perf_maps}


def parse_economy_tab(html_text: str, map_info_by_gid: dict, segments: list[dict] = None, overview_teams: tuple[str, str] = ("Team 1", "Team 2")) -> dict | None:
    if not html_text or "Stats from this map are not available yet" in html_text or "not available yet" in html_text.lower():
        return None

    parser = HTMLParser(html_text)
    game_blocks = parser.css("div.vm-stats-game")
    if not game_blocks:
        return None

    round_history_by_map = {}
    if segments:
        for seg in segments:
            round_history_by_map[seg.get("map_id")] = seg.get("round_history", [])

    econ_maps = {}

    for gb in game_blocks:
        gid = gb.attributes.get("data-game-id", "")
        if not gid:
            continue

        minfo = map_info_by_gid.get(gid, {"map_id": "all_maps" if gid == "all" else gid, "map_name": "All Maps" if gid == "all" else gid})
        map_id = minfo["map_id"]
        map_name = minfo["map_name"]

        econ_tables = gb.css("table.mod-econ")
        summary_dict = {}

        if len(econ_tables) >= 1:
            econ_table = econ_tables[0]
            rows = econ_table.css("tbody tr") if econ_table.css("tbody tr") else econ_table.css("tr")
            valid_rows = []
            for r in rows:
                cells = r.css("td")
                if len(cells) >= 6 and clean_text(cells[0].text()) != "":
                    valid_rows.append(cells)

            t1_full = overview_teams[0] if len(overview_teams) > 0 else "Team 1"
            t2_full = overview_teams[1] if len(overview_teams) > 1 else "Team 2"

            for vr_idx, cells in enumerate(valid_rows):
                team_name = t1_full if vr_idx == 0 else (t2_full if vr_idx == 1 else clean_text(cells[0].text()))
                pistol_str = cells[1].text(strip=True)
                pistol_won = int(pistol_str) if pistol_str.isdigit() else 0

                eco = parse_played_won(cells[2].text(strip=True))
                semi_eco = parse_played_won(cells[3].text(strip=True))
                semi_buy = parse_played_won(cells[4].text(strip=True))
                full_buy = parse_played_won(cells[5].text(strip=True))

                total_rounds = eco["played"] + semi_eco["played"] + semi_buy["played"] + full_buy["played"]

                summary_dict[team_name] = {
                    "total_rounds": total_rounds,
                    "pistol_won": pistol_won,
                    "eco": eco,
                    "semi_eco": semi_eco,
                    "semi_buy": semi_buy,
                    "full_buy": full_buy
                }

        map_entry = {
            "map_name": map_name,
            "economy_summary": summary_dict
        }

        if gid != "all" and len(econ_tables) >= 2:
            r_table = econ_tables[1]
            td0 = r_table.css_first("td")
            teams_in_timeline = []
            if td0:
                teams_in_timeline = [clean_text(t.text()) for t in td0.css(".team")]

            rh_list = round_history_by_map.get(map_id, [])
            rh_map = {}
            for r_item in rh_list:
                rh_map[r_item.get("round_num")] = (r_item.get("winner"), r_item.get("side"))

            round_econ_list = []
            t1_full = overview_teams[0] if len(overview_teams) > 0 else "Team 1"
            t2_full = overview_teams[1] if len(overview_teams) > 1 else "Team 2"

            for tr in r_table.css("tr"):
                for td in tr.css("td")[1:]:
                    r_num_el = td.css_first(".round-num")
                    if not r_num_el or not r_num_el.text(strip=True).isdigit():
                        continue
                    r_num = int(r_num_el.text(strip=True))

                    sqs = td.css(".rnd-sq")
                    if len(sqs) < 2:
                        continue

                    r_winner, r_side = rh_map.get(r_num, (None, None))

                    def parse_sq(sq, team_idx):
                        val = sq.attributes.get("title", "0")
                        bank = int(val) if val.isdigit() else 0
                        symbol = clean_text(sq.text())
                        buy_type = BUY_TYPE_MAP.get(symbol, "eco")
                        t_name = t1_full if team_idx == 0 else t2_full

                        won = (t_name == r_winner) if r_winner else False
                        if r_winner and r_side:
                            if t_name == r_winner:
                                side_str = "attack" if r_side == "t" else "defense"
                            else:
                                side_str = "defense" if r_side == "t" else "attack"
                        else:
                            side_str = "unknown"

                        return {
                            "map_id": map_id,
                            "round": r_num,
                            "team": t_name,
                            "side": side_str,
                            "bank": bank,
                            "buy_type": buy_type,
                            "symbol": symbol,
                            "won": won
                        }

                    round_econ_list.append(parse_sq(sqs[0], 0))
                    round_econ_list.append(parse_sq(sqs[1], 1))

            map_entry["round_economy"] = round_econ_list

        econ_maps[map_id] = map_entry

    return {"maps": econ_maps}


def parse_vlr_match(match_id_or_url: str) -> dict:
    """
    Fetches and parses Overview, Performance, and Economy tabs for a VLR match.
    Returns complete analytics-grade JSON schema v1.0.
    """
    match_id = match_id_or_url.strip("/").split("/")[0]
    scraped_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Overview Tab
    overview_url = f"{VLR_BASE_URL}/{match_id}"
    logger.info(f"Fetching VLR Overview page from {overview_url}...")
    overview_html = fetch_url_with_curl(overview_url)
    if not overview_html:
        logger.error("Failed to fetch overview page.")
        return {
            "schema_version": "1.0",
            "scraper_version": "2026-08-06",
            "source": {
                "site": "vlr.gg",
                "match_url": overview_url,
                "scraped_at": scraped_at
            },
            "match_id": match_id,
            "overview": None,
            "performance": None,
            "economy": None
        }

    metadata, segments, map_info_by_gid = parse_overview_tab(overview_html)
    t1_name = metadata.get("teams", {}).get("team1", "Team 1")
    t2_name = metadata.get("teams", {}).get("team2", "Team 2")

    # 2. Performance Tab
    perf_url = f"{VLR_BASE_URL}/{match_id}/?tab=performance"
    logger.info(f"Fetching VLR Performance page from {perf_url}...")
    perf_html = fetch_url_with_curl(perf_url)
    perf_parsed = parse_performance_tab(perf_html, map_info_by_gid)
    if perf_parsed is None:
        performance = None
        perf_reason = "Stats from this map are not available yet"
    else:
        performance = perf_parsed
        perf_reason = None

    # 3. Economy Tab
    econ_url = f"{VLR_BASE_URL}/{match_id}/?tab=economy"
    logger.info(f"Fetching VLR Economy page from {econ_url}...")
    econ_html = fetch_url_with_curl(econ_url)
    econ_parsed = parse_economy_tab(econ_html, map_info_by_gid, segments=segments, overview_teams=(t1_name, t2_name))
    if econ_parsed is None:
        economy = None
        econ_reason = "Stats from this map are not available yet"
    else:
        economy = econ_parsed
        econ_reason = None

    result = {
        "schema_version": "1.0",
        "scraper_version": "2026-08-06",
        "source": {
            "site": "vlr.gg",
            "match_url": overview_url,
            "scraped_at": scraped_at
        },
        "match_id": match_id,
        "overview": {
            "metadata": metadata,
            "segments": segments
        },
        "performance": performance,
        "economy": economy
    }

    if perf_reason:
        result["performance_reason"] = perf_reason
    if econ_reason:
        result["economy_reason"] = econ_reason

    return result


if __name__ == "__main__":
    logger.info("Running analytics spec test VLR Scrape...")
    test_id = "712822"
    res = parse_vlr_match(test_id)
    if res:
        print(f"Scraped match {res.get('match_id')} successfully!")
        print(json.dumps(res, indent=2)[:1500] + "...")
    else:
        print("Scrape failed.")
