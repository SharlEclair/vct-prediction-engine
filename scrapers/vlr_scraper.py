import os
import re
import json
import logging
import random
import time
from curl_cffi import requests
from selectolax.parser import HTMLParser

logger = logging.getLogger("vlr_scraper")
logging.basicConfig(level=logging.INFO)

VLR_BASE_URL = "https://www.vlr.gg"

_session = None

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

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def parse_vlr_match(match_id_or_url: str) -> list[dict]:
    """
    Parses VLR Match page, performance tabs, and economy tabs.
    Consolidates them into a list of map-level standardized JSONs containing:
    patch, event, map, team1, team2, winner, composition, players, round_history, performance, economy
    """
    # Normalize match_id
    match_id = match_id_or_url.strip("/").split("/")[0]
    
    # 1. Fetch base match page
    match_url = f"{VLR_BASE_URL}/{match_id}"
    logger.info(f"Fetching VLR base match page from {match_url}...")
    html_text = fetch_url_with_curl(match_url)
    if not html_text:
        logger.error("Failed to fetch base match page.")
        return []
        
    parser = HTMLParser(html_text)
    
    # Extract Event
    event_elem = parser.css_first(".match-header-super a")
    event = clean_text(event_elem.text()) if event_elem else ""
    if not event:
        event_div = parser.css_first(".match-header-super")
        if event_div:
            event = clean_text(event_div.text())
            
    # Extract Patch
    patch = "Unknown"
    note_elem = parser.css_first(".match-header-note")
    if note_elem:
        note_text = clean_text(note_elem.text())
        patch_match = re.search(r'Patch\s+([0-9.]+)', note_text)
        if patch_match:
            patch = patch_match.group(1).strip()
            
    # Extract Date
    date_str = ""
    date_elem = parser.css_first(".match-header-date")
    if date_elem:
        date_str = clean_text(date_elem.text())
    if patch and patch != "Unknown" and "Patch" not in date_str:
        date_str = f"{date_str} Patch {patch}"

    # Extract Teams
    team1_elem = parser.css_first(".match-header-link-name.mod-1")
    team1 = clean_text(team1_elem.text()) if team1_elem else "Team 1"
    
    team2_elem = parser.css_first(".match-header-link-name.mod-2")
    team2 = clean_text(team2_elem.text()) if team2_elem else "Team 2"
    
    # Extract Winner
    winner = ""
    scores = parser.css(".match-header-vs-score span")
    scored_spans = [s for s in scores if s.text(strip=True).isdigit()]
    if len(scored_spans) >= 2:
        cls0 = scored_spans[0].attributes.get("class", "")
        cls1 = scored_spans[1].attributes.get("class", "")
        if "winner" in cls0:
            winner = team1
        elif "winner" in cls1:
            winner = team2
            
    # Extract Vetoes
    vetoes = []
    note_elems = parser.css(".match-header-note")
    for el in note_elems:
        txt = clean_text(el.text())
        parts = re.split(r'[;\n\r]', txt)
        for part in parts:
            part = part.strip()
            if any(w in part.lower() for w in ["ban", "pick", "remains", "decider"]):
                vetoes.append(part)

    # Game IDs for tabs
    game_ids = []
    for item in parser.css(".vm-stats-gamesnav-item"):
        gid = item.attributes.get("data-game-id", "")
        if gid and gid != "all":
            game_ids.append(gid)
            
    # Parse games (maps)
    map_segments = []
    game_blocks = parser.css("div.vm-stats-game")
    
    # Filter out 'all' map blocks
    valid_game_blocks = []
    for gb in game_blocks:
        gid = gb.attributes.get("data-game-id", "")
        if gid and gid != "all":
            valid_game_blocks.append(gb)
            
    for idx, gb in enumerate(valid_game_blocks):
        game_id = game_ids[idx] if idx < len(game_ids) else ""
        
        # Map Name
        map_name = "Unknown"
        map_elem = gb.css_first(".map")
        if map_elem:
            pick_el = map_elem.css_first(".picked") or map_elem.css_first(".pick")
            dur_el = map_elem.css_first(".map-duration")
            sub = ""
            if pick_el:
                sub += pick_el.text()
            if dur_el:
                sub += dur_el.text()
            map_name = clean_text(map_elem.text().replace(sub, ""))
            
        # Map Winner
        map_winner = ""
        scores_block = gb.css(".team")
        if len(scores_block) >= 2:
            s1_el = scores_block[0].css_first(".score")
            s2_el = scores_block[1].css_first(".score")
            s1 = int(s1_el.text()) if s1_el and s1_el.text().isdigit() else 0
            s2 = int(s2_el.text()) if s2_el and s2_el.text().isdigit() else 0
            if s1 > s2:
                map_winner = team1
            elif s2 > s1:
                map_winner = team2
                
        # Composition & Players
        composition = {team1: [], team2: []}
        players_list = []
        
        tables = gb.css("table.wf-table-inset.mod-overview")
        for t_idx, table in enumerate(tables):
            current_team = team1 if t_idx == 0 else team2
            for row in table.css("tbody tr"):
                cells = row.css("td")
                if len(cells) < 5:
                    continue
                
                # Player Name
                p_cell = cells[0]
                p_name_el = p_cell.css_first(".text-of")
                p_name = clean_text(p_name_el.text()) if p_name_el else clean_text(p_cell.text())
                
                # Agent
                agent = ""
                img = cells[1].css_first("img")
                if img:
                    agent = img.attributes.get("title", "") or img.attributes.get("alt", "")
                
                if agent:
                    composition[current_team].append(agent)
                    
                # Stats
                def get_val(c) -> str:
                    both = c.css_first(".side.mod-both")
                    return both.text(strip=True) if both else c.text(strip=True)
                    
                rating = get_val(cells[2])
                acs = get_val(cells[3])
                kills = get_val(cells[4])
                deaths = get_val(cells[5])
                assists = get_val(cells[6])
                kast = get_val(cells[8])
                adr = get_val(cells[9])
                hs_pct = get_val(cells[10])
                fk = get_val(cells[11])
                fd = get_val(cells[12])
                
                players_list.append({
                    "name": p_name,
                    "team": current_team,
                    "agent": agent,
                    "rating": rating,
                    "acs": acs,
                    "kills": kills,
                    "deaths": deaths,
                    "assists": assists,
                    "kast": kast,
                    "adr": adr,
                    "hs_pct": hs_pct,
                    "fk": fk,
                    "fd": fd
                })
                
        # Round History
        round_history = []
        rounds_row = gb.css(".vlr-rounds .vlr-rounds-row")
        r_num = 0
        for r_row in rounds_row:
            for col in r_row.css(".vlr-rounds-row-col"):
                if "mod-spacing" in col.attributes.get("class", ""):
                    continue
                sqs = col.css(".rnd-sq")
                if not sqs:
                    continue
                r_num += 1
                r_winner = ""
                side = ""
                for sq_idx, sq in enumerate(sqs):
                    sq_cls = sq.attributes.get("class", "")
                    if "mod-win" in sq_cls:
                        r_winner = team1 if sq_idx == 0 else team2
                        if "mod-ct" in sq_cls:
                            side = "ct"
                        elif "mod-t" in sq_cls:
                            side = "t"
                        break
                round_history.append({
                    "round_num": r_num,
                    "winner": r_winner,
                    "side": side
                })
                
        # Fetch dynamic sub-tabs
        performance_data = {}
        economy_data = []
        
        if game_id:
            # Fetch Performance Tab
            perf_url = f"{VLR_BASE_URL}/match/tab/performance?match_id={match_id}&game_id={game_id}"
            logger.info(f"Fetching performance tab for game {game_id}...")
            perf_html = fetch_url_with_curl(perf_url)
            if perf_html:
                perf_parser = HTMLParser(perf_html)
                
                # Advanced stats
                adv_table = perf_parser.css_first("table.wf-table-inset.mod-adv-stats")
                advanced_list = []
                if adv_table:
                    headers = [clean_text(th.text()) for th in adv_table.css("thead tr th")]
                    for row in adv_table.css("tbody tr"):
                        cells = row.css("td")
                        if not cells:
                            continue
                        p_name = clean_text(cells[0].text())
                        stat_dict = {"player": p_name}
                        for cell_idx, cell in enumerate(cells[1:], start=1):
                            label = headers[cell_idx] if cell_idx < len(headers) else str(cell_idx)
                            stat_dict[label] = clean_text(cell.text())
                        advanced_list.append(stat_dict)
                performance_data = {"advanced_stats": advanced_list}
                
            # Fetch Economy Tab
            econ_url = f"{VLR_BASE_URL}/match/tab/economy?match_id={match_id}&game_id={game_id}"
            logger.info(f"Fetching economy tab for game {game_id}...")
            econ_html = fetch_url_with_curl(econ_url)
            if econ_html:
                econ_parser = HTMLParser(econ_html)
                econ_table = econ_parser.css_first("table.wf-table-inset.mod-econ")
                if econ_table:
                    headers = [clean_text(th.text()) for th in econ_table.css("thead tr th")]
                    for row in econ_table.css("tbody tr"):
                        cells = row.css("td")
                        if not cells:
                            continue
                        row_dict = {}
                        for cell_idx, cell in enumerate(cells):
                            label = headers[cell_idx] if cell_idx < len(headers) else str(cell_idx)
                            row_dict[label] = clean_text(cell.text())
                        economy_data.append(row_dict)
                        
        map_segments.append({
            "date": date_str,
            "patch": patch,
            "event": event,
            "map": map_name,
            "team1": team1,
            "team2": team2,
            "winner": map_winner,
            "composition": composition,
            "players": players_list,
            "round_history": round_history,
            "performance": performance_data,
            "economy": economy_data,
            "vetoes": vetoes
        })
        
    return map_segments

if __name__ == "__main__":
    # Test scrape for paper-rex-vs-leviatan (id: 353198 or similar, or 670471)
    logger.info("Running test VLR Scrape...")
    test_id = "670471"
    res = parse_vlr_match(test_id)
    if res:
        print(f"Scraped {len(res)} maps successfully!")
        print("Map 1 Details:")
        print(json.dumps(res[0], indent=2)[:500] + "...")
    else:
        print("Scrape failed.")
