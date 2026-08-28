import os
import json
import time
import random
import logging
import re
from datetime import datetime
from curl_cffi import requests
from selectolax.parser import HTMLParser

# Configure Logging
log_file = "./data/raw/harvest_all_t1.log"
os.makedirs("./data/raw", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("harvest_all_t1")

VLR_BASE_URL = "https://www.vlr.gg"

# HTML Parsing Helpers
_STRIP_RE = re.compile(
    r'<(script|style)[^>]*>.*?</\1>|<!--.*?-->',
    re.DOTALL | re.IGNORECASE,
)

def strip_html(html_str: str) -> str:
    return _STRIP_RE.sub("", html_str)

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def parse_html(text: str) -> HTMLParser:
    return HTMLParser(strip_html(text))

def extract_text_content(element, strip: bool = True) -> str:
    if not element:
        return ""
    text = element.text(strip=False)
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip() if strip else collapsed

def normalize_image_url(src: str) -> str:
    if not src:
        return ""
    if src.startswith("//"):
        return "https:" + src
    elif src.startswith("/"):
        return VLR_BASE_URL + src
    else:
        return src

def build_full_url(href: str) -> str:
    if not href:
        return ""
    return VLR_BASE_URL + href if href.startswith("/") else href

def parse_href_id_slug(href: str) -> tuple[str, str]:
    if not href:
        return "", ""
    parts = href.strip("/").split("/")
    for i, part in enumerate(parts):
        if part.isdigit():
            slug = parts[i + 1] if i + 1 < len(parts) else ""
            return part, slug
    return "", ""

# Match Details Parsing Logic
def _parse_event_info(html) -> dict:
    event_name = ""
    event_series = ""
    event_logo = ""
    super_elem = html.css_first(".match-header-super")
    if super_elem:
        first_div = super_elem.css_first("div")
        if first_div:
            anchor = first_div.css_first("a")
            if anchor:
                event_name = extract_text_content(anchor)
            else:
                event_name = extract_text_content(first_div)
        series_elem = super_elem.css_first(".match-header-event-series")
        if series_elem:
            event_series = extract_text_content(series_elem)
    logo_elem = html.css_first(".match-header-event img")
    if logo_elem:
        src = logo_elem.attributes.get("src", "")
        event_logo = normalize_image_url(src)
    return {"name": event_name, "series": event_series, "logo": event_logo}

def _parse_match_header(html) -> dict:
    date = ""
    patch = ""
    status = ""
    date_elem = html.css_first(".match-header-date")
    if date_elem:
        date = extract_text_content(date_elem)
    note_elem = html.css_first(".match-header-note")
    if note_elem:
        patch = extract_text_content(note_elem)
    vs_note_elem = html.css_first(".match-header-vs-note")
    if vs_note_elem:
        status = extract_text_content(vs_note_elem)
    return {"date": date, "map_vetos": patch, "status": status}

def _parse_teams(html) -> list[dict]:
    teams = []
    for mod in ("mod-1", "mod-2"):
        team_id = ""
        link_elem = html.css_first(f".match-header-link.{mod}")
        if link_elem:
            href = link_elem.attributes.get("href", "")
            team_id, _ = parse_href_id_slug(href)
        name_elem = html.css_first(f".match-header-link-name.{mod}")
        name = ""
        tag = ""
        if name_elem:
            full_text = name_elem.text()
            lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
            if lines:
                name = lines[0]
            if len(lines) > 1:
                tag = lines[1]
        teams.append({
            "id": team_id,
            "name": name,
            "tag": tag,
            "logo": "",
            "score": "",
            "is_winner": False,
        })
    vs_elem = html.css_first(".match-header-vs")
    if vs_elem:
        logos = vs_elem.css("img")
        for idx, img in enumerate(logos[:2]):
            src = img.attributes.get("src", "")
            if src:
                teams[idx]["logo"] = normalize_image_url(src)
    score_elems = html.css(".match-header-vs-score span")
    winner_idx = -1
    scored_spans = [
        (span.attributes.get("class") or "", span.text(strip=True))
        for span in score_elems
        if span.text(strip=True).isdigit()
    ]
    if len(scored_spans) >= 2:
        cls0, val0 = scored_spans[0]
        cls1, val1 = scored_spans[1]
        teams[0]["score"] = val0
        teams[1]["score"] = val1
        if "match-header-vs-score-winner" in cls0:
            winner_idx = 0
        elif "match-header-vs-score-winner" in cls1:
            winner_idx = 1
    if winner_idx >= 0:
        teams[winner_idx]["is_winner"] = True
    return teams

def _parse_streams_vods(html) -> tuple[list[dict], list[dict]]:
    streams = []
    vods = []
    for btn in html.css(".match-streams-btn"):
        href = btn.attributes.get("href", "")
        name = extract_text_content(btn)
        if name or href:
            streams.append({"name": name, "url": build_full_url(href)})
    vods_container = html.css_first(".match-vods")
    if vods_container:
        for anchor in vods_container.css("a"):
            href = anchor.attributes.get("href", "")
            name = extract_text_content(anchor)
            if name or href:
                vods.append({"name": name, "url": href})
    return streams, vods

def _parse_player_row(cells: list) -> dict:
    def cell_val(cell) -> str:
        if not cell:
            return ""
        both = cell.css_first(".side.mod-both")
        if both:
            return both.text(strip=True)
        return cell.text(strip=True)

    def safe_val(idx: int) -> str:
        return cell_val(cells[idx]) if idx < len(cells) else ""

    player_name = ""
    if cells:
        player_cell = cells[0]
        name_div = player_cell.css_first(".text-of")
        if name_div:
            player_name = name_div.text(strip=True)
        else:
            player_name = player_cell.text(strip=True)

    agent = ""
    if len(cells) > 1:
        img = cells[1].css_first("img")
        if img:
            agent = img.attributes.get("title", "") or img.attributes.get("alt", "")

    return {
        "name": player_name,
        "agent": agent,
        "rating": safe_val(2),
        "acs": safe_val(3),
        "kills": safe_val(4),
        "deaths": safe_val(5),
        "assists": safe_val(6),
        "kd_diff": safe_val(7),
        "kast": safe_val(8),
        "adr": safe_val(9),
        "hs_pct": safe_val(10),
        "fk": safe_val(11),
        "fd": safe_val(12),
        "fk_diff": safe_val(13),
    }

def _parse_map_players(game_elem) -> dict:
    team1_players = []
    team2_players = []
    tables = game_elem.css("table.wf-table-inset.mod-overview")

    def parse_table_rows(table) -> list[dict]:
        players = []
        for row in table.css("tbody tr"):
            cells = row.css("td")
            if not cells:
                continue
            if len(cells) < 5:
                continue
            try:
                players.append(_parse_player_row(cells))
            except Exception as exc:
                logger.debug(f"Skipping player row due to parse error: {exc}")
        return players

    if len(tables) >= 1:
        team1_players = parse_table_rows(tables[0])
    if len(tables) >= 2:
        team2_players = parse_table_rows(tables[1])

    return {"team1": team1_players, "team2": team2_players}

def _parse_map_scores(game_elem) -> dict:
    result = {
        "score": {"team1": "", "team2": ""},
        "score_ct": {"team1": "", "team2": ""},
        "score_t": {"team1": "", "team2": ""},
        "score_ot": {"team1": "", "team2": ""},
    }
    header = game_elem.css_first(".vm-stats-game-header")
    if not header:
        return result
    team_blocks = header.css(".team")
    keys = ["team1", "team2"]
    for idx, block in enumerate(team_blocks[:2]):
        key = keys[idx]
        score_el = block.css_first(".score")
        if score_el:
            val = score_el.text(strip=True)
            try:
                result["score"][key] = int(val)
            except (ValueError, TypeError):
                result["score"][key] = val
        ct_el = block.css_first(".mod-ct")
        if ct_el:
            result["score_ct"][key] = ct_el.text(strip=True)
        t_el = block.css_first(".mod-t")
        if t_el:
            result["score_t"][key] = t_el.text(strip=True)
        ot_el = block.css_first(".mod-ot")
        if ot_el:
            result["score_ot"][key] = ot_el.text(strip=True)
    return result

def _parse_rounds(game_elem) -> list[dict]:
    rounds = []
    rounds_container = game_elem.css_first(".vlr-rounds")
    if not rounds_container:
        return rounds
    round_num = 0
    for row in rounds_container.css(".vlr-rounds-row"):
        for col in row.css(".vlr-rounds-row-col"):
            cls = col.attributes.get("class", "")
            if "mod-spacing" in cls:
                continue
            sqs = col.css(".rnd-sq")
            if not sqs:
                continue
            round_num += 1
            winner = ""
            winning_side = ""
            for idx, sq in enumerate(sqs):
                sq_cls = sq.attributes.get("class", "")
                if "mod-win" in sq_cls:
                    winner = "team1" if idx == 0 else "team2"
                    if "mod-ct" in sq_cls:
                        winning_side = "ct"
                    elif "mod-t" in sq_cls:
                        winning_side = "t"
                    break
            rounds.append({
                "round_num": round_num,
                "winner": winner,
                "side": winning_side,
            })
    return rounds

def _parse_maps(html) -> list[dict]:
    maps = []
    for game_elem in html.css("div.vm-stats-game"):
        game_id = game_elem.attributes.get("data-game-id", "")
        if game_id == "all":
            continue
        map_name = ""
        picked_by = ""
        map_container = game_elem.css_first(".vm-stats-game-header .map")
        if map_container:
            pick_elem = map_container.css_first(".picked") or map_container.css_first(".pick")
            dur_elem = map_container.css_first(".map-duration")
            if pick_elem:
                picked_by = pick_elem.text(strip=True)
            full_text = map_container.text(strip=True)
            subtract = ""
            if pick_elem:
                subtract += pick_elem.text(strip=True)
            if dur_elem:
                subtract += dur_elem.text(strip=True)
            map_name = re.sub(r"\s+", " ", full_text.replace(subtract, "")).strip()

        duration = ""
        dur_elem = game_elem.css_first(".map-duration")
        if dur_elem:
            duration = dur_elem.text(strip=True)

        scores = _parse_map_scores(game_elem)
        players = _parse_map_players(game_elem)
        rounds = _parse_rounds(game_elem)

        maps.append({
            "map_name": map_name,
            "picked_by": picked_by,
            "duration": duration,
            "score": scores["score"],
            "score_ct": scores["score_ct"],
            "score_t": scores["score_t"],
            "score_ot": scores["score_ot"],
            "players": players,
            "rounds": rounds,
        })
    return maps

def _parse_head_to_head(html) -> list[dict]:
    h2h = []
    container = html.css_first(".match-h2h-matches")
    if not container:
        return h2h
    for row in container.css(".wf-module-item"):
        team_elems = row.css(".match-h2h-matches-team")
        teams = []
        for te in team_elems:
            cls = te.attributes.get("class", "")
            is_winner = "mod-win" in cls
            teams.append({"name": extract_text_content(te), "is_winner": is_winner})
        score_elem = row.css_first(".match-h2h-matches-score")
        score = extract_text_content(score_elem) if score_elem else ""
        event_elem = row.css_first(".match-h2h-matches-event-name")
        event = extract_text_content(event_elem) if event_elem else ""
        date_elem = row.css_first(".match-h2h-matches-date")
        date = extract_text_content(date_elem) if date_elem else ""
        href = row.attributes.get("href", "")
        url = build_full_url(href)
        h2h.append({
            "event": event,
            "date": date,
            "teams": teams,
            "score": score,
            "url": url,
        })
    return h2h

def _extract_game_ids(html) -> list[str]:
    game_ids = []
    for item in html.css(".vm-stats-gamesnav-item"):
        gid = item.attributes.get("data-game-id", "")
        if gid and gid != "all":
            game_ids.append(gid)
    return game_ids

def _parse_kill_matrix(html) -> list[dict]:
    matrix = []
    table = html.css_first("table.wf-table-inset.mod-matrix.mod-normal")
    if not table:
        return matrix
    header_row = table.css_first("thead tr")
    opponents = []
    if header_row:
        for th in header_row.css("th"):
            opponents.append(extract_text_content(th))
    for row in table.css("tbody tr"):
        cells = row.css("td")
        if not cells:
            continue
        player_cell = cells[0]
        player_name = extract_text_content(player_cell)
        kills_vs = {}
        for idx, cell in enumerate(cells[1:], start=1):
            opponent = opponents[idx] if idx < len(opponents) else str(idx)
            kills_vs[opponent] = extract_text_content(cell)
        matrix.append({"player": player_name, "kills_vs": kills_vs})
    return matrix

def _parse_advanced_stats(html) -> list[dict]:
    advanced = []
    table = html.css_first("table.wf-table-inset.mod-adv-stats")
    if not table:
        return advanced
    header_row = table.css_first("thead tr")
    headers = []
    if header_row:
        for th in header_row.css("th"):
            headers.append(extract_text_content(th))
    for row in table.css("tbody tr"):
        cells = row.css("td")
        if not cells:
            continue
        player_name = extract_text_content(cells[0]) if cells else ""
        stat_dict = {"player": player_name}
        for idx, cell in enumerate(cells[1:], start=1):
            label = headers[idx] if idx < len(headers) else str(idx)
            stat_dict[label] = extract_text_content(cell)
        advanced.append(stat_dict)
    return advanced

def _parse_economy(html) -> list[dict]:
    economy = []
    table = html.css_first("table.wf-table-inset.mod-econ")
    if not table:
        return economy
    header_row = table.css_first("thead tr")
    headers = []
    if header_row:
        for th in header_row.css("th"):
            headers.append(extract_text_content(th))
    for row in table.css("tbody tr"):
        cells = row.css("td")
        if not cells:
            continue
        row_dict = {}
        for idx, cell in enumerate(cells):
            label = headers[idx] if idx < len(headers) else str(idx)
            row_dict[label] = extract_text_content(cell)
        economy.append(row_dict)
    return economy

# Whitelist / Blacklist Filters
def is_tier1_event(event_name: str) -> bool:
    event_lower = event_name.lower()
    
    # Exclude Filters
    exclude_tokens = ["challengers", "ascension", "game changers", "gc", "open qualifier", "showmatch"]
    if any(tok in event_lower for tok in exclude_tokens):
        return False
        
    # Allow Filters
    allow_tokens = ["champions", "masters", "kickoff", "americas", "emea", "pacific"]
    if any(tok in event_lower for tok in allow_tokens):
        return True
        
    # Word boundary check for "CN"
    if re.search(r'\b(CN|cn)\b', event_name):
        return True
        
    return False

# Networking with Jitter and Exponential Backoff
def fetch_with_backoff(session, url, headers, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Jittered delay before the request (3.0 + random.uniform(0.5, 2.5) seconds)
            delay = 3.0 + random.uniform(0.5, 2.5)
            logger.info(f"Sleeping for {delay:.2f}s before hitting VLR.gg...")
            time.sleep(delay)
            
            logger.info(f"Fetching URL: {url}")
            response = session.get(url, headers=headers, timeout=20.0)
            
            # Check for blocking status codes
            if response.status_code in [429, 403]:
                penalty = (2 ** attempt) * 10
                logger.warning(f"Challenged/Blocked (status {response.status_code}) on {url}. Backoff penalty: sleeping for {penalty}s...")
                time.sleep(penalty)
                continue
                
            if response.status_code != 200:
                logger.warning(f"Received non-200 status code {response.status_code} for {url}. Retrying...")
                continue
                
            return response.text
            
        except Exception as e:
            penalty = (2 ** attempt) * 10
            logger.error(f"Networking exception on {url} (attempt {attempt+1}/{max_retries}): {e}. Backoff penalty: sleeping for {penalty}s...")
            time.sleep(penalty)
            
    logger.critical(f"Failed to fetch {url} after {max_retries} attempts.")
    return None

def fetch_game_tab(session, base_url, game_id, tab, headers):
    url = f"{base_url}/?game={game_id}&tab={tab}"
    html_text = fetch_with_backoff(session, url, headers)
    if not html_text:
        return None
    return parse_html(html_text)

# Scrape Match Page + tab details
def scrape_match_details(session, match_id, headers) -> dict:
    base_url = f"{VLR_BASE_URL}/{match_id}"
    base_html_text = fetch_with_backoff(session, base_url, headers)
    if not base_html_text:
        return {"status": "error", "message": f"Failed to fetch base match page for {match_id}"}
        
    base_html = parse_html(base_html_text)
    
    # Parse games (maps)
    game_ids = _extract_game_ids(base_html)
    first_game_id = game_ids[0] if game_ids else None
    
    performance_by_game = {}
    economy_by_game = {}
    
    for game_id in game_ids:
        # Performance
        perf_html = fetch_game_tab(session, base_url, game_id, "performance", headers)
        if perf_html:
            performance_by_game[game_id] = {
                "kill_matrix": _parse_kill_matrix(perf_html),
                "advanced_stats": _parse_advanced_stats(perf_html)
            }
        else:
            performance_by_game[game_id] = {"kill_matrix": [], "advanced_stats": []}
            
        # Economy
        econ_html = fetch_game_tab(session, base_url, game_id, "economy", headers)
        if econ_html:
            economy_by_game[game_id] = _parse_economy(econ_html)
        else:
            economy_by_game[game_id] = []
            
    event_info = _parse_event_info(base_html)
    header_info = _parse_match_header(base_html)
    teams = _parse_teams(base_html)
    streams, vods = _parse_streams_vods(base_html)
    maps = _parse_maps(base_html)
    h2h = _parse_head_to_head(base_html)
    
    for index, map_data in enumerate(maps):
        game_id = game_ids[index] if index < len(game_ids) else ""
        map_data["performance"] = performance_by_game.get(
            game_id, {"kill_matrix": [], "advanced_stats": []}
        )
        map_data["economy"] = economy_by_game.get(game_id, [])
        
    first_game_performance = performance_by_game.get(
        first_game_id or "", {"kill_matrix": [], "advanced_stats": []}
    )
    first_game_economy = economy_by_game.get(first_game_id or "", [])
    
    segment = {
        "match_id": match_id,
        "event": event_info,
        "date": header_info["date"],
        "map_vetos": header_info["map_vetos"],
        "status": header_info["status"],
        "teams": teams,
        "streams": streams,
        "vods": vods,
        "maps": maps,
        "head_to_head": h2h,
        "performance": {
            "kill_matrix": first_game_performance["kill_matrix"],
            "advanced_stats": first_game_performance["advanced_stats"],
            "by_map": [
                {"game_id": gid, **performance_by_game.get(gid, {"kill_matrix": [], "advanced_stats": []})}
                for gid in game_ids
            ]
        },
        "economy": first_game_economy,
        "economy_by_map": [
            {"game_id": gid, "rows": economy_by_game.get(gid, [])}
            for gid in game_ids
        ]
    }
    
    return {"status": "success", "data": {"status": 200, "segments": [segment]}}

# Main Orchestration Loop
def run_harvester(dry_run=False):
    logger.info("Initializing VCT Tier 1 Consolidated Ingestion Engine...")
    session = requests.Session(impersonate="chrome")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    
    target_dir = "./data/raw"
    os.makedirs(target_dir, exist_ok=True)
    
    page = 1
    total_processed = 0
    total_skipped = 0
    total_downloaded = 0
    
    while True:
        url = f"{VLR_BASE_URL}/matches/results?page={page}"
        logger.info(f"Traversing results page {page}...")
        html_text = fetch_with_backoff(session, url, headers)
        if not html_text:
            logger.error(f"Failed to fetch results page {page}. Retrying on next cycle...")
            # If page fails completely after all backoffs, safety stop
            break
            
        parser = HTMLParser(html_text)
        col = parser.css_first(".col.mod-1")
        if not col:
            logger.info(f"Page {page}: no col container found. End of results.")
            break
            
        divs = col.css("div")
        if not divs:
            logger.info(f"Page {page}: no child divs. End of results.")
            break
            
        current_date_str = ""
        current_date_dt = None
        matches_on_page = 0
        
        # Traverse direct children divs in DOM order
        for div in divs:
            parent = div.parent
            if not parent:
                continue
            parent_cls = parent.attributes.get("class", "") if parent.attributes else ""
            if "col" not in parent_cls or "mod-1" not in parent_cls:
                continue
                
            cls = div.attributes.get("class", "") if div.attributes else ""
            if "wf-label" in cls and "mod-large" in cls:
                date_text = clean_text(div.text())
                for suffix in ["Today", "Yesterday"]:
                    if date_text.endswith(suffix):
                        date_text = date_text[:-len(suffix)].strip()
                current_date_str = date_text
                try:
                    current_date_dt = datetime.strptime(current_date_str, "%a, %B %d, %Y")
                    logger.info(f"Encountered date boundary: {current_date_str} (parsed year: {current_date_dt.year})")
                except ValueError:
                    current_date_dt = None
                    logger.warning(f"Could not parse date text: '{current_date_str}'")
                    
            elif "wf-card" in cls:
                if current_date_dt and current_date_dt.year <= 2022:
                    logger.info(f"Reached terminal year boundary ({current_date_dt.year} <= 2022). Stopping harvest.")
                    return
                    
                matches = div.css("a.wf-module-item")
                for m in matches:
                    matches_on_page += 1
                    href = m.attributes.get("href", "")
                    match_id, _ = parse_href_id_slug(href)
                    if not match_id:
                        continue
                        
                    tourney = clean_text(m.css_first(".match-item-event").text()) if m.css_first(".match-item-event") else ""
                    if not is_tier1_event(tourney):
                        # Drop lower tier match
                        continue
                        
                    total_processed += 1
                    logger.info(f"Found Tier 1 Match: {match_id} under event '{tourney}' ({current_date_str})")
                    
                    # Cache check
                    out_path = os.path.join(target_dir, f"match_{match_id}.json")
                    if os.path.exists(out_path):
                        total_skipped += 1
                        logger.info(f"Skipping cached match {match_id} (exists at {out_path}).")
                        continue
                        
                    logger.info(f"Ingesting missing match details for ID {match_id}...")
                    match_data = scrape_match_details(session, match_id, headers)
                    
                    if match_data.get("status") == "success":
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(match_data, f, indent=4, ensure_ascii=False)
                        total_downloaded += 1
                        logger.info(f"Successfully cached match_{match_id}.json ({total_downloaded} downloaded in this session).")
                    else:
                        logger.error(f"Failed to scrape match {match_id}: {match_data.get('message')}")
                        
                    if dry_run and total_processed >= 1:
                        logger.info("Dry-run target met. Stopping.")
                        return
                        
        logger.info(f"Page {page} complete. Found {matches_on_page} matches.")
        
        # Stop check for page limits (safety cap at 200 pages)
        if page >= 200:
            logger.info("Reached maximum page traversal limit (page 200). Stopping.")
            break
            
        page += 1

if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    run_harvester(dry_run=dry)
