"""
Safe API probe script for VFL endpoints.
Prints key structure only — never dumps full payloads.
"""
import json
import re
import httpx

BASE = "https://api.valorantfantasyleague.net/api"
CURRENT_EVENT_URL = f"{BASE}/event/currentevent"
ALL_PLAYERS_URL   = f"{BASE}/player/allplayers"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

def safe_keys(obj, depth=0, max_depth=4, prefix=""):
    """Recursively print key names and types up to max_depth."""
    indent = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            vtype = type(v).__name__
            extra = f" (len={len(v)})" if isinstance(v, (dict, list)) else f" = {repr(v)[:60]}"
            print(f"{indent}{prefix}{k} ({vtype}){extra}")
            if depth < max_depth:
                safe_keys(v, depth + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{indent}[List, {len(obj)} items]")
        if obj and depth < max_depth:
            print(f"{indent}  -- first item --")
            safe_keys(obj[0], depth + 1, max_depth)

with httpx.Client(timeout=20.0, follow_redirects=True) as client:
    # ── Step A: current event ──────────────────────────────────────────────
    print("=" * 60)
    print("Step A: GET", CURRENT_EVENT_URL)
    r = client.get(CURRENT_EVENT_URL, headers=HEADERS)
    print(f"Status: {r.status_code}  |  Content-Type: {r.headers.get('content-type','?')}")
    
    if r.status_code == 200:
        event_data = r.json()
        print("\n-- currentevent structure --")
        safe_keys(event_data)

        # Try common id field names
        event_id = None
        if isinstance(event_data, dict):
            for key in ("id", "_id", "eventId", "event_id", "Id"):
                if key in event_data:
                    event_id = event_data[key]
                    print(f"\n  >> Resolved event_id via key '{key}': {event_id}")
                    break
        elif isinstance(event_data, list) and event_data:
            first = event_data[0]
            for key in ("id", "_id", "eventId", "event_id", "Id"):
                if key in first:
                    event_id = first[key]
                    print(f"\n  >> Resolved event_id from list[0] key '{key}': {event_id}")
                    break

        # ── Step B: all players ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"Step B: GET {ALL_PLAYERS_URL}?eventId={event_id}")
        r2 = client.get(ALL_PLAYERS_URL, params={"eventId": event_id}, headers=HEADERS)
        print(f"Status: {r2.status_code}  |  Content-Type: {r2.headers.get('content-type','?')}")
        
        if r2.status_code == 200:
            players_data = r2.json()
            print(f"\n-- allplayers response type: {type(players_data).__name__} --")
            safe_keys(players_data, max_depth=3)

            # Figure out actual player list
            player_list = None
            if isinstance(players_data, list):
                player_list = players_data
            elif isinstance(players_data, dict):
                for k, v in players_data.items():
                    if isinstance(v, list) and len(v) > 0:
                        player_list = v
                        print(f"\n  >> Player list found under key: '{k}'")
                        break

            if player_list:
                print(f"\n  >> Total players: {len(player_list)}")
                print("\n  -- Sample player (first item) full dump --")
                # Print full first item — it's a single player dict, size manageable
                for k, v in player_list[0].items():
                    print(f"    {k}: {repr(v)[:120]}")
            else:
                print("\n  Could not locate player list in response.")
        else:
            print(f"  Error body (first 300 chars): {r2.text[:300]}")
    else:
        print(f"  Error body (first 300 chars): {r.text[:300]}")
