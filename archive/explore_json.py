import json, os, httpx
from collections import Counter

# ── Probe the raw API role integers ──────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

with httpx.Client(timeout=20) as c:
    ev = c.get("https://api.valorantfantasyleague.net/api/event/currentevent", headers=HEADERS).json()
    event_id = ev["id"]
    players = c.get(
        "https://api.valorantfantasyleague.net/api/player/allplayers",
        params={"eventId": event_id}, headers=HEADERS
    ).json()

role_counts = Counter(p["playerRole"] for p in players)
print(f"Total players: {len(players)}")
print("Raw playerRole integer distribution:", dict(sorted(role_counts.items())))

# Show a sample of each role integer
for role_int in sorted(role_counts.keys()):
    sample = [p for p in players if p["playerRole"] == role_int][:3]
    print(f"\nrole={role_int} ({len([p for p in players if p['playerRole']==role_int])} players):")
    for p in sample:
        print(f"  {p['player']['name']:15} team={p['team']['shortName']}")
