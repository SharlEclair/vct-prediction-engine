import json

with open("./data/raw/match_670466.json", "r", encoding="utf-8") as f:
    match_data = json.load(f)

seg = match_data["data"]["segments"][0]
print(f"Match Date: {seg['date']}")
print(f"Teams: {seg['teams'][0]['name']} vs {seg['teams'][1]['name']}")

for map_data in seg.get("maps", []):
    print(f"\nMap Name: {map_data['map_name']}")
    for team_key in ['team1', 'team2']:
        print(f"  Team: {seg['teams'][0]['name'] if team_key == 'team1' else seg['teams'][1]['name']}")
        for p in map_data.get('players', {}).get(team_key, []):
            if p['name'].lower() == 'something':
                print(f"    Player: {p['name']} | Agent: {p.get('agent')} | ACS: {p.get('acs')}")
