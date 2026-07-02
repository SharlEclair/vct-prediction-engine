import httpx
import json

BASE_URL = "http://localhost:3000"

url = f"{BASE_URL}/v2/match?q=results&from_page=1&to_page=1"
try:
    r = httpx.get(url, timeout=30.0)
    data = r.json()
    print("Keys of JSON response:", data.keys())
    if "data" in data:
        segments = data["data"].get("segments", [])
        print(f"Number of segments returned: {len(segments)}")
        if segments:
            print("First segment keys:", segments[0].keys())
            print("First segment sample:", json.dumps(segments[0], indent=2))
        else:
            print("No segments in 'data'.")
    else:
        print("No 'data' key in response.")
except Exception as e:
    print("Error:", e)
