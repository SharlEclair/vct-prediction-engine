import json
import os
import sys
import httpx
from selectolax.parser import HTMLParser

URL = "https://www.valorantfantasyleague.net/playerstats"
DEBUG_JSON_PATH = os.path.join("data", "processed", "vfl_debug_output.json")

def test_scrape():
    print(f"Fetching VFL player stats from: {URL}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(URL, headers=headers)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Failed to fetch. Status: {response.status_code}")
                return
                
            html = response.text
            print(f"HTML length: {len(html)} characters")
            
            # Use selectolax to search for __NEXT_DATA__
            parser = HTMLParser(html)
            next_data_script = parser.css_first('script[id="__NEXT_DATA__"]')
            
            if next_data_script:
                print("Found <script id=\"__NEXT_DATA__\"> element!")
                script_text = next_data_script.text()
                
                try:
                    data = json.loads(script_text)
                    print("Successfully parsed __NEXT_DATA__ as JSON.")
                    
                    # Ensure directory exists and save to debug file
                    os.makedirs(os.path.dirname(DEBUG_JSON_PATH), exist_ok=True)
                    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved complete JSON payload to: {DEBUG_JSON_PATH}")
                    
                    # Print high-level keys
                    print("Root keys:", list(data.keys()))
                    if "props" in data:
                        print("Props keys:", list(data["props"].keys()))
                        if "pageProps" in data["props"]:
                            print("pageProps keys:", list(data["props"]["pageProps"].keys()))
                            # Try to find common data list keys (like players, roster, stats, etc.)
                            for k, v in data["props"]["pageProps"].items():
                                if isinstance(v, list):
                                    print(f"  Found list key in pageProps: '{k}' (length: {len(v)})")
                                    if len(v) > 0:
                                        print(f"  Sample item from '{k}':")
                                        print(json.dumps(v[0], indent=2))
                                elif isinstance(v, dict):
                                    print(f"  Found dict key in pageProps: '{k}' keys: {list(v.keys())}")
                                    for sub_k, sub_v in v.items():
                                        if isinstance(sub_v, list):
                                            print(f"    Sub-key list: '{sub_k}' (length: {len(sub_v)})")
                                            if len(sub_v) > 0:
                                                print(f"    Sample item from '{sub_k}':")
                                                print(json.dumps(sub_v[0], indent=2))
                    
                except Exception as e:
                    print(f"Error parsing script inner text as JSON: {e}")
            else:
                print("Could NOT find <script id=\"__NEXT_DATA__\"> element.")
                # Look for API calls in the HTML
                print("Searching for api.valorantfantasyleague.net or similar in raw HTML...")
                matches = []
                for line in html.splitlines():
                    if "api.valorantfantasyleague" in line:
                        matches.append(line.strip())
                if matches:
                    print(f"Found {len(matches)} occurrences containing api.valorantfantasyleague.net:")
                    for match in matches[:5]:
                        print(f"  {match[:120]}...")
                else:
                    print("No references to api.valorantfantasyleague.net found in the HTML.")
                    
    except Exception as e:
        print(f"Error during request/parsing: {e}")

if __name__ == "__main__":
    test_scrape()
