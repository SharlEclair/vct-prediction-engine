import httpx

BASE_URL = "http://localhost:3000"

def get_page_date(page):
    url = f"{BASE_URL}/v2/match?q=results&from_page={page}&to_page={page}"
    try:
        r = httpx.get(url, timeout=30.0)
        data = r.json()
        segments = data.get("data", {}).get("segments", [])
        if not segments:
            return None
        # Get date of the first and last match on the page
        first_date = segments[0].get("date", "")
        last_date = segments[-1].get("date", "")
        return first_date, last_date
    except Exception as e:
        return f"Error: {e}"

# Probe some page numbers
pages_to_probe = [1, 50, 100, 150, 200, 250, 300]
for p in pages_to_probe:
    res = get_page_date(p)
    print(f"Page {p}: {res}")
