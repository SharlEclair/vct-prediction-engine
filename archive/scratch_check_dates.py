import httpx

BASE_URL = "http://localhost:3000"

def check_page(page):
    url = f"{BASE_URL}/v2/match?q=results&from_page={page}&to_page={page}"
    try:
        r = httpx.get(url, timeout=30.0)
        data = r.json()
        segs = data.get("data", {}).get("segments", [])
        if segs:
            print(f"Page {page}: first='{segs[0].get('time_completed')}' last='{segs[-1].get('time_completed')}'")
        else:
            print(f"Page {page}: Empty")
    except Exception as e:
        print(f"Page {page}: Error: {e}")

check_page(100)
check_page(150)
check_page(200)
check_page(250)
check_page(300)
check_page(350)
check_page(400)
check_page(450)
