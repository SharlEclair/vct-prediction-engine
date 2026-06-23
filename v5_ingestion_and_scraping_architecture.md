````markdown id="8s9b4a"
# 1. Bypassing WAF / Edge Firewalls (The 403 Fix)

The HTTP `403 Forbidden` error currently breaking the Patch Notes Ingestor (Fandom Wiki) and historically plaguing VLR.gg scrapers is caused by Cloudflare and Akamai Web Application Firewalls (WAF).

Standard Python `requests` and `httpx` libraries have predictable JA3/TLS fingerprints that are instantly flagged by these edge networks.

## The Solution

The ingestion pipeline must universally transition to `curl_cffi`.

By compiling native `libcurl` bindings, the scraper mimics the exact HTTP/2 window settings and cipher suites of modern browsers:

```python
impersonate="chrome"
````

This will successfully fetch the Fandom MediaWiki raw text and bypass VLR.gg's dynamic blocks.

---

# 2. VLR.gg Deep Scraping Pipeline

The VLR match telemetry is divided between server-rendered HTML and dynamically loaded internal endpoints.

The V5 scraping pipeline must execute a sequential extraction graph:

## 1. Base HTML Extraction

```http
GET /{match_url}
```

Extracts:

* Map vetoes
* Standard scorelines
* Base player stats
* Base agent compositions

---

## 2. Performance Endpoint

```http
GET /match/tab/performance?match_id={id}&game_id={id}
```

Extracts micro-stats:

* First Kills
* First Deaths
* Clutches
* Multikills

---

## 3. Economy Endpoint

```http
GET /match/tab/economy?match_id={id}&game_id={id}
```

Extracts:

* Eco-win rates
* Buy-round conversions

---

## 4. Data Consolidation

These three streams must be unified into a single hierarchical JSON object per match.

```
```
