````markdown
# 1. Scraping Architecture & Target Spectrum

To fully populate the V5 Simulation Engine's historical feature matrices, the telemetry pipeline must ingest a comprehensive historical dataset spanning from 2023 through the current 2026 season.

This multi-year data horizon is required to properly calculate the exponential moving averages (EMA) of team performance and establish stable baseline player comfort metrics.

The collection task executes across the standard VLR endpoint hierarchy, parsing match-level metadata from the main results pages.

### Base Pagination Target

```text
https://www.vlr.gg/matches/results?page={n}
````

### Temporal Boundaries

The ingestion engine recursively traverses pages backwards from the current 2026 match list, systematically processing all records until it hits the terminal boundary at the beginning of the 2023 competitive season.

---

# 2. Rigid Tier 1 Token Filtering Matrix

To protect the downstream models from low-tier statistical noise, the scraping engine employs a strict multi-layer boolean match filter on the parsed tournament event name string.

Every match row discovered must pass a strict whitelist validation and bypass a comprehensive blacklist match.

## The Whitelist (Inclusion Criteria)

A match is exclusively retained if its event string matches one or more of the following core Tier 1 tokens:

* `"Champions"` (e.g., Champions Seoul, Champions Berlin)
* `"Masters"` (e.g., Masters London 2026, Masters Madrid)
* `"Kickoff"` (Regional season-opening qualifiers)

### International Leagues

* `"Americas"`
* `"EMEA"`
* `"Pacific"`
* `"CN"`

## The Blacklist (Exclusion Criteria)

Even if a whitelist token is triggered, the match is instantly dropped if any of the following Tier 2/3, developmental, or casual tokens are detected:

* `"Challengers"` (Regional Tier-2 leagues)
* `"Ascension"` (Tier-2 promotion tournaments)
* `"Game Changers"` / `"GC"` (Impact/Developmental circuits)
* `"Open Qualifier"` (Amateur/Open brackets)
* `"Showmatch"` (Off-season exhibition games)

---

# 3. Anti-Friction Resilience Layer

Given that the host system was previously shadow-banned, the ingestion framework wraps all networking tasks inside an active browser-impersonation layer utilizing `curl_cffi`.

Requests scale dynamically to throw off signature tracking:

$$
\Delta t_{\text{sleep}} = 3.0 + \mathcal{U}(0.5, 2.5)
$$

If an edge security wall issues a challenge or connection reset, the client catches the error, holds the thread for an exponential penalty sleep cycle:

$$
2^k \times 10 \text{ seconds}
$$

and attempts a re-route.

```
```
