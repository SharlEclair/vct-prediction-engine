# 1. Mitigating Cloudflare Firewalls & Network Timeouts

Automated programmatic collection from `vlr.gg` frequently triggers edge security mitigation layers (such as Cloudflare or automated anti-bot firewalls).

This results in:

- Connection drops
- SSL handshake errors
- Classic `ERR_TIMED_OUT` status failures

Traditional standard HTTP clients (e.g., standard Python `requests` or `httpx`) fail because their TLS client hello signatures and JA3 fingerprints do not match real browser execution profiles.

To bypass these edge-network constraints, the V5 ingestion framework transitions to an impersonation architecture using **`curl_cffi`**.

This library compiles native `libcurl` bindings to accurately mirror the exact browser fingerprints, cipher suites, and HTTP/2 window settings of modern browsers (e.g., Chrome, Safari).

---

# 2. Asynchronous Rate-Limiting & Jitter Control

To maintain stable session persistence and avoid IP address throttling, requests must simulate human browsing behaviors via a non-linear temporal sequence:

$$
\Delta t_{\text{sleep}} = t_{\text{base}} + \mathcal{U}(0, \text{jitter})
$$

Where:

- $t_{\text{base}}$ is the mandatory baseline throttling interval (e.g., $3.0$ seconds).

- $\mathcal{U}(0, \text{jitter})$ injects a randomized uniform float component to completely scramble periodic fingerprint patterns.

---

# 3. Deterministic Tournament Structural Hierarchies

To ensure historical matrices are populated purely with relevant elite telemetry, the data pipeline implements explicit structural lookups for Tier 1 context filtering, programmatically excluding non-professional tiers.

| Target Inclusion (Tier 1 Allowed) | Explicit Suppression (Tier 2/3 Excluded) |
|---|---|
| VCT Champions, VCT Masters | VCT Challengers, Open Qualifiers |
| VCT International Leagues (Americas, EMEA, Pacific, CN) | VCT Game Changers (GC), VCT Ascension |
| VCT Regional Kickoffs | Off-Season Showmatches & Community Cups |