# VFL Model Performance Audit: GW1 vs GW2
## Model Efficiency Analysis & Hypothesis Report

---

## Executive Summary

The model achieved meaningful **Predicted EV** projections in both weeks but made critically suboptimal roster selections relative to hindsight. The key metrics:

| Metric | GW1 | GW2 |
|---|---|---|
| **Hindsight Optimal Score** | ~175–199 Pts | **199.0 Pts** |
| **Model Predicted Roster Actual** | ~107 Pts | **140.0 Pts** |
| **Model Efficiency Ratio** | ~54–61% | **~70.4%** |
| **GW Player Mean Score** | 8.76 Pts | 7.80 Pts (all) / 8.89 (active) |
| **GW Score Std Dev** | 3.66 | **5.42** |
| **Players scoring 0 Pts** | 0 | 7 (HYUNMIN, keznit, zerona, free1ng, BeYN, Crws, Paincakes) |
| **Mean absolute GW1→GW2 swing** | — | **5.13 Pts/player** |
| **Max player score** | 18.0 Pts | 22.0 Pts (RieNs) |

The model's efficiency improved from GW1 (~54%) to GW2 (~70%), but the baseline problem is shared: **roster composition optimizes on stale historical averages rather than gameweek-specific structural factors**.

---

## Observed Failure Patterns

### **GW2: Big Drop Players the Model Relied On**

| Player | GW1 | GW2 | Drop | Team | VP |
|---|---|---|---|---|---|
| Zeus | 17.0 | **−1.0** | −18.0 | Team Secret | 9.0 |
| free1ng | 17.0 | **0.0** | −17.0 | DRX | 8.0 |
| JitBoyS | 17.0 | **1.0** | −16.0 | FULL SENSE | 9.0 |
| lukxo | 17.0 | **2.0** | −15.0 | LOUD | 10.0 |
| Favian | 18.0 | **4.0** | −14.0 | Eternal Fire | 11.5 |
| MaKo | 13.0 | **−1.0** | −14.0 | DRX | 9.0 |
| Darker | 16.0 | **2.0** | −14.0 | LOUD | 8.5 |

### **GW2: Breakout Players the Model Missed**

| Player | GW1 | GW2 | Rise | Team | VP |
|---|---|---|---|---|---|
| Mazino | 3.0 | **16.0** | +13.0 | MIBR | 8.5 |
| benjyfishy | 2.0 | **14.0** | +12.0 | Team Heretics | 9.0 |
| Wo0t | 5.0 | **15.0** | +10.0 | Team Heretics | 12.5 |
| neT | 3.0 | **12.0** | +9.0 | GIANTX | 5.5 |
| trent | 4.0 | **13.0** | +9.0 | G2 Esports | 10.5 |

### **GW2 Price-Performance Reality Check**

| Tier | n | GW1 Avg | GW2 Avg | Shift |
|---|---|---|---|---|
| Cheap (≤7VP) | 42 | 7.00 | 5.79 | −1.21 |
| Mid (7–10VP) | 80 | 9.04 | 6.90 | **−2.14** |
| Expensive (>10VP) | 58 | 9.66 | **10.50** | +0.84 |

In GW2, the expensive bracket *outperformed expectations* while the mid-tier bracket the model heavily relies on *sharply underperformed*.

---

---

# Hypotheses

---

## Hypothesis 1: The Model Confuses "Was Good Last Week" with "Will Be Good This Week"

> **The model's primary EV signal is historical PPG (per-game average across all historical matches), which anchors heavily on recency-weighted performance without accounting for schedule-driven volatility.**

### Supporting Evidence
- All 7 big-drop players (Zeus, free1ng, JitBoyS, lukxo, Favian, MaKo, Darker) had strong GW1 outputs and were likely ranked highly by global PPG.
- The model predicted `Autumn` at **25.0 EV** — she scored **34.0** (close). But `zerona` at **12.0 EV** scored **0.0**, `skuba` at **19.0 EV** scored **7.0**, and `d4v41` at **22.0 EV** scored **14.0**. The model's ordering of who scores big is largely PPG-driven, not matchup-driven.
- `free1ng` scored 17.0 in GW1 and was included in the model's predicted roster. In GW2, DRX played against different opposition on different maps and scored 0 (negative map points).
- The **mean absolute GW1→GW2 change is 5.13 Pts** — that's 59% of the mean score of 8.76. This scale of volatility is inherent in the game and not modelled.

### Unsupporting Evidence
- Some players with high PPG *did* follow through: `Autumn` scored 34 pts (IGL), `Asuna` scored 14 pts vs predicted 14.5 pts.
- The model correctly *included* neT in the predicted roster (Hit), showing some signal validity.
- Historical PPG is a reasonable prior; the issue is in *precision* not *direction*.

### What Could Be Changed
- **Introduce recency decay weighting** in `compute_all_players_historical_stats()`. The current implementation averages all matches equally. Matches from the last 30 days should carry 3–5x the weight of matches from 90+ days ago.
- **Add a map-count normaliser**: A player with 3 maps played has a wildly different PPG distribution than one with 15. Track confidence intervals explicitly and penalise thin samples.
- **Implement a "momentum score"** — a simple 3-game rolling average that can capture form state.

---

## Hypothesis 2: The H2H Blending Engine Is Largely Inert for VCT 2026 Champions

> **The H2H blend activates only when a player has ≥3 maps against their specific upcoming opponent. For a new 2026 event with 2 completed gameweeks, most VCT Champions matchups are new pairings — so the engine falls back to global PPG for nearly every player.**

### Supporting Evidence
- Looking at the GW2 schedule: `GIANTX vs Karmine Corp`, `Team Heretics vs Team Liquid`, `G2 vs 100T`, etc. — many of these are cross-regional pairings that rarely occurred pre-2026.
- The `blend_ev()` function requires `n_maps >= 3` to activate H2H weighting. For newly-formed rosters or cross-regional pairings at VCT Champions, this threshold is almost never met.
- For surgers like `benjyfishy`, `Wo0t` (Team Heretics), and `trent` (G2) — the H2H system would fall back to global PPG because Team Heretics vs Team Liquid hadn't happened enough in the raw data.
- The model weights H2H linearly from 30% (at 3 maps) to 70% (at 10 maps). Even where H2H *does* activate, it's diluted significantly by global PPG.

### Unsupporting Evidence
- There *are* regional matchups (EMEA teams have played each other repeatedly) where H2H data is denser, so the system isn't completely inert.
- The blend formula is sensible for data-rich matchups. The problem is coverage, not design.

### What Could Be Changed
- **Team-level H2H proxies**: When player-level H2H is insufficient, aggregate team-level outcomes. If `GIANTX` as a team consistently under/overperforms against EMEA controllers, that signal can transfer to individual player projections.
- **Lower the activation threshold to 2 maps** (from 3) with a corresponding lower max weight of 50% (from 70%).
- **Cross-regional EV correction**: Introduce a "big event" multiplier. Players who perform well in large international events (Champions, Masters) historically spike significantly — this isn't captured by regular season PPG.
- **Use VLR team-level recent match data** as a proxy H2H signal before per-player data accumulates.

---

## Hypothesis 3: Map Pool and Map Win-Rate Are Not Weighted in Player EV

> **The VFL scoring system awards map points (round margin, win/loss bonus) which are highly deterministic of a team's strength on a specific map. The model does not predict map outcomes before computing EV.**

### Supporting Evidence
- In GW2, the breakout performers all had strong map points: `RieNs` (K:8 M:9 B:5), `Mazino` (K:5 M:7 B:4), `LewN` (K:7 M:7 B:4), `spike` (K:7 M:6 B:4).
- The big drop players all had near-zero or negative map points: `free1ng` (K:1 M:−1 B:0 = 0 total), `Zeus` (negative total), `MaKo` (negative total). These were map losses against difficult opponents.
- Map points are the **single most reliable component** because they're team-driven. If a team is projected to win a map, all 5 players get +1 to +2 bonus pts. The model currently computes kill points independently of this.
- The `VetoPredictor` is loaded at startup and has compiled veto stats from 1,925 matches — but there's no evidence it feeds into EV calculation for the weekly lineup optimization.

### Unsupporting Evidence
- Kill points *are* correlated with map wins (better players on winning teams get more kills). So PPG implicitly captures some of this.
- Map selection (veto) is only probabilistic — the predicted map might not be played.

### What Could Be Changed
- **Integrate VetoPredictor outputs** into `optimize_roster()`. If a team is predicted to win Map A (where they average 5 players × 1.5 pts = 7.5 team bonus pts), add a prorated map EV adjustment per player.
- **Add a map-win EV component** to `blend_ev()`: `map_ev = P(team wins map) × avg_map_pts_per_player_on_winning_side`.
- **Separate kill EV and map EV** in the projection and weight them differently in the objective function. Map EV has lower variance than kill EV.

---

## Hypothesis 4: The MILP Objective Over-Indexes on Expensive Premium Players When Mid-Tier Provides Better Value

> **In GW2, the expensive player bracket (>10VP) had the best actual performance (avg 10.50 Pts vs predicted ~9.5). But in GW1, mid-tier (7–10VP, avg 9.04) was the best value. The MILP budget allocation isn't adapting to which tier is producing value each week.**

### Supporting Evidence
- The GW2 model predicted roster included many mid-tier players at 8–11VP who dramatically underperformed (zerona 8VP → 0pts, skuba 9VP → 7pts, Chronicle 11VP → 10pts).
- The actual optimal GW2 roster included `RieNs` at 10.5VP (22 pts IGL), `spike` at 12VP (17 pts), `LewN` at 10.5VP (18 pts) — all expensive. The model did not fully load up on the top tier.
- The budget cap of 99VP for 11 players allows ~9VP average. The model appears biased toward 8–10VP players by treating the salary cap constraint as a hard floor as well as ceiling.
- The `soft penalty for cost > 48 VP` mentioned in the docstring suggests there's already budget utilisation aversion baked in.

### Unsupporting Evidence
- In GW1, mid-tier was correct — `lukxo` (10VP, 17pts), `free1ng` (8VP, 17pts), `tkzin` (8.5VP, 16pts) all excellent.
- Over-indexing on expensive players would be just as wrong. The problem is *adaptability* not a systematic tier preference.

### What Could Be Changed
- **Remove or relax the cost penalty** in the MILP objective. The soft penalty discourages spending the full budget even when projections justify it.
- **Add a "value tier tracking"** feature: compute which price tier had the best PPG per VP ratio in the previous 2 gameweeks and use that as a prior weight in the MILP objective coefficient.
- **Introduce a price-adjusted EV term**: `adj_EV = computed_ppg / price` as a secondary objective signal weighted at 0.1–0.2 alongside the primary EV objective.

---

## Hypothesis 5: The Survival Threshold Filter Is Too Permissive and Lets Fragile Teams Through

> **The `survival_threshold=0.35` filters out teams with a win rate below 35%. However, teams like DRX (who produced free1ng scoring 0 in GW2) pass this filter easily because their *aggregate* VLR win rate is adequate, even if their current form is declining.**

### Supporting Evidence
- `DRX` is a historically strong team with a win rate well above 35%, yet in GW2 their players scored 0 (negative map points). The filter doesn't know DRX was in poor form going into this specific week.
- `FULL SENSE` similarly passed the filter. `JitBoyS` dropped from 17 to 1 and `Killua` dropped from 10 to −2.
- `zerona` (Evil Geniuses, 8VP) scored 0 in GW2. Evil Geniuses' GW2 opponent may have been a historically strong counter for them.
- The GW2 zero-scorers (`free1ng`, `BeYN`, `HYUNMIN` — all DRX) strongly suggest DRX had a match where they were heavily outclassed, not that they didn't play.

### Unsupporting Evidence
- A single weak week doesn't mean a team is failing — variance is high in best-of-3 formats.
- Lowering the threshold aggressively would filter too many legitimate picks.
- The model correctly avoids teams that are genuinely eliminated.

### What Could Be Changed
- **Add a "recent form filter"**: Track each team's PPG in the last 3 matches specifically. If a team's last-3-match average is more than 1.5 std devs below their historical average, apply a discount multiplier to their players' EVs.
- **Use VLR match result recency** to compute a "form score" separate from the all-time win rate. The survival threshold should be `max(historical_wr × 0.5, form_wr × 0.5)`.
- **Incorporate ELO-style opponent-adjusted ratings** rather than raw win rates. Beating low-ranked teams doesn't translate to scoring well against Champions-level competition.

---

## Hypothesis 6: The IGL Selection Method Is Not Calibrated Correctly

> **The IGL is selected as the player with the highest `floor = PPG − 1.0 × sigma`. This makes sense for risk-aversion but doesn't account for the fact that IGLs in VCT typically show high upside rather than high floor.**

### Supporting Evidence
- `RieNs` (Team Heretics IGL) was the actual GW2 optimal IGL with 22 pts (11 base × 2x). RieNs' `ppg` was only `22.0/2 = 11.0` pts base — not necessarily the highest floor player.
- `Autumn` (Global Esports IGL) scored 17 in GW2 base × 2x = 34 which appeared in the predicted roster correctly.
- The floor metric (`ppg - sigma`) would disadvantage a high-upside volatile IGL vs a consistent average player, even if the high-upside IGL is strictly better in EV terms.
- In GW1, `basic` (FURIA IGL) scored 13 base which multipled to 26 — the highest value per unit of cost. This wouldn't be predicted if the IGL selection only looks at floor.

### Unsupporting Evidence
- High-floor IGL selection is appropriate for avoiding catastrophic low scores (e.g., choosing an IGL who gets −5 pts would cost you 10 points from the 2x multiplier).
- Picking a volatile IGL is a risk/reward tradeoff that some managers might prefer.

### What Could Be Changed
- **Switch IGL selection to an EV-maximising criterion**: `igl_score = computed_ppg × 2 − price`. Pick the player with the best adjusted value.
- **Introduce a risk dial**: Allow the user to configure `igl_selection_mode = "conservative" | "aggressive"`. Conservative = floor-based, aggressive = EV-based.
- **Model IGL-specific PPG separately**: Analyse how players' scores change *when they are IGL* vs not. Some players systematically score higher as IGL (anchoring, reading maps better) — this isn't captured.

---

---

# Cross-Cutting Structural Issues

## 1. The Model Has No "Matchup Difficulty" Encoding

The GW2 optimal roster is dominated by **EMEA teams** (Team Heretics, Karmine Corp, MIBR, GIANTX, LEVIATÁN). This is because EMEA teams were playing match-ups they were favoured in, producing more maps won and thus more map points per player. The model doesn't account for:

- **Match favouriteness** — the probability that team A wins 2–0 vs 1–2 vs 0–2.
- **Expected maps played** — a 2-0 series produces fewer total kill points than a 2-1, because fewer maps are played. A 3-1 BO5 produces more total kill points across all 5 players than a 3-0.
- **Series format effects** — BO3 vs BO5 dramatically changes the EV range of kill points. The VetoPredictor has this data but it isn't being injected.

**Recommendation**: Add a `maps_expected` multiplier to player EV: `adj_ppg = global_ppg × (expected_maps / avg_maps_historically)`.

---

## 2. The Volatility Model Is Static

The model computes `sigma` from historical data but uses it identically regardless of context. GW2 had a much higher std dev (5.42 vs 3.66 in GW1). This means:

- The MILP picked players with adequate EV but insufficient expected ceiling.
- High-sigma players who could hit 16–22 pts were undervalued because the optimiser treats sigma as risk, not opportunity.
- For lineups in high-variance weeks, a **CVaR-90 maximising strategy** (pick the roster whose best-case scenario is highest) would outperform a mean-maximising strategy.

**Recommendation**: Add a week-type classifier. If the scheduled matches include many competitive BO5s or inter-regional clashes (historically higher variance), shift the MILP objective coefficient toward `cvar_90` and away from `floor`.

---

## 3. The Model Cannot Learn Between Gameweeks

The most significant structural gap: **the model does not update its priors from GW1 data to improve GW2 predictions**. Every gameweek starts fresh from the full historical database.

In GW1, `Mazino` scored 3 pts (poor). In GW2, he scored 16 pts. If the model had treated GW1 as a low observation (signal: "Mazino underperformed vs his historical avg, likely due to tough matchup") and GW2 opponent as different/easier, it would have upgraded Mazino's projection.

**Recommendation**: After each gameweek completes, run a **post-hoc calibration pass**: compare predicted EV vs actual for each player, compute per-player error terms, and add a `calibration_factor` to the player's EV for the next gameweek. Players who systematically over/underperform against the model should have their EV adjusted.

---

## Priority Optimisation Ranking

| Priority | Change | Expected Impact | Effort |
|---|---|---|---|
| 🔴 **Critical** | Recency-weighted PPG in `compute_all_players_historical_stats()` | High — reduces stale star-player overconfidence | Medium |
| 🔴 **Critical** | VetoPredictor → map EV injection into `blend_ev()` | High — map points are the most predictable scoring component | High |
| 🟡 **High** | Match difficulty / `maps_expected` multiplier | Medium-High — captures BO3 vs BO5 variance | Medium |
| 🟡 **High** | Post-GW calibration pass | Medium-High — builds learning loop | Medium |
| 🟡 **High** | IGL selection: EV-maximising vs floor-maximising | Medium — single player but 2× impact | Low |
| 🟠 **Medium** | Recent form filter (`last_3_match_ppg`) | Medium — catches declining teams | Low |
| 🟠 **Medium** | Lower H2H threshold to 2 maps + team-level proxy | Medium — increases H2H activation coverage | Low |
| 🟢 **Nice-to-have** | Price-tier adaptive weighting in MILP | Low-Medium — only matters when tier differential is clear | Medium |
| 🟢 **Nice-to-have** | Week-type variance classifier (CVaR vs mean objective) | Low-Medium — requires tuning | High |
