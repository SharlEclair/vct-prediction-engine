# V3 Core Blueprint: Fantasy Engine & UI Architecture

## [x] Phase 3.1: Structural Ingestion & Bo5 Veto Expansion
- [x] Refactor `api_client.py` and parsers to natively handle best-of-five (`Bo5`) data layers (parsing 5 map segments instead of 3).
- [x] Train a parallel multi-label classifier or sequential Markov chain model to predict the map veto sequence (Bans/Picks) based on historical team map preferences.
- [x] Adjust veto vector weights to support up to 5 entries dynamically.

## [x] Phase 3.2: Map Score & Composition Generation Pipelines
- [x] Build a round-level regression model to predict exact scores (e.g., 13-9) per map. Ensure it accommodates both the dynamically predicted veto and custom user overrides.
- [x] Build a multi-class generative matrix to predict the most probable 5-agent composition per team per map based on recent patch meta and player comfort picks.

## [x] Phase 3.3: Valorant Fantasy Score Engine
Implement a strict scoring rule engine evaluating predicted/simulated match metrics:
- **Gameweek Aggregation**: Cap score calculations at the top 2 maps per gameweek.
- **Kills Metrics**: Apply piecewise penalties/rewards (-3 for 0 kills, -1 for 1-4, +1 baseline at 10 kills, +1 per 5 kills thereafter).
- **Multi-Kills Vector**: Parse/simulate round performance to award 4K (+1), 5K (+3), 6K (+5), and 7K (+10) points.
- **Map Out-of-Bounds Metrics**: Evaluate round deltas (+5 for 13-0, -5 for 0-13, +2 for 10+ diff win, +1 for 5-9 diff win, -1 for 10+ diff loss).
- **Series Scale Modifiers**: Calculate 2-0 series (+2), 3-0 (+4), and 3-1 (+1) series bonuses.
- **VLR Rating Bonuses**: Sort predicted player match ratings to allocate top 3 placements (+3, +2, +1) and absolute scaling modifiers (+1 for 1.5+, +2 for 1.75+, +3 for 2.0+).

## [x] Phase 3.4: Aesthetic Web UI Construction
- [x] Spin up a frontend web server (Streamlit or FastAPI+Tailwind).
- [x] Create an immersive dashboard displaying team match-ups, predicted map vetoes, projected maps scores, and visual agent composition grids using official asset image links.
- [x] Render a live leaderboard for the calculated Valorant Fantasy Scores.