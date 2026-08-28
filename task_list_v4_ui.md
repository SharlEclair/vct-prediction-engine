# V4.1 Core Blueprint: Streamlit UI Ghost Nerf Integration

## [x] Phase 4.1.1: Meta Health Dashboard
- [x] Refactor `app.py` to securely load `./data/processed/automated_patch_nerf_registry.json` on startup.
- [x] Add a new visual component inside the "VFL Fantasy Manager Hub" tab called **"Live Meta Radar"**. 
- [x] Parse the latest patch key (e.g., "9.02") from the registry and display a styled leaderboard of the top 3 most penalized agents (Direct and Ghost Nerfs combined).

## [x] Phase 4.1.2: Roster Optimization Tagging
- [x] Update the **Stage 2 Optimal Lineup** visual cards in `app.py`.
- [x] If an optimal player's most played agent currently carries a penalty $> 0.10$ in the registry, append a visual warning badge (e.g., ⚠️ Meta Penalty: 0.98) to their roster card so the user knows they are a volatile asset.

## [x] Phase 4.1.3: The Ghost Nerf Transfer Advisor
- [x] Overhaul the **3-Transfer Advisor** UI logic. 
- [x] Program the advisor to explicitly prioritize transferring *out* players whose agents appear in the latest patch registry with severe penalties.
- [x] Add a text explanation below each suggested transfer citing the exact numerical Ghost Nerf penalty causing the trade.