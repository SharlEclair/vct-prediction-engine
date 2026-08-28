# V4.1 Core Blueprint: Production MediaWiki Patch Ingestion & UI Radar

## [ ] Phase 4.1.0: Codebase Git Push & Branching
- [ ] Stage all existing files, commit the stable V3.2 manual patch registry and global JSD baseline code, and push to the remote GitHub origin.
- [ ] Create and checkout a fresh working branch named `v4-autonomous-meta`.

## [ ] Phase 4.1.1: Fandom Wiki Live Patch Ingestion (`patch_ingestor.py`)
- [ ] Install environment prerequisites (`spacy`, `scikit-learn`) and download the `en_core_web_sm` model language core.
- [ ] Create `patch_ingestor.py` to scrape patch text using `httpx` from `https://valorant.fandom.com/wiki/Patch_Notes/{patch_version}?action=raw`.
- [ ] Read the local `patch_notes.csv` file to extract all valid historical patch strings to scrape sequentially.
- [ ] Implement a hierarchical tree-state machine parsing loop utilizing `v4_parsing_skills.py` to ingest categories, subjects, and semantic change metrics.

## [ ] Phase 4.1.2: RBF Matrix Automation & Ghost Nerf Calculation
- [ ] Combine parsed wiki metrics into the `patch_analyzer.py` engine to automatically calculate the Agent design state vector $\mathbf{v}_{a,t}$.
- [ ] Compute the RBF-driven $\Delta P_{agent}$ and cross-reference weapon dependencies from telemetry logs to assign $\Delta P_{ghost}$.
- [ ] Save the multi-patch dynamic output compilation directly into `./data/processed/automated_patch_nerf_registry.json`.

## [ ] Phase 4.1.3: Visual Roster UI Badges (`app.py`)
- [ ] Update `app.py` to securely read the automated registry.
- [ ] Inject a **"Live Meta Radar Leaderboard"** detailing highly penalized agents.
- [ ] Dynamically attach active warnings (`⚠️ Meta Penalty: 0.98`) onto the optimal Stage 2 roster display cards and map explicit trade justifications to the **3-Transfer Advisor** block.