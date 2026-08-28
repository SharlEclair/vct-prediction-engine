# Phase 12: Roster Integrity & Strict Map Veto Rules
**Objective:** Eliminate roster pollution (Academy/GC/Inactive players), build a UI override tool, and enforce strict VCT map veto sequences for BO1, BO3, and BO5 formats.

## 1. Roster Ingestion & UI Override
* **The Bug:** Scraping scripts are improperly merging players from Academy rosters, Game Changers (GC) rosters, and inactive benches into the main Tier-1 team rosters (e.g., MIBR getting Academy players; ENVY getting inactive players).
* **Fix 1 (Filtering):** Update the scraper parsing logic to explicitly ignore players marked with "Inactive" tags, and isolate/exclude rosters with suffixes like "Academy", "GC", "Black", or "Blue".
* **Fix 2 (UI Override):** In `app.py` (under System Administration), add a "Roster Management Override" UI. It should allow users to manually define the exact active 5-man (or 6-man) roster for a specific team. Save these overrides to `data/processed/roster_overrides.json`. The pipeline must check this JSON first before defaulting to scraped data.

## 2. Map Veto Information & Engine Update
* **The Bug:** The current Contextual Bandit map veto logic does not perfectly mirror official VCT match rules for Team A/B selection and the precise ban/pick sequence.
* **Fix 1 (Team A/B Selection):**
  * **BO1:** The left-sided team on the match page chooses Team A or Team B.
  * **BO3:** The higher-seeded team (Swiss stage) chooses.
  * **BO5:** The Upper Bracket winner chooses.
  * **Equal Seeding:** A 1v1 Skirmish determines the winner, who then chooses.
* **Fix 2 (Strict Veto Sequences):** Update the map veto simulator to enforce these exact sequences:
  * **BO1:** A Ban 1 -> B Ban 1 -> A Ban 2 -> B Ban 2 -> A Ban 3 -> B Picks Map 1 -> A Picks Side.
  * **BO3:** A Ban 1 -> B Ban 1 -> A Picks Map 1 (B Side) -> B Picks Map 2 (A Side) -> A Ban 2 -> B Ban 2 -> Map 3 Remains (A Side).
  * **BO5:** A Ban 1 -> B Ban 1 -> A Picks Map 1 (B Side) -> B Picks Map 2 (A Side) -> A Picks Map 3 (B Side) -> B Picks Map 4 (A Side) -> Map 5 Remains (B Side).