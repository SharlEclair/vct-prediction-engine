Phase 18: VFL Strict Transfer Rules (v7.6)

Objective: Enforce official VFL roster building and transfer rules within the Knapsack solver and the Streamlit Transfer Advisor.

1. Mathematical Roster Constraints (fantasy_engine.py)

File Target: fantasy_engine.py (specifically optimize_roster and suggest_transfers).

The Flaw: The current solver optimizes for maximum EV but ignores the real-world team cap, and doesn't explicitly model the 4 Core Roles + 2 Wildcards requirement perfectly alongside the 3-transfer limit.

The Fix: Inject new strict constraints into the PuLP Linear Programming (LP) model:

Team Cap Constraint: Extract all unique team names from the slate. For each team, add a constraint: lpSum(player_vars for players on this team) <= 2.

Role/Wildcard Constraint: To correctly model "Role-Locked Swaps vs Wildcards", the solver simply needs to enforce the final 6-man roster composition:

Total Players == 6

Duelists >= 1, Initiators >= 1, Controllers >= 1, Sentinels >= 1.

By enforcing exactly 6 players and $\ge 1$ of each core role, the remaining 2 slots mathematically act as unrestricted Wildcards.

Transfer Cap Constraint: When current_roster is provided to the solver, add a constraint that the sum of player variables not in the current roster must be $\le$ max_transfers.

2. Transfer Advisor UI Enhancements (app.py)

File Target: app.py (Tab 3: 🧠 Roster Optimizer).

The Flaw: The user has no visual feedback confirming that their lineup is legal under VFL rules, and trade suggestions don't highlight when a Wildcard slot is being utilized.

The Fix:

Legality Check: Add a validation function that checks the user's current 6-man roster. If they have >2 players from one team, or are missing a core role, display a red st.error ("⚠️ VFL Rule Violation: Max 2 players per team / Missing core role").

Trade Output Enhancement: When rendering the OUT and IN trade suggestions, add a small info badge highlighting if the solver utilized a Wildcard swap to achieve the optimal EV.