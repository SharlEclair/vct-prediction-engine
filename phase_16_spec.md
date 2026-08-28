# Phase 16: Stateful Economy Simulator (v7.4)
**Objective:** Upgrade the memoryless round simulator to a Stateful Economy Simulator that tracks multi-round loss-streaks, dynamic credit injections, and survival penalties.

## 1. Upgrade the Round Simulator
* **File Target:** `v5_simulation_engine.py` (specifically `SideConditionedMarkovSimulator` or the round simulation loop).
* **The Flaw:** The current round simulator is a memoryless Markov process. It ignores Valorant's strict loss-bonus economy mechanics, which dictate team momentum and loadout deltas.
* **The Fix:** Rename the simulator to `StatefulEconomySimulator` and implement explicit round-to-round memory:
  * **Loss Streak Tracking:** Initialize `loss_streak_a = 0` and `loss_streak_b = 0`.
  * **Credit Injection Logic:** On a round win, the winning team resets their loss streak to `0` and gains base economy power (e.g., $3000). The losing team increments their loss streak (capped at `3`) and gains economy based on the streak:
    * Streak 1: $1900
    * Streak 2: $2400
    * Streak 3+: $2900
  * **Survival Penalty ("Saving"):** If a team loses, inject a probability (e.g., `15%`) that a player "saves" a weapon. If a save occurs, apply the Valorant survival penalty (credit income restricted to $1000 for that player), but carry over a portion of their previous round's loadout value to the next round.
  
## 2. Dynamic Loadout Delta
* **Action:** Link the new economy state directly to the round win probability log-odds ($Z$).
* **Logic:** Instead of an abstracted or static `delta_loadout`, calculate the true `delta_loadout` for round $t$ by evaluating the accumulated economy power (Base Credits + Loss Bonus + Saved Weapons) of Team A vs. Team B.
* **Impact:** This mathematically forces the engine to simulate "Eco rounds" (where the team with a massive economy deficit has a drastically lowered win probability) and "Momentum swings" (where a max loss bonus allows a team to finally afford a full buy).