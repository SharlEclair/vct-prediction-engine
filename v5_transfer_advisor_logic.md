# 1. Roster & Budget Optimization Constraints

The 3-Transfer Advisor is mathematically invalid if it suggests trades that violate the user's available liquidity.

The optimization solver must enforce a strict knapsack constraint:

$$
\text{Incoming Player Cost} \leq \text{Outgoing Player Cost} + \text{Available Bank Balance}
$$

The "Select $n$ matches" option is obsolete in a predictive expected value (EV) framework and must be removed to prevent user confusion.

---

# 2. Expected Value (EV) Multipliers (The IGL Toggle)

The VFL scoring rules dictate that the In-Game Leader (IGL) receives a $2\times$ points multiplier.

The UI must allow the user to explicitly flag one player as the IGL.

The optimizer will then project the team's total "Score Velocity" by applying this multiplier to the designated IGL's base expected points.