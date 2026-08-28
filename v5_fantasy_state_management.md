````markdown id="6x8s0d"
# 1. Dynamic Liquidity & Budget Auto-Calculation

The user should never have to manually calculate their remaining bank balance.

The system must derive it mathematically based on the global salary cap:

$$
50.0\text{ VP}
$$

and the current roster state.

Let $R$ be the set of players currently on the user's roster.

### Current Roster Cost

$$
C_{\text{roster}} = \sum_{p \in R} \text{Cost}(p)
$$

### Floating Bank (Unspent VP)

$$
B_{\text{float}} = 50.0 - C_{\text{roster}}
$$

When the algorithm suggests swapping out a specific player $p_{\text{out}}$, the maximum allowable cost for the incoming player $p_{\text{in}}$ is dynamically calculated as:

$$
\text{Max Swap Budget} = \text{Cost}(p_{\text{out}}) + B_{\text{float}}
$$

---

# 2. IGL Designation & EV Velocity

The user must be able to designate exactly one player $p \in R$ as the In-Game Leader (IGL).

The total Expected Value (EV) of the roster applies a $2\times$ multiplier strictly to that player:

$$
\text{Total EV} =
\sum_{p \in R \setminus \{\text{IGL}\}} \text{EV}(p)
+
(2 \times \text{EV}(\text{IGL}))
$$

---

# 3. State Persistence (Save/Load Roster)

To allow managers to carry their team from Gameweek to Gameweek, the roster state must be persisted to the local disk.

Implement:

```text
data/user_roster_state.json
````

Provide a:

```
💾 Save Current Roster
```

button that writes the current 6 UUIDs/Names and the designated IGL to this file.

On application load, if this file exists, automatically populate the Transfer Advisor's starting roster.

```
```
