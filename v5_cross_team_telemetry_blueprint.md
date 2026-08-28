````markdown id="7v3m8q"
# 2. Global Player Telemetry Ledger (Team Decoupling)

## The Core Problem

Currently, the engine maps player stats under team anchors or evaluates historical performance based on the specific context of their current roster configuration.

When a star player moves teams (e.g., a high-profile transfer), the model suffers from an artificial data-sparsity shock, failing to carry over their mechanical skill ceiling (ACS, ADR) and role baseline to their new home.

---

## Architectural Solution

We must decouple player data from team definitions by implementing a centralized **Global Player Entity Ledger**.

```text
[Raw VLR Scraper JSON] ──> Extracts Global Player Key (Name/ID)
                                │
                                ▼
            [Global Player Telemetry Ledger]
         Stores: Global ACS, Map Comfort, Role Profiles
         (Independent of Team Context, Tagged with Team-History Timestamps)
                                │
                                ▼
[Simulation Engine Feature Builder] ──> Resolves active 5-man roster identities
````

---

## Mathematical Adjustments

### Decoupled Comfort Calculation

The Bayesian-smoothed comfort rating is calculated across a player's entire global career timeline, completely independent of what jersey they wore:

$$
\text{GlobalComfort}(Player_i, Agent_j, Map_k)
$$

### Team Synergy Scaling

To account for the adjustment period of a new team transfer, introduce a Roster Cohesion Coefficient ($CF$).

If a player's career history with their current team signature is low ($< 5$ matches), their individual variance boundaries ($\sigma$ in the Dirichlet distribution) are slightly expanded to reflect early tactical friction, while keeping their raw mechanical skill priors intact.

---

## The Consultative Antigravity Prompt

Copy and paste this prompt to your agent to initiate the architectural review.

This is explicitly flag-guarded to brainstorm and formulate ideas without touching code.

```
```
