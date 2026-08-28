````markdown id="4n8k2m"
# 1. Dynamic Temporal Map Pool Registry

## The Core Problem

Valorant active competitive map pools are non-stationary.

For example, a match simulated or parsed in mid-2023 operates in a completely different 7-map universe (e.g., with Pearl and Fracture) compared to a match in 2026.

If the MapVetoBandit evaluates utilities across a hardcoded static pool, it will hallucinate obsolete map picks or fail to evaluate historical strengths on retired maps.

---

## Architectural Solution

We must implement a **Temporal Map Registry Wrapper** that resolves the active map pool as a function of the match's timestamp or patch version.

```json
{
  "patch_ranges": [
    {
      "start_patch": "6.0",
      "end_patch": "7.04",
      "maps": ["Ascent", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
    },
    {
      "start_patch": "9.0",
      "end_patch": "2026_latest",
      "maps": ["Ascent", "Bind", "Haven", "Icebox", "Lotus", "Abyss", "Sunset"]
    }
  ]
}
````

---

## Downstream Pipeline Alignment

### Bandit Arm Restrictions

Before `MapVetoBandit` initializes its multi-armed bandit context vector, it requests the active pool from the registry using the target match date.

Arms ($\mathcal{A}$) are dynamically pruned to match only active maps.

### Context Vector Scaling

Historical map performance achieved during a map's previous rotation flight is decayed differently than active consecutive appearances to protect against meta shifts.

```
```
