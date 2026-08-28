# Refactoring Plan: Patch-Analysis Pipeline Redesign

This document serves as both the `REFACTOR_PLAN` and the `implementation_plan` to fundamentally repair the feature extraction and patch analyzer bridge in the Valorant prediction model.

## User Review Required
> [!WARNING]
> This plan proposes removing the hardcoded `[slideCount, runSpeedMultiplier]` API vector in favor of **sparse semantic delta updates**. Because the Valorant API does not provide exhaustive baseline ability stats (like cooldowns or projectile speeds), the analyzer will dynamically calculate impact using the `old_value` and `new_value` directly from the patch text to measure relative change (e.g., `% change`). 

## Open Questions
> [!IMPORTANT]
> 1. **Scoring Directional Rules:** Are we strictly bounding `automated_patch_nerf_registry.json` scores between `0.0` (Buffs / No Nerf) and `1.0` (Max Nerf)? Or should massive buffs yield negative values (e.g., `-0.5`)? 
> 2. **Ghost Nerfs:** Should ghost-nerfs (weapon shifts impacting agent viability) also be clamped/separated from buffs, or do they follow the same new scaling rules?

---

## 1. Architecture Review

### Current Architecture
* **`feature_builder.py`**: Extracts raw text, blindly checks for substrings like "speed", and forces output into one of 5 rigid categories. Assigns `weight` but the output is an untyped dictionary.
* **`patch_analyzer.py`**: Iterates through JSON, discards features that don't match the 5 rigid categories. Replaces weights with `[1.0, 1.0]`. Treats all absolute vector distance as a penalty. Overwrites baseline speed with projectile/duration variables, blowing up the StandardScaler.

### Proposed Architecture
* **`feature_builder.py`**: Uses context-aware extraction to map text to a comprehensive `FEATURE_SCHEMA` (combat, ability, movement, projectile, economy). Outputs clean, strictly-typed schemas.
* **`patch_analyzer.py`**: Eliminates rigid API vector mapping for undocumented stats. Instead, computes **Patch Impact Distance** directly from the delta percentage `(new - old) / old`. Modulates the distance using the feature's `weight` and `change_direction` to cleanly separate *nerf impact* from *buff impact*.

---

## 2. Feature Schema Expansion

We will introduce a new `FEATURE_SCHEMA` object definition. All patches will be mapped into this space.

```json
{
  "agent": "Neon",
  "ability": "High Gear",
  "category": "movement",      // combat, ability, movement, projectile, economy, general
  "feature_name": "slideCount",// semantic name, e.g., cooldown, duration, ultimate_cost, projectile_velocity
  "old_value": 2.0,
  "new_value": 1.0,
  "unit": "charges",           // s, credits, charges, multiplier, units
  "change_direction": "nerf",  // buff, nerf, adjustment
  "weight": 0.8,               // 0.0 to 1.0 based on impact
  "confidence": 0.95,
  "source_text": "Slides 2 >>> 1"
}
```

Instead of collapsing "duration" and "projectile speed" into `runSpeedMultiplier`, they will correctly map to:
- `"category": "ability", "feature_name": "duration"` (Clove)
- `"category": "projectile", "feature_name": "velocity"` (Breach)

---

## 3. Fix Scoring Model & Respect Feature Weights

The current formula `1 - exp(-distance)` treats all change as a nerf.
We will separate **Distance** from **Nerf Impact**.

1. **Calculate Delta %**: `delta = (new_value - old_value) / old_value`
2. **Apply Semantic Weight**: `impact = abs(delta) * feature.weight`
3. **Determine Direction**: 
   - If `feature.change_direction == 'nerf'`, `nerf_score += impact`
   - If `feature.change_direction == 'buff'`, `buff_score += impact`
4. **Final Nerf Penalty**: `penalty = 1 - exp(-gamma * nerf_score)`
5. **Final Buff Benefit** (if needed): `benefit = 1 - exp(-gamma * buff_score)`

The registry will only store the *Nerf Penalty* (or a combined net score if you prefer) to preserve downstream consumer compatibility.

---

## 4. Preservation & Migration Strategy

**Consumers:** `v5_simulation_engine.py`, `predict_match.py`, `model_pipeline.py`, etc., currently load `automated_patch_nerf_registry.json` and expect a dictionary of floats: `{"10.04": {"Clove": 1.0}}`. 

**Strategy:** We will maintain the exact JSON schema of `automated_patch_nerf_registry.json`. The only difference will be that the float values will be mathematically sound representations of true nerf impact. No downstream consumer code will need to be touched.

---

## 5. Verification Plan

### Automated Regression Tests
We will create `test_pipeline_regression.py` that mocks `feature_builder` outputs and verifies `patch_analyzer` math:
- **Test 1**: Breach projectile speed increase results in a `buff_score > 0`, but `nerf_score == 0` (Registry = 0.0).
- **Test 2**: Clove duration reduction results in `nerf_score > 0`.
- **Test 3**: Neon slide count reduction results in `nerf_score > 0` and is scaled by its `weight`.
- **Test 4**: `runSpeedMultiplier` is completely isolated from "projectile speed" inputs.

### Manual Validation
1. Re-run `patch_analyzer.generate_patch_distances()`.
2. Generate the **Validation Report** to compare the old values (0.632, 1.0, 1.0) with the new, semantically correct values.
3. Verify that the output `automated_patch_nerf_registry.json` still parses correctly in `predict_match.py`.
