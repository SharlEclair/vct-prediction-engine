# Valorant Patch Parser Validation Report

This report summarizes the parsing metrics and extracted data across the target patch notes.

| Patch Version | Release Date | Sections Populated | Agents Detected | Weapons Detected | Numeric Changes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 9.0 | June 25th, 2024 | agent_changes, competitive_changes, bug_fixes, player_behavior_changes | Iso | None | 0 |
| 9.01 | July 16th, 2024 | agent_changes, bug_fixes | General | None | 0 |
| 9.02 | July 30th, 2024 | agent_changes, bug_fixes, player_behavior_changes | General | None | 0 |
| 9.03 | August 13th, 2024 | agent_changes, performance_changes, bug_fixes | General | None | 0 |
| 9.04 | August 27th, 2024 | agent_changes, competitive_changes, bug_fixes | General | None | 0 |
| 12.09 | May 12th, 2026 | agent_changes, weapon_changes, competitive_changes, performance_changes, bug_fixes | Neon | All shotguns, Bucky, Judge, Shorty | 23 |

**Total Numeric Transitions Extracted**: 23
**Total Ingestion Failures**: 0

## Example Outputs from Patch 12.09

### ⚡ Neon (Agent Changes)
- **Ability**: `High Gear` (Nerf)
  - *Description*: Jumping with High Gear active no longer provides any speed bonus while Neon is airborne. Instead, Neon’s air speed while sprinting will match melee speed.
- **Ability**: `High Gear` (Adjustment)
  - *Description*: Improved VFX to more clearly communicate the direction and origin of her slide
- **Ability**: `Energy` (Nerf)
  - *Description*: Fuel will only regenerate with a kill when Neon's ultimate is active
- **Ability**: `Energy` (Adjustment)
  - *Description*: Fuel will still regenerate passively

### 🔫 Bucky (Weapon Changes)
- **General**: (Adjustment) - Bucky pellet damage in the 0-8m range decreased
- **Stat**: `Head` (Nerf)
  - *Transition*: `40.0` >>> `34.0`
- **Stat**: `Body` (Nerf)
  - *Transition*: `20.0` >>> `17.0`
- **Stat**: `Legs` (Nerf)
  - *Transition*: `17.0` >>> `14.0`
- **Stat**: `Minimum spread` (Nerf)
  - *Transition*: `2.6` >>> `3.0`
- **Stat**: `Walking spread` (Nerf)
  - *Transition*: `0.075` >>> `1.0`
- **Stat**: `Running spread` (Nerf)
  - *Transition*: `0.1` >>> `2.0`
- **Stat**: `Crouch-walk spread` (Nerf)
  - *Transition*: `0.05` >>> `0.5`
- **Stat**: `Jump spread` (Nerf)
  - *Transition*: `1.25` >>> `4.0`

### 🔫 Judge (Weapon Changes)
- **Stat**: `Minimum spread` (Nerf)
  - *Transition*: `2.25` >>> `2.5`
- **General**: (Adjustment) - Note: this will only apply to PC for now- this spread change will apply to console in Patch 12.11
- **Stat**: `Walking spread` (Nerf)
  - *Transition*: `0.075` >>> `1.0`
- **Stat**: `Running spread` (Nerf)
  - *Transition*: `0.75` >>> `2.0`
- **Stat**: `Crouch-walk spread` (Nerf)
  - *Transition*: `0.05` >>> `0.5`
- **Stat**: `Jump spread` (Nerf)
  - *Transition*: `2.25` >>> `4.0`

### 🔫 Shorty (Weapon Changes)
- **Stat**: `Fire rate` (Nerf)
  - *Transition*: `3.33` >>> `3.0`
- **Stat**: `Walking spread` (Nerf)
  - *Transition*: `0.075` >>> `1.0`
- **Stat**: `Running spread` (Nerf)
  - *Transition*: `0.1` >>> `2.0`
- **Stat**: `Crouch-walk spread` (Nerf)
  - *Transition*: `0.05` >>> `0.5`
- **Stat**: `Jump spread` (Nerf)
  - *Transition*: `1.25` >>> `4.0`