# Model Card: match_winner_v1

## Model Overview
- **Model Name**: `match_winner_v1`
- **Version**: `10.0.0`
- **Algorithm**: LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`)
- **Release Date**: `2026-08-27`

## Intended Use
- **Primary Use Case**: Predict win probability for VCT Valorant esports matches.
- **Out of Scope Use**: Real-money betting optimization without risk constraints or non-VCT amateur matches.

## Training Data & Features
- **Training Window**: 2022-2025
- **Dataset Size**: 2028 matches
- **Feature Categories**: Team historical win rates, map pick/win form, economy conversion (full buy, semi buy, eco, pistol), entry fragging rates, and player combat EMAs.

## Quantitative Performance
| Metric | Value |
|---|---|
| **Validation Accuracy** | `0.5948` |
| **ROC-AUC** | `0.6289` |
| **Log Loss** | `0.6808` |
| **Brier Score** | `0.2428` |
| **Expected Calibration Error (ECE)** | `0.0843` |
| **Maximum Calibration Error (MCE)** | `0.1959` |

## Hyperparameters
```json
{
  "n_estimators": 100,
  "learning_rate": 0.05,
  "max_depth": 4,
  "num_leaves": 15,
  "objective": "binary"
}
```

## Known Limitations & Risks
- **Schema Legacy Fallbacks**: Legacy matches (Gen 1 / Gen 2) prior to Schema v1.0 lack detailed economy and clutch telemetry.
- **Roster Rework Sensitivity**: Mid-season roster substitutions exhibit initial feature variance until trailing EMAs stabilize.
- **Patch Meta Shifts**: Major game patch balance shifts (agent reworks) may cause temporary calibration drift until sufficient patch observations accrue.
