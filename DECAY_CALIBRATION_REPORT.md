# Decay Calibration Analysis

This report documents the performance of the historical performance concept drift model under alternative lambda values.

## Mathematical Scaling of exp(lambda * delta)

The decay is computed as `exp(lambda * delta_p_agent)`. We evaluated five candidate lambda values on the complete VCT match dataset:

| Lambda Coefficient | Overall ACS MAE | Overall ACS RMSE |
| --- | --- | --- |
| -1.0 | 35.18473 | 45.51195 |
| -1.5 | 35.18665 | 45.51606 |
| -2.0 | 35.18887 | 45.52030 |
| -2.5 | 35.19159 | 45.52463 |
| -3.0 | 35.19442 | 45.52901 |

**Best Lambda:** `-1.0`

## Correlation Analysis

- **Pearson Correlation** (BLOPS score vs observed performance error increase): `0.11272`
- **Spearman Correlation** (BLOPS ranking vs observed disruption ranking): `0.01891`
