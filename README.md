# Valorant VCT Prediction Engine

![Version](https://img.shields.io/badge/version-10.0.0-blue.svg)
![Schema](https://img.shields.io/badge/schema-v1.0-green.svg)
![Adapter](https://img.shields.io/badge/adapter-v1.0-orange.svg)

An end-to-end Valorant Champions Tour (VCT) match prediction, simulation, and feature engineering platform.

## Architecture

```
data/raw JSONs
      │
      ▼
utils.match_adapter (Schema v1.0, Gen 2, Gen 1)
      │
      ▼
ml.feature_builder (Team, Player, Map Features)
      │
      ▼
ml.dataset_builder (Match, Map, Score Datasets)
      │
      ▼
ml.train & ml.evaluate (LogReg, LightGBM, XGBoost)
      │
      ▼
ml.backtest & v5_simulation_engine
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete ML pipeline
python pipeline.py

# Run test suite
pytest tests/
```

## Version Metadata

- **VERSION**: `10.0.0`
- **SCHEMA_VERSION**: `1.0`
- **ADAPTER_VERSION**: `1.0`
