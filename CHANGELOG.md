# Changelog

All notable changes to the VCT Prediction Engine will be documented in this file.

## [10.0.0] - 2026-08-07

### Added
- VLR Scraper upgraded to **Schema v1.0**.
- Central `utils/match_adapter.py` supporting Schema v1.0, Gen 2, and Gen 1 data formats.
- `utils/team_registry.py` canonical team resolution.
- `ml/` package for feature store, dataset builder, model training, evaluation, and backtesting.
- `config/version.py` and `config/ml.yaml` metadata.
- Production pipeline `pipeline.py`.
- Comprehensive test suite under `tests/`.
