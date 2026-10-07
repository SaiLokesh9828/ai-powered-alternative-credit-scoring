# Notebook workflow

The production-oriented implementation lives in `src/`. If you want a notebook, create one that imports the same functions rather than duplicating preprocessing logic. Recommended sequence:

1. `from src.data_processing import load_raw, clean_raw, validate_raw`
2. `from src.features import engineer_features`
3. `from src.modeling import build_models, split_xy`
4. Inspect `data/processed/evaluation_results.json` after `python -m src.pipeline`.
