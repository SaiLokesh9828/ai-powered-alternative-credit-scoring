# Results status

Model metrics and trained artifacts are intentionally **not pre-populated** in the repository because they must be generated from the real public dataset during execution. No metrics in this project are fabricated.

Run:

```bash
python -m src.data_download
python -m src.eda
python -m src.pipeline
python -m src.shap_explain
pytest -q
```

After execution, the pipeline writes `evaluation_results.json`, split metadata, plots, and `models/credit_risk_model.joblib`.
