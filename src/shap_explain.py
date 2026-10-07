from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from .config import MODELS_DIR, FIGURES_DIR, DROP_FROM_MODEL, TARGET
from .features import engineer_features


def _prepare(model, df):
    X = engineer_features(df.copy())
    if TARGET in X.columns:
        X = X.drop(columns=[TARGET])
    X = X.drop(columns=[c for c in DROP_FROM_MODEL if c in X.columns])
    prep = model.named_steps['preprocessor']
    Xt = prep.transform(X)
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    return Xt, prep.get_feature_names_out()


def _explainer(model, background):
    estimator = model.named_steps['model']
    name = estimator.__class__.__name__.lower()
    if name.startswith(('randomforest', 'xgb')):
        return shap.TreeExplainer(estimator)
    return shap.LinearExplainer(estimator, background)


def individual_explanation(customer_data: dict, model_path=None, top_k=5):
    model_path = Path(model_path or MODELS_DIR / 'credit_risk_model.joblib')
    if not model_path.exists():
        raise FileNotFoundError('Trained model not found. Run: python -m src.pipeline')
    model = joblib.load(model_path)
    raw = pd.DataFrame([customer_data])
    Xt, names = _prepare(model, raw)
    explainer = _explainer(model, Xt)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[0]
    pairs = sorted(zip(names, values), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
    return [{'feature': str(name), 'shap_value': float(value), 'impact': 'increases risk' if value > 0 else 'decreases risk'} for name, value in pairs]


def explain(model_path=None, data_path=None, max_samples=1000):
    model_path = Path(model_path or MODELS_DIR / 'credit_risk_model.joblib')
    data_path = Path(data_path or Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'clean_credit_default.csv')
    if not model_path.exists() or not data_path.exists():
        raise FileNotFoundError('Run the training pipeline first.')
    model = joblib.load(model_path)
    raw = pd.read_csv(data_path)
    sample = raw.sample(min(max_samples, len(raw)), random_state=42)
    Xt, names = _prepare(model, sample)
    explainer = _explainer(model, Xt)
    values = explainer.shap_values(Xt)
    if isinstance(values, list): values = values[1]
    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, Xt, feature_names=names, show=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / 'shap_global_summary.png', dpi=160, bbox_inches='tight'); plt.close()
    return values


if __name__ == '__main__':
    explain()
