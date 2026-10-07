from pathlib import Path
import joblib
import pandas as pd
from .config import MODELS_DIR
from .features import engineer_features
from .shap_explain import individual_explanation


def risk_category(p: float) -> str:
    if p < 0.30:
        return 'LOW'
    if p < 0.50:
        return 'MEDIUM'
    return 'HIGH'


def predict_credit_risk(customer_data: dict) -> dict:
    model_path = MODELS_DIR / 'credit_risk_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError('Trained model not found. Run: python -m src.pipeline')
    model = joblib.load(model_path)
    df = engineer_features(pd.DataFrame([customer_data]))
    probability = float(model.predict_proba(df)[:, 1][0])
    factors = individual_explanation(customer_data, model_path=model_path, top_k=5)
    return {'risk_probability': probability, 'risk_category': risk_category(probability), 'top_risk_factors': factors}
