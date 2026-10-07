from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import DROP_FROM_MODEL, MODELS_DIR, RANDOM_STATE, TARGET


def split_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X = X.drop(columns=[c for c in DROP_FROM_MODEL if c in X.columns])
    return X, y


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ], remainder='drop')


def build_models(X, y):
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    scale_pos_weight = neg / pos
    prep = make_preprocessor(X)
    return {
        'logistic_regression': Pipeline([('preprocessor', prep), ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))]),
        'random_forest': Pipeline([('preprocessor', prep), ('model', RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=5))]),
        'xgboost': Pipeline([('preprocessor', prep), ('model', XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=4))]),
    }


def metrics_at_threshold(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        'threshold': threshold,
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'tn': int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 0]),
        'fp': int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 1]),
        'fn': int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 0]),
        'tp': int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 1]),
    }


def evaluate(y_true, prob, thresholds=(0.30,0.40,0.50,0.60,0.70,0.80)):
    out = {
        'roc_auc': roc_auc_score(y_true, prob),
        'pr_auc': average_precision_score(y_true, prob),
        'thresholds': [metrics_at_threshold(y_true, prob, t) for t in thresholds],
    }
    return out


def save_model(model, name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f'{name}.joblib'
    joblib.dump(model, path)
    return path


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
