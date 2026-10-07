from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from .data_processing import load_raw, clean_raw, validate_raw, save_clean
from .features import engineer_features
from .modeling import build_models, split_xy, evaluate, save_model, save_json
from .leakage import detect_obvious_leakage
from .evaluation import save_evaluation_plots
from .config import DATA_PROCESSED, RANDOM_STATE, TARGET


def run():
    df = clean_raw(load_raw())
    validate_raw(df)
    leakage = detect_obvious_leakage(df)
    if leakage:
        raise ValueError(f'Potential target leakage detected: {leakage}')
    save_clean(df)
    df = engineer_features(df)

    train_val, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[TARGET], random_state=RANDOM_STATE)

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    models = build_models(X_train, y_train)
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        results[name] = {'validation': evaluate(y_val, val_prob)}
        fitted[name] = model

    # Small, reproducible tuning grid on the best validation ROC-AUC model.
    best_name = max(results, key=lambda n: results[n]['validation']['roc_auc'])
    best_model = fitted[best_name]
    if best_name == 'xgboost':
        param_grid = {'model__max_depth': [3, 4], 'model__learning_rate': [0.03, 0.05], 'model__n_estimators': [250, 400]}
    elif best_name == 'random_forest':
        param_grid = {'model__max_depth': [None, 12], 'model__min_samples_leaf': [3, 5]}
    else:
        param_grid = {'model__C': [0.25, 0.5, 1.0]}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(best_model, param_grid, scoring='average_precision', cv=cv, n_jobs=1, refit=True)
    grid.fit(X_train, y_train)
    tuned = grid.best_estimator_
    val_prob = tuned.predict_proba(X_val)[:, 1]
    test_prob = tuned.predict_proba(X_test)[:, 1]
    results['selected_model'] = best_name
    results['tuned_best_params'] = grid.best_params_
    results['tuned_validation'] = evaluate(y_val, val_prob)
    results['test'] = evaluate(y_test, test_prob)
    best_threshold = max(results['tuned_validation']['thresholds'], key=lambda x: x['f1'])['threshold']
    results['selected_threshold'] = best_threshold
    save_evaluation_plots(y_test, test_prob, threshold=best_threshold, prefix='test')

    # Refit selected model on train+validation before saving for inference.
    X_trainval, y_trainval = split_xy(train_val)
    tuned.fit(X_trainval, y_trainval)
    save_model(tuned, 'credit_risk_model')
    save_json(results, DATA_PROCESSED / 'evaluation_results.json')
    (DATA_PROCESSED / 'split_sizes.json').write_text(json.dumps({'train': len(train), 'validation': len(val), 'test': len(test)}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    run()
