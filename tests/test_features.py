import pandas as pd
from src.features import engineer_features


def test_feature_engineering_creates_risk_features():
    row = {
        'LIMIT_BAL': 10000, 'PAY_0': 2, 'PAY_2': 0, 'PAY_3': -1, 'PAY_4': 1, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': 5000, 'BILL_AMT2': 4500, 'BILL_AMT3': 4000, 'BILL_AMT4': 3500, 'BILL_AMT5': 3000, 'BILL_AMT6': 2500,
        'PAY_AMT1': 1000, 'PAY_AMT2': 1000, 'PAY_AMT3': 1000, 'PAY_AMT4': 1000, 'PAY_AMT5': 1000, 'PAY_AMT6': 1000,
    }
    out = engineer_features(pd.DataFrame([row]))
    assert out.loc[0, 'max_payment_delay'] == 2
    assert 'avg_utilization' in out.columns
