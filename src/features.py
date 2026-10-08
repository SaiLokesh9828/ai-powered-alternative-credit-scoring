import numpy as np
import pandas as pd

PAY_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Convert UCI repayment code -1 (paid duly) to 0 delay for interpretable aggregates.
    delay = out[PAY_COLS].replace(-1, 0).clip(lower=0)
    out['max_payment_delay'] = delay.max(axis=1)
    out['avg_payment_delay'] = delay.mean(axis=1)
    out['months_with_delay'] = (delay > 0).sum(axis=1)
    out['severe_delay_months'] = (delay >= 2).sum(axis=1)

    bills = out[BILL_COLS]
    payments = out[PAY_AMT_COLS]
    limit = out['LIMIT_BAL'].replace(0, np.nan)
    out['avg_bill_amount'] = bills.mean(axis=1)
    out['bill_volatility'] = bills.std(axis=1).fillna(0)
    out['avg_payment_amount'] = payments.mean(axis=1)
    out['payment_volatility'] = payments.std(axis=1).fillna(0)
    out['avg_utilization'] = (bills.div(limit, axis=0)).mean(axis=1).replace([np.inf, -np.inf], np.nan)
    out['max_utilization'] = (bills.div(limit, axis=0)).max(axis=1).replace([np.inf, -np.inf], np.nan)
    out['avg_payment_ratio'] = (
    payments.sum(axis=1)
    / bills.abs().sum(axis=1).replace(0, np.nan)
    )
    out['recent_bill_change'] = out['BILL_AMT1'] - out['BILL_AMT6']
    out['recent_payment_change'] = out['PAY_AMT1'] - out['PAY_AMT6']
    out['total_billed_6m'] = bills.sum(axis=1)
    out['total_paid_6m'] = payments.sum(axis=1)
    out['total_payment_ratio_6m'] = out['total_paid_6m'] / out['total_billed_6m'].abs().replace(0, np.nan)

    # These raw monthly columns are retained because tree/linear models can learn
    # temporal repayment patterns; no post-outcome fields are introduced.
    return out.replace([np.inf, -np.inf], np.nan)
