import pandas as pd
from .config import TARGET


def detect_obvious_leakage(df: pd.DataFrame) -> list[str]:
    """Return columns that look like target/post-outcome leakage by name or identity."""
    target_tokens = {'default', 'outcome', 'delinquency_next', 'future', 'chargeoff'}
    findings = []
    for c in df.columns:
        if c == TARGET:
            continue
        normalized = str(c).lower().replace(' ', '_')
        if normalized in {'default_payment_next_month', 'default.payment.next.month'}:
            findings.append(c)
        elif any(token in normalized for token in target_tokens) and c != TARGET:
            findings.append(c)
    return findings
