from pathlib import Path
import pandas as pd
from .config import TARGET, DATA_RAW, DATA_PROCESSED, DATASET_XLS

UCI_COLUMNS = {
    'default.payment.next.month': TARGET,
    'default payment next month': TARGET,
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATASET_XLS
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run: python -m src.data_download')
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=UCI_COLUMNS)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError('Dataset is empty.')
    if TARGET not in df.columns:
        raise ValueError(f'Missing target column {TARGET!r}.')
    if df[TARGET].isna().any():
        raise ValueError('Target contains missing values.')
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError('Target must be binary 0/1.')


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates().reset_index(drop=True)
    # UCI uses 0 for unknown categories in some categorical fields.
    # Keep them as valid categories rather than silently imputing a value.
    for c in out.columns:
        if c != TARGET and out[c].dtype == 'object':
            out[c] = out[c].replace({'?': pd.NA})
    return out


def save_clean(df: pd.DataFrame) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / 'clean_credit_default.csv'
    df.to_csv(path, index=False)
    return path


if __name__ == '__main__':
    df = clean_raw(load_raw())
    validate_raw(df)
    path = save_clean(df)
    print(f'Saved {len(df):,} rows to {path}')
