from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from .data_processing import load_raw, clean_raw, validate_raw
from .config import FIGURES_DIR, TARGET


def run_eda():
    df = clean_raw(load_raw())
    validate_raw(df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = df.describe(include='all').transpose()
    summary.to_csv(FIGURES_DIR.parent / 'eda_summary.csv')
    plt.figure(figsize=(6, 4))
    df[TARGET].value_counts().sort_index().plot(kind='bar')
    plt.title('Target class distribution')
    plt.xlabel('Default')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'target_distribution.png', dpi=160)
    plt.close()
    return summary


if __name__ == '__main__':
    run_eda()
