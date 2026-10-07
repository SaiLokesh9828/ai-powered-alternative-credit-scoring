from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from .config import FIGURES_DIR


def save_evaluation_plots(y_true, probability, threshold=0.50, prefix='test'):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, probability)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / f'{prefix}_roc_curve.png', dpi=160); plt.close()
    PrecisionRecallDisplay.from_predictions(y_true, probability)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / f'{prefix}_pr_curve.png', dpi=160); plt.close()
    pred = (probability >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, pred)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / f'{prefix}_confusion_matrix.png', dpi=160); plt.close()
