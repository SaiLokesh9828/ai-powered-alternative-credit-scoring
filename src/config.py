from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
FIGURES_DIR = ROOT / 'reports' / 'figures'

DATASET_ZIP = DATA_RAW / 'uci_credit_default.zip'
DATASET_XLS = DATA_RAW / 'default of credit card clients.xls'
DATASET_URL = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'

TARGET = 'default_payment_next_month'
RANDOM_STATE = 42

# Conservative production-style baseline: do not train on direct demographic attributes.
# They can still be used for offline fairness auditing if explicitly enabled.
AUDIT_ONLY_COLUMNS = ['SEX', 'EDUCATION', 'MARRIAGE']
ID_COLUMNS = ['ID']
DROP_FROM_MODEL = ID_COLUMNS + AUDIT_ONLY_COLUMNS

LOW_THRESHOLD = 0.30
MEDIUM_THRESHOLD = 0.50
