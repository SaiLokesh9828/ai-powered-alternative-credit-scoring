from pathlib import Path
import urllib.request
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
URL = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
ZIP_PATH = RAW / 'uci_credit_default.zip'
XLS_PATH = RAW / 'default of credit card clients.xls'


def download_dataset() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    if not XLS_PATH.exists():
        print(f'Downloading UCI dataset to {ZIP_PATH} ...')
        urllib.request.urlretrieve(URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW)
    if not XLS_PATH.exists():
        raise FileNotFoundError(f'Expected dataset file not found after extraction: {XLS_PATH}')
    print(f'Dataset ready: {XLS_PATH}')
    return XLS_PATH


if __name__ == '__main__':
    download_dataset()
