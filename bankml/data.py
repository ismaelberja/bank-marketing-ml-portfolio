"""UCI original data loader, integrity checks, and leakage-aware feature policy."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = 'y'
# Duration reveals information only known after the call. Demographic variables
# are held out of model training and used only for aggregate subgroup auditing.
EXCLUDED = frozenset({'y', 'duration', 'age', 'marital'})
REQUIRED = frozenset({'y', 'duration', 'age', 'marital', 'job', 'contact', 'campaign', 'pdays', 'previous'})
NUMERIC = ('campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
           'cons.conf.idx', 'euribor3m', 'nr.employed')
CATEGORICAL = ('job', 'education', 'default', 'housing', 'loan', 'contact',
               'month', 'day_of_week', 'poutcome')
FEATURES = NUMERIC + CATEGORICAL


def read_data(path: str | Path) -> pd.DataFrame:
    """Keep original row order, which UCI documents as chronological."""
    df = pd.read_csv(path, sep=';', keep_default_na=False)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    missing_features = set(FEATURES).difference(df.columns)
    if missing_features:
        raise ValueError(f'Missing predictor columns: {sorted(missing_features)}')
    if not set(df[TARGET].unique()).issubset({'yes', 'no'}):
        raise ValueError("Target must contain only 'yes'/'no' labels")
    if len(df) < 200:
        raise ValueError('Need at least 200 observations for time-ordered splits')
    if df[list(FEATURES)].isnull().all(axis=0).any():
        raise ValueError('A predictor contains only missing values')
    return df


def make_splits(df: pd.DataFrame, train_fraction: float = .60,
                validation_fraction: float = .20):
    """Contiguous, non-overlapping chronological train / validation / test.

    UCI's original full additional dataset is ordered by date but contains no full
    calendar timestamp; the dataset index is therefore used as an order proxy.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError('Fractions must be between 0 and 1')
    if train_fraction + validation_fraction >= 1:
        raise ValueError('Need a positive test fraction')
    n = len(df)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    if min(a, b-a, n-b) < 30:
        raise ValueError('Each split must contain at least 30 observations')
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def xy(df: pd.DataFrame):
    x = df.loc[:, list(FEATURES)].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    for col in CATEGORICAL:
        x[col] = x[col].astype(str).replace('unknown', 'unknown')
    return x, df[TARGET].eq('yes').astype(int).to_numpy()
