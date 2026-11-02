"""Illustrative batch scoring, not a deployed customer decision system.

WARNING: joblib models must only be loaded from a trusted source.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from bankml.data import FEATURES, NUMERIC


def score(input_csv, model_file, output_csv):
    blob = joblib.load(model_file)
    df = pd.read_csv(input_csv)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required scoring columns: {sorted(missing)}')
    x = df[list(FEATURES)].copy()
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    scores = blob['model'].predict_proba(x)[:, 1]
    # Avoid carrying source/customer attributes into the result; row references
    # are local only and should not be treated as durable identifiers.
    scored = pd.DataFrame({'input_row_index': range(len(df)), 'subscription_score': scores})
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    return scored


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--model', default='models/champion.joblib')
    p.add_argument('--output', default='reports/generated/scored.csv')
    a = p.parse_args()
    print(score(a.input, a.model, a.output).head())
