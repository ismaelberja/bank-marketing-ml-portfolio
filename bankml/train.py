"""End-to-end training, validation-only model choice and untouched holdout reporting."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bankml.data import read_data, make_splits, xy, NUMERIC, CATEGORICAL, FEATURES, EXCLUDED
from bankml.metrics import evaluate, top_k_mask, subgroup_audit
from bankml.plots import save_plots


def preprocessing():
    # Dense representation is shared by logistic baseline and boosted trees;
    # all transformations are fit INSIDE each pipeline on training data only.
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categories = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                           ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', numeric, list(NUMERIC)),
                              ('category', categories, list(CATEGORICAL))],
                             remainder='drop')


def candidates(fast: bool = False):
    return {
        'logistic_regression': Pipeline([
            ('preprocessing', preprocessing()),
            ('classifier', LogisticRegression(max_iter=800, C=0.5))]),
        'gradient_boosting': Pipeline([
            ('preprocessing', preprocessing()),
            ('classifier', HistGradientBoostingClassifier(max_iter=35 if fast else 175,
                max_leaf_nodes=15, max_depth=5, learning_rate=0.07,
                l2_regularization=3, early_stopping=False, random_state=42))]),
    }


def percent(n):
    return 'n/a' if n is None else f'{n:.1%}'


def concise(num, decimals=3):
    return 'n/a' if num is None else f'{num:.{decimals}f}'


def fmt_lift(value):
    return 'n/a' if value is None else f'{value:.2f}x'


def report(dataset, manifest, holdout_metrics, validation_metrics, audit, importance):
    a = manifest['assumptions']
    winner = manifest['selected_model']
    details = holdout_metrics[winner]
    description = ('**ORIGINAL DATA** — 41,188 records verified against the SHA-256 manifest saved '
                   'by the original-source downloader.' if manifest['original_data_verified'] else
                   '**SYNTHETIC TEST FIXTURE** — development-only test data. Do not publish these scores '
                   'as observations about the UCI dataset or any bank.')
    lines = [
        '# Bank Marketing ML — Experiment report', '',
        f'> {description}', '',
        '## Executive summary', '',
        f'- **Business question:** among historically contacted customers, can we rank likely deposit subscriptions before a call?',
        f'- **Selected model:** `{winner}`, selected solely using validation-set average precision.',
        f'- **Held-out period:** last **20%** of chronologically ordered observations (row order is a date proxy).',
        f'- **Test average precision:** {concise(details["average_precision"])}; **ROC AUC:** {concise(details["roc_auc"])}.',
        f'- **Test precision at {a["capacity_fraction"]:.0%} capacity:** {percent(details["precision_at_capacity"])}; '
        f'**lift vs test prevalence:** {concise(details["lift_at_capacity"], 2)}x.',
        f'- **Scenario-only incremental value vs random ranking:** '
        f'€{details["simulated_incremental_vs_random"]:,.0f}, '
        'not an observed profit or a causal impact estimate.', '',
        '## Dataset and split', '',
        '| Split (original chronological order) | Records | Positive target rate |',
        '|---|---:|---:|',
    ]
    for key in ('train', 'validation', 'test'):
        x = manifest['splits'][key]
        lines.append(f'| {key.title()} | {x["records"]:,} | {x["positive_rate"]:.2%} |')
    lines += ['', '## Model comparison — untouched test', '',
              '| Model | Average precision | ROC AUC | Brier score | Precision @ capacity | Lift @ capacity |',
              '|---|---:|---:|---:|---:|---:|']
    for model_name, m in holdout_metrics.items():
        lines.append(f'| {model_name} | {concise(m["average_precision"])} | '
                     f'{concise(m["roc_auc"])} | {concise(m["brier"])} | '
                     f'{percent(m["precision_at_capacity"])} | '
                     f'{fmt_lift(m["lift_at_capacity"])} |')
    lines += ['', '**Baseline note:** a constant prior score cannot rank records; top-capacity precision and lift are not defined for this baseline.', '', '## Campaign simulation', '',
              f'- **Contact capacity:** top {a["capacity_fraction"]:.0%} of the test records ranked by score.',
              f'- **Hypothetical value per observed subscription:** €{a["value_per_conversion"]:,.0f}.',
              f'- **Hypothetical cost per contact:** €{a["contact_cost"]:,.0f}.',
              f'- **Contacts selected:** {details["selected"]:,}.',
              f'- **Subscriptions within selected rows:** {details["conversions_captured"]:,}.',
              f'- **Recall of historical subscribers:** {percent(details["recall_at_capacity"])}.',
              f'- **Simulated net value for selected rows:** €{details["simulated_net_value"]:,.0f}.',
              '',
              '**Warning:** all records were contacted historically; this retrospective exercise cannot tell '
              'us what would have happened to customers who were never contacted. '
              'The hypothetical financial values are user-configurable assumptions and not bank revenue data.', '',
              '## Model selection protocol', '',
              '- Data are contiguous: first 60% train, next 20% validation, final 20% test.',
              '- Candidate choice uses **validation average precision only**, never test scores.',
              '- Selected model is refit on train+validation before the final, one-off test evaluation.',
              '- Preprocessing is learned only on each fitting fold; unseen categories are ignored safely.',
              '- Target leakage: **`duration` excluded** (unknown until a call finishes).',
              '- Demographic fields **`age` and `marital` excluded** from prediction; age used for aggregate audit only.',
              '- Monthly economic indicators and other features may act as time proxies; concept drift is expected.',
              '', '## Validation-only comparison', '',
              '| Model | Validation average precision | Validation ROC AUC |',
              '|---|---:|---:|']
    for name, m in validation_metrics.items():
        lines.append(f'| {name} | {concise(m["average_precision"])} | {concise(m["roc_auc"])} |')
    lines += ['', '## Aggregate age-band audit (test; descriptive)', '',
              '| Age band | N | Actual positive rate | Selection rate | TPR | FPR | Precision |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in audit:
        lines.append(f'| {r["group"]} | {r["n"]} | {percent(r["base_rate"])} | '
                     f'{percent(r["selection_rate"])} | {percent(r["true_positive_rate"])} | '
                     f'{percent(r["false_positive_rate"])} | {percent(r["precision"])} |')
    lines += ['', '**Not a fairness certification.** Age groups may differ in contact history, '
              'outcome prevalence and sample size. Report only aggregate results.', '',
              '## Model interpretation', '',
              'Permutation importance measures the loss in **test average precision** when one '
              'input feature is shuffled. It indicates predictive reliance, **not causality**.', '',
              '| Feature | Mean decrease in average precision |', '|---|---:|']
    for k, v in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        lines.append(f'| {k} | {v:.4f} |')
    lines += ['', '## Reproducible figures', '',
              '![Precision recall](figures/precision_recall.png)',
              '![ROC curve](figures/roc.png)',
              '![Calibration](figures/calibration.png)',
              '![Campaign selection confusion matrix](figures/contact_matrix.png)',
              '![Permutation importance](figures/permutation_importance.png)', '',
              '## Limitations and responsible use', '',
              '- Data: Portuguese bank telemarketing from **2008–2010**, not a modern deployment population.',
              '- UCI documents row order as chronological, but exact timestamps and customer IDs are unavailable.',
              '- Repeat observations by the same person cannot be ruled out and group leakage cannot be tested.',
              '- Selection / calibration may shift over time. These are observational, historical associations.',
              '- Job, education and financial predictors may be demographic proxies and require governance.',
              '- No production deployment, lending/eligibility decision or automatic customer targeting is advised.',
              '- Reported experiments were developed with AI coding assistance; interpretation and ownership require review.',
              '', '## Provenance and verification', '',
              f'- Source: [UCI Bank Marketing, DOI 10.24432/C5K306](https://archive.ics.uci.edu/dataset/222/bank+marketing).',
              '- Dataset license: CC BY 4.0. Original data are not redistributed here.',
              f'- SHA256 of source CSV: `{manifest["data_sha256"]}`.',
              '- See `experiment.json` for machine-readable settings and results.', '']
    return '\n'.join(lines)


def run(data_path: str | Path, output: str | Path, fast=False, importance_limit=1200,
        capacity=.20, value_per_conversion=100., contact_cost=5., synthetic=False):
    path, out = Path(data_path), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    raw = read_data(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not synthetic:
        provenance_path = path.parent / 'source_provenance.json'
        if not provenance_path.is_file():
            raise ValueError('Original UCI data must be downloaded via python -m bankml.download first')
        meta = json.loads(provenance_path.read_text(encoding='utf-8'))
        if (meta.get('source_url') != 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
                or meta.get('sha256') != digest or meta.get('records') != 41_188
                or len(raw) != 41_188 or path.name != 'bank-additional-full.csv'):
            raise ValueError('Source dataset does not match the verified UCI downloader manifest')
    train, valid, test = make_splits(raw)
    xtr, ytr = xy(train)
    xva, yva = xy(valid)
    xte, yte = xy(test)
    xtrainval, ytrainval = xy(pd.concat([train, valid], axis=0))
    assert not set(EXCLUDED).intersection(xtr.columns)
    if min(ytr.sum(), yva.sum(), yte.sum()) == 0:
        raise ValueError('Each chronological split must have positive examples')
    validation = {}
    models = candidates(fast=fast)
    trained = {}
    for name, pipe in models.items():
        print(f'Training candidate: {name}', flush=True)
        pipe.fit(xtr, ytr)
        trained[name] = pipe
        proba = pipe.predict_proba(xva)[:, 1]
        validation[name] = evaluate(yva, proba, capacity, value_per_conversion, contact_cost)
        print(f'  validation AP={validation[name]["average_precision"]:.4f}', flush=True)
    winner = max(validation, key=lambda name: validation[name]['average_precision'])
    print(f'Selected based on validation only: {winner}', flush=True)
    selected_model = models[winner]
    selected_model.fit(xtrainval, ytrainval)
    dummy = DummyClassifier(strategy='prior')
    dummy.fit(xtrainval, ytrainval)
    results = {}
    predictions = {}
    for name, model in [('constant_prevalence_baseline', dummy), (winner, selected_model)]:
        proba = model.predict_proba(xte)[:, 1]
        predictions[name] = proba
        results[name] = evaluate(yte, proba, capacity, value_per_conversion, contact_cost)
        if name == 'constant_prevalence_baseline':
            for key in ('conversions_captured', 'precision_at_capacity', 'recall_at_capacity',
                        'lift_at_capacity', 'simulated_net_value', 'simulated_incremental_vs_random'):
                results[name][key] = None
    select = top_k_mask(predictions[winner], capacity)
    audit = subgroup_audit(yte, predictions[winner], test['age'].to_numpy(), capacity, min_n=20 if fast else 100)
    imp = permutation_importance(selected_model, xte.iloc[:importance_limit], yte[:importance_limit],
                                 scoring='average_precision', n_repeats=2 if fast else 3,
                                 random_state=42, n_jobs=1)
    importance = {col: float(imp.importances_mean[i]) for i, col in enumerate(FEATURES)}
    save_plots(yte, predictions, float(yte.mean()), out / 'figures', select.astype(int), importance)
    splits = {}
    for name, chunk in [('train', train), ('validation', valid), ('test', test)]:
        splits[name] = dict(records=int(len(chunk)), positive_rate=float(chunk['y'].eq('yes').mean()),
                            row_start=int(chunk.index[0]), row_end_inclusive=int(chunk.index[-1]))
    manifest = {'dataset': 'SYNTHETIC TEST FIXTURE' if synthetic else 'UCI bank-additional-full.csv',
                'original_data_verified': not synthetic, 'data_sha256': digest,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'splits': splits, 'selected_model': winner, 'feature_columns': list(FEATURES),
                'excluded_columns': sorted(EXCLUDED),
                'assumptions': dict(capacity_fraction=capacity, value_per_conversion=value_per_conversion,
                                    contact_cost=contact_cost, currency='EUR (hypothetical)'),
                'validation': validation, 'untouched_test': results,
                'age_band_audit': audit, 'permutation_importance': importance}
    (out / 'experiment.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'RESULTS.md').write_text(report(raw, manifest, results, validation, audit, importance), encoding='utf-8')
    model_path = Path('models') / 'champion.joblib'
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({'model': selected_model, 'feature_columns': list(FEATURES),
                 'source_sha256': digest, 'selected_model': winner}, model_path)
    print(f'Completed: results={out / "RESULTS.md"}; model={model_path}; AP={results[winner]["average_precision"]:.4f}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/bank-additional-full.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/published'))
    parser.add_argument('--fast', action='store_true', help='Faster test-only experimentation')
    parser.add_argument('--capacity', type=float, default=.20)
    parser.add_argument('--conversion-value', type=float, default=100.)
    parser.add_argument('--contact-cost', type=float, default=5.)
    args = parser.parse_args()
    run(args.data, args.output, fast=args.fast, capacity=args.capacity,
        value_per_conversion=args.conversion_value, contact_cost=args.contact_cost)
