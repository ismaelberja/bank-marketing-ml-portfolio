"""Unit and end-to-end tests use ONLY SYNTHETIC, labeled test data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankml.data import read_data, make_splits, xy, FEATURES, EXCLUDED
from bankml.fixtures import synthetic_fixture
from bankml.metrics import top_k_mask, evaluate, subgroup_audit
from bankml.train import run
from bankml.predict import score


def test_no_target_leakage_in_training_features():
    df = synthetic_fixture()
    X, y = xy(df)
    assert set(X.columns) == set(FEATURES)
    assert not set(X.columns) & EXCLUDED
    assert 'duration' not in X and 'age' not in X and 'marital' not in X
    assert y.dtype.kind in 'iu'


def test_chronological_split_indices_are_contiguous_and_disjoint():
    df = synthetic_fixture()
    train, validation, test = make_splits(df)
    assert list(train.index) == list(range(0, 360))
    assert list(validation.index) == list(range(360, 480))
    assert list(test.index) == list(range(480, 600))
    assert len(train) + len(validation) + len(test) == len(df)


def test_invalid_split_sizes_rejected():
    with pytest.raises(ValueError):
        make_splits(synthetic_fixture(), .85, .2)


def test_top_k_selects_exact_capacity_with_stable_ties():
    p = [0.5] * 10
    selected = top_k_mask(p, .2)
    assert selected.sum() == 2
    assert selected[:2].all()
    assert not selected[2:].any()


def test_rank_metrics_expected_business_value():
    y = [1,0,1,0,0,0,0,0,0,0]
    p = [0.9,.8,.7,.6,.5,.4,.3,.2,.1,.0]
    metrics = evaluate(y, p, .2, value=100, cost=5)
    assert metrics['selected'] == 2
    assert metrics['conversions_captured'] == 1
    assert metrics['precision_at_capacity'] == .5
    assert metrics['lift_at_capacity'] == pytest.approx(2.5)
    assert metrics['simulated_net_value'] == 90
    assert metrics['simulated_incremental_vs_random'] == pytest.approx(60)


def test_probabilities_out_of_range_rejected():
    with pytest.raises(ValueError):
        evaluate([0,1],[.2,1.1])


def test_subgroup_audit_only_outputs_aggregates():
    y = np.array([0,1]*60)
    p = np.linspace(.02,.98,120)
    ages = np.concatenate([np.full(40,27),np.full(40,44),np.full(40,65)])
    audit = subgroup_audit(y,p,ages,capacity=.2,min_n=20)
    assert len(audit) == 3
    assert all('group' in entry and 'selection_rate' in entry and entry['n']==40 for entry in audit)
    assert all('age' not in entry for entry in audit)


def test_uci_columns_validation(tmp_path):
    df = synthetic_fixture(450)
    csv = tmp_path/'synthetic.csv'
    df.to_csv(csv, sep=';', index=False)
    assert len(read_data(csv)) == 450
    df.drop(columns=['duration']).to_csv(csv,sep=';',index=False)
    with pytest.raises(ValueError,match='duration'):
        read_data(csv)


def test_full_synthetic_workflow_runs_and_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path/'synthetic-test-only.csv'
    synthetic_fixture(n=650).to_csv(data,sep=';',index=False)
    results = run(data, tmp_path/'reports',fast=True,importance_limit=100,synthetic=True)
    assert (tmp_path/'reports/RESULTS.md').is_file()
    assert len(list((tmp_path/'reports/figures').glob('*.png'))) == 5
    assert (tmp_path/'models/champion.joblib').is_file()
    assert results['selected_model'] in ('gradient_boosting','logistic_regression')
    assert 'duration' not in results['feature_columns']
    manifest = json.loads((tmp_path/'reports/experiment.json').read_text())
    assert manifest['dataset'] == 'SYNTHETIC TEST FIXTURE'
    assert not manifest['original_data_verified']
    assert sum(x['records'] for x in manifest['splits'].values()) == 650
    # These synthetic results are explicitly never claimed as real-data performance.
    candidates = synthetic_fixture(3).drop(columns=['y']).copy()
    input_csv = tmp_path/'scoring.csv'
    candidates.to_csv(input_csv,index=False)
    out = score(input_csv,tmp_path/'models/champion.joblib',tmp_path/'scored.csv')
    assert len(out) == 3
    assert out['subscription_score'].between(0,1).all()


def test_cannot_label_synthetic_as_original(tmp_path):
    source = tmp_path / 'synthetic.csv'
    synthetic_fixture(n=650).to_csv(source, sep=';', index=False)
    with pytest.raises(ValueError, match='must be downloaded'):
        run(source, tmp_path/'reports', fast=True)
