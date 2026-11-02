"""Ranking, classification, cost-sensitive and subgroup metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             log_loss)


def top_k_mask(probs, capacity: float = .20):
    """Select exactly floor(capacity * n) observations (min 1), stable tie break."""
    p = np.asarray(probs, dtype=float)
    if len(p) == 0 or not np.isfinite(p).all() or not (0 < capacity <= 1):
        raise ValueError('Need finite, nonempty probabilities and 0 < capacity <= 1')
    k = max(1, int(np.floor(capacity * len(p))))
    selected = np.zeros(len(p), dtype=bool)
    selected[np.argsort(-p, kind='stable')[:k]] = True
    return selected


def evaluate(y, probs, capacity: float = .20, value: float = 100., cost: float = 5.) -> dict:
    """Evaluate rank performance and observed outcomes for a simulated campaign.

    Economic quantities are illustrative ASSUMPTIONS, not facts about the bank.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(probs, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError('Target must be nonempty and binary with same length as scores')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    selected = top_k_mask(p, capacity)
    tp = int(y[selected].sum())
    total_positives = int(y.sum())
    n_selected = int(selected.sum())
    prevalence = float(y.mean())
    precision_at_k = float(tp / n_selected)
    recall_at_k = float(tp / total_positives) if total_positives else 0.
    out = dict(n=int(len(y)), positive_rate=prevalence, selected=n_selected,
               conversions_captured=tp, precision_at_capacity=precision_at_k,
               recall_at_capacity=recall_at_k,
               lift_at_capacity=precision_at_k/prevalence if prevalence else None,
               simulated_net_value=value*tp-cost*n_selected,
               expected_random_net_value=(value*prevalence-cost)*n_selected,
               simulated_incremental_vs_random=value*(tp-prevalence*n_selected))
    if len(np.unique(y)) == 2:
        out.update(average_precision=float(average_precision_score(y, p)),
                   roc_auc=float(roc_auc_score(y, p)),
                   brier=float(brier_score_loss(y, p)),
                   log_loss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))))
    else:
        out.update(average_precision=None, roc_auc=None, brier=float(brier_score_loss(y, p)),
                   log_loss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0,1])))
    return out


def subgroup_audit(y, probs, ages, capacity=.20, min_n=100):
    """Aggregate observational subgroup diagnostics; not a fairness certification."""
    p = np.asarray(probs)
    ages = np.asarray(ages, dtype=float)
    y = np.asarray(y, dtype=int)
    mask = top_k_mask(p, capacity)
    groups = [('18-35', (ages >= 18) & (ages <= 35)),
              ('36-55', (ages >= 36) & (ages <= 55)),
              ('56+', ages >= 56)]
    result = []
    for name, group in groups:
        n = int(group.sum())
        if n < min_n:
            continue
        yg, pg = y[group], mask[group]
        tp, fp = int((yg & pg).sum()), int(((1-yg) & pg).sum())
        positives, negatives = int(yg.sum()), int((1-yg).sum())
        result.append(dict(group=name, n=n,
                           base_rate=float(yg.mean()),
                           selection_rate=float(pg.mean()),
                           true_positive_rate=tp/positives if positives else None,
                           false_positive_rate=fp/negatives if negatives else None,
                           precision=tp/int(pg.sum()) if pg.sum() else None))
    return result
