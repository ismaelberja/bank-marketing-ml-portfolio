"""Matplotlib plots generated only from evaluated, original-data model results."""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (precision_recall_curve, roc_curve, confusion_matrix,
                             )
from sklearn.calibration import calibration_curve


def save_plots(y, predictions, baseline_rate, target, selection, importance=None):
    target.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, scores in predictions.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, label=name, linewidth=2)
    ax.axhline(baseline_rate, linestyle='--', alpha=.7, label='positive prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–recall | untouched chronological test', xlim=(0,1), ylim=(0,1))
    ax.legend()
    fig.tight_layout(); fig.savefig(target / 'precision_recall.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, scores in predictions.items():
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=name, linewidth=2)
    ax.plot([0,1],[0,1],linestyle='--',alpha=.6)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC | untouched chronological test')
    ax.legend(); fig.tight_layout(); fig.savefig(target / 'roc.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, scores in predictions.items():
        try:
            prob_true, prob_pred = calibration_curve(y, scores, n_bins=8, strategy='quantile')
            ax.plot(prob_pred, prob_true, marker='o', linewidth=1.5, label=name)
        except ValueError:
            pass
    ax.plot([0,1],[0,1],linestyle='--',label='ideal calibration',alpha=.6)
    ax.set(xlabel='Mean predicted score', ylabel='Observed subscription frequency', title='Calibration (diagnostic only)', xlim=(0,1), ylim=(0,1))
    ax.legend(); fig.tight_layout(); fig.savefig(target / 'calibration.png', dpi=160); plt.close(fig)

    cm = confusion_matrix(y, selection, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Not selected', 'Selected'])
    ax.set_yticks([0, 1], labels=['No subscription', 'Subscription'])
    for (i, j), count in np.ndenumerate(cm):
        ax.text(j, i, f'{count:,}', ha='center', va='center', color='white' if count > cm.max()/2 else 'black')
    ax.set(title='Top-capacity contact selection vs historical outcome', xlabel='Campaign selection', ylabel='Actual subscription')
    fig.tight_layout(); fig.savefig(target/'contact_matrix.png',dpi=160); plt.close(fig)

    if importance:
        pairs = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:14]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([k for k,v in pairs][::-1], [v for k,v in pairs][::-1])
        ax.set(xlabel='Decrease in test average precision when shuffled',title='Permutation importance (diagnostic; test set)')
        fig.tight_layout(); fig.savefig(target/'permutation_importance.png', dpi=160); plt.close(fig)
