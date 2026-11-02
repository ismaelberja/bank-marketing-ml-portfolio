# Model card — Bank Marketing propensity ranking (research / demonstration only)

## Intended use

Education and professional portfolio: predict a historical **term-deposit subscription label** on already-contacted records, using features plausibly available before a marketing call. This is not a financial eligibility model or a production-ready targeting system.

## Original dataset

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. Licensed CC BY 4.0. Full `bank-additional-full.csv`, 41,188 time-ordered records, May 2008–November 2010.

## Leakage and feature policy

- **Exclude `duration`:** not available until the call ends; using it would give an unrealistic benchmark.
- **Exclude `age` and `marital`:** avoid direct use of two demographic features in the scoring system. The `age` variable is used only for aggregate subgroup diagnostics.
- Retained: contact channel, proposed/recorded month/day, prior campaign and previous-contact history, other CRM attributes and macroeconomic indicators. The operational availability and compliance of each feature would need verification in a real deployment.
- All preprocessing is fit inside the training pipeline. Outcome `y` cannot enter model inputs.

## Evaluation approach

One chronological split: 60% fit, 20% candidate selection, 20% untouched test. Model selection uses **validation average precision**, to accommodate asymmetric class prevalence. No accuracy-only claims. Final model is refit on 80%, tested only once. Report precision–recall AUC, ROC AUC, Brier, PR curves, calibration, subgroup audit and capacity-constrained rankings.

## Verified original-data evaluation (historical 2008–2010)

The complete GitHub Actions experiment ran on the UCI `bank-additional-full.csv` file (41,188 records), with source integrity verified via SHA-256. A fixed chronological 60/20/20 split yielded 24,712 train, 8,238 validation and 8,238 test rows.

- **Champion:** logistic regression, selected by validation average precision when compared with histogram gradient boosting.
- **Validation:** class prevalence 11.07%, average precision **0.099**, ROC AUC **0.490**. These weak scores indicate limited validation ranking performance.
- **Untouched final test:** class prevalence 30.83%, average precision **0.529**, ROC AUC **0.749**, Brier score **0.191**.
- **Capacity ranking:** precision **59.4%** among the highest-scored 20% of historically contacted records (1,647 records), representing **1.93×** the test positive prevalence.
- **Strong temporal drift:** the positive rate rises from 4.81% (train) to 11.07% (validation) to 30.83% (test); the difference in validation vs test performance undermines claims of stable deployment generalization.
- A constant-prevalence baseline has no meaningful ranking, so top-capacity precision and lift are intentionally **not applicable**.

These are historical diagnostic outcomes, not a stable modern forecast or measured incremental profit. For source hashes, reproducible figures, subgroup comparisons and detailed model outputs, see [results](../reports/published/RESULTS.md) and [experiment manifest](../reports/published/experiment.json).

## Business experiment limitations

Illustrative contact value €100 and call cost €5 are **assumptions**, not amounts provided by UCI or by any bank. Expected random targeting uses the test-set prevalence solely as a historical comparator. Retrospective outcomes are not a randomized uplift or an estimated causal treatment effect.

## Ethical and scientific limitations

Age may correlate with other retained predictors. Reported subgroup TPR/FPR disparities do not establish fairness. No complete timestamps or persistent customer IDs prevent group leakage checks and sophisticated longitudinal validation. Results are highly context dependent; 2008–2010 bank data should never be used for present-day targeting without governance and renewed validation. Job and education can act as proxies for protected characteristics. The source includes individuals' historical records and is intended for statistical research, not individual decisions.

## Reproducibility

Original source is downloaded from UCI and hashed SHA-256. No client-level source rows are committed. Random seeds are fixed. Model serialization is local (`models/`) and ignored by Git; do not unpickle untrusted model files.

## AI assistance and attribution

This project was created with AI coding assistance. The Git repository is a practical learning and portfolio artifact; the owner should run, understand, review, explain and revise the pipeline and its results before representing it in applications.
