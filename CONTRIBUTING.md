# Reproducible contributions

- Run `python -m pytest -q` before proposing changes.
- Do not commit raw original data, customer-level predictions, local model binary files or tokens.
- Do not introduce features unknown at call-decision time.
- Do not use the final chronological test split for model selection or iterative tuning.
- Keep synthetic fixture results labeled as synthetic; only original-data training can support real business claims.
- When interpreting findings, distinguish historical correlation from causal outreach effectiveness.
- Update MODEL_CARD.md when adding any new predictors, thresholds or evaluation protocols.
