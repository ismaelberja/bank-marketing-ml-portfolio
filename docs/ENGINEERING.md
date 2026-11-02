# Engineering decisions / interview discussion

## Why a chronological holdout rather than a random split?

The original UCI *bank-additional-full* dataset is described as ordered by date. A shuffled random split would let future market conditions leak into the training period and could make evaluation overly optimistic. Calendar timestamps are not provided, so observation order is used as a proxy. Strong seasonal and macroeconomic drift remains possible.

## Why remove call duration?

`duration` is only known after the marketing call. If the objective is deciding whom to contact, it is unavailable at decision time and creates **target leakage**. The dataset's documentation explicitly warns about this feature.

## Why average precision, not accuracy?

Most examples are non-subscribers. A near-all-negative model could have high accuracy and still fail to identify subscribers. AP summarizes the precision–recall ranking while explicitly benchmarking against the prevalence. ROC AUC and Brier score provide different perspectives.

## Why a fixed contact capacity?

A marketing team often has a limited number of calls. Selecting the top 20% by score provides a comparable allocation policy across models. The budget and value assumptions are editable. The retrospective comparison **does not estimate uplift** or forecast incremental customer behavior.

## Why model selection on validation only?

The final test set represents unseen later events. Trying many features and selecting based on the test would overfit the benchmark. Fit on the early 60%, select on the next 20%, refit on 80%, and finally evaluate once on the last 20%.

## Production upgrades to discuss

- Verify availability of every feature at the exact decision timestamp.
- Construct person-level and timestamped splits (not possible with this public extract).
- Quantify uncertainty using time-block bootstrapping or rolling-window backtesting.
- Monitor prevalence drift, score calibration, referral rates, subgroup stability and cost assumptions.
- Develop a randomized control group and uplift model to estimate *causal incremental conversion*.
- Apply consent, GDPR/privacy, data retention and anti-discrimination governance before contacting people.
- Add schema validation, model registry, artifacts and monitoring for an actual production deployment.

## Verification checklist

- [ ] Run all synthetic tests
- [ ] Download original UCI data (41,188 rows)
- [ ] Execute the real experiment (not the synthetic fixture)
- [ ] Inspect business results and subgroup metrics
- [ ] Check all five plots
- [ ] Review test model assumptions and limitations
- [ ] Explain potential leakage and model choice in your own words
