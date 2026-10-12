## Missingness Monitoring — BILL_AMT1

### Selected Feature
- **Feature:** `BILL_AMT1`
- **Description:** Most recent bill statement amount
- **Data type:** Numerical
- **Monitoring purpose:** Detect missing values in incoming data batches

### Healthy Training Reference
- Training observations: 21,000
- Reference missingness: 0%
- Mean: 51,194.22
- Median: 22,225.50
- Standard deviation: 73,633.98
- Minimum: -165,580
- Maximum: 964,511

These statistics were calculated using training data only.

### Preliminary Failure Detection Rule

Missingness increase = Current batch missingness − Healthy reference missingness.

A missing-data failure is flagged when the increase exceeds 5 percentage points.

### Visualization Test Cases

| Simulated missingness | Expected detection |
|---|---|
| 0% | No failure |
| 5% | No failure |
| 15% | Failure detected |
| 30% | Failure detected |

The interactive visualization illustrates the proposed detection threshold. It does not yet evaluate missingness in actual experimental data batches.

### Proposed Healing Action

When a missingness failure is detected, the system will attempt median imputation using the healthy training median of 22,225.50.

The corrected pipeline will be evaluated against predefined recovery criteria before the correction is accepted.

If the correction fails the acceptance criteria, the system will reject it and retain the previously approved pipeline.

### Implementation Status

- Train/validation/test split: Completed
- Healthy training reference for BILL_AMT1: Completed
- Interactive detection visualization: Implemented
- Actual missing-value injection: Not yet implemented
- Automated failure detection on experimental batches: Not yet implemented
- Median-imputation recovery experiment: Not yet implemented
- Recovery acceptance and rollback: Not yet implemented


**Duplicate Record Investigation**

The training dataset contains 21,000 observations. After excluding the identifier column, 36 records belong to duplicate groups, corresponding to 18 duplicate occurrences beyond the first (approximately 0.086%).

These records may represent different customers with identical feature values and target labels. Because their validity has not been disproven, they will be retained during initial profiling.

No duplicate records have been removed.

### Training Numerical Feature Profile

Five numerical features were profiled using the 21,000 training observations: `LIMIT_BAL`, `AGE`, `BILL_AMT1`, `PAY_AMT1`, and `BILL_AMT2`.

All five features have a missingness rate of 0%.

The monetary features exhibit evidence of right-skewed distributions, with means greater than medians. This motivates evaluating median imputation as an initial missing-value recovery strategy.

Negative bill statement amounts were observed in `BILL_AMT1` and `BILL_AMT2`. These values will be investigated before being classified as invalid.

`LIMIT_BAL` was selected as the preliminary numerical distribution-drift monitoring candidate because it is interpretable and exhibits substantial variation.

The healthy reference statistics were calculated exclusively from the training dataset to avoid using held-out validation or test information.

### Training Categorical Feature Profile

Five categorical or ordinal features were examined using the 21,000 training observations: `SEX`, `EDUCATION`, `MARRIAGE`, `PAY_0`, and `PAY_2`.

All five features have 0% missingness.

A preliminary rare-category threshold of less than 1% was applied.

- **SEX:** 2 unique categories; no rare categories.
- **EDUCATION:** 7 unique categories; rare codes 0, 4, and 6.
- **MARRIAGE:** 4 unique categories; rare code 0.
- **PAY_0:** 11 unique categories; rare codes 4, 5, 6, 7, and 8.
- **PAY_2:** 11 unique categories; rare codes 1, 4, 5, 6, 7, and 8.

The most common categories were SEX = 2 (60.36%), EDUCATION = 2 (46.82%), MARRIAGE = 2 (53.37%), PAY_0 = 0 (48.98%), and PAY_2 = 0 (52.31%).

The rare-category findings differ slightly from the full-dataset profile, reinforcing the importance of establishing the operational reference using training observations only.

Rare categories will be retained during initial profiling. Undocumented codes will be investigated before any preprocessing decisions are made.


## Step 6 — Monitoring Candidates

The SHML system will initially monitor five aspects of data quality and distribution.

**1. Schema Monitoring — LIMIT_BAL**

- Potential failure: Required column missing or incompatible data type.
- Evidence: Schema validation failure.
- Initial response: Reject the affected batch and report the schema problem.

**2. Missingness Monitoring — BILL_AMT1**

- Healthy training missingness: 0%.
- Preliminary threshold: More than 5 percentage points above the training reference.
- Initial response: Flag the batch for median-imputation recovery evaluation.
- Training median: 22,225.50.

**3. Numerical Drift Monitoring — LIMIT_BAL**

- Healthy training median: 140,000.
- Potential failure: Meaningful change in credit-limit distribution.
- Initial response: Flag for investigation.
- Detection threshold: To be established in a later experiment.

**4. Categorical Shift Monitoring — PAY_0**

- Most common category: 0 (48.98%).
- Potential failure: New repayment-status codes or meaningful distribution changes.
- Initial response: Flag for investigation. Do not automatically remap categories.
- Detection threshold: To be established later.

**5. Class-Balance Monitoring — default payment next month**

- Healthy positive-class proportion: 22.12%.
- Potential failure: Meaningful change in default prevalence.
- Initial response: Investigate possible population or label-distribution changes when ground-truth labels become available.
- Detection threshold: To be established later.

These candidates were selected using training-only reference statistics. Automated monitoring and recovery have not yet been implemented.
