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

### Implementation Status — Phase 2

**Completed in Phase 1 (Preliminary Experiments)**

- Established a logistic regression baseline using a stratified 70/15/15 training, validation, and test split with random seed 42.
- Evaluated healthy baseline performance on the validation set.
- Injected missing values across all 23 predictor features at overall rates of 5%, 15%, and 30%.
- Implemented an overall missingness-rate detector and an inference-success check.
- Applied median imputation fitted on healthy training data.
- Evaluated corrected validation batches using F1, recall, ROC-AUC, and accuracy.
- Implemented preliminary recovery acceptance decisions using a 10% relative F1 degradation tolerance.
- Retained the approved model configuration when a candidate recovery failed validation; a full production rollback mechanism has not yet been demonstrated.

**Completed in Phase 2 (Healthy Reference Profile)**

- Established a healthy reference using 21,000 training observations and 23 predictor features.
- Documented feature distributions, missingness, class balance, rare categorical codes, and negative billing amounts.
- Selected `BILL_AMT1` for feature-specific missingness monitoring.
- Selected `LIMIT_BAL` for numerical drift monitoring and `PAY_0` for categorical shift monitoring.
- Defined five preliminary monitoring candidates: schema, missingness, numerical drift, categorical shift, and class balance.
- Proposed a `BILL_AMT1` missingness alert threshold of more than 5 percentage points above the healthy training reference.

**Planned for Later Phases**

- Implement and evaluate feature-specific missingness injection into `BILL_AMT1`.
- Test the proposed `BILL_AMT1` detection rule on actual corrupted batches.
- Evaluate detection reliability, including missed failures and false alerts.
- Investigate numerical drift and categorical shift using controlled experiments.
- Formalize recovery acceptance criteria, including a critical-class recall guardrail.
- Implement and verify model-state preservation and rollback behavior.

**Experimental distinction:** Phase 1 used overall missingness across all predictor features. The next controlled experiment will target `BILL_AMT1` specifically. The results of these experiments must be reported separately.

**Limitation:** Recovery evaluation currently uses known validation labels in an offline experiment. In a deployed system, true labels may not be immediately available, so performance-based acceptance may require delayed evaluation.

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


## Connection to the Self-Healing Machine Learning System

### Purpose of the Healthy Reference Profile

The healthy reference profile establishes the expected characteristics of the training data before introducing controlled failures.

It was created using 21,000 training observations and 23 predictor features. The profile provides a baseline for comparing future incoming data batches and identifying potential data-quality or distribution changes.

### How the Reference Profile Supports Failure Detection

The SHML system will use the healthy reference to evaluate five monitoring conditions:

1. **Schema validation:** Verify that incoming batches contain the expected predictor features with compatible data types.
2. **Missingness monitoring:** Compare incoming missing-value rates with the healthy reference. `BILL_AMT1` is the selected feature for the next controlled missingness experiment.
3. **Numerical drift:** Compare the distribution of `LIMIT_BAL` against its training reference distribution.
4. **Categorical shift:** Compare `PAY_0` category frequencies and identify previously unseen codes.
5. **Class balance:** Monitor changes in the default rate when actual outcome labels become available.

### Connection to Diagnosis and Correction

When the system detects a potential failure, it will identify the affected feature and type of problem before selecting an appropriate response.

For the initial missingness experiment, the proposed correction is median imputation using the median calculated from healthy training data.

Schema failures, numerical drift, and categorical shifts may require investigation rather than automatic correction.

### Recovery Validation and Safety

After an approved correction is applied, the corrected pipeline must be evaluated before the recovery is accepted.

The preliminary Phase 1 experiments used F1-score as the primary recovery metric and allowed a maximum relative F1 degradation of 10% compared with the healthy baseline.

A correction that fails the recovery criteria must not replace the approved model or pipeline state. The failure should be recorded and escalated for further investigation.

Future experiments will also examine recall-based safeguards because identifying clients who actually default is important for this classification problem.

### Scope and Limitations

Phase 2 establishes the reference profile and proposed monitoring rules. It does not implement new automated recovery experiments.

Preliminary missingness injection, median imputation, and recovery acceptance experiments were performed in Phase 1.

The current recovery evaluation uses labeled validation data. A deployed system may require delayed performance evaluation when true outcomes are not immediately available.

The reference profile and monitoring rules provide the foundation for subsequent failure-detection, diagnosis, and recovery experiments.
