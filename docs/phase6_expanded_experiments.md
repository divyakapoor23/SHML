# Phase 6 — Reproducibility and Robustness Verification

## 1. Overview

Phase 6 evaluates the reproducibility and safety behavior of the Self-Healing Machine Learning (SHML) system developed during Phases 1–5.

Earlier phases established a healthy credit-default classification baseline, introduced controlled missing-data failures, implemented rule-based failure detection and diagnosis, applied median-imputation recovery, and evaluated recovery outcomes using predefined performance and safety criteria.

Phase 6 extends this work by independently reproducing the experimental results, verifying recovery decision rules, and testing whether simulated pipeline states are protected from unsafe recovery candidates.

The primary focus is **experimental reproducibility, safety verification, and state preservation**, rather than introducing additional failure types or machine-learning models.

---

## 2. Research Objectives

The objectives of Phase 6 are:

1. Verify that Phase 5 experimental results can be reproduced using the same experimental configuration.
2. Compare reproduced evaluation metrics against previously saved results.
3. Test whether the safety controller correctly rejects recovery candidates that violate predefined criteria.
4. Verify that unsafe candidates cannot replace an approved pipeline state in the simulated controller.
5. Confirm that valid candidates can be promoted successfully.
6. Save experiment results and verification logs for future analysis.

### Research Question

Can the SHML experimental workflow reproduce previously recorded results and consistently apply predefined recovery safety rules under controlled conditions?

### Hypothesis

Under a fixed dataset, preprocessing configuration, model configuration, and random seed, the SHML workflow will reproduce its recorded evaluation metrics within numerical tolerance and correctly distinguish acceptable from unacceptable recovery candidates in controlled safety tests.

---

## 3. Experimental Configuration

The experiment uses the same dataset and model configuration established in earlier phases.

| Component | Configuration |
|---|---|
| Dataset | UCI Default of Credit Card Clients |
| Observations | 30,000 |
| Predictor variables | 23 |
| Prediction target | `default payment next month` |
| Positive class | Default (1) |
| Baseline model | Logistic Regression |
| Preprocessing | StandardScaler |
| Training split | 70% |
| Validation split | 15% |
| Test split | 15% |
| Random seed | 42 |
| Monitored feature | `BILL_AMT1` |
| Failure type | Artificial missingness |
| Missingness levels | 0%, 5%, 15%, 30% |
| Recovery strategy | Training-median imputation |
| Primary performance metric | F1-score |
| Additional safety metric | Recall |
| Maximum relative F1 loss | 10% |
| Maximum relative recall loss | 10% |

The missingness experiments operate on copies of the validation dataset. The original dataset remains unchanged.

The training data is used to fit the scaler and Logistic Regression model and to calculate the median used for recovery.

---

## 4. Step 6.1 — Verification of Saved Experimental Artifacts

### Objective

Verify that the Phase 5 experiment artifacts exist, contain valid data, and can be used as reference results.

### Method

The following saved files were loaded:

- `results/phase5/phase5_recovery_performance.csv`
- `results/phase5/phase5_safety_decisions.csv`
- `results/phase5/phase5_final_experiment_summary.csv`

Validation checks examined the required columns, missingness scenarios, performance metric values, and safety decisions.

### Result

All required checks passed.

**Verification output:**

`Phase 6.1 verification passed. All saved experiment artifacts are structurally valid.`

### Interpretation

The saved Phase 5 results were structurally suitable for use as reference artifacts in the reproducibility experiment.

This check does not independently verify the scientific correctness of every saved result.

---

## 5. Steps 6.2–6.3 — Experimental Reproducibility

### Objective

Determine whether the Phase 5 missingness experiments can be reproduced using the same dataset, preprocessing procedure, model configuration, and random seed.

### Method

The experimental pipeline was independently reconstructed:

1. Load the original credit-default dataset.
2. Separate the predictors and target variable.
3. Recreate stratified training, validation, and test splits using random seed 42.
4. Fit StandardScaler on the training data.
5. Train the Logistic Regression baseline.
6. Inject missing values into `BILL_AMT1` at the four predefined levels.
7. Apply median imputation using the training-derived median.
8. Generate predictions and predicted probabilities.
9. Calculate accuracy, precision, recall, F1-score, and ROC-AUC.
10. Compare the reproduced results against the saved Phase 5 reference.

The comparison used NumPy's `assert_allclose()` with:

- Relative tolerance: `1e-6`
- Absolute tolerance: `1e-8`

### Results

| Missingness | Reference F1 | Reproduced F1 | Verification |
|---|---:|---:|---|
| 0% | 0.356983 | 0.356983 | PASS |
| 5% | 0.358898 | 0.358898 | PASS |
| 15% | 0.360947 | 0.360947 | PASS |
| 30% | 0.373723 | 0.373723 | PASS |

All five evaluation metrics matched the reference results within the specified numerical tolerances.

**Verification output:**

`Phase 6.3 verification passed. All reproduced metrics match Phase 5.`

### Interpretation

The experiment demonstrated reproducibility under the same experimental configuration and execution environment.

The results support the reliability of the implemented experimental procedure under controlled conditions.

They do not establish reproducibility across different random seeds, software environments, or datasets.

### Important Observation

The recovered F1-score increased slightly as the injected `BILL_AMT1` missingness level increased.

This observation should not be interpreted as evidence that missingness generally improves predictive performance.

The behavior may depend on the selected feature, corruption pattern, trained model, and recovery method.

---

## 6. Step 6.4 — Recovery Safety Verification

### Objective

Verify that the recovery safety rules reject candidates that violate performance or data-integrity requirements and accept candidates that satisfy the rules.

### Safety Criteria

A recovery candidate is rejected when:

- Relative F1 degradation exceeds 10%.
- Relative recall degradation exceeds 10%.
- Data-integrity validation fails.
- Required performance metrics are invalid.

### Method

A simplified safety verification function was implemented using the same primary acceptance criteria established in Phase 5.

Controlled synthetic performance values were generated to test four conditions.

### Results

| Test Scenario | Expected Decision | Actual Decision | Result |
|---|---|---|---|
| Excessive F1 loss | REJECT | REJECT | PASS |
| Excessive recall loss | REJECT | REJECT | PASS |
| Data-integrity failure | REJECT | REJECT | PASS |
| Valid recovery candidate | ACCEPT | ACCEPT | PASS |

**Verification output:**

`Phase 6.4 verification passed. All safety controller tests passed.`

### Interpretation

All four controlled safety tests produced the expected decisions.

The results demonstrate that the simplified verification function correctly applies the tested acceptance and rejection conditions.

### Limitation

These tests were performed using a separate verification function rather than directly invoking the complete original Phase 5 safety controller.

Therefore, the results verify the tested safety rules but do not constitute comprehensive testing of the original controller implementation.

---

## 7. Step 6.5 — State Preservation and Candidate Promotion

### Objective

Verify that unsafe recovery candidates do not replace the previously approved pipeline state and that acceptable candidates can be promoted.

### Method

A simulated approved-state record was created with:

- Version: `baseline_v1`
- Model: Logistic Regression
- Status: APPROVED
- Reference F1 and recall values

A state-transition function evaluated candidate metrics using the safety verification rules.

Two scenarios were tested.

### Experiment A — Unsafe Candidate Rejection

A candidate with a 30% reduction in F1 and recall was evaluated.

**Results:**

| Property | Result |
|---|---|
| Candidate | `candidate_unsafe_v2` |
| Safety decision | REJECT |
| Action | PRESERVE_APPROVED_STATE |
| Active version | `baseline_v1` |
| Verification | PASS |

The unsafe candidate was rejected.

The approved-state record remained unchanged.

### Experiment B — Safe Candidate Promotion

A candidate with a 2% reduction in F1 and recall was evaluated.

**Results:**

| Property | Result |
|---|---|
| Candidate | `candidate_safe_v2` |
| Safety decision | ACCEPT |
| Action | PROMOTE_CANDIDATE |
| Resulting version | `candidate_safe_v2` |
| Verification | PASS |

The valid candidate was accepted, and the returned state reflected the promoted version.

The original input state was not mutated.

### Interpretation

Both tests passed, demonstrating the expected state-transition behavior in the local simulation.

This supports the feasibility of separating recovery validation from candidate promotion.

### Limitation

The experiment used in-memory state records. It did not deploy or replace a serialized machine-learning model, execute production rollback, or manage live prediction traffic.

---

## 8. Step 6.6 — Experiment Artifacts and Verification Logs

### Objective

Preserve the evidence generated during Phase 6 for reproducibility, analysis, and reporting.

### Generated Files

All artifacts were saved under `results/phase6/`.

| File | Records | Description |
|---|---:|---|
| `phase6_reproducibility_comparison.csv` | 4 | Comparison of reference and reproduced metrics |
| `phase6_safety_test_results.csv` | 4 | Controlled safety-controller verification |
| `phase6_state_transition_logs.csv` | 2 | Simulated rejection and promotion outcomes |

### Verification

Each file was checked for existence and nonempty contents.

The saved safety and state-transition logs were also checked to confirm passing outcomes.

**Verification output:**

`Phase 6.6 verification passed.`

All three artifacts were verified successfully.

---

## 9. Summary of Phase 6 Results

| Step | Experiment | Result |
|---|---|---|
| 6.1 | Saved artifact verification | PASS |
| 6.2 | Experiment reproduction | Completed |
| 6.3 | Reference metric comparison | PASS |
| 6.4 | Safety-controller verification | 4/4 PASS |
| 6.5 | State preservation and promotion | 2/2 PASS |
| 6.6 | Artifact generation and verification | PASS |

### Key Findings

**Finding 1 — Controlled reproducibility**

The SHML experiment reproduced the saved Phase 5 evaluation metrics within the specified numerical tolerances.

**Finding 2 — Safety-rule consistency**

The simplified safety verification function produced the expected acceptance and rejection decisions for all four tested scenarios.

**Finding 3 — Approved-state preservation**

The simulated controller prevented an unsafe candidate from replacing the approved state.

**Finding 4 — Controlled candidate promotion**

The simulated controller promoted a candidate that satisfied the predefined safety criteria.

**Finding 5 — Traceable experimental evidence**

The reproducibility comparison, safety-test results, and state-transition outcomes were saved as CSV artifacts.

---

## 10. Known Limitations and Threats to Validity

### Experimental Scope

The reproducibility experiment uses one structured dataset, one baseline model, and one monitored feature.

The results cannot automatically be generalized to other machine-learning models, datasets, or failure types.

### Randomness

The experiment uses a fixed random seed of 42.

Multiple-seed reproducibility and robustness testing remain future work.

### Environment Reproducibility

The experiment was reproduced within the same execution environment.

Reproducibility across different Python versions, operating systems, and library versions has not yet been established.

### Safety Verification

The controlled safety tests used a simplified verification function.

Direct testing of the original Phase 5 safety controller remains an additional validation opportunity.

### State Management

State preservation and promotion were simulated locally.

Actual model artifact replacement, production rollback, cloud deployment, and live monitoring were not evaluated in Phase 6.

### Performance Evaluation

The experiment uses labeled validation data.

In real-world deployments, target labels may not be immediately available, which can delay performance-based recovery validation.

---

## 11. Reproducibility Instructions

The Phase 6 experiment is implemented in:

`phase6_expanded_experiments.ipynb`

### Execution Procedure

1. Install the dependencies listed in `requirements.txt`.
2. Make the UCI credit-default dataset available in the expected local data directory.
3. Open the Phase 6 notebook from the repository root.
4. Execute the notebook cells in order.
5. Verify that the reproducibility and safety assertions pass.
6. Inspect the generated CSV files under `results/phase6/`.

### Expected Results

- Four reproduced missingness scenarios.
- Matching Phase 5 metrics within tolerance.
- Four passing safety-controller tests.
- Two passing state-transition tests.
- Three verified Phase 6 output artifacts.

The dataset must be available locally in the expected format and location for the notebook to run successfully.

---

## 12. Research Conclusion

Phase 6 demonstrates that the SHML experimental workflow can reproduce its recorded credit-default classification results under controlled, fixed-seed conditions.

The controlled safety tests further show that predefined performance and data-integrity rules can distinguish acceptable and unacceptable recovery candidates.

The simulated state-transition experiments demonstrate how a recovery controller can preserve an approved state when a candidate fails validation and return an updated state when a candidate passes.

Together, these findings support the feasibility of a reproducible and safety-aware recovery evaluation workflow for the selected missing-data experiment.

The results provide evidence for the tested configuration rather than establishing general reliability across all failure types or production environments.

---

## 13. Future Work

Potential extensions include:

- Testing multiple random seeds and execution environments.
- Directly testing the complete Phase 5 safety controller.
- Packaging reusable detection, diagnosis, recovery, and validation components.
- Implementing cloud deployment and model artifact versioning.
- Evaluating additional numerical and categorical failure types.
- Conducting subgroup-level performance analysis.
- Comparing alternative models and recovery strategies.
- Investigating LLM-assisted diagnosis and automated recovery planning.

---

## 14. Phase Status

**Phase 6: Completed — Controlled Reproducibility and Safety Verification**

The experimental results and verification logs have been generated and validated.

The next academic milestone is to consolidate the experimental evidence, complete any final reproducibility checks, and prepare the cloud demonstration and research report.