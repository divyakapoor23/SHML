# Phase 4 — Experiment Log

## Experiment Title
Automated Diagnosis and Recovery of Feature-Specific Missingness

## Research Objective

Evaluate whether a rule-based Self-Healing Machine Learning (SHML) pipeline can automatically detect, diagnose, and correct feature-specific missingness in incoming validation data without manual intervention during pipeline execution.

The experiment also examines the consistency, reproducibility, and traceability of the integrated detection, diagnosis, and recovery components.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset | UCI Default of Credit Card Clients |
| Prediction task | Binary classification |
| Baseline model | Logistic Regression |
| Training samples | 21,000 |
| Validation samples | 4,500 |
| Test samples | 4,500 (reserved) |
| Monitored feature | BILL_AMT1 |
| Failure type | Numerical missingness |
| Healthy reference missingness | 0% |
| Missingness scenarios | 0%, 5%, 15%, 30% |
| Detection threshold | Greater than 5 percentage points |
| Diagnosis approach | Rule-based classification |
| Recovery strategy | Training-median imputation |
| Random seed | 42 |
| Evaluation approach | Functional verification and data-integrity checks |

## Experimental Procedure

1. Reproduced the healthy training and validation datasets using the original stratified split.
2. Recreated the logistic regression baseline and feature-specific missingness detector.
3. Injected controlled missingness into BILL_AMT1 at four predefined levels.
4. Used the detector to identify batches exceeding the monitoring threshold.
5. Implemented a rule-based diagnosis engine to classify missingness severity and select a recovery action.
6. Applied training-derived median imputation to affected values.
7. Integrated detection, diagnosis, recovery, and audit logging into one pipeline.
8. Developed verification checks for missingness correction, data integrity, and repeatability.

## Experimental Results

### Detection and Diagnosis

| Missingness | Alert Triggered | Diagnosis Severity | Selected Action |
|---|---|---|---|
| 0% | No | None | No correction |
| 5% | No | Below alert threshold | Routine median preprocessing |
| 15% | Yes | Moderate | Median imputation |
| 30% | Yes | Severe | Median imputation |

The detector followed the predefined monitoring rule. The 15% and 30% scenarios triggered alerts, while the 0% and 5% scenarios did not.

The diagnosis engine used the detected condition to select the appropriate preprocessing or recovery action.

### Data Recovery

| Missingness | Missing Before | Expected Missing After | Values Imputed |
|---|---:|---:|---:|
| 0% | 0 | 0 | 0 |
| 5% | 225 | 0 | 225 |
| 15% | 675 | 0 | 675 |
| 30% | 1,350 | 0 | 1,350 |

The recovery implementation was designed to replace missing BILL_AMT1 values using the training-derived median while preserving existing values and unrelated features.

**Verification status:** The table records expected recovery outcomes. Confirm all four PASS results in the final verification DataFrame before reporting these outcomes as fully verified.

## What Worked

### 1. Feature-Specific Failure Detection

The monitoring logic correctly distinguished between missingness rates that exceeded the threshold and those that did not.

This demonstrated that a simple, interpretable rule could identify the selected data-quality condition.

### 2. Rule-Based Failure Diagnosis

The diagnosis engine classified the detected condition by feature, failure type, and severity.

This allowed the system to select a recovery action without manually choosing a correction for each batch.

### 3. Training-Derived Recovery Strategy

The recovery engine used a median calculated from the healthy training dataset rather than from the corrupted validation batch.

This approach avoided using validation data to estimate the replacement statistic.

### 4. Modular System Architecture

Detection, diagnosis, and recovery were implemented as separate functions and connected through an integrated pipeline.

This design improved the organization of the system and created a foundation for supporting additional failure types.

### 5. Experiment Traceability

The pipeline was designed to generate audit records containing the detected condition, severity, selected action, and number of values corrected.

These records support reproducibility and future comparison of recovery strategies.

## What Did Not Work or Required Adjustment

### Issue 1 — Undefined Integrated Pipeline Function

**Observed error:** `NameError: name 'run_shml_recovery_pipeline' is not defined`

**Cause:** The integrated pipeline function was unavailable in the active Jupyter kernel when the verification cell executed.

**Correction:** The pipeline function was defined and its cell executed before running the verification tests.

**Lesson:** Notebook execution order must be managed carefully to ensure that all dependencies are available.

### Issue 2 — Missing Field in Pipeline Audit Log

**Observed error:** `KeyError: 'values_imputed'`

**Cause:** The returned pipeline log did not contain the `values_imputed` field required by the verification code.

**Correction:** The pipeline function was revised to calculate the number of imputed values from the difference between missing-value counts before and after recovery and include this field in the returned log.

**Lesson:** Consistent data structures and interfaces are essential when integrating independently developed system components.

### Issue 3 — Verification DataFrame Not Defined

**Observed error:** `NameError: name 'verification_df' is not defined`

**Cause:** An earlier verification cell terminated before creating the DataFrame, leaving the CSV-export code without its required input.

**Correction:** The verification, DataFrame creation, and CSV-export operations were reorganized to ensure that results are generated before being saved.

**Lesson:** Experiment logging should depend on successful verification rather than assuming that earlier notebook cells executed correctly.

### Overall Debugging Observation

The observed errors were associated with notebook execution order and component integration rather than evidence of incorrect missingness-detection logic.

These issues highlighted the importance of automated verification, consistent audit-log schemas, and reproducible execution.

## Known Limitations

1. The current diagnosis and recovery implementation supports only BILL_AMT1 missingness.
2. The detection and severity thresholds were predefined rather than statistically optimized.
3. Only one recovery method, median imputation, was implemented.
4. The experiment used one validation split and one random seed.
5. Correcting missing values does not guarantee restoration of predictive performance.
6. The current system does not automatically compare multiple recovery strategies.
7. Model rollback and production deployment have not yet been implemented.
8. The system has not yet been evaluated against concept drift, categorical shift, or multiple simultaneous failures.

## Lessons Learned

**Modularity:** Separating detection, diagnosis, and recovery simplifies debugging and makes the system easier to extend.

**Data leakage prevention:** Recovery statistics should be estimated using healthy training data rather than corrupted validation batches.

**Monitoring versus preprocessing:** A batch may require preprocessing even when its missingness does not exceed the monitoring alert threshold.

**Interface consistency:** The integrated pipeline must return standardized outputs so that verification and logging components can operate reliably.

**Reproducibility:** Notebook execution order, random seeds, and saved experimental results are essential for repeatable research.

**Recovery validation:** Successful data correction must be followed by predictive-performance evaluation before a recovered batch can be considered acceptable.

## Phase 4 Conclusion

Phase 4 established a rule-based architecture for integrating feature-specific failure detection, diagnosis, and median-imputation recovery.

The implementation demonstrates how the SHML system can select and apply a predefined corrective action based on an observed data-quality condition.

The integration process also revealed practical software-engineering challenges involving notebook execution order, audit-log consistency, and experiment verification.

The remaining requirement for formal Phase 4 completion is confirmation that the final verification checks pass for all four experimental scenarios and that the results are saved successfully.

## Next Research Activity — Phase 5

Phase 5 will evaluate whether the recovered validation batches preserve acceptable predictive performance.

The evaluation will compare recovered-batch metrics against the healthy logistic regression baseline:

| Metric | Healthy Baseline |
|---|---:|
| Accuracy | 0.8087 |
| Precision | 0.6948 |
| Recall | 0.2402 |
| F1-score | 0.3570 |
| ROC-AUC | 0.7130 |

An initial acceptance criterion will use a maximum 10% relative reduction in F1-score, alongside examination of recall and other safety considerations.

The next objective is to distinguish successful data correction from successful model-performance recovery and implement explicit acceptance or rejection decisions.