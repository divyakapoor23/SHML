# Self-Healing Machine Learning (SHML)

**Automated Failure Detection, Diagnosis, Recovery, and Safety Validation for Machine Learning Systems**

## 1. Project Overview

Machine-learning models can become unreliable after deployment due to missing data, feature drift, distribution shifts, schema changes, and other unexpected conditions. Traditional recovery often requires manual investigation, preprocessing changes, model retraining, and redeployment.

This project investigates whether a Self-Healing Machine Learning (SHML) system can automatically detect data-quality failures, diagnose their likely causes, apply corrective actions, evaluate recovery effectiveness, and determine whether the corrected state satisfies predefined safety requirements.

The initial implementation focuses on structured binary classification and controlled missing-data failures.

The longer-term objective is to develop an extensible framework that supports multiple failure types, alternative recovery strategies, automated model management, and cloud deployment.

## 2. Research Objectives

The project investigates the following questions:

1. How reliably can a monitoring system detect predefined data-quality failures?
2. Can a rule-based diagnosis engine select an appropriate recovery action?
3. Does correcting missing data preserve acceptable predictive performance?
4. Can quantitative safety criteria distinguish acceptable and unacceptable recovery outcomes?
5. How can the framework be extended to support additional failures, recovery methods, and deployment environments?

## 3. Trial 1 — Credit Default Prediction

### Dataset

**UCI Default of Credit Card Clients**

The dataset contains 30,000 observations and 23 predictor variables describing customer demographics, credit limits, payment histories, billing amounts, and previous payments.

**Prediction target:** `default payment next month`

- Class 0: No default
- Class 1: Default

### Data Splitting

| Dataset | Observations | Percentage |
|---|---:|---:|
| Training | 21,000 | 70% |
| Validation | 4,500 | 15% |
| Testing | 4,500 | 15% |

The data was split using stratified sampling with random seed 42.

The test dataset is reserved for final evaluation.

### Baseline Model

Logistic Regression implemented using scikit-learn.

Preprocessing uses StandardScaler fitted exclusively on the training dataset.

### Healthy Baseline Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8087 |
| Precision | 0.6948 |
| Recall | 0.2402 |
| F1-score | 0.3570 |
| ROC-AUC | 0.7130 |

The baseline model achieves approximately 80.87% accuracy but only 24.02% recall for the default class.

This demonstrates why accuracy alone is insufficient for evaluating model reliability on an imbalanced classification problem.

F1-score and recall are therefore important metrics for recovery validation.

## 4. SHML System Architecture

The current experimental workflow consists of the following components:

1. **Data Profiling:** Establish the healthy reference characteristics of the dataset.
2. **Baseline Modeling:** Train and evaluate the healthy classification model.
3. **Failure Injection:** Introduce controlled data-quality failures.
4. **Failure Detection:** Identify conditions that violate predefined monitoring rules.
5. **Failure Diagnosis:** Classify the failure and select an appropriate correction.
6. **Recovery Engine:** Apply the selected data correction.
7. **Performance Validation:** Evaluate predictive performance after recovery.
8. **Safety Decision Engine:** Accept or reject the corrected batch according to predefined criteria.
9. **Fallback Controller:** Determine whether to approve, block, or escalate a recovery outcome.
10. **Experiment Logging:** Record metrics, decisions, and experimental results.

### Pipeline

```text
Dataset
   |
   v
Data Profiling
   |
   v
Healthy Baseline Model
   |
   v
Controlled Failure Injection
   |
   v
Failure Detection
   |
   v
Failure Diagnosis
   |
   v
Recovery Engine
   |
   v
Performance Evaluation
   |
   v
Safety Validation
   |
   +------ ACCEPT ------> Approve for Offline Evaluation
   |
   +------ REJECT ------> Block / Escalate
```

The current implementation evaluates recovery and fallback decisions offline. It does not yet perform production deployment, automated traffic routing, or actual model rollback.

## 5. Project Development Progress

### Phase 1 — Problem Definition and Dataset Preparation

**Status: Completed**

- Defined the binary classification research problem.
- Selected the UCI credit-default dataset.
- Established the prediction target and feature set.
- Created stratified training, validation, and test splits.
- Documented dataset characteristics and data-quality observations.
- Created the Trial 1 dataset contract.

### Phase 2 — Data Profiling and Monitoring Design

**Status: Completed**

- Examined missingness and feature distributions.
- Investigated numerical and categorical characteristics.
- Identified rare categorical values and unusual numerical observations.
- Established healthy reference characteristics.
- Selected monitoring candidates for numerical missingness, numerical drift, and categorical shift.

### Phase 3 — Baseline Modeling and Failure Detection

**Status: Completed**

- Trained the healthy logistic regression baseline.
- Evaluated accuracy, precision, recall, F1-score, and ROC-AUC.
- Compared performance with a majority-class classifier.
- Implemented feature-specific missingness injection.
- Developed a rule-based missingness detector.
- Tested missingness scenarios of 0%, 5%, 15%, and 30%.

The detector triggers when BILL_AMT1 missingness increases by more than five percentage points relative to the healthy reference.

### Phase 4 — Automated Diagnosis and Recovery

**Status: Implemented**

- Developed a rule-based diagnosis component.
- Classified missingness conditions by severity.
- Selected recovery actions based on the diagnosed condition.
- Applied median imputation using training-derived statistics.
- Integrated detection, diagnosis, and recovery logic.
- Added experiment logs and verification checks.

The training median for BILL_AMT1 was 22,225.5.

### Phase 5 — Recovery Validation and Safety

**Status: Implemented; experimental results recorded**

- Evaluated model performance after recovery.
- Calculated relative F1 and recall degradation.
- Implemented explicit acceptance and rejection criteria.
- Developed a fallback decision controller.
- Tested the rejection pathway using a simulated unsafe performance scenario.
- Exported experiment results and decision logs.

The safety and fallback components currently support offline decisions rather than production execution.

## 6. Formal Trial 1 Missingness Experiment

### Experimental Configuration

| Parameter | Configuration |
|---|---|
| Monitored feature | BILL_AMT1 |
| Failure type | Numerical missingness |
| Missingness levels | 0%, 5%, 15%, 30% |
| Detection threshold | Greater than 5 percentage points |
| Recovery strategy | Training-median imputation |
| Baseline model | Logistic Regression |
| Random seed | 42 |

Missing values were introduced into copies of the validation dataset. The original raw dataset remained unchanged.

### Failure Detection

| Missingness | Detection Alert |
|---|---|
| 0% | No |
| 5% | No |
| 15% | Yes |
| 30% | Yes |

The 5% scenario does not trigger an alert because the detector uses a strict greater-than comparison.

### Recovery Performance Results

| Missingness | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0% | 0.8087 | 0.6948 | 0.2402 | 0.3570 | 0.7130 |
| 5% | 0.8087 | 0.6925 | 0.2422 | 0.3589 | 0.7132 |
| 15% | 0.8080 | 0.6835 | 0.2452 | 0.3609 | 0.7090 |
| 30% | 0.8093 | 0.6827 | 0.2573 | 0.3737 | 0.7111 |

The recovered batches maintained F1-scores at or above the healthy baseline.

However, precision decreased as missingness increased, while recall increased.

These results should not be interpreted as evidence that missingness improves model performance generally.

## 7. Recovery Safety Criteria

The initial safety framework uses the following requirements:

- No unsupported missing values remain.
- Required features and schema are preserved.
- Relative F1 loss does not exceed 10%.
- Relative recall loss does not exceed 10%.
- Evaluation metrics are valid.

ROC-AUC is monitored but is not currently an enforced acceptance threshold.

### Minimum Acceptable F1

```text
Healthy F1 = 0.356983
Maximum relative F1 loss = 10%

Minimum acceptable F1 = 0.321285
```

### Phase 5 Safety Results

| Missingness | Relative F1 Loss | Safety Decision |
|---|---:|---|
| 0% | 0.00% | ACCEPT |
| 5% | -0.54% | ACCEPT |
| 15% | -1.11% | ACCEPT |
| 30% | -4.69% | ACCEPT |

Negative relative loss indicates that the recovered F1-score exceeded the healthy baseline.

All four scenarios satisfied the implemented offline safety criteria.

The fallback controller recorded `APPROVE_FOR_OFFLINE_EVALUATION` for all four scenarios.

These decisions do not constitute approval for production deployment.

## 8. Preliminary Missingness Experiment

Before the formal feature-specific experiments, an exploratory missingness experiment was conducted across multiple numerical and categorical predictors.

### Preliminary Results

| Missingness | F1 After Imputation | Relative F1 Loss | Decision |
|---|---:|---:|---|
| 5% | 0.3366 | 5.70% | ACCEPT |
| 15% | 0.3008 | 15.74% | REJECT |
| 30% | 0.2410 | 32.49% | REJECT |

These exploratory results are not directly comparable to the formal BILL_AMT1-only experiment because the corruption scope differs.

They demonstrate that the impact of missingness depends on which features are affected and how much information is lost.

This motivates broader failure experiments and recovery-strategy comparisons.

## 9. Key Research Findings

### Finding 1 — Data-Quality Failure Does Not Always Cause Predictive Degradation

The formal experiment detected elevated BILL_AMT1 missingness at 15% and 30%, yet recovered-batch F1-scores remained at or above the healthy baseline.

This suggests that the impact of a data-quality failure depends on the affected feature, model, and corruption pattern.

### Finding 2 — Successful Data Correction Does Not Guarantee Performance Recovery

The preliminary experiment demonstrated that median imputation can restore complete inputs while predictive performance remains below the acceptance threshold.

Therefore, recovery should be evaluated using model-performance criteria rather than missing-value counts alone.

### Finding 3 — Multiple Performance Metrics Are Necessary

In the formal experiment, F1 and recall increased slightly while precision decreased.

This highlights the importance of monitoring several metrics rather than relying exclusively on accuracy or F1-score.

### Finding 4 — Safety Decisions Should Be Explicit

The acceptance engine provides auditable decisions based on predefined data-integrity and performance requirements.

A simulated unsafe recovery outcome was used to exercise the rejection pathway.

## 10. Known Limitations

- Formal recovery experiments currently focus on one numerical feature.
- Only logistic regression and median imputation have been evaluated in the formal recovery workflow.
- The detection and safety thresholds are predefined rather than statistically optimized.
- Results are based on one validation split and one random seed.
- The current experiment has not established recovery effectiveness across all failure types.
- Offline predictive-performance evaluation requires validation labels, which may not be available immediately in production.
- The fallback controller does not execute actual production rollback.
- Cloud deployment and end-to-end production integration remain future work.

## 11. Repository Structure

```text
SHML/
├── Phase1.ipynb
├── Phase2.ipynb
├── phase3_baseline_model.ipynb
├── phase3_failure_detection.ipynb
├── phase4_diagnosis_recovery.ipynb
├── phase5_recovery_validation.ipynb
├── data/
├── docs/
│   ├── trial1_dataset_contract.md
│   ├── phase5_recovery_validation.md
│   └── ...
├── results/
│   ├── phase4/
│   └── phase5/
│       ├── phase5_recovery_performance.csv
│       ├── phase5_degradation_analysis.csv
│       ├── phase5_safety_decisions.csv
│       ├── phase5_fallback_decisions.csv
│       └── phase5_final_experiment_summary.csv
├── requirements.txt
└── Readme.md
```

The repository contains additional experiment documentation and results in their corresponding directories.

## 12. Technology Stack

**Programming and analysis:** Python, NumPy, Pandas

**Machine learning:** scikit-learn

**Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC

**Visualization:** Matplotlib

**Development environment:** Jupyter Notebook

**Version control:** Git and GitHub

**Planned cloud platform:** Microsoft Azure

## 13. Next Steps

### Phase 6 — Expanded Failure Experiments

- Evaluate numerical drift in LIMIT_BAL.
- Evaluate categorical distribution shift in PAY_0.
- Compare alternative recovery strategies.
- Repeat experiments using multiple random seeds.
- Analyze detection effectiveness and recovery reliability.

### Phase 7 — System Integration

- Improve component interfaces and experiment reproducibility.
- Consolidate the detection, diagnosis, recovery, and safety workflow.
- Add integration tests and structured logging.

### Phase 8 — Research Evaluation and Reporting

- Consolidate experimental findings.
- Prepare research figures, tables, and conclusions.
- Evaluate system limitations and threats to validity.
- Prepare the capstone presentation and final report.

### Cloud Deployment

Deploy and evaluate the integrated SHML system in a Microsoft Azure environment as part of the capstone's planned deployment work.

### Future Extensions

- Compare logistic regression with a TensorFlow/Keras neural network.
- Investigate automated retraining and model version management.
- Explore LLM-assisted diagnosis and retrieval-augmented explanations.
- Evaluate additional datasets, including multimodal datasets.

## 14. Project Status

**Current milestone:** Phase 5 implemented and experimental results uploaded.

**Next milestone:** Phase 6 — Expanded Failure Experiments and Recovery Strategy Comparison.

The project has established an initial framework for automated missingness detection, rule-based diagnosis, median-imputation recovery, offline performance validation, and explicit safety decisions.

The next stage will test whether this framework remains reliable under more challenging and varied failure conditions.