# Self-Healing Machine Learning (SHML) System

## Project Overview

This project investigates whether a machine-learning system can automatically
detect data or model failures, apply an appropriate correction, validate the
recovery, and safely accept or reject the corrected state.

The initial implementation focuses on binary classification and controlled
missing-data failures.

## Trial 1

### Dataset
UCI Default of Credit Card Clients

### Prediction Task
Predict whether a credit-card client will default on payment in the following
month.

- Class 0: No default
- Class 1: Default

### Baseline Model
Logistic Regression

### Healthy Baseline

| Metric | Score |
|---|---:|
| F1 | 0.3570 |
| Recall | 0.2402 |
| ROC-AUC | 0.7130 |
| Accuracy | 0.8087 |

## Failure Experiment

Missing values were artificially injected into validation data at three levels:

- 5%
- 15%
- 30%

The original raw dataset remains unchanged.

### Detection

The system monitors:

- Missing-value rate
- Model inference failure
- Predictive performance after recovery

### Healing Strategy

Median imputation is used as the initial approved correction for missing
numeric values.

### Safety Rule

Recovered performance must remain within 10% of the healthy F1 baseline.

Minimum acceptable F1:

0.3213

## Preliminary Results

| Missingness | Healed F1 | Relative F1 Drop | Decision |
|---:|---:|---:|---|
| 5% | 0.3366 | 5.70% | ACCEPT |
| 15% | 0.3008 | 15.74% | REJECT |
| 30% | 0.2410 | 32.49% | REJECT |

Median imputation restored technical inference capability at all three
missingness levels. However, predictive recovery met the predefined safety
requirement only at 5% missingness.

This demonstrates that correcting a data-quality failure does not necessarily
restore acceptable model performance. Post-correction validation is therefore
required before a recovered state can be accepted.

## Current SHML Pipeline

Dataset
→ Data Profiling
→ Baseline Model
→ Failure Injection
→ Failure Detection
→ Missing-Value Diagnosis
→ Median Imputation
→ Performance Validation
→ Accept / Reject
→ Escalation

## Next Steps

1. Create the Trial 1 dataset contract.
2. Define the healthy reference data profile.
3. Formalize data-quality rules.
4. Complete candidate-model retraining and rollback logic.
5. Add the TensorFlow/Keras comparison model.
6. Expand to additional failure types.