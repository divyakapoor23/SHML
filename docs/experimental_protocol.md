
# SHML Final Experimental Protocol

## Research Scope

This experiment evaluates a Self-Healing Machine Learning
system for detecting and responding to missing values in
BILL_AMT1 using the UCI Default of Credit Card Clients dataset.

## Dataset

- Total observations: 30,000
- Predictors: 23
- Target: Default payment next month

## Data Partitioning

- Training: 21,000 observations (70%)
- Validation: 4,500 observations (15%)
- Testing: 4,500 observations (15%)
- Stratified splitting with random_state=42

## Baseline Model

- Algorithm: Logistic Regression
- Scaling: StandardScaler fitted on training data only
- max_iter: 1000
- random_state: 42

## Failure Injection

- Feature: BILL_AMT1
- Failure type: Randomly injected missing values
- Missingness levels: 0%, 5%, 15%, and 30%
- Detection threshold: greater than 5%
- Random seeds: 0–19 for repeated-seed validation

## Recovery Strategies

1. Zero imputation (preferred candidate)
2. Training-median imputation (fallback)
3. Abstention if both candidates fail safety checks

## Safety Constraints

- Maximum relative F1 loss: 10%
- Maximum relative recall loss: 10%
- Experimental maximum relative ROC-AUC loss: 0.5%
- Data integrity checks required

The ROC-AUC guardrail is exploratory and was examined
through validation-set sensitivity analysis. It has not
been independently established as an optimal threshold.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 score (primary)
- ROC-AUC (additional safety metric)

## Experimental Results

At 15% BILL_AMT1 missingness across 20 seeds:

- Preferred recovery accepted: 16
- Median fallback accepted: 2
- Abstention: 2
- Integrated policy matched notebook results: 20/20

## Methodological Limitations

- Recovery selection uses validation labels in an offline
  research setting.
- Missingness did not produce consistent F1 degradation.
- Missingness masks are repeated on the same validation set.
- The 0.5% ROC-AUC guardrail was explored on validation data.
- Results do not establish deployment-time autonomous healing.

## Test-Set Protection

The test partition is reserved for one final evaluation
after the experimental protocol and selection rules are
frozen.

No test-set results will be used to modify recovery
strategies, metric tolerances, or model hyperparameters.
