# Phase 5 — Recovery Validation and Safety

## Experiment Title

Predictive Performance Validation and Safety Evaluation Following Automated Missingness Recovery

## Research Objective

Evaluate whether the SHML system preserves acceptable predictive performance after correcting feature-specific missingness and whether predefined safety criteria can prevent unacceptable recovery outcomes from being approved.

## Research Questions

1. Does median imputation preserve predictive performance after BILL_AMT1 missingness is introduced?
2. How do F1-score, recall, precision, and ROC-AUC change as missingness increases?
3. Can a rule-based safety engine distinguish acceptable and unacceptable recovery outcomes?
4. Can a fallback controller prevent rejected recovery outcomes from being approved?

## Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset | UCI Default of Credit Card Clients |
| Baseline model | Logistic Regression |
| Validation samples | 4,500 |
| Monitored feature | BILL_AMT1 |
| Failure type | Numerical missingness |
| Missingness scenarios | 0%, 5%, 15%, 30% |
| Recovery method | Training-median imputation |
| Training median | 22,225.5 |
| Random seed | 42 |
| Maximum relative F1 loss | 10% |
| Provisional maximum relative recall loss | 10% |
| Evaluation environment | Offline validation |

## Healthy Baseline

| Metric | Value |
|---|---:|
| Accuracy | 0.808667 |
| Precision | 0.694767 |
| Recall | 0.240201 |
| F1-score | 0.356983 |
| ROC-AUC | 0.713025 |

## Recovery Performance Results

| Missingness | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0% | 0.808667 | 0.694767 | 0.240201 | 0.356983 | 0.713025 |
| 5% | 0.808667 | 0.692529 | 0.242211 | 0.358898 | 0.713210 |
| 15% | 0.808000 | 0.683473 | 0.245226 | 0.360947 | 0.708958 |
| 30% | 0.809333 | 0.682667 | 0.257286 | 0.373723 | 0.711135 |

## Performance Degradation Findings

The recovered batches maintained or slightly exceeded the healthy baseline F1-score across all tested missingness levels.

At 30% missingness, the recovered F1-score increased from approximately 0.3570 to 0.3737.

However, precision decreased while recall increased, indicating a change in the model's prediction behavior.

ROC-AUC remained relatively stable, with a small reduction in the 15% scenario.

The observed improvements should not be interpreted as evidence that increasing missingness improves model quality.

## What Worked

### Recovery Performance Evaluation

The recovered validation batches were evaluated using the original training-fitted scaler and healthy logistic regression model.

This allowed direct comparison without retraining the model.

### Performance Monitoring

The experiment measured accuracy, precision, recall, F1-score, and ROC-AUC for all four missingness scenarios.

Relative F1 loss was calculated against the healthy baseline.

### Safety Decision Engine

A rule-based safety controller was implemented to evaluate data integrity, F1 degradation, and recall degradation.

The design separates successful data correction from acceptable predictive performance.

### Rejection Testing

A synthetic unsafe performance scenario was constructed to test the rejection pathway.

This test was designed to verify that the safety engine rejects outcomes that exceed the defined performance-loss tolerances.

### Fallback Decision Logic

A fallback controller was implemented to determine the intended action following acceptance or rejection.

The controller supports offline approval, blocking, escalation, and selection of a hypothetical independently validated fallback.

## What Did Not Work or Requires Improvement

### Limited Evidence of Performance Degradation

The tested BILL_AMT1 missingness scenarios did not produce substantial F1-score degradation after median imputation.

As a result, the experiment provides evidence of performance preservation under the tested conditions but does not establish recovery from a severely degraded model.

### Fixed Recovery Strategy

Only median imputation was evaluated.

Alternative strategies, such as KNN imputation, iterative imputation, or model retraining, have not yet been compared.

### Limited Failure Coverage

The experiment focuses on one numerical feature and does not evaluate numerical drift, categorical shift, schema failures, or simultaneous feature corruption.

### Production Rollback Not Implemented

The fallback controller records intended actions but does not restore a deployed model or execute production traffic routing.

## Known Limitations

1. One dataset and one baseline model were evaluated.
2. One numerical feature was corrupted.
3. One validation split and one random seed were used.
4. The F1 and recall tolerances were selected experimentally rather than calibrated against operational risk.
5. No confidence intervals or repeated-trial variability estimates were calculated.
6. Validation labels were available for offline performance evaluation; they would not necessarily be available during real-time inference.
7. No production deployment or actual rollback was performed.
8. The safety controller has not been evaluated against every possible invalid-input condition.

## Lessons Learned

**Data-quality recovery and performance recovery are not equivalent.**

Successfully imputing missing values does not automatically prove that the model has recovered from predictive degradation.

**Multiple metrics are necessary.**

F1-score increased in the tested scenarios, but precision decreased, demonstrating why safety evaluation should not rely on one metric alone.

**Explicit safety rules improve traceability.**

Acceptance and rejection criteria allow the system to explain why a recovery outcome is considered acceptable or unacceptable.

**Synthetic rejection tests are useful.**

When real experiments do not produce unsafe outcomes, controlled unit tests can verify whether the safety controller handles rejection correctly.

**Production rollback requires additional infrastructure.**

A decision to reject or restore a model is different from executing a rollback in a deployed environment.

## Phase 5 Conclusion

Phase 5 evaluated predictive performance after median-imputation recovery for four BILL_AMT1 missingness scenarios.

The observed results indicate that the recovered batches maintained F1-scores at or above the healthy baseline.

The experiment also established an offline framework for evaluating recovery safety through explicit data-integrity and performance criteria.

These results support the feasibility of combining automated correction with quantitative recovery validation.

However, broader claims about self-healing reliability require additional failure types, repeated experiments, alternative recovery strategies, and deployment-level testing.

## Next Research Activity

Phase 6 will expand the experimental evaluation to determine whether the SHML system can detect and recover from more challenging failures.

Potential experiments include numerical drift, categorical shift, alternative imputation strategies, and repeated corruption trials.

The goal is to establish stronger evidence of reliability beyond the initial BILL_AMT1 missingness experiment.