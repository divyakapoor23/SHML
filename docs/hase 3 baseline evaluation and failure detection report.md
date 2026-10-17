# Phase 3 — Baseline Model Evaluation and Failure Detection

**Project:** Self-Healing Machine Learning (SHML)  
**Trial:** 1  
**Dataset:** UCI Default of Credit Card Clients  
**Status:** Core experiments completed

## 1. Objective

Establish a reproducible healthy machine-learning baseline and implement a feature-specific data-quality failure detector.

Phase 3 addresses two research questions:

1. How well does the baseline model predict credit default under healthy data conditions?
2. Can the system detect controlled missingness in an incoming validation batch?

## 2. Experimental Configuration

| Component | Configuration |
|---|---|
| Prediction task | Binary classification |
| Baseline model | Logistic Regression |
| Preprocessing | StandardScaler |
| Training samples | 21,000 |
| Validation samples | 4,500 |
| Test samples | 4,500 (reserved) |
| Random seed | 42 |
| Monitored feature | BILL_AMT1 |
| Healthy missingness | 0% |
| Detection threshold | Greater than 5 percentage points |
| Injected missingness | 0%, 5%, 15%, 30% |

## 3. Healthy Baseline Evaluation

The logistic regression model was trained on the training dataset and evaluated on the held-out validation dataset.

| Metric | Result |
|---|---:|
| Accuracy | 0.8087 |
| Precision | 0.6948 |
| Recall | 0.2402 |
| F1-score | 0.3570 |
| ROC-AUC | 0.7130 |

### Confusion Matrix

| | Predicted No Default | Predicted Default |
|---|---:|---:|
| Actual No Default | 3,400 | 105 |
| Actual Default | 756 | 239 |

The model correctly identified 239 default cases but missed 756 actual defaults.

Although accuracy exceeded 80%, positive-class recall was approximately 24%.

This indicates that accuracy alone is insufficient to assess the reliability of the credit-default classifier.

## 4. Majority-Class Benchmark

A majority-class classifier was used as a simple reference model. It always predicts no default.

| Metric | Logistic Regression | Majority Class |
|---|---:|---:|
| Accuracy | 0.8087 | 0.7789 |
| Precision | 0.6948 | 0.0000 |
| Recall | 0.2402 | 0.0000 |
| F1-score | 0.3570 | 0.0000 |

Logistic regression outperformed the majority-class classifier across the evaluated metrics.

However, its relatively low recall demonstrates that identifying default cases remains challenging.

F1-score and positive-class recall were therefore selected as important measures for subsequent recovery evaluation.

## 5. Controlled Failure Injection

Missing values were injected exclusively into BILL_AMT1 within the validation dataset.

| Scenario | Injected Missingness | Observed Missingness |
|---|---:|---:|
| Healthy | 0% | 0% |
| Mild | 5% | 5% |
| Moderate | 15% | 15% |
| Severe | 30% | 30% |

The observed missingness matched the intended injection rate in all four scenarios.

The original validation data was copied before corruption to avoid modifying the healthy reference batch.

## 6. Failure Detection

A rule-based detector was implemented to compare incoming BILL_AMT1 missingness against the healthy training reference.

**Detection rule:**

Failure detected if:

Current missingness − Reference missingness > 0.05

### Detection Results

| Missingness | Expected Alert | Actual Alert | Result |
|---|---|---|---|
| 0% | False | False | Pass |
| 5% | False | False | Pass |
| 15% | True | True | Pass |
| 30% | True | True | Pass |

All four scenarios produced the expected decisions.

The 5% scenario did not trigger an alert because the detection rule uses a strict greater-than comparison.

## 7. What Worked

- The original stratified data split was reproduced.
- Logistic regression produced the expected healthy baseline metrics.
- The majority-class benchmark provided a useful comparison.
- Controlled missingness was injected successfully into BILL_AMT1.
- The detector correctly calculated observed missingness.
- All four detection decisions matched the predefined rules.
- The experimental design preserved the test dataset for later evaluation.

## 8. What Did Not Work or Required Improvement

No specific Phase 3 execution failures have been documented in the available experiment record.

However, the baseline model demonstrated limited sensitivity to actual default cases, identifying only 239 of 995 positive validation examples.

This is a model-performance limitation rather than a software execution failure.

The detection threshold also requires further evaluation because its ability to identify meaningful predictive-performance degradation has not yet been established.

## 9. Known Limitations

1. Only BILL_AMT1 missingness was tested in the formal Phase 3 detection experiment.
2. The 5-percentage-point threshold was predefined rather than statistically calibrated.
3. Four missingness scenarios and one random seed were evaluated.
4. Successful detection of missingness does not establish model-performance degradation.
5. Other failure types, including numerical drift, categorical shift, and schema violations, remain outside this experiment.
6. No automated recovery was implemented in Phase 3.

## 10. Lessons Learned

**Accuracy can be misleading:** A classifier can achieve relatively high accuracy while missing many positive cases.

**Healthy reference profiles matter:** Detection requires a documented baseline against which incoming data can be compared.

**Controlled experiments support reproducibility:** Fixed random seeds and feature-specific corruption make it easier to interpret detection behavior.

**Detection and recovery are different tasks:** Identifying a data-quality issue does not automatically correct it or restore predictive performance.

## 11. Phase 3 Conclusion

Phase 3 established a reproducible logistic regression baseline and demonstrated deterministic detection of feature-specific missingness.

The experiments provide the monitoring foundation for the SHML architecture.

The detector correctly identified batches exceeding the predefined missingness threshold, while the baseline evaluation established reference metrics for later recovery experiments.

These findings support moving to Phase 4, where detected failures are diagnosed and corrected automatically.

## 12. Associated Files

- `phase3_baseline_model.ipynb`
- `phase3_failure_detection.ipynb`
- `docs/phase3_baseline_failure_detection.md`

## 13. Next Phase

**Phase 4 — Automated Failure Diagnosis and Recovery**

Develop a rule-based diagnosis engine, apply training-derived median imputation, integrate the components, and record recovery actions for subsequent performance validation.