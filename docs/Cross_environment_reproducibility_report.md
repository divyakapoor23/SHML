# SHML — Cross-Environment Reproducibility Report

## 1. Objective

Evaluate whether the Self-Healing Machine Learning (SHML) pipeline produces consistent experimental results when executed in two different computing environments: a local macOS computer and a Google Cloud Compute Engine virtual machine running Ubuntu Linux.

## 2. Experimental Configuration

- **Dataset:** UCI Default of Credit Card Clients
- **Observations:** 30,000
- **Predictors:** 23
- **Model:** Logistic Regression
- **Training observations:** 21,000
- **Validation observations:** 4,500
- **Test observations:** 4,500
- **Failure variable:** `BILL_AMT1`
- **Injected missingness:** 0%, 5%, 15%, and 30%
- **Recovery method:** Training-median imputation
- **Safety tolerance:** 10% relative performance degradation

The same standalone SHML workflow was executed in both environments.

## 3. Computing Environments

| Component | Local | Google Cloud |
|---|---|---|
| Operating system | macOS | Ubuntu Linux |
| Python | 3.12.14 | 3.12.16 |
| Execution | Conda environment | Conda environment |
| NumPy | 2.3.4 | 2.3.4 specified |
| Pandas | 2.2.3 | 2.2.3 specified |
| Scikit-learn | 1.8.0 | 1.8.0 specified |
| SciPy | 1.16.3 | 1.16.3 specified |
| xlrd | 2.0.2 | 2.0.2 specified |

The recorded local Git commit was `af2471c1a1f26161897d6e6dfe12b806d218ccf0`. The cloud commit was not independently recorded.

## 4. Experimental Results

| Missingness | F1 | Recall | ROC-AUC | Decision |
|---|---:|---:|---:|---|
| 0% | 0.356983 | 0.240201 | 0.713025 | ACCEPT |
| 5% | 0.358898 | 0.242211 | 0.713210 | ACCEPT |
| 15% | 0.360947 | 0.245226 | 0.708958 | ACCEPT |
| 30% | 0.373723 | 0.257286 | 0.711135 | ACCEPT |

The results were identical in the exported local and cloud CSV files.

## 5. Reproducibility Verification

Both environments successfully completed the standalone SHML pipeline, including dataset verification, baseline model evaluation, controlled missingness injection, recovery evaluation, safety decisions, and CSV export.

The resulting CSV files were compared using:

`pd.testing.assert_frame_equal(local, cloud, check_exact=True)`

The comparison passed without errors, confirming exact equality of the parsed CSV DataFrames.

## 6. Interpretation and Limitations

The experiment demonstrates reproducible execution of the tested SHML workflow across macOS and Ubuntu Linux.

The results support automated detection and correction of missing values, together with evaluation and safety-based acceptance of recovery candidates.

However, the experiments do not establish that predictive performance first degraded and was subsequently restored. All four corrected scenarios satisfied the safety acceptance criteria.

The findings are limited to one dataset, one baseline model, one affected feature, one failure type, and four controlled missingness levels.

## 7. Conclusion

The SHML standalone pipeline successfully reproduced identical recorded experimental results in local and