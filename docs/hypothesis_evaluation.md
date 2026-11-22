
# SHML — Research Hypothesis Evaluation

## Experimental Scope

This evaluation examines controlled missingness in
BILL_AMT1 using the UCI Default of Credit Card Clients
dataset and a Logistic Regression classifier.

The project uses 21,000 training, 4,500 validation,
and 4,500 test observations.

## H1 — Model Degradation

Status: Partially supported

The experiments demonstrate metric-dependent degradation.

On the held-out test set, ROC-AUC decreased from
0.716954 (healthy) to 0.712371 (30% missingness,
zero imputation).

However, F1 increased from 0.348416 to 0.378733.

Therefore, degradation was observed in ROC-AUC
but not in F1 under this experimental configuration.

## H2 — Diagnosis-Guided Recovery

Status: Not established

The system implemented failure diagnosis and compared
zero and training-median imputation.

Zero imputation achieved higher F1 at the evaluated
nonzero missingness levels on the held-out test set.

However, the experiments did not establish recovery
of performance lost through F1 degradation.

A controlled comparison against untreated failure
and generic retraining was not completed.

## H3 — Validation and Rollback

Status: Partially supported

The system implements performance-based safety checks
using F1, recall, and an exploratory ROC-AUC guardrail.

At 15% missingness across 20 validation seeds:

- Zero imputation accepted: 16
- Median fallback accepted: 2
- Both rejected / abstention: 2

The integrated implementation reproduced all 20
notebook decisions.

On the final test set, both recovery strategies
violated the frozen ROC-AUC tolerance at 30%
missingness in a retrospective safety audit.

These experiments demonstrate offline rejection and
fallback behavior, not production model rollback.

## H4 — Failure Severity

Status: Partially supported

Increasing BILL_AMT1 missingness produced declining
ROC-AUC in the held-out test experiment.

However, F1 increased with missingness under both
evaluated imputation strategies.

The results indicate that the relationship between
failure severity and model performance depends on
the evaluation metric.

## H5 — Reproducibility

Status: Supported within the tested configuration

The experiment used controlled dataset partitions,
model settings, and random seeds.

The 20-seed integrated recovery experiment reproduced
the research notebook decisions exactly:

- 20 of 20 decisions matched.

Automated regression tests also passed.

This supports computational reproducibility on
the tested local setup.

Independent cross-environment reproducibility
has not yet been established.

## H6 — Subgroup Performance and Reliability

Status: Not tested

The completed experiments focused on aggregate
predictive performance.

Subgroup-level evaluation remains necessary
to establish whether recovery strategies have
unequal effects across different populations.

## H7 — Exploratory LLM/RAG

Status: Not tested

LLM/RAG-based diagnosis and recovery were not
implemented as part of the core BILL_AMT1 experiment.

This hypothesis remains a possible future extension.

## Overall Research Conclusion

SHML demonstrates a reproducible offline workflow
for controlled missingness detection, diagnosis,
recovery comparison, and safety evaluation.

The strongest findings concern:

1. Metric-dependent effects of missingness.
2. Tradeoffs between F1 and ROC-AUC.
3. Safety-based acceptance, fallback, and abstention.
4. Reproducibility of integrated recovery decisions.

The current experiments do not establish restoration
from F1 degradation, fully autonomous recovery,
or subgroup reliability.
