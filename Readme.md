### Current Research Focus

The current research focuses on rigorously investigating the effects of controlled missingness in the `BILL_AMT1` feature of the UCI Default of Credit Card Clients dataset.

Although the initial SHML framework implements automated failure detection, diagnosis, recovery, performance evaluation, and safety validation, further experimentation is required to determine whether the system can effectively restore predictive performance following data corruption.

The study investigates missingness severity, feature importance and redundancy, recovery strategies, statistical reliability, reproducibility, and subgroup performance.

The immediate objective is to establish scientifically defensible evidence of when data corruption affects predictions, whether recovery improves performance, and whether the safety mechanism makes appropriate decisions.

Additional failure types, model architectures, and cloud deployment remain future extensions.



## 2. Research Questions and Hypotheses

### Primary Research Question

To what extent can a self-healing machine learning system detect, diagnose, and recover from controlled missingness in `BILL_AMT1` while maintaining predictive performance, enforcing safety constraints, and producing reproducible results?

### Supporting Research Questions

**RQ1 — Failure Detection and Degradation**

How does increasing missingness in `BILL_AMT1` affect predictive performance, and can the system reliably detect missingness associated with meaningful degradation?

**H1:** Increasing missingness in `BILL_AMT1` produces measurable predictive degradation at one or more severity levels, and the detection mechanism identifies data-quality failures exceeding its monitoring threshold.

**RQ2 — Recovery Effectiveness**

To what extent do different imputation strategies restore predictive performance compared with untreated or benchmark approaches?

**H2:** At least one recovery strategy improves predictive performance relative to an appropriate control under one or more missingness conditions.

**RQ3 — Safety Validation**

Can the safety mechanism distinguish acceptable from unacceptable recovery candidates and preserve the approved state after rejection?

**H3:** The safety mechanism correctly accepts candidates satisfying predefined criteria and rejects candidates violating those criteria.

**RQ4 — Missingness Severity and Feature Resilience**

How do missingness severity and relationships between `BILL_AMT1` and other predictors influence model degradation and recovery?

**H4:** Missingness severity affects predictive performance and recovery effectiveness, while information retained in related predictors may reduce degradation.

**RQ5 — Reproducibility and Statistical Reliability**

Are experimental outcomes reproducible and statistically reliable across repeated missingness injections?

**H5:** The experimental pipeline reproduces results under fixed configurations, and repeated experiments provide estimable uncertainty around degradation and recovery effects.

**RQ6 — Subgroup Robustness**

Does missingness affect predictive performance and recovery effectiveness differently across demographic subgroups?

**H6:** The magnitude of predictive degradation and recovery effectiveness differs across at least some predefined demographic subgroups.

These hypotheses represent claims to be tested rather than established findings.


### Phase 7 — System Integration and Standalone Pipeline

**Status: Implemented; rigorous experimental validation pending**

- Integrated failure detection, diagnosis, recovery, evaluation, and safety modules.
- Developed the standalone execution pipeline in `src/run_pipeline.py`.
- Verified dataset dimensions, splitting, and recorded baseline metrics.
- Executed controlled `BILL_AMT1` missingness scenarios at 0%, 5%, 15%, and 30%.
- Implemented deterministic reproducibility assertions and result exports.
- Verified the existing pipeline against its recorded expected outcomes.

These checks support consistency with the existing implementation but do not independently establish successful predictive recovery.

### Current Research Milestone — Rigorous BILL_AMT1 Validation

**Status: Planned; experimental audit is the next task**

The project is moving from prototype development to rigorous scientific evaluation of the existing system, including controlled comparisons, repeated experiments, statistical validation, and detailed analysis of feature behavior.


### Interpretation of Existing Results

The current `BILL_AMT1` experiment reports post-imputation F1-scores of approximately 0.3570, 0.3589, 0.3609, and 0.3737 at 0%, 5%, 15%, and 30% missingness, respectively.

These results demonstrate that the recovered validation batches maintained F1-scores at or above the recorded healthy baseline under the tested configuration.

However, the experiment does not yet measure predictive performance on the corrupted data before recovery. Consequently, the observed F1 values cannot establish whether the recovery intervention improved performance relative to an untreated or benchmark condition.

Additional experiments will investigate whether the apparent resilience results from feature redundancy, model sensitivity, imputation behavior, or random variation.


## 9. Preliminary Observations and Open Research Questions

1. **Missingness detection:** The rule-based detector identified missingness exceeding the configured 5-percentage-point threshold.

2. **Post-repair performance:** Recorded F1-scores remained near or above the healthy baseline in the formal `BILL_AMT1` experiment. The reason for this behavior requires further investigation.

3. **Recovery effectiveness:** Before-versus-after recovery effectiveness has not yet been established using appropriate untreated and benchmark controls.

4. **Safety behavior:** The offline safety mechanism generated acceptance decisions in the formal experiment. Simulated unsafe scenarios tested rejection behavior, but integrated rejection and rollback validation requires further work.

5. **Reproducibility:** Earlier experiments demonstrated consistency under tested configurations. Statistical reliability across repeated missingness injections remains an open research objective.

These observations motivate the next stage of controlled experimentation and should not be interpreted as confirmation of the research hypotheses.


### Additional Research Limitations

- The standalone experiment does not explicitly compare untreated corrupted predictions against recovered predictions.
- Current results are primarily based on one missingness injection seed and one validation split.
- The effect of `BILL_AMT1` feature redundancy has not been thoroughly investigated.
- Only training-median imputation is implemented in the formal recovery pipeline.
- The scientific justification for the detection and safety thresholds requires further evaluation.
- Existing safety tests do not fully demonstrate integrated rollback behavior.
- Statistical confidence intervals and repeated-injection comparisons have not yet been completed.
- Subgroup performance under missingness has not been rigorously evaluated.
- Automated diagnosis has not been compared across multiple failure types.
- Earlier exploratory and formal experimental configurations require provenance verification before their outcomes can be compared.



## 13. Next Steps — Rigorous BILL_AMT1 Research

The next research milestone is a systematic investigation of `BILL_AMT1` missingness before extending SHML to additional failure types.

### Phase A — Experimental Audit

- Reproduce and verify existing baseline and experimental results.
- Confirm data splits, preprocessing, classification thresholds, and experiment configurations.
- Reconcile differences between exploratory and formal experiments.
- Review safety decisions and reproducibility checks.

### Phase B — Feature Analysis

- Investigate the distribution of `BILL_AMT1`.
- Examine relationships with `BILL_AMT2` and other predictors.
- Evaluate feature importance, coefficient behavior, and model sensitivity.
- Investigate possible explanations for model resilience.

### Phase C — Controlled Recovery Experiments

- Evaluate missingness levels of 0%, 5%, 15%, 30%, and 60%.
- Measure predictive performance before and after recovery.
- Compare different imputation and benchmark strategies.
- Quantify recovery effectiveness.

### Phase D — Statistical Validation

- Repeat missingness experiments using multiple random seeds.
- Estimate confidence intervals and performance variability.
- Analyze severity-response relationships.
- Evaluate the justification and sensitivity of performance thresholds.

### Phase E — Safety and Subgroup Validation

- Test integrated acceptance and rejection behavior.
- Verify preservation of the approved state following rejection.
- Evaluate performance across predefined demographic subgroups.
- Review reproducibility and experimental robustness.

### Phase F — Research Documentation and Visualization

- Produce feature analysis and correlation visualizations.
- Create missingness-versus-performance plots with uncertainty intervals.
- Compare recovery strategies.
- Visualize safety decisions.
- Prepare the final hypothesis evidence matrix, research report, and capstone presentation.

### Future Research

Potential extensions include `PAY_0` categorical shift, `LIMIT_BAL` numerical drift, additional datasets, alternative machine-learning architectures, LLM/RAG-assisted diagnosis, and cloud deployment.

## 14. Project Status

**Current stage:** Transition from implemented offline SHML prototype to rigorous experimental validation.

**Current research feature:** `BILL_AMT1`

**Next milestone:** Phase A — Experimental Audit

**Immediate deliverable:** A verified baseline, reconciled experimental results, and an auditable configuration for subsequent research experiments.