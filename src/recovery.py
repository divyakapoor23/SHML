"""SHML missing-value recovery. Supports legacy and frozen policy actions."""


def recover_missingness(X_batch, diagnosis, training_median):
    feature = diagnosis["feature"]
    action = diagnosis["recovery_action"]
    if feature != "BILL_AMT1":
        raise ValueError(f"Unsupported recovery feature: {feature}")
    supported = {
        "No correction", "Routine median preprocessing",
        "Median imputation", "Zero imputation"
    }
    if action not in supported:
        raise ValueError(f"Unsupported recovery action: {action}")

    X_recovered = X_batch.copy(deep=True)
    missing_before = int(X_recovered[feature].isna().sum())

    if action in {"Routine median preprocessing", "Median imputation"}:
        X_recovered[feature] = X_recovered[feature].fillna(training_median)
    elif action == "Zero imputation":
        X_recovered[feature] = X_recovered[feature].fillna(0)

    missing_after = int(X_recovered[feature].isna().sum())
    log = {
        "feature": feature,
        "severity": diagnosis["severity"],
        "action": action,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "values_imputed": missing_before - missing_after,
        "recovery_applied": missing_before > missing_after,
    }
    return X_recovered, log
