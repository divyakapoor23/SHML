
# src/recovery.py
# SHML missingness recovery module

def recover_missingness(
    X_batch,
    diagnosis,
    training_median
):
    """
    Apply the diagnosed recovery action without
    modifying the original validation batch.
    """

    feature = diagnosis["feature"]
    action = diagnosis["recovery_action"]

    if feature != "BILL_AMT1":
        raise ValueError("Unsupported recovery feature")

    # Preserve the original corrupted batch
    X_recovered = X_batch.copy(deep=True)

    missing_before = int(
        X_recovered[feature].isna().sum()
    )

    supported_actions = {
        "No correction",
        "Routine median preprocessing",
        "Median imputation"
    }

    if action not in supported_actions:
        raise ValueError(f"Unsupported action: {action}")

    if action in {
        "Routine median preprocessing",
        "Median imputation"
    }:
        X_recovered[feature] = (
            X_recovered[feature].fillna(training_median)
        )

    missing_after = int(
        X_recovered[feature].isna().sum()
    )

    recovery_log = {
        "feature": feature,
        "severity": diagnosis["severity"],
        "action": action,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "values_imputed": missing_before - missing_after,
        "recovery_applied": missing_before > missing_after
    }

    return X_recovered, recovery_log
