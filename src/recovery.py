"""SHML missing-value recovery for the supported BILL_AMT1 policy actions."""

import math
from numbers import Real

import pandas as pd


_SUPPORTED_ACTIONS = {
    "No correction",
    "Routine median preprocessing",
    "Median imputation",
    "Zero imputation",
}
_MEDIAN_ACTIONS = {
    "Routine median preprocessing",
    "Median imputation",
}


def recover_missingness(X_batch, diagnosis, training_median):
    """Apply a supported recovery action to a copy and validate its integrity."""
    if not isinstance(X_batch, pd.DataFrame):
        raise TypeError("X_batch must be a pandas DataFrame")

    feature = diagnosis["feature"]
    action = diagnosis["recovery_action"]
    if feature != "BILL_AMT1":
        raise ValueError(f"Unsupported recovery feature: {feature}")
    if feature not in X_batch.columns:
        raise ValueError(f"Recovery feature is missing from X_batch: {feature}")
    if action not in _SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported recovery action: {action}")
    if action in _MEDIAN_ACTIONS and (
        not isinstance(training_median, Real)
        or not math.isfinite(training_median)
    ):
        raise ValueError("training_median must be a finite number")

    X_recovered = X_batch.copy(deep=True)
    missing_mask = X_batch[feature].isna()
    observed_mask = ~missing_mask
    missing_before = int(missing_mask.sum())

    if action in _MEDIAN_ACTIONS:
        X_recovered[feature] = X_recovered[feature].fillna(training_median)
        expected_fill = training_median
    elif action == "Zero imputation":
        X_recovered[feature] = X_recovered[feature].fillna(0)
        expected_fill = 0
    else:
        expected_fill = None

    missing_after = int(X_recovered[feature].isna().sum())
    feature_values_preserved = (
        X_recovered.loc[observed_mask, feature]
        .eq(X_batch.loc[observed_mask, feature])
        .all()
    )
    if action == "No correction":
        missing_values_preserved = (
            X_recovered.loc[missing_mask, feature].isna().all()
        )
    else:
        missing_values_preserved = (
            X_recovered.loc[missing_mask, feature]
            .eq(expected_fill)
            .all()
        )

    integrity_passed = (
        X_recovered.shape == X_batch.shape
        and X_recovered.index.equals(X_batch.index)
        and X_recovered.columns.equals(X_batch.columns)
        and X_recovered.drop(columns=[feature]).equals(
            X_batch.drop(columns=[feature])
        )
        and feature_values_preserved
        and missing_values_preserved
        and (
            missing_after == missing_before
            if action == "No correction"
            else missing_after == 0
        )
    )
    if not integrity_passed:
        raise RuntimeError("Recovery failed data-integrity validation")

    log = {
        "feature": feature,
        "severity": diagnosis["severity"],
        "action": action,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "values_imputed": missing_before - missing_after,
        "recovery_applied": missing_before > missing_after,
        "integrity_passed": integrity_passed,
    }
    return X_recovered, log
