

# src/detection.py
# SHML feature missingness detection

def detect_feature_missingness(
    X_batch,
    feature,
    reference_missingness=0.0,
    threshold=0.05
):
    """
    Detect whether a feature's missingness rate has increased
    beyond the allowed threshold relative to the healthy reference.
    """

    if feature not in X_batch.columns:
        raise ValueError(f"Feature '{feature}' is missing")

    current_missingness = X_batch[feature].isna().mean()

    missingness_increase = (
        current_missingness - reference_missingness
    )

    failure_detected = missingness_increase > threshold

    return {
        "feature": feature,
        "current_missingness": current_missingness,
        "reference_missingness": reference_missingness,
        "missingness_increase": missingness_increase,
        "threshold": threshold,
        "failure_detected": bool(failure_detected)
    }

def diagnose_failure(detection_result):
    """
    Diagnose BILL_AMT1 missingness and recommend
    an appropriate recovery action.
    """

    feature = detection_result["feature"]
    missingness = detection_result["current_missingness"]
    failure_detected = detection_result["failure_detected"]

    # Validate supported feature
    if feature != "BILL_AMT1":
        raise ValueError(
            f"No diagnosis rule configured for {feature}"
        )

    # Determine severity and recovery strategy
    if missingness == 0:
        severity = "None"
        failure_type = "No missingness"
        recovery_action = "No correction"

    elif not failure_detected:
        severity = "Below alert threshold"
        failure_type = "Numerical missingness"
        recovery_action = "Routine median preprocessing"

    elif missingness <= 0.20:
        severity = "Moderate"
        failure_type = "Numerical missingness"
        recovery_action = "Median imputation"

    else:
        severity = "Severe"
        failure_type = "Numerical missingness"
        recovery_action = "Median imputation"

    explanation = (
        f"{feature} has {missingness:.2%} missing values. "
        f"Severity: {severity}. "
        f"Recommended action: {recovery_action}."
    )

    return {
        "feature": feature,
        "failure_type": failure_type,
        "severity": severity,
        "failure_detected": failure_detected,
        "recovery_action": recovery_action,
        "explanation": explanation
    }
