"""Failure diagnosis rules extracted from Phase 4."""


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
