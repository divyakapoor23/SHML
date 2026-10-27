"""Recovery safety checks extracted from Phase 5."""

import numpy as np


def evaluate_recovery_safety(
    recovered_metrics,
    healthy_metrics,
    data_integrity_passed,
    max_f1_loss=0.10,
    max_recall_loss=0.10
):
    """
    Evaluate offline recovery acceptance criteria.
    """

    required_metrics = [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ]

    metrics_valid = all(
        metric in recovered_metrics
        and np.isfinite(recovered_metrics[metric])
        and 0 <= recovered_metrics[metric] <= 1
        for metric in required_metrics
    )

    baseline_valid = all(
        metric in healthy_metrics
        and np.isfinite(healthy_metrics[metric])
        and 0 < healthy_metrics[metric] <= 1
        for metric in ["f1", "recall"]
    )

    if not metrics_valid or not baseline_valid:
        return {
            "decision": "REJECT",
            "reason": "Invalid evaluation metrics or baseline",
            "f1_loss_pct": np.nan,
            "recall_loss_pct": np.nan
        }

    f1_loss = (
        healthy_metrics["f1"] - recovered_metrics["f1"]
    ) / healthy_metrics["f1"]

    recall_loss = (
        healthy_metrics["recall"] - recovered_metrics["recall"]
    ) / healthy_metrics["recall"]

    reasons = []

    if not data_integrity_passed:
        reasons.append("Data integrity check failed")

    if f1_loss > max_f1_loss:
        reasons.append("F1 loss exceeds tolerance")

    if recall_loss > max_recall_loss:
        reasons.append("Recall loss exceeds tolerance")

    decision = "REJECT" if reasons else "ACCEPT"

    return {
        "decision": decision,
        "reason": "; ".join(reasons) if reasons else "All checks passed",
        "f1_loss_pct": f1_loss * 100,
        "recall_loss_pct": recall_loss * 100
    }
