"""Validate the Phase H recovery policy using simulated missingness."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnosis import diagnose_failure
from src.detection import detect_feature_missingness
from src.evaluation import evaluate_model
from src.recovery import recover_missingness
from src.safety import evaluate_recovery_safety


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "working"
    / "default of credit card clients.xls"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase7"
FEATURE = "BILL_AMT1"
MISSINGNESS_LEVELS = [0.00, 0.05, 0.15, 0.30]
MAX_AUC_LOSS = 0.005


def inject_missingness(X_batch, rate, seed=42):
    """
    Deliberately mask BILL_AMT1 values in a copy of the input batch.

    The default seed=42 preserves previous
    pipeline behavior.
    """

    if not 0 <= rate <= 1:
        raise ValueError(
            f"Missingness rate must be between 0 and 1: {rate}"
        )

    if FEATURE not in X_batch.columns:
        raise ValueError(
            f"Missing feature: {FEATURE}"
        )

    X_corrupted = X_batch.copy(deep=True)

    n_missing = int(len(X_corrupted) * rate)

    rng = np.random.default_rng(seed)

    positions = rng.choice(
        len(X_corrupted),
        size=n_missing,
        replace=False
    )

    X_corrupted.iloc[
        positions,
        X_corrupted.columns.get_loc(FEATURE)
    ] = np.nan

    return X_corrupted


def assess_candidate(
    X_batch,
    y_true,
    diagnosis,
    action,
    training_median,
    healthy_metrics,
    model,
    scaler,
    max_auc_loss=None,
):
    candidate_diagnosis = {
        **diagnosis,
        "recovery_action": action,
    }
    X_candidate, recovery_log = recover_missingness(
        X_batch=X_batch,
        diagnosis=candidate_diagnosis,
        training_median=training_median,
    )

    missing_mask = X_batch[FEATURE].isna()
    observed_mask = ~missing_mask
    expected_fill = 0 if action == "Zero imputation" else training_median
    integrity_passed = (
        X_candidate.shape == X_batch.shape
        and X_candidate.index.equals(X_batch.index)
        and X_candidate.columns.equals(X_batch.columns)
        and not X_candidate.isna().any().any()
        and X_candidate.drop(columns=[FEATURE]).equals(
            X_batch.drop(columns=[FEATURE])
        )
        and X_candidate.loc[observed_mask, FEATURE].equals(
            X_batch.loc[observed_mask, FEATURE]
        )
        and X_candidate.loc[missing_mask, FEATURE].eq(expected_fill).all()
    )

    if not integrity_passed:
        return {
            "action": action,
            "decision": "REJECT",
            "reason": "Data integrity check failed",
            "metrics": None,
            "log": recovery_log,
            "integrity": False,
        }

    metrics = evaluate_model(
        X_batch=X_candidate,
        y_true=y_true,
        model=model,
        scaler=scaler,
    )
    safety = evaluate_recovery_safety(
        recovered_metrics=metrics,
        healthy_metrics=healthy_metrics,
        data_integrity_passed=integrity_passed,
        max_auc_loss=max_auc_loss,
    )
    return {
        "action": action,
        "decision": safety["decision"],
        "reason": safety["reason"],
        "metrics": metrics,
        "log": recovery_log,
        "integrity": integrity_passed,
    }


def select_recovery(
    X_corrupted,
    y_val,
    diagnosis,
    training_median,
    healthy_metrics,
    model,
    scaler,
    max_auc_loss=None,
):
    if not X_corrupted[FEATURE].isna().any():
        chosen = assess_candidate(
            X_corrupted, y_val, diagnosis, "No correction",
            training_median, healthy_metrics, model, scaler,
            max_auc_loss=max_auc_loss,
        )
        return chosen, chosen, None, "NO_REPAIR_NEEDED"

    preferred = assess_candidate(
        X_corrupted, y_val, diagnosis, "Zero imputation",
        training_median, healthy_metrics, model, scaler,
        max_auc_loss=max_auc_loss,
    )
    if preferred["decision"] == "ACCEPT":
        return preferred, preferred, None, "PREFERRED_ACCEPTED"

    fallback = assess_candidate(
        X_corrupted, y_val, diagnosis, "Median imputation",
        training_median, healthy_metrics, model, scaler,
        max_auc_loss=max_auc_loss,
    )
    if fallback["decision"] == "ACCEPT":
        return fallback, preferred, fallback, "FALLBACK_ACCEPTED"
    return None, preferred, fallback, "NO_SAFE_RECOVERY_ABSTAIN"


def main():
    """Evaluate simulated missingness on copies of complete validation data."""
    print("SHML Phase H offline recovery pipeline starting...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, header=1, engine="xlrd")
    target_col = "default payment next month"
    X = df.drop(columns=["ID", target_col])
    y = df[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    assert (len(df), X.shape[1], len(X_train), len(X_val), len(X_test)) == (
        30000, 23, 21000, 4500, 4500
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    healthy_metrics = evaluate_model(
        X_batch=X_val, y_true=y_val, model=model, scaler=scaler
    )
    assert abs(healthy_metrics["f1"] - 0.356983) < 0.0001
    assert abs(healthy_metrics["recall"] - 0.240201) < 0.0001
    training_median = float(X_train[FEATURE].median())

    print("Healthy validation F1:", round(healthy_metrics["f1"], 6))
    print("Healthy validation recall:", round(healthy_metrics["recall"], 6))
    print("Training median:", training_median)
    print("Untouched test observations:", len(X_test))

    records = []
    for rate in MISSINGNESS_LEVELS:
        X_corrupted = inject_missingness(X_val, rate)
        detection = detect_feature_missingness(
            X_batch=X_corrupted, feature=FEATURE,
            reference_missingness=0.0, threshold=0.05
        )
        diagnosis = diagnose_failure(detection)

        # Preserve the 5% detector boundary (strict > 5%, not >= 5%).
        assert detection["failure_detected"] == (rate > 0.05)

        # Apply the validation-selected policy independently of the
        # diagnosis's historical median-only recovery suggestion.
        chosen, preferred, fallback, status = select_recovery(
            X_corrupted=X_corrupted,
            y_val=y_val,
            diagnosis=diagnosis,
            training_median=training_median,
            healthy_metrics=healthy_metrics,
            model=model,
            scaler=scaler,
            max_auc_loss=MAX_AUC_LOSS,
        )

        metrics = chosen["metrics"] if chosen is not None else None
        record = {
            "missingness": rate,
            "detected": detection["failure_detected"],
            "severity": diagnosis["severity"],
            "status": status,
            "selected_strategy": chosen["action"] if chosen else "None",
            "preferred_decision": preferred["decision"],
            "preferred_reason": preferred["reason"],
            "fallback_decision": fallback["decision"] if fallback else "NOT_EVALUATED",
            "fallback_reason": fallback["reason"] if fallback else "",
            "values_imputed": chosen["log"]["values_imputed"] if chosen else 0,
            "integrity_passed": chosen["integrity"] if chosen else False,
            "f1": metrics["f1"] if metrics else np.nan,
            "recall": metrics["recall"] if metrics else np.nan,
            "roc_auc": metrics["roc_auc"] if metrics else np.nan,
        }
        records.append(record)
        print(f"{rate:.0%}: {status} | {record['selected_strategy']} "
              f"| F1={record['f1']:.6f}")

    output = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "recovery_policy_validation_results.csv"
    output.to_csv(csv_path, index=False)
    assert csv_path.exists() and len(pd.read_csv(csv_path)) == len(MISSINGNESS_LEVELS)
    print("\nValidation results saved:", csv_path)
    print("Test labels and test performance have NOT been used.")
    print("SHML Phase H offline integration completed.")
    return output


if __name__ == "__main__":
    main()
