
# ============================================================
# SELF-HEALING MACHINE LEARNING (SHML)
# Phase 7.5 — Standalone Academic Pipeline
#
# Dataset: UCI Default of Credit Card Clients
# Model: Logistic Regression
# Failure: BILL_AMT1 Missingness
# Scenarios: 0%, 5%, 15%, 30%
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.detection import detect_feature_missingness
from src.diagnosis import diagnose_failure
from src.recovery import recover_missingness
from src.evaluation import evaluate_model
from src.safety import evaluate_recovery_safety


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "working"
    / "default of credit card clients.xls"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase7"


def main():

    print("SHML standalone pipeline starting...")

    # ========================================================
    # PHASE 7.5.3.2 — DATASET PREPARATION
    # ========================================================

    df = pd.read_excel(
        DATA_PATH,
        header=1,
        engine="xlrd"
    )

    target_col = "default payment next month"

    X = df.drop(columns=["ID", target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDataset information:")
    print("Dataset rows:", len(df))
    print("Number of predictors:", X.shape[1])
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    assert len(df) == 30000
    assert X.shape[1] == 23
    assert len(X_train) == 21000
    assert len(X_val) == 4500
    assert len(X_test) == 4500

    print("\nPhase 7.5.3.2 dataset verification passed.")

    # ========================================================
    # PHASE 7.5.3.3 — HEALTHY BASELINE
    # ========================================================

    scaler = StandardScaler()

    # Fit scaler on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    baseline_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    baseline_model.fit(X_train_scaled, y_train)

    healthy_metrics = evaluate_model(
        X_batch=X_val,
        y_true=y_val,
        model=baseline_model,
        scaler=scaler
    )

    print("\nHealthy baseline metrics:")

    for metric, value in healthy_metrics.items():
        print(f"{metric}: {value:.4f}")

    expected_baseline = {
        "accuracy": 0.8087,
        "precision": 0.6948,
        "recall": 0.2402,
        "f1": 0.3570,
        "roc_auc": 0.7130
    }

    for metric, expected in expected_baseline.items():
        assert abs(healthy_metrics[metric] - expected) < 0.0001, (
            f"Baseline mismatch for {metric}"
        )

    print("\nPhase 7.5.3.3 baseline verification passed.")

    # ========================================================
    # PHASE 7.5.3.4 — MISSINGNESS INJECTION
    # ========================================================

    missingness_levels = [0.00, 0.05, 0.15, 0.30]

    corrupted_batches = {}

    for rate in missingness_levels:

        # Preserve original validation data
        X_corrupted = X_val.copy(deep=True)

        # Deterministic random number generator
        rng = np.random.default_rng(42)

        n_missing = int(len(X_corrupted) * rate)

        missing_indices = rng.choice(
            len(X_corrupted),
            size=n_missing,
            replace=False
        )

        # Inject missing values into BILL_AMT1 only
        X_corrupted.iloc[
            missing_indices,
            X_corrupted.columns.get_loc("BILL_AMT1")
        ] = np.nan

        corrupted_batches[rate] = X_corrupted

        actual_missing = int(
            X_corrupted["BILL_AMT1"].isna().sum()
        )

        print(
            f"Missingness: {rate:.0%} | "
            f"Missing values: {actual_missing}"
        )

        assert actual_missing == n_missing

    # Original validation data must remain unchanged
    assert X_val["BILL_AMT1"].isna().sum() == 0

    print("\nPhase 7.5.3.4 missingness verification passed.")

    # ========================================================
    # PHASE 7.5.3.5 — INTEGRATED SHML PIPELINE
    # ========================================================

    training_median = X_train["BILL_AMT1"].median()

    expected_detection = {
        0.00: False,
        0.05: False,
        0.15: True,
        0.30: True
    }

    expected_f1 = {
        0.00: 0.356983,
        0.05: 0.358898,
        0.15: 0.360947,
        0.30: 0.373723
    }

    pipeline_results = []

    for rate in missingness_levels:

        X_batch = corrupted_batches[rate]

        # ----------------------------------------------------
        # STEP 1 — FAILURE DETECTION
        # ----------------------------------------------------

        detection = detect_feature_missingness(
            X_batch=X_batch,
            feature="BILL_AMT1",
            reference_missingness=0.0,
            threshold=0.05
        )

        # ----------------------------------------------------
        # STEP 2 — FAILURE DIAGNOSIS
        # ----------------------------------------------------

        diagnosis = diagnose_failure(detection)

        # ----------------------------------------------------
        # STEP 3 — AUTOMATED RECOVERY
        # ----------------------------------------------------

        X_recovered, recovery_log = recover_missingness(
            X_batch=X_batch,
            diagnosis=diagnosis,
            training_median=training_median
        )

        # ----------------------------------------------------
        # STEP 4 — DATA INTEGRITY VERIFICATION
        # ----------------------------------------------------

        other_features = X_batch.columns.drop("BILL_AMT1")

        observed_mask = X_batch["BILL_AMT1"].notna()
        missing_mask = X_batch["BILL_AMT1"].isna()

        integrity_passed = (
            X_recovered.shape == X_batch.shape
            and X_recovered.columns.equals(X_batch.columns)
            and X_recovered.isna().sum().sum() == 0
            and X_recovered[other_features].equals(
                X_batch[other_features]
            )
            and X_recovered.loc[
                observed_mask, "BILL_AMT1"
            ].equals(
                X_batch.loc[observed_mask, "BILL_AMT1"]
            )
            and (
                X_recovered.loc[missing_mask, "BILL_AMT1"]
                == training_median
            ).all()
        )

        # ----------------------------------------------------
        # STEP 5 — MODEL EVALUATION
        # ----------------------------------------------------

        recovered_metrics = evaluate_model(
            X_batch=X_recovered,
            y_true=y_val,
            model=baseline_model,
            scaler=scaler
        )

        # ----------------------------------------------------
        # STEP 6 — SAFETY CONTROLLER
        # ----------------------------------------------------

        safety = evaluate_recovery_safety(
            recovered_metrics=recovered_metrics,
            healthy_metrics=healthy_metrics,
            data_integrity_passed=integrity_passed,
            max_f1_loss=0.10,
            max_recall_loss=0.10
        )

        # ----------------------------------------------------
        # STEP 7 — REPRODUCIBILITY VERIFICATION
        # ----------------------------------------------------

        assert (
            detection["failure_detected"]
            == expected_detection[rate]
        ), f"Detection mismatch at {rate:.0%}"

        assert (
            abs(recovered_metrics["f1"] - expected_f1[rate])
            < 0.0001
        ), f"F1 mismatch at {rate:.0%}"

        assert integrity_passed, (
            f"Data integrity failed at {rate:.0%}"
        )

        assert safety["decision"] == "ACCEPT", (
            f"Unexpected safety decision at {rate:.0%}"
        )

        # ----------------------------------------------------
        # STEP 8 — RECORD EXPERIMENT RESULTS
        # ----------------------------------------------------

        pipeline_results.append({
            "missingness": f"{rate:.0%}",
            "detected": detection["failure_detected"],
            "severity": diagnosis["severity"],
            "values_imputed": recovery_log["values_imputed"],
            "integrity_passed": integrity_passed,
            "f1": round(recovered_metrics["f1"], 6),
            "recall": round(recovered_metrics["recall"], 6),
            "roc_auc": round(recovered_metrics["roc_auc"], 6),
            "decision": safety["decision"]
        })

    # ========================================================
    # PHASE 7.5.4 — SAVE REPRODUCIBILITY ARTIFACTS
    # ========================================================

    results_df = pd.DataFrame(pipeline_results)

    print("\nIntegrated SHML experiment results:")
    print(results_df.to_string(index=False))

    # Create output directory if necessary
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "standalone_pipeline_results.csv"
    )

    # Export experiment results
    results_df.to_csv(
        output_file,
        index=False
    )

    # Verify exported results
    assert output_file.exists()

    saved_results = pd.read_csv(output_file)

    assert len(saved_results) == 4
    assert saved_results["decision"].eq("ACCEPT").all()

    print("\nResults saved to:", output_file)

    print(
        "Phase 7.5.4 results export "
        "verification passed."
    )

    print(
        "\nPhase 7.5 standalone SHML "
        "pipeline completed successfully."
    )


# ============================================================
# SCRIPT ENTRY POINT
# Must remain outside main() at the bottom of the file
# ============================================================

if __name__ == "__main__":
    main()
