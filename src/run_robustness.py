
"""Reproduce the SHML 20-seed BILL_AMT1 safety experiment."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_pipeline import (
    DATA_PATH,
    FEATURE,
    MAX_AUC_LOSS,
    inject_missingness,
    select_recovery,
)
from src.detection import detect_feature_missingness
from src.diagnosis import diagnose_failure
from src.evaluation import evaluate_model


OUTPUT_DIR = PROJECT_ROOT / "results" / "bill_amt1"

MISSINGNESS_RATE = 0.15
SEEDS = range(20)

STATUS_TO_NOTEBOOK = {
    "PREFERRED_ACCEPTED": "ZERO_ACCEPTED",
    "FALLBACK_ACCEPTED": "MEDIAN_FALLBACK_ACCEPTED",
    "NO_SAFE_RECOVERY_ABSTAIN": "ABSTAIN",
}


def prepare_validation_data():
    """Reproduce the training and validation setup."""

    df = pd.read_excel(
        DATA_PATH,
        header=1,
        engine="xlrd",
    )

    target = "default payment next month"
    X = df.drop(columns=["ID", target])
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    assert (
        len(X_train), len(X_val), len(X_test)
    ) == (21000, 4500, 4500)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    healthy_metrics = evaluate_model(
        X_batch=X_val,
        y_true=y_val,
        model=model,
        scaler=scaler,
    )

    training_median = float(X_train[FEATURE].median())

    # X_test and y_test are never evaluated.
    return (
        X_val,
        y_val,
        model,
        scaler,
        healthy_metrics,
        training_median,
    )


def run_experiment():
    """Run 20 independently seeded missingness masks."""

    (
        X_val,
        y_val,
        model,
        scaler,
        healthy_metrics,
        training_median,
    ) = prepare_validation_data()

    print("SHML Integrated Robustness Experiment")
    print("Feature:", FEATURE)
    print("Missingness:", MISSINGNESS_RATE)
    print("Seeds:", len(SEEDS))
    print("Healthy F1:", round(healthy_metrics["f1"], 6))
    print("Healthy ROC-AUC:",
          round(healthy_metrics["roc_auc"], 6))
    print("ROC-AUC tolerance:", MAX_AUC_LOSS)
    print()

    records = []

    for seed in SEEDS:

        X_corrupted = inject_missingness(
            X_batch=X_val,
            rate=MISSINGNESS_RATE,
            seed=seed,
        )

        detection = detect_feature_missingness(
            X_batch=X_corrupted,
            feature=FEATURE,
            reference_missingness=0.0,
            threshold=0.05,
        )

        diagnosis = diagnose_failure(detection)

        assert detection["failure_detected"]
        assert int(X_corrupted[FEATURE].isna().sum()) == 675

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

        metrics = chosen["metrics"] if chosen else None

        record = {
            "seed": seed,
            "missingness": MISSINGNESS_RATE,
            "missing_count": int(
                X_corrupted[FEATURE].isna().sum()
            ),
            "detected": detection["failure_detected"],
            "status": status,
            "selected_action": (
                chosen["action"] if chosen else "None"
            ),
            "preferred_decision": preferred["decision"],
            "preferred_reason": preferred["reason"],
            "fallback_decision": (
                fallback["decision"]
                if fallback else "NOT_EVALUATED"
            ),
            "fallback_reason": (
                fallback["reason"] if fallback else ""
            ),
            "f1": metrics["f1"] if metrics else np.nan,
            "recall": metrics["recall"] if metrics else np.nan,
            "roc_auc": (
                metrics["roc_auc"] if metrics else np.nan
            ),
        }

        records.append(record)

        print(
            f"Seed {seed:02d}: "
            f"{status} | {record['selected_action']}"
        )

    results = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        OUTPUT_DIR / "integrated_20_seed_results.csv"
    )

    results.to_csv(output_file, index=False)

    print("\nFinal integrated outcomes:")
    print(results["status"].value_counts())

    expected_counts = {
        "PREFERRED_ACCEPTED": 16,
        "FALLBACK_ACCEPTED": 2,
        "NO_SAFE_RECOVERY_ABSTAIN": 2,
    }

    actual_counts = results["status"].value_counts().to_dict()

    counts_match = actual_counts == expected_counts

    print("\nCounts match notebook:", counts_match)

    # Compare decisions for each individual seed.
    reference_file = (
        OUTPUT_DIR / "combined_safety_policy_results.csv"
    )

    if reference_file.exists():
        reference = pd.read_csv(reference_file)

        reference = reference[
            ["seed", "final_decision"]
        ]

        integrated = results[["seed", "status"]].copy()
        integrated["integrated_decision"] = (
            integrated["status"].map(STATUS_TO_NOTEBOOK)
        )

        comparison = integrated.merge(
            reference,
            on="seed",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        comparison["matches"] = (
            comparison["_merge"].eq("both")
            & comparison["integrated_decision"].eq(
                comparison["final_decision"]
            )
        )

        comparison.drop(
            columns=["_merge"],
            inplace=True,
        )

        comparison_file = (
            OUTPUT_DIR / "integrated_vs_notebook.csv"
        )
        comparison.to_csv(comparison_file, index=False)

        print(
            "Matching seed decisions:",
            int(comparison["matches"].sum()),
            "out of",
            len(comparison),
        )
        print("Comparison saved:", comparison_file)

        if not comparison["matches"].all():
            print("\nMismatched decisions:")
            print(
                comparison.loc[
                    ~comparison["matches"]
                ].to_string(index=False)
            )
    else:
        print("Notebook reference CSV not found.")

    print("\nResults saved:", output_file)
    print("Test partition remains untouched.")

    return results


if __name__ == "__main__":
    run_experiment()
