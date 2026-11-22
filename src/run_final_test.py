
"""Evaluate frozen SHML strategies on held-out copies with simulated missingness."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.run_pipeline import (
    DATA_PATH,
    FEATURE,
    MAX_AUC_LOSS,
    inject_missingness,
)
from src.evaluation import evaluate_model
from src.safety import evaluate_recovery_safety


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_test"

SEED = 42
MISSINGNESS_LEVELS = [0.0, 0.05, 0.15, 0.30]


def main():
    print("SHML Final Held-Out Evaluation")

    df = pd.read_excel(DATA_PATH, header=1, engine="xlrd")

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

    assert len(X_train) == 21000
    assert len(X_val) == 4500
    assert len(X_test) == 4500

    scaler = StandardScaler()
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(scaler.fit_transform(X_train), y_train)

    training_median = float(X_train[FEATURE].median())

    healthy_metrics = evaluate_model(
        X_batch=X_test,
        y_true=y_test,
        model=model,
        scaler=scaler,
    )

    print("Healthy test F1:", round(healthy_metrics["f1"], 6))
    print("Healthy test AUC:", round(healthy_metrics["roc_auc"], 6))

    records = []

    for rate in MISSINGNESS_LEVELS:
        X_corrupted = inject_missingness(
            X_test,
            rate,
            seed=SEED,
        )

        for strategy, fill_value in [
            ("Fixed median", training_median),
            ("Zero imputation", 0),
        ]:
            X_candidate = X_corrupted.copy(deep=True)
            X_candidate[FEATURE] = (
                X_candidate[FEATURE].fillna(fill_value)
            )

            metrics = evaluate_model(
                X_batch=X_candidate,
                y_true=y_test,
                model=model,
                scaler=scaler,
            )

            # Post-hoc audit only:
            # test labels do NOT select the recovery strategy.
            audit = evaluate_recovery_safety(
                recovered_metrics=metrics,
                healthy_metrics=healthy_metrics,
                data_integrity_passed=True,
                max_auc_loss=MAX_AUC_LOSS,
            )

            records.append({
                "missingness": rate,
                "strategy": strategy,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "f1_vs_healthy": (
                    metrics["f1"] - healthy_metrics["f1"]
                ),
                "auc_vs_healthy": (
                    metrics["roc_auc"] - healthy_metrics["roc_auc"]
                ),
                "posthoc_safety_decision": audit["decision"],
                "posthoc_safety_reason": audit["reason"],
            })

    results = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "final_test_results.csv"
    results.to_csv(output_path, index=False)

    print("\nFinal evaluation results:")
    print(results.round(6).to_string(index=False))

    print("\nSaved:", output_path)
    print("Test results must not be used for policy tuning.")

    return results


if __name__ == "__main__":
    main()
