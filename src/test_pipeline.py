import io
import sys
import unittest
from contextlib import redirect_stdout
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import run_pipeline
from src.detection import detect_feature_missingness
from src.diagnosis import diagnose_failure
from src.evaluation import evaluate_model
from src.recovery import recover_missingness
from src.run_pipeline import select_recovery
from src.safety import evaluate_recovery_safety


def candidate(action, decision):
    return {"action": action, "decision": decision}


class SelectRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.corrupted = pd.DataFrame(
            {"BILL_AMT1": [100.0, float("nan")]}
        )
        self.complete = pd.DataFrame(
            {"BILL_AMT1": [100.0, 200.0]}
        )
        self.args = {
            "y_val": pd.Series([0, 1]),
            "diagnosis": {
                "feature": "BILL_AMT1",
                "severity": "Moderate",
                "recovery_action": "Median imputation",
            },
            "training_median": 150.0,
            "healthy_metrics": {},
            "model": object(),
            "scaler": object(),
        }

    @patch("src.run_pipeline.assess_candidate")
    def test_no_missing_data_does_not_attempt_imputation(self, assess):
        expected = candidate("No correction", "ACCEPT")
        assess.return_value = expected

        chosen, preferred, fallback, status = select_recovery(
            X_corrupted=self.complete, **self.args
        )

        self.assertEqual(status, "NO_REPAIR_NEEDED")
        self.assertIs(chosen, expected)
        self.assertIs(preferred, expected)
        self.assertIsNone(fallback)
        assess.assert_called_once()
        self.assertEqual(assess.call_args.args[3], "No correction")

    @patch("src.run_pipeline.assess_candidate")
    def test_preferred_zero_imputation_is_accepted(self, assess):
        expected = candidate("Zero imputation", "ACCEPT")
        assess.return_value = expected

        chosen, preferred, fallback, status = select_recovery(
            X_corrupted=self.corrupted, **self.args
        )

        self.assertEqual(status, "PREFERRED_ACCEPTED")
        self.assertIs(chosen, preferred)
        self.assertIs(chosen, expected)
        self.assertIsNone(fallback)
        assess.assert_called_once()
        self.assertEqual(assess.call_args.args[3], "Zero imputation")

    @patch("src.run_pipeline.assess_candidate")
    def test_fallback_accepted_when_zero_rejected(self, assess):
        assess.side_effect = [
            candidate("Zero imputation", "REJECT"),
            candidate("Median imputation", "ACCEPT"),
        ]

        chosen, preferred, fallback, status = select_recovery(
            X_corrupted=self.corrupted, **self.args
        )

        self.assertEqual(status, "FALLBACK_ACCEPTED")
        self.assertEqual(preferred["decision"], "REJECT")
        self.assertEqual(chosen["action"], "Median imputation")
        self.assertIs(chosen, fallback)
        self.assertEqual(
            [call.args[3] for call in assess.call_args_list],
            ["Zero imputation", "Median imputation"],
        )

    @patch("src.run_pipeline.assess_candidate")
    def test_both_rejected_results_in_abstention(self, assess):
        assess.side_effect = [
            candidate("Zero imputation", "REJECT"),
            candidate("Median imputation", "REJECT"),
        ]

        chosen, preferred, fallback, status = select_recovery(
            X_corrupted=self.corrupted, **self.args
        )

        self.assertEqual(status, "NO_SAFE_RECOVERY_ABSTAIN")
        self.assertIsNone(chosen)
        self.assertEqual(preferred["decision"], "REJECT")
        self.assertEqual(fallback["decision"], "REJECT")
        self.assertEqual(
            [call.args[3] for call in assess.call_args_list],
            ["Zero imputation", "Median imputation"],
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.healthy = pd.DataFrame(
            {
                "BILL_AMT1": [100.0 * (row + 1) for row in range(20)],
                "BILL_AMT2": [300 + row for row in range(20)],
            },
            index=range(100, 120),
        )
        self.labels = pd.Series([0] * 10 + [1] * 10, index=self.healthy.index)
        self.corrupted = run_pipeline.inject_missingness(self.healthy, 0.15)
        self.diagnosis = diagnose_failure(
            detect_feature_missingness(self.corrupted, "BILL_AMT1")
        )
        self.training_median = self.healthy["BILL_AMT1"].median()

    def test_recovery_preserves_unaffected_columns_and_existing_values(self):
        original = self.corrupted.copy(deep=True)
        missing = original["BILL_AMT1"].isna()
        for action, fill_value in (
            ("Zero imputation", 0.0),
            ("Median imputation", self.training_median),
        ):
            with self.subTest(action=action):
                recovered, _ = recover_missingness(
                    self.corrupted,
                    {**self.diagnosis, "recovery_action": action},
                    self.training_median,
                )
                expected = original.copy(deep=True)
                expected.loc[missing, "BILL_AMT1"] = fill_value
                pd.testing.assert_frame_equal(recovered, expected)
                pd.testing.assert_frame_equal(self.corrupted, original)

    def test_detection_and_diagnosis_at_configured_missingness_levels(self):
        for rate, detected, severity, action in (
            (0.00, False, "None", "No correction"),
            (0.05, False, "Below alert threshold", "Routine median preprocessing"),
            (0.15, True, "Moderate", "Median imputation"),
            (0.30, True, "Severe", "Median imputation"),
        ):
            with self.subTest(rate=rate):
                corrupted = run_pipeline.inject_missingness(self.healthy, rate)
                detection = detect_feature_missingness(corrupted, "BILL_AMT1")
                diagnosis = diagnose_failure(detection)

                self.assertAlmostEqual(detection["current_missingness"], rate)
                self.assertIs(detection["failure_detected"], detected)
                self.assertIs(diagnosis["failure_detected"], detected)
                self.assertEqual(diagnosis["severity"], severity)
                self.assertEqual(diagnosis["recovery_action"], action)

    def test_zero_and_median_imputation_produce_valid_metrics(self):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.healthy)
        model = LogisticRegression(random_state=42).fit(scaled, self.labels)
        healthy_metrics = evaluate_model(self.healthy, self.labels, model, scaler)
        for action in ("Zero imputation", "Median imputation"):
            with self.subTest(action=action):
                evaluated = run_pipeline.assess_candidate(
                    self.corrupted, self.labels, self.diagnosis, action,
                    self.training_median, healthy_metrics, model, scaler,
                )
                self.assertTrue(evaluated["integrity"])
                self.assertEqual(evaluated["log"]["values_imputed"], 3)
                metrics = evaluated["metrics"]
                self.assertEqual(
                    set(metrics), {"accuracy", "precision", "recall", "f1", "roc_auc"}
                )
                for name, value in metrics.items():
                    self.assertTrue(isfinite(value) and 0 <= value <= 1, (name, value))

    def test_safety_rejects_unsafe_candidate_and_selects_safe_fallback(self):
        healthy_metrics = dict.fromkeys(
            ("accuracy", "precision", "recall", "f1", "roc_auc"), 0.8
        )
        unsafe_metrics = {**healthy_metrics, "f1": 0.5}
        with patch.object(
            run_pipeline, "evaluate_model",
            side_effect=[unsafe_metrics, healthy_metrics],
        ):
            chosen, preferred, fallback, status = select_recovery(
                self.corrupted, self.labels, self.diagnosis,
                self.training_median, healthy_metrics, object(), object(),
            )

        self.assertTrue(preferred["integrity"])
        self.assertEqual(preferred["action"], "Zero imputation")
        self.assertEqual(preferred["decision"], "REJECT")
        self.assertIn("F1 loss exceeds tolerance", preferred["reason"])
        self.assertEqual(status, "FALLBACK_ACCEPTED")
        self.assertIs(chosen, fallback)
        self.assertEqual(chosen["action"], "Median imputation")
        self.assertEqual(chosen["decision"], "ACCEPT")

    def test_optional_roc_auc_guardrail_rejects_recovery(self):
        healthy = {
            "accuracy": 0.80,
            "precision": 0.70,
            "recall": 0.25,
            "f1": 0.36,
            "roc_auc": 0.713,
        }
        recovered = {
            "accuracy": 0.81,
            "precision": 0.69,
            "recall": 0.26,
            "f1": 0.37,
            "roc_auc": 0.708,
        }

        original = evaluate_recovery_safety(recovered, healthy, True)
        combined = evaluate_recovery_safety(
            recovered, healthy, True, max_auc_loss=0.005
        )

        print("Original policy:", original["decision"])
        print("Combined policy:", combined["decision"])
        print("Reason:", combined["reason"])
        print("ROC-AUC loss (%):", round(combined["auc_loss_pct"], 4))

        self.assertEqual(original["decision"], "ACCEPT")
        self.assertEqual(combined["decision"], "REJECT")
        self.assertEqual(combined["reason"], "ROC-AUC loss exceeds tolerance")
        self.assertAlmostEqual(combined["auc_loss_pct"], 0.7013, places=4)

    def test_repeated_pipeline_runs_produce_identical_results(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output_dir = Path(directory)
            with patch.object(run_pipeline, "OUTPUT_DIR", output_dir / "first"):
                first = run_pipeline.main()
            with patch.object(run_pipeline, "OUTPUT_DIR", output_dir / "second"):
                second = run_pipeline.main()

            pd.testing.assert_frame_equal(first, second, check_exact=True)
            csv_name = "recovery_policy_validation_results.csv"
            self.assertEqual(
                (output_dir / "first" / csv_name).read_bytes(),
                (output_dir / "second" / csv_name).read_bytes(),
            )

class AUCSafetyGuardrailTests(unittest.TestCase):

    def test_auc_guardrail_rejects_unsafe_candidate(self):
        healthy = {
            "accuracy": 0.80,
            "precision": 0.70,
            "recall": 0.25,
            "f1": 0.36,
            "roc_auc": 0.713
        }

        recovered = {
            "accuracy": 0.81,
            "precision": 0.69,
            "recall": 0.26,
            "f1": 0.37,
            "roc_auc": 0.708
        }

        # Original safety policy
        original = evaluate_recovery_safety(
            recovered, healthy, True
        )

        # Experimental ROC-AUC safety policy
        combined = evaluate_recovery_safety(
            recovered, healthy, True,
            max_auc_loss=0.005
        )

        self.assertEqual(original["decision"], "ACCEPT")
        self.assertEqual(combined["decision"], "REJECT")

        self.assertIn(
            "ROC-AUC loss exceeds tolerance",
            combined["reason"]
        )

        self.assertAlmostEqual(
            combined["auc_loss_pct"],
            0.701262,
            places=4
        )


if __name__ == "__main__":
    unittest.main()
