import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_pipeline import select_recovery


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


if __name__ == "__main__":
    unittest.main()
