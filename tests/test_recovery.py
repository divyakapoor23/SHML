import unittest

import pandas as pd

from src.recovery import recover_missingness


class RecoverMissingnessTests(unittest.TestCase):
    def setUp(self):
        self.X_batch = pd.DataFrame(
            {
                "BILL_AMT1": [1000, None, 5000, None],
                "BILL_AMT2": [900, 2000, 4500, 3000],
                "AGE": [25, 31, 42, 50],
            },
            index=[10, 20, 30, 40],
        )
        self.original = self.X_batch.copy(deep=True)
        self.diagnosis = {
            "feature": "BILL_AMT1",
            "severity": "moderate",
            "recovery_action": "Zero imputation",
        }

    def test_zero_imputation_returns_correct_values_and_log(self):
        recovered, log = recover_missingness(
            self.X_batch, self.diagnosis, training_median=22225.5
        )

        self.assertEqual(
            recovered["BILL_AMT1"].tolist(),
            [1000, 0, 5000, 0],
        )
        self.assertEqual(log["missing_before"], 2)
        self.assertEqual(log["missing_after"], 0)
        self.assertEqual(log["values_imputed"], 2)
        self.assertTrue(log["recovery_applied"])
        self.assertTrue(log["integrity_passed"])
        self.assertTrue(recovered.index.equals(self.original.index))
        self.assertTrue(recovered.columns.equals(self.original.columns))
        pd.testing.assert_frame_equal(
            recovered[["BILL_AMT2", "AGE"]],
            self.original[["BILL_AMT2", "AGE"]],
        )
        self.assertEqual(
            recovered.loc[[10, 30], "BILL_AMT1"].tolist(),
            [1000, 5000],
        )
        pd.testing.assert_frame_equal(self.X_batch, self.original)

    def test_median_fallback_uses_training_median(self):
        diagnosis = {
            **self.diagnosis,
            "recovery_action": "Median imputation",
        }

        recovered, log = recover_missingness(
            self.X_batch, diagnosis, training_median=22225.5
        )

        self.assertEqual(
            recovered["BILL_AMT1"].tolist(),
            [1000, 22225.5, 5000, 22225.5],
        )
        self.assertEqual(log["values_imputed"], 2)
        self.assertEqual(log["missing_after"], 0)
        self.assertTrue(log["integrity_passed"])
        pd.testing.assert_frame_equal(self.X_batch, self.original)

    def test_no_correction_preserves_missing_values_and_input(self):
        diagnosis = {**self.diagnosis, "recovery_action": "No correction"}

        recovered, log = recover_missingness(
            self.X_batch, diagnosis, training_median=22225.5
        )

        pd.testing.assert_frame_equal(recovered, self.original)
        self.assertEqual(log["missing_before"], 2)
        self.assertEqual(log["missing_after"], 2)
        self.assertEqual(log["values_imputed"], 0)
        self.assertFalse(log["recovery_applied"])
        self.assertTrue(log["integrity_passed"])

    def test_rejects_unsupported_recovery_action(self):
        diagnosis = {**self.diagnosis, "recovery_action": "Unknown action"}

        with self.assertRaisesRegex(ValueError, "Unsupported recovery action"):
            recover_missingness(
                self.X_batch, diagnosis, training_median=22225.5
            )

    def test_rejects_unsupported_recovery_feature(self):
        diagnosis = {**self.diagnosis, "feature": "AGE"}

        with self.assertRaisesRegex(ValueError, "Unsupported recovery feature"):
            recover_missingness(
                self.X_batch, diagnosis, training_median=22225.5
            )

    def test_median_recovery_requires_finite_training_median(self):
        diagnosis = {
            **self.diagnosis,
            "recovery_action": "Median imputation",
        }

        for invalid_median in (None, float("nan"), float("inf")):
            with self.subTest(training_median=invalid_median):
                with self.assertRaisesRegex(
                    ValueError, "training_median must be a finite number"
                ):
                    recover_missingness(
                        self.X_batch, diagnosis, invalid_median
                    )


if __name__ == "__main__":
    unittest.main()
