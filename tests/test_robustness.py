
"""Regression tests for SHML's integrated robustness experiment."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src import run_robustness


class RobustnessRegressionTests(unittest.TestCase):

    def test_integrated_20_seed_results(self):
        """
        Verify that the integrated SHML policy reproduces
        all 20 validation-based notebook decisions.
        """

        reference_path = (
            run_robustness.PROJECT_ROOT
            / "results"
            / "bill_amt1"
            / "combined_safety_policy_results.csv"
        )

        self.assertTrue(
            reference_path.exists(),
            f"Missing notebook reference: {reference_path}"
        )

        # Use a temporary directory so tests never
        # overwrite the research results.
        with TemporaryDirectory() as temp_dir:

            temp_path = Path(temp_dir)

            copyfile(
                reference_path,
                temp_path / "combined_safety_policy_results.csv"
            )

            with patch.object(
                run_robustness,
                "OUTPUT_DIR",
                temp_path
            ):
                with redirect_stdout(io.StringIO()):
                    results = run_robustness.run_experiment()

            # Check 20 runs and seeds 0-19.
            self.assertEqual(len(results), 20)

            self.assertEqual(
                results["seed"].tolist(),
                list(range(20))
            )

            # Validate the three policy outcomes.
            counts = results["status"].value_counts()

            self.assertEqual(
                counts.get("PREFERRED_ACCEPTED", 0), 16
            )

            self.assertEqual(
                counts.get("FALLBACK_ACCEPTED", 0), 2
            )

            self.assertEqual(
                counts.get("NO_SAFE_RECOVERY_ABSTAIN", 0), 2
            )

            # Check that all 20 scenarios had
            # the intended corruption and detection.
            self.assertTrue(
                (results["missing_count"] == 675).all()
            )

            self.assertTrue(results["detected"].all())

            # Verify expected decisions for specific seeds.
            decisions = results.set_index("seed")["status"]

            self.assertEqual(
                decisions.loc[0], "FALLBACK_ACCEPTED"
            )
            self.assertEqual(
                decisions.loc[2], "FALLBACK_ACCEPTED"
            )
            self.assertEqual(
                decisions.loc[8], "NO_SAFE_RECOVERY_ABSTAIN"
            )
            self.assertEqual(
                decisions.loc[11], "NO_SAFE_RECOVERY_ABSTAIN"
            )

            # Confirm that the saved notebook decisions
            # match the integrated pipeline for every seed.
            comparison_file = (
                temp_path / "integrated_vs_notebook.csv"
            )

            self.assertTrue(comparison_file.exists())

            comparison = pd.read_csv(comparison_file)

            self.assertEqual(len(comparison), 20)
            self.assertTrue(comparison["matches"].all())

            # Verify selected output file exists.
            self.assertTrue(
                (
                    temp_path / "integrated_20_seed_results.csv"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
