"""Check the regression in label handling and verify recorded test predictions."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analyze_results import all_tables, score_predictions


class ReproductionChecks(unittest.TestCase):
    def test_label_mask_regression(self):
        """Nonempty False strings must never be treated as labeled examples."""
        mask = ["True", "False", "False", "True"]
        released = [bool(value) for value in mask]
        corrected = [value == "True" for value in mask]
        self.assertEqual(sum(released), 4)
        self.assertEqual(sum(corrected), 2)
        from author_runner import stage_method
        import inspect
        patch = inspect.getsource(stage_method)
        self.assertIn('.eq("True")', patch)

    def test_recorded_mask_counts(self):
        """Check the actual saved BANKING77 mask audit."""
        from analyze_results import read
        rows = read(ROOT / "results/table_mask_audit.csv")
        observed = next(row for row in rows if row["split"] == "train"
                        and row["known_ratio"] == "0.25" and row["fold"] == "0")
        self.assertEqual(int(observed["released_knncon"]), 2303)
        self.assertEqual(int(observed["corrected_knncon"]), 229)

    def test_known_and_unknown_metrics(self):
        """Check pooled-unknown F1 against a hand-calculated confusion pattern."""
        rows = [dict(gold="known", predicted="known"), dict(gold="known", predicted="unknown"),
                dict(gold="unknown", predicted="unknown"), dict(gold="unknown", predicted="known")]
        metrics, errors = score_predictions(rows, "unknown")
        self.assertEqual(metrics["K-F1"], 50)
        self.assertEqual(metrics["N-F1"], 50)
        self.assertEqual(errors["unknown accepted"], 1)
        self.assertEqual(errors["known rejected"], 1)

    def test_saved_prediction_evidence(self):
        """Detect absent outputs, scale errors, or metrics inconsistent with predictions."""
        all_tables(ROOT / "results")

    def test_analysis_under_work_ancestor(self):
        """A checkout inside a directory named work must retain its trial evidence."""
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "work" / "checkout" / "results"
            shutil.copytree(ROOT / "results", copied)
            all_tables(copied)
            self.assertTrue((copied / "tables/deepunk_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
