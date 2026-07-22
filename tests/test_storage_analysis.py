import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_storage_accounting import linear_slope  # noqa: E402


class StorageAnalysisTest(unittest.TestCase):
    def test_training_growth_slope_uses_mib_per_active_task(self):
        self.assertAlmostEqual(linear_slope([(1, 10.0), (2, 15.0), (3, 20.0)]), 5.0)

    def test_degenerate_growth_is_not_reported(self):
        self.assertIsNone(linear_slope([(2, 1.0), (2, 2.0)]))


if __name__ == "__main__":
    unittest.main()
