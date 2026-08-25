import unittest

from privacy_metrics import roc_auc_from_scores


class PrivacyMetricsTest(unittest.TestCase):
    def test_all_ties_are_chance(self):
        self.assertAlmostEqual(roc_auc_from_scores([1, 1], [1, 1]), 0.5)

    def test_partial_ties_use_average_ranks(self):
        self.assertAlmostEqual(roc_auc_from_scores([1, 2], [1, 0]), 0.875)

    def test_perfect_ordering(self):
        self.assertAlmostEqual(roc_auc_from_scores([2, 3], [0, 1]), 1.0)
        self.assertAlmostEqual(roc_auc_from_scores([0, 1], [2, 3]), 0.0)

    def test_empty_group_is_undefined(self):
        self.assertIsNone(roc_auc_from_scores([], [1]))


if __name__ == "__main__":
    unittest.main()
