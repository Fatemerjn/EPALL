import unittest

import torch

from reference_metrics import paired_reference_batch_sums


class RetrainingReferenceMetricsTest(unittest.TestCase):
    def test_identical_outputs_have_exact_identity_metrics(self):
        logits = torch.tensor([[2.0, -1.0], [0.2, 0.8]])
        features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

        result = paired_reference_batch_sums(logits, logits, features, features)

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["agreement_sum"], 2.0)
        self.assertAlmostEqual(result["js_sum"], 0.0, places=7)
        self.assertAlmostEqual(result["logit_l2_sum"], 0.0, places=7)
        self.assertAlmostEqual(result["feature_cosine_sum"], 2.0, places=7)

    def test_different_outputs_are_finite_and_sample_aligned(self):
        logits_a = torch.tensor([[4.0, -2.0], [0.0, 1.0]])
        logits_b = torch.tensor([[-1.0, 3.0], [0.2, 0.8]])
        features_a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        features_b = torch.tensor([[0.0, 1.0], [1.0, -1.0]])

        result = paired_reference_batch_sums(logits_a, logits_b, features_a, features_b)

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["agreement_sum"], 1.0)
        self.assertGreater(result["js_sum"], 0.0)
        self.assertGreater(result["logit_l2_sum"], 0.0)
        for key in ("js_sum", "logit_l2_sum", "feature_cosine_sum"):
            self.assertTrue(torch.isfinite(torch.tensor(result[key])).item())

    def test_rejects_misaligned_logits(self):
        with self.assertRaises(ValueError):
            paired_reference_batch_sums(torch.zeros(2, 2), torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
