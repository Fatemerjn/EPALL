import unittest

import torch

from models.subnet_layers import SubnetLinear


class SparseBackupTest(unittest.TestCase):
    def test_store_merge_and_apply_preserve_selected_historical_values(self):
        layer = SubnetLinear(4, 1, bias=False, sparsity=0.5)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        layer.store_backup(5, weight_mask=torch.tensor([[1, 0, 1, 0]], dtype=torch.bool))

        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[10.0, 20.0, 30.0, 40.0]]))
        layer.store_backup(5, weight_mask=torch.tensor([[0, 0, 1, 1]], dtype=torch.bool))

        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[100.0, 200.0, 300.0, 400.0]]))
        restored, _ = layer._apply_backup(layer.weight, None, 5)
        self.assertTrue(torch.equal(restored, torch.tensor([[1.0, 200.0, 30.0, 40.0]])))

        entry = layer.backup_weights[5]
        self.assertEqual(entry["weight_indices"].numel(), 3)
        self.assertEqual(entry["weight_values"].numel(), 3)
        self.assertNotIn("weight_mask", entry)
        self.assertNotIn("weight", entry)

    def test_unselected_task_uses_live_weights_without_clone(self):
        layer = SubnetLinear(2, 1, bias=False, sparsity=0.5)
        weight, bias = layer._apply_backup(layer.weight, None, 99)
        self.assertIs(weight, layer.weight)
        self.assertIsNone(bias)

    def test_legacy_dense_entry_remains_readable(self):
        layer = SubnetLinear(3, 1, bias=False, sparsity=0.5)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[7.0, 8.0, 9.0]]))
        layer.backup_weights[1] = {
            "weight": torch.tensor([[1.0, 2.0, 3.0]]),
            "weight_mask": torch.tensor([[0, 1, 0]], dtype=torch.bool),
            "bias": None,
            "bias_mask": None,
        }
        restored, _ = layer._apply_backup(layer.weight, None, 1)
        self.assertTrue(torch.equal(restored, torch.tensor([[7.0, 2.0, 9.0]])))


if __name__ == "__main__":
    unittest.main()
