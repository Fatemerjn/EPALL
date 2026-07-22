import unittest

import torch
import torch.nn as nn

from methods.base import Base


class _Memory:
    def __init__(self):
        self.buffer = {
            0: {
                "X": torch.zeros(2, 3, dtype=torch.float32),
                "Y": torch.zeros(2, dtype=torch.int64),
                "H": torch.zeros(2, 4, dtype=torch.float32),
            }
        }


class _BackupLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backup_weights = {
            0: {
                "weight_indices": torch.tensor([1, 3], dtype=torch.int64),
                "weight_values": torch.tensor([0.2, 0.4], dtype=torch.float32),
            }
        }


class StorageAccountingTest(unittest.TestCase):
    def test_explicit_categories_sum_to_total(self):
        model = Base.__new__(Base)
        nn.Module.__init__(model)
        model.net = nn.Sequential(nn.Linear(3, 2, bias=False), _BackupLayer())
        model.side_nets = {1: nn.Linear(2, 2, bias=False)}
        model.per_task_masks = {0: {"layer": torch.ones(5, dtype=torch.bool)}}
        model.archived_task_masks = {}
        model.memory = _Memory()

        report = model.storage_accounting()

        self.assertEqual(report["model_parameter_bytes"], 6 * 4)
        self.assertEqual(report["side_network_parameter_bytes"], 4 * 4)
        self.assertEqual(report["active_side_networks"], 1)
        self.assertEqual(report["subnet_mask_bytes"], 5)
        self.assertEqual(report["backup_index_bytes"], 2 * 8)
        self.assertEqual(report["backup_value_bytes"], 2 * 4)
        self.assertEqual(report["replay_image_bytes"], 6 * 4)
        self.assertEqual(report["replay_label_bytes"], 2 * 8)
        self.assertEqual(report["replay_logit_bytes"], 8 * 4)
        expected = sum(
            report[key]
            for key in (
                "model_parameter_bytes",
                "side_network_parameter_bytes",
                "subnet_mask_bytes",
                "backup_index_bytes",
                "backup_value_bytes",
                "replay_image_bytes",
                "replay_label_bytes",
                "replay_logit_bytes",
            )
        )
        self.assertEqual(report["accounted_total_bytes"], expected)


if __name__ == "__main__":
    unittest.main()
