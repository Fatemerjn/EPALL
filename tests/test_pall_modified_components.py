import unittest
from types import SimpleNamespace

import torch
from torch import nn

from methods.pall_base import PALLBase


class SelectionHarness:
    def __init__(self):
        self.net = nn.Module()
        self.net.register_parameter(
            "first", nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        )
        self.net.register_parameter(
            "second", nn.Parameter(torch.tensor([0.5, 0.6, 0.7]))
        )
        self.args = SimpleNamespace(seed=7)


class PALLModifiedComponentSelectionTest(unittest.TestCase):
    def setUp(self):
        self.model = SelectionHarness()
        self.candidates = {
            "first": torch.tensor([1, 1, 1, 1], dtype=torch.bool),
            "second": torch.tensor([1, 1, 1], dtype=torch.bool),
        }

    def select(self, *args, **kwargs):
        return PALLBase._select_exact_budget_masks(self.model, *args, **kwargs)

    def test_ranked_selection_is_global_and_exact(self):
        scores = {
            "first": torch.tensor([1.0, 8.0, 2.0, 7.0]),
            "second": torch.tensor([3.0, 9.0, 4.0]),
        }
        selected = self.select(self.candidates, 3, score_map=scores)
        self.assertEqual(sum(int(mask.sum()) for mask in selected.values()), 3)
        self.assertTrue(torch.equal(selected["first"], torch.tensor([0, 1, 0, 1], dtype=torch.bool)))
        self.assertTrue(torch.equal(selected["second"], torch.tensor([0, 1, 0], dtype=torch.bool)))

    def test_random_control_is_reproducible_exact_and_in_support(self):
        first = self.select(self.candidates, 4, random_seed=2027)
        second = self.select(self.candidates, 4, random_seed=2027)
        self.assertEqual(sum(int(mask.sum()) for mask in first.values()), 4)
        for key in self.candidates:
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertFalse(torch.any(first[key] & ~self.candidates[key]))

    def test_budget_is_clamped_to_candidate_count(self):
        selected = self.select(self.candidates, 99, random_seed=1)
        self.assertEqual(sum(int(mask.sum()) for mask in selected.values()), 7)

    def test_equal_budget_controls_use_the_intended_candidate_support(self):
        target = {"first": torch.tensor([1, 1, 1, 1], dtype=torch.bool)}
        shared = {"first": torch.tensor([0, 1, 0, 1], dtype=torch.bool)}
        random_support = PALLBase._modified_control_candidates(
            "random_budget", target, shared
        )
        ranking_support = PALLBase._modified_control_candidates(
            "ranking_no_overlap", target, shared
        )
        self.assertIs(random_support, shared)
        self.assertIs(ranking_support, target)


if __name__ == "__main__":
    unittest.main()
