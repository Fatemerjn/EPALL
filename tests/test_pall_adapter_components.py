import unittest
from types import SimpleNamespace

import torch

from methods.pall_adapter import PALLAdapter


def make_args(mode):
    return SimpleNamespace(
        arch="adapter_resnet18",
        n_tasks=2,
        class_per_task=2,
        sparsity=0.8,
        norm_params=False,
        adapter_bottleneck=2,
        adapter_shared_bottleneck=2,
        adapter_location="residual",
        pretrained_backbone="none",
        pretrained_weights=None,
        device=torch.device("cpu"),
        lr=1e-2,
        momentum=0.0,
        weight_decay=0.0,
        mem_budget=8,
        dim_input=(512,),
        mem_type="random",
        k_shot=1,
        adapter_train_classifier=True,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        optim="sgd",
        seed=0,
        retrain_steps=1,
        retrain_epochs=None,
        adapter_shared_forget_ratio=0.5,
        adapter_shared_protect_ratio=0.5,
        adapter_shared_protect_strength=None,
        adapter_shared_forget_lr=None,
        adapter_forget_steps=1,
        adapter_forget_mode="uniform_loop",
        adapter_mask_mode="discrete",
        adapter_conflict_gamma=1.0,
        protect_importance="gradient",
        eval_bound=False,
        adapter_component_mode=mode,
        eval_component_stages=True,
    )


def build_model(mode):
    torch.manual_seed(7)
    model = PALLAdapter(make_args(mode))
    model.net.features_are_precomputed = True
    model.prev_tasks = [0, 1]
    model.task_status = {0: "T", 1: "T"}
    model.memory.buffer = {
        0: {
            "X": torch.randn(4, 512),
            "Y": torch.tensor([0, 1, 0, 1]),
            "num_seen": 4,
        },
        1: {
            "X": torch.randn(4, 512),
            "Y": torch.tensor([2, 3, 2, 3]),
            "num_seen": 4,
        },
    }
    return model


def eval_stub(_stage):
    return {"accuracy": [0.75, 0.50]}


class PALLAdapterComponentModesTest(unittest.TestCase):
    def test_reset_only_skips_shared_classifier_and_repair(self):
        model = build_model("reset_only")
        shared_before = [param.detach().clone() for _, param in model._shared_adapter_param_items()]
        info = model.forget_with_diagnostics(1, eval_fn=eval_stub, remaining_tasks=[0])
        shared_after = [param.detach() for _, param in model._shared_adapter_param_items()]

        self.assertTrue(all(torch.equal(a, b) for a, b in zip(shared_before, shared_after)))
        self.assertEqual(info["classifier_forget_param_count"], 0)
        self.assertEqual(info["finetune_diag"]["retrain_steps"], 0)
        self.assertNotIn(1, model.memory.buffer)
        self.assertEqual(set(info["stage_evals"]), {
            "after_target_reset", "after_shared_update",
            "after_classifier_ascent", "after_retained_repair",
        })

    def test_full_mode_runs_all_components(self):
        model = build_model("full")
        info = model.forget_with_diagnostics(1, eval_fn=eval_stub, remaining_tasks=[0])

        self.assertGreater(info["shared_effective_forget_params"], 0)
        self.assertGreater(info["classifier_forget_param_count"], 0)
        self.assertEqual(info["finetune_diag"]["retrain_steps"], 1)
        self.assertTrue(all(info["component_stages"].values()))

    def test_reset_repair_skips_forgetting_updates_but_repairs(self):
        model = build_model("reset_repair")
        info = model.forget_with_diagnostics(1, eval_fn=eval_stub, remaining_tasks=[0])

        self.assertEqual(info["shared_effective_forget_params"], 0)
        self.assertEqual(info["classifier_forget_param_count"], 0)
        self.assertEqual(info["finetune_diag"]["retrain_steps"], 1)
        self.assertEqual(
            info["component_stages"],
            {
                "target_reset": True,
                "shared_update": False,
                "classifier_ascent": False,
                "retained_repair": True,
            },
        )

    def test_uniform_unprotected_uses_full_selected_support(self):
        model = build_model("uniform_unprotected")
        info = model.forget_with_diagnostics(1, eval_fn=eval_stub, remaining_tasks=[0])

        self.assertEqual(info["protected_adapter_params"], 0)
        self.assertEqual(info["shared_soft_update_params"], 0)
        self.assertEqual(info["shared_effective_forget_params"], info["shared_forget_candidates"])

    def test_mask_no_ascent_skips_classifier_update(self):
        model = build_model("mask_no_ascent")
        info = model.forget_with_diagnostics(1, eval_fn=eval_stub, remaining_tasks=[0])

        self.assertGreater(info["shared_effective_forget_params"], 0)
        self.assertEqual(info["classifier_forget_param_count"], 0)
        self.assertEqual(info["finetune_diag"]["retrain_steps"], 1)


if __name__ == "__main__":
    unittest.main()
