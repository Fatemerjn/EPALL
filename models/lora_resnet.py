import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import *
from .resnet import BasicBlock, Bottleneck, conv3x3

__all__ = ["LoRAResNet", "lora_resnet18", "lora_resnet34", "lora_resnet50"]


class TaskLoRA(nn.Module):
    """Per-task LoRA module: x + (alpha/r) * B(A x), no nonlinearity.

    Mirrors ``TaskBottleneckAdapter`` but with a plain low-rank update instead of
    the down->ReLU->up bottleneck. ``down`` is A (dim->r), ``up`` is B (r->dim),
    initialized so the module starts as the identity (B=0).
    """

    def __init__(self, dim, rank, alpha):
        super(TaskLoRA, self).__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.scaling * self.up(self.down(x))


class SharedLoRA(nn.Module):
    """Optional shared LoRA applied to the feature for all tasks."""

    def __init__(self, dim, rank, alpha):
        super(SharedLoRA, self).__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.scaling * self.up(self.down(x))


class LoRAResNet(BaseModel):
    """ResNet with a frozen backbone and per-task LoRA modules on the feature.

    API mirrors ``AdapterResNet`` (forward(x, task), freeze_backbone,
    freeze_backbone_batchnorm, reset_classifier_slice, count_* helpers); the
    only renamed method is ``reset_task_lora`` (the adapter's
    ``reset_task_adapter``).
    """

    def __init__(
        self,
        block,
        num_blocks,
        num_classes,
        nf,
        n_tasks,
        lora_rank=8,
        lora_alpha=16,
        lora_shared_rank=0,
        norm_params=False,
        pretrained_backbone="none",
        pretrained_weights=None,
    ):
        super(LoRAResNet, self).__init__()

        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.n_tasks = n_tasks
        self.norm_params = norm_params
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_shared_rank = lora_shared_rank
        self.pretrained_backbone = pretrained_backbone

        # Feature extractor: from-scratch stem (default) or a frozen ImageNet
        # backbone. Both yield a 512-d pooled feature; the LoRA/classifier below
        # are identical either way.
        from .pretrained_backbone import build_frozen_backbone
        self.frozen_backbone = build_frozen_backbone(pretrained_backbone, pretrained_weights)
        if self.frozen_backbone is not None:
            self.feature_dim = self.frozen_backbone.feature_dim
        else:
            self.conv1 = conv3x3(3, nf * 1)
            self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=self.norm_params, affine=self.norm_params)
            self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = nf * 8 * block.expansion

        self.shared_lora = None
        if lora_shared_rank > 0:
            self.shared_lora = SharedLoRA(self.feature_dim, lora_shared_rank, lora_alpha)
        self.loras = nn.ModuleList(
            [TaskLoRA(self.feature_dim, lora_rank, lora_alpha) for _ in range(n_tasks)]
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_params))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def extract_backbone_features(self, x):
        if getattr(self, "features_are_precomputed", False):
            return x  # x is an already-cached 512-d backbone feature (see feature_cache.py)
        if self.frozen_backbone is not None:
            return self.frozen_backbone(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def forward(self, x, task=-1, mask=None, mode="train", returnt="out"):
        del mask, mode
        feature = self.extract_backbone_features(x)
        if self.shared_lora is not None:
            feature = self.shared_lora(feature)
        if 0 <= task < self.n_tasks:
            feature = self.loras[task](feature)

        if returnt == "features":
            return feature

        out = self.classifier(feature)
        if returnt == "out":
            return out
        if returnt == "all":
            return out, feature
        raise NotImplementedError("Unknown return type")

    def freeze_backbone(self, train_classifier=False):
        for param in self.parameters():
            param.requires_grad = False
        if self.shared_lora is not None:
            for param in self.shared_lora.parameters():
                param.requires_grad = True
        for param in self.loras.parameters():
            param.requires_grad = True
        if train_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def freeze_backbone_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def reset_task_lora(self, task_id):
        self.loras[task_id].reset_parameters()

    def reset_classifier_slice(self, task_id, class_per_task):
        start = task_id * class_per_task
        end = start + class_per_task
        replacement = torch.empty_like(self.classifier.weight.data[start:end])
        nn.init.kaiming_uniform_(replacement, a=math.sqrt(5))
        self.classifier.weight.data[start:end].copy_(replacement)

    def lora_parameters(self, task_id=None):
        if task_id is None:
            return self.loras.parameters()
        return self.loras[task_id].parameters()

    def count_task_adapter_params(self, task_id=None):
        params = self.lora_parameters(task_id)
        return sum(param.numel() for param in params)

    def count_total_params(self):
        return sum(param.numel() for param in self.parameters())

    def count_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def count_adapter_params(self, task_id=None):
        return self.count_task_adapter_params(task_id)

    def count_shared_adapter_params(self):
        if self.shared_lora is None:
            return 0, 0
        total_params = sum(param.numel() for param in self.shared_lora.parameters())
        return total_params, total_params

    def count_total_adapter_params(self):
        task_params = self.count_task_adapter_params()
        shared_params, _ = self.count_shared_adapter_params()
        return int(task_params + shared_params)


def lora_resnet18(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=None,
                  lora_rank=8, lora_alpha=16, lora_shared_rank=0,
                  pretrained_backbone="none", pretrained_weights=None):
    del sparsity
    return LoRAResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, n_tasks=n_tasks,
                      lora_rank=lora_rank, lora_alpha=lora_alpha,
                      lora_shared_rank=lora_shared_rank, norm_params=norm_params,
                      pretrained_backbone=pretrained_backbone, pretrained_weights=pretrained_weights)


def lora_resnet34(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=None,
                  lora_rank=8, lora_alpha=16, lora_shared_rank=0,
                  pretrained_backbone="none", pretrained_weights=None):
    del sparsity
    return LoRAResNet(BasicBlock, [3, 4, 6, 3], num_classes, nf, n_tasks=n_tasks,
                      lora_rank=lora_rank, lora_alpha=lora_alpha,
                      lora_shared_rank=lora_shared_rank, norm_params=norm_params,
                      pretrained_backbone=pretrained_backbone, pretrained_weights=pretrained_weights)


def lora_resnet50(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=None,
                  lora_rank=8, lora_alpha=16, lora_shared_rank=0,
                  pretrained_backbone="none", pretrained_weights=None):
    del sparsity
    return LoRAResNet(Bottleneck, [3, 4, 6, 3], num_classes, nf, n_tasks=n_tasks,
                      lora_rank=lora_rank, lora_alpha=lora_alpha,
                      lora_shared_rank=lora_shared_rank, norm_params=norm_params,
                      pretrained_backbone=pretrained_backbone, pretrained_weights=pretrained_weights)
