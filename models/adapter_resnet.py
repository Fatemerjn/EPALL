import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import *
from .resnet import BasicBlock, Bottleneck, conv3x3

__all__ = ["AdapterResNet", "adapter_resnet18", "adapter_resnet34", "adapter_resnet50"]


class TaskBottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck):
        super(TaskBottleneckAdapter, self).__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class SharedBottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck):
        super(SharedBottleneckAdapter, self).__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class AdapterResNet(BaseModel):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes,
        nf,
        n_tasks,
        adapter_bottleneck=16,
        adapter_shared_bottleneck=0,
        adapter_location="residual",
        norm_params=False,
    ):
        super(AdapterResNet, self).__init__()
        if adapter_location != "residual":
            raise ValueError(f"Unsupported adapter_location={adapter_location!r}; expected 'residual'.")

        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.n_tasks = n_tasks
        self.norm_params = norm_params
        self.adapter_bottleneck = adapter_bottleneck
        self.adapter_shared_bottleneck = adapter_shared_bottleneck
        self.adapter_location = adapter_location

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=self.norm_params, affine=self.norm_params)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = nf * 8 * block.expansion
        self.shared_adapter = None
        if adapter_shared_bottleneck > 0:
            self.shared_adapter = SharedBottleneckAdapter(self.feature_dim, adapter_shared_bottleneck)
        self.adapters = nn.ModuleList(
            [TaskBottleneckAdapter(self.feature_dim, adapter_bottleneck) for _ in range(n_tasks)]
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
        if self.shared_adapter is not None:
            feature = self.shared_adapter(feature)
        if 0 <= task < self.n_tasks:
            feature = self.adapters[task](feature)

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
        if self.shared_adapter is not None:
            for param in self.shared_adapter.parameters():
                param.requires_grad = True
        for param in self.adapters.parameters():
            param.requires_grad = True
        if train_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def freeze_backbone_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def reset_task_adapter(self, task_id):
        self.adapters[task_id].reset_parameters()

    def reset_classifier_slice(self, task_id, class_per_task):
        start = task_id * class_per_task
        end = start + class_per_task
        replacement = torch.empty_like(self.classifier.weight.data[start:end])
        nn.init.kaiming_uniform_(replacement, a=math.sqrt(5))
        self.classifier.weight.data[start:end].copy_(replacement)

    def adapter_parameters(self, task_id=None):
        if task_id is None:
            return self.adapters.parameters()
        return self.adapters[task_id].parameters()

    def count_total_params(self):
        return sum(param.numel() for param in self.parameters())

    def count_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def count_adapter_params(self, task_id=None):
        params = self.adapter_parameters(task_id)
        return sum(param.numel() for param in params)

    def count_shared_adapter_params(self):
        if self.shared_adapter is None:
            return 0, 0
        total_params = 0
        for param in self.shared_adapter.parameters():
            total_params += param.numel()
        return total_params, total_params


def adapter_resnet18(
    num_classes,
    nf=64,
    norm_params=False,
    n_tasks=1,
    sparsity=None,
    adapter_bottleneck=16,
    adapter_shared_bottleneck=0,
    adapter_location="residual",
):
    del sparsity
    return AdapterResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        nf,
        n_tasks=n_tasks,
        adapter_bottleneck=adapter_bottleneck,
        adapter_shared_bottleneck=adapter_shared_bottleneck,
        adapter_location=adapter_location,
        norm_params=norm_params,
    )


def adapter_resnet34(
    num_classes,
    nf=64,
    norm_params=False,
    n_tasks=1,
    sparsity=None,
    adapter_bottleneck=16,
    adapter_shared_bottleneck=0,
    adapter_location="residual",
):
    del sparsity
    return AdapterResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        nf,
        n_tasks=n_tasks,
        adapter_bottleneck=adapter_bottleneck,
        adapter_shared_bottleneck=adapter_shared_bottleneck,
        adapter_location=adapter_location,
        norm_params=norm_params,
    )


def adapter_resnet50(
    num_classes,
    nf=64,
    norm_params=False,
    n_tasks=1,
    sparsity=None,
    adapter_bottleneck=16,
    adapter_shared_bottleneck=0,
    adapter_location="residual",
):
    del sparsity
    return AdapterResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        nf,
        n_tasks=n_tasks,
        adapter_bottleneck=adapter_bottleneck,
        adapter_shared_bottleneck=adapter_shared_bottleneck,
        adapter_location=adapter_location,
        norm_params=norm_params,
    )
