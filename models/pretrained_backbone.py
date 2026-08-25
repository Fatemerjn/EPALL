import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FrozenImageNetBackbone", "build_frozen_backbone"]

_DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenImageNetBackbone(nn.Module):
    """ImageNet-pretrained ResNet-18 feature extractor: frozen, offline-safe.

    Wraps ``torchvision.models.resnet18(weights=None)`` and loads weights from a
    LOCAL ``.pth`` file (no network access). The final fc layer is replaced by an
    identity, so a forward pass returns the 512-d pooled feature. 32x32 inputs are
    bilinearly upsampled to 224 inside ``forward`` (so data.py needs no change).
    All backbone parameters are frozen and BatchNorm is forced to eval mode (and
    kept there across ``.train()`` toggles), giving a fixed feature extractor that
    the PEFT adapters/LoRA modules adapt on top of.
    """

    feature_dim = 512

    def __init__(self, weights_path, input_size=224, input_dataset=None,
                 input_norm="imagenet"):
        super(FrozenImageNetBackbone, self).__init__()
        from torchvision.models import resnet18

        net = resnet18(weights=None)
        if weights_path is None:
            raise ValueError("FrozenImageNetBackbone requires --pretrained_weights (a local .pth path).")
        from pathlib import Path
        if not Path(str(weights_path)).is_file():
            raise FileNotFoundError(f"pretrained weights not found: {weights_path}")
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state, strict=True)
        net.fc = nn.Identity()  # drop the classifier -> forward yields the 512-d feature

        self.backbone = net
        self.input_size = int(input_size)
        self.input_dataset = input_dataset
        self.input_norm = input_norm
        if input_norm not in {"imagenet", "legacy_dataset_stats"}:
            raise ValueError(f"unknown pretrained input normalization: {input_norm}")
        if input_norm == "imagenet":
            if input_dataset not in _DATASET_STATS:
                raise ValueError(
                    f"ImageNet input conversion requires a known dataset, got {input_dataset!r}"
                )
            source_mean, source_std = _DATASET_STATS[input_dataset]
            self.register_buffer("source_mean", torch.tensor(source_mean).view(1, 3, 1, 1))
            self.register_buffer("source_std", torch.tensor(source_std).view(1, 3, 1, 1))
            self.register_buffer("imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
            self.preprocessing_id = f"{input_dataset}_to_imagenet_v2"
        else:
            self.preprocessing_id = "legacy_dataset_stats_v1"
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        # The frozen backbone must stay in eval mode (fixed BN stats) regardless
        # of the parent module's train/eval toggling.
        super(FrozenImageNetBackbone, self).train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        if self.input_norm == "imagenet":
            # data.py supplies dataset-normalized tensors. Recover [0,1] pixels,
            # then apply the normalization expected by ImageNet ResNet-18.
            x = x * self.source_std.to(dtype=x.dtype) + self.source_mean.to(dtype=x.dtype)
            x = (x - self.imagenet_mean.to(dtype=x.dtype)) / self.imagenet_std.to(dtype=x.dtype)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_frozen_backbone(pretrained_backbone, pretrained_weights, input_dataset=None,
                          input_norm="imagenet"):
    """Return a FrozenImageNetBackbone for the named option, or None for 'none'."""
    if pretrained_backbone in (None, "none"):
        return None
    if pretrained_backbone == "imagenet_resnet18":
        return FrozenImageNetBackbone(
            pretrained_weights,
            input_dataset=input_dataset,
            input_norm=input_norm,
        )
    raise ValueError(f"unknown pretrained_backbone={pretrained_backbone!r}")
