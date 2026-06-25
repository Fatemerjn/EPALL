import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FrozenImageNetBackbone", "build_frozen_backbone"]


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

    def __init__(self, weights_path, input_size=224):
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
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_frozen_backbone(pretrained_backbone, pretrained_weights):
    """Return a FrozenImageNetBackbone for the named option, or None for 'none'."""
    if pretrained_backbone in (None, "none"):
        return None
    if pretrained_backbone == "imagenet_resnet18":
        return FrozenImageNetBackbone(pretrained_weights)
    raise ValueError(f"unknown pretrained_backbone={pretrained_backbone!r}")
