"""Frozen-backbone feature caching for the pretrained PEFT path.

When ``--pretrained_backbone imagenet_resnet18 --cache_features`` is set (for
``pall_adapter`` or ``lora``), each image's 512-d pooled feature is precomputed
ONCE with the frozen backbone (the same upsample-to-224 + torchvision ResNet-18)
and cached to disk, keyed by dataset + split + underlying image index. Training
and evaluation then run the adapters/classifier directly on the cached features,
so the (frozen, expensive) backbone is never re-run per epoch -> large speedup and
lower memory (the backbone is dropped from the training model afterwards).

Augmentation note: features are cached AUGMENTATION-FREE (the test transform:
ToTensor + Normalize, no RandomCrop / horizontal flip). This removes train-time
augmentation for the PEFT head -- a small accuracy trade-off accepted in exchange
for the speedup. Caching is a no-op unless the flag is set AND a frozen pretrained
backbone is active, so default behaviour is unchanged.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# CIFAR normalization constants (must match data.py's test transforms).
_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _augfree_base_dataset(dataset_name, data_dir, split):
    """Raw CIFAR split with AUGMENTATION-FREE transforms (ToTensor + Normalize)."""
    if dataset_name not in _NORM:
        raise ValueError(f"feature caching supports cifar10/cifar100, not {dataset_name!r}")
    mean, std = _NORM[dataset_name]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    return cls(data_dir, train=(split == "train"), download=True, transform=tf)


@torch.no_grad()
def compute_features(base_dataset, backbone, device, batch_size=256):
    """Run ``backbone`` over ``base_dataset`` (in index order) -> (N, 512) CPU tensor."""
    backbone.eval()
    loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    for x, _ in loader:
        feats.append(backbone(x.to(device)).detach().float().cpu())
    return torch.cat(feats, 0)


def build_or_load_cache(dataset_name, split, data_dir, backbone, device, cache_dir):
    """Return the (N, 512) CPU feature tensor for a split, loading a matching
    on-disk cache if present, otherwise computing and saving it."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset_name}_{split}_imagenet_resnet18.pt"
    base = _augfree_base_dataset(dataset_name, data_dir, split)
    n = len(base)
    if path.is_file():
        feats = torch.load(path, map_location="cpu")
        if feats.shape[0] == n and feats.shape[1] == backbone.feature_dim:
            print(f"[cache] loaded {split} features from {path} {tuple(feats.shape)}", flush=True)
            return feats
        print(f"[cache] stale cache {path} (have {tuple(feats.shape)}, need ({n}, "
              f"{backbone.feature_dim})); recomputing", flush=True)
    print(f"[cache] computing {split} features for {dataset_name} ({n} images) ...", flush=True)
    feats = compute_features(base, backbone, device)
    torch.save(feats, path)
    print(f"[cache] saved {split} features to {path} {tuple(feats.shape)}", flush=True)
    return feats


class CachedFeatureDataset(Dataset):
    """Wrap a ``data.SubDataset`` to yield ``(cached_512d_feature, remapped_label)``
    instead of ``(image, label)``, mirroring SubDataset's permutation label remap."""

    def __init__(self, sub_dataset, features):
        self.sub_indices = list(sub_dataset.sub_indices)
        self._targets = sub_dataset.dataset.targets
        self.permutation = list(sub_dataset.permutation)
        self.features = features

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        under = self.sub_indices[index]
        label = self._targets[under]
        label = int(label.item()) if torch.is_tensor(label) else int(label)
        return self.features[under], self.permutation.index(label)


def apply_feature_cache(args, model, train_datasets, test_datasets):
    """Build/load the caches, wrap the task datasets, free the backbone from the
    training model, and switch the net to consume precomputed features.

    Returns wrapped ``(train_datasets, test_datasets)``. Requires an active frozen
    pretrained backbone (``model.net.frozen_backbone``).
    """
    backbone = getattr(model.net, "frozen_backbone", None)
    if backbone is None:
        raise RuntimeError("--cache_features requires an active frozen pretrained backbone "
                           "(--pretrained_backbone imagenet_resnet18 on pall_adapter/lora).")
    device = model.device
    cache_dir = Path(args.data_dir) / "feature_cache"
    train_feats = build_or_load_cache(args.dataset, "train", args.data_dir, backbone, device, cache_dir)
    test_feats = build_or_load_cache(args.dataset, "test", args.data_dir, backbone, device, cache_dir)
    train_datasets = [CachedFeatureDataset(d, train_feats) for d in train_datasets]
    test_datasets = [CachedFeatureDataset(d, test_feats) for d in test_datasets]
    # The backbone is no longer needed by the feature-space model: drop it (memory)
    # and flip the net into precomputed-feature mode.
    model.net.frozen_backbone = None
    model.net.features_are_precomputed = True
    print("[cache] feature caching active: adapters/classifier now train on cached "
          "512-d features (backbone freed).", flush=True)
    return train_datasets, test_datasets
