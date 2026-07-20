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
from torchvision.datasets.folder import default_loader

# Normalization constants (must match data.py's test transforms).
_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
_FEATURE_DIM = 512
_TINYIMAGENET_NUM_CLASSES = 200


def _backbone_feature_dim(backbone):
    feature_dim = getattr(backbone, "feature_dim", None)
    if feature_dim is None:
        return _FEATURE_DIM
    return int(feature_dim)


def _load_tinyimagenet_wnids(root):
    wnids_path = root / "wnids.txt"
    if not wnids_path.is_file():
        raise FileNotFoundError(f"TinyImageNet wnids.txt not found: {wnids_path}")
    with open(wnids_path, "r", encoding="utf-8") as handle:
        wnids = [line.strip() for line in handle if line.strip()]
    if len(wnids) != _TINYIMAGENET_NUM_CLASSES:
        raise ValueError(
            f"Expected {_TINYIMAGENET_NUM_CLASSES} TinyImageNet classes in {wnids_path}, "
            f"found {len(wnids)}."
        )
    return wnids


def _tinyimagenet_root(data_dir):
    root = Path(data_dir) / "tiny-imagenet-200"
    missing = [
        str(root / name)
        for name in ("train", "val", "test")
        if not (root / name).is_dir()
    ]
    if not root.is_dir() or missing or not (root / "wnids.txt").is_file():
        details = f" Missing: {missing}" if missing else ""
        raise FileNotFoundError(
            "TinyImageNet not found. Expected args.data_dir/tiny-imagenet-200 "
            "with train/, val/, test/, and wnids.txt."
            f"{details}"
        )
    return root


class TinyImageNetTrainCacheDataset(datasets.ImageFolder):
    """TinyImageNet train split with class ids fixed by wnids.txt order."""

    def __init__(self, root, wnids, transform=None):
        self.wnids = list(wnids)
        self._wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        super().__init__(str(root), transform=transform)
        self.targets = [target for _, target in self.samples]

    def find_classes(self, directory):
        missing = [wnid for wnid in self.wnids if not (Path(directory) / wnid).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Missing TinyImageNet train class directories under {directory}: {missing[:5]}"
            )
        return list(self.wnids), dict(self._wnid_to_idx)


class TinyImageNetAnnotatedCacheDataset(Dataset):
    """TinyImageNet annotated val/test split with class ids from wnids.txt."""

    def __init__(self, root, wnids, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.loader = default_loader
        self.classes = list(wnids)
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.classes)}

        annotations_path = self.root / f"{self.root.name}_annotations.txt"
        if self.root.name == "val":
            annotations_path = self.root / "val_annotations.txt"
        images_dir = self.root / "images"
        if not annotations_path.is_file():
            raise FileNotFoundError(
                f"TinyImageNet annotations not found for split {self.root.name!r}: "
                f"{annotations_path}"
            )
        if not images_dir.is_dir():
            raise FileNotFoundError(f"TinyImageNet image directory not found: {images_dir}")

        self.samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    raise ValueError(f"Unknown TinyImageNet wnid {wnid!r} in {annotations_path}.")
                image_path = images_dir / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(f"TinyImageNet image not found: {image_path}")
                self.samples.append((str(image_path), self.class_to_idx[wnid]))

        self.targets = [target for _, target in self.samples]
        self.imgs = list(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _tinyimagenet_imagefolder_split(root, split, wnids, transform):
    split_root = root / split
    if not any((split_root / wnid).is_dir() for wnid in wnids):
        return None
    return TinyImageNetTrainCacheDataset(split_root, wnids, transform=transform)


def _tinyimagenet_dataset(data_dir, split, transform):
    root = _tinyimagenet_root(data_dir)
    wnids = _load_tinyimagenet_wnids(root)
    if split == "train":
        return TinyImageNetTrainCacheDataset(root / "train", wnids, transform=transform)
    if split == "val":
        return TinyImageNetAnnotatedCacheDataset(root / "val", wnids, transform=transform)
    if split == "test":
        imagefolder_dataset = _tinyimagenet_imagefolder_split(root, "test", wnids, transform)
        if imagefolder_dataset is not None:
            return imagefolder_dataset
        return TinyImageNetAnnotatedCacheDataset(root / "test", wnids, transform=transform)
    raise ValueError("TinyImageNet feature caching split must be one of: 'train', 'val', 'test'.")


def _augfree_base_dataset(dataset_name, data_dir, split):
    """Raw split with AUGMENTATION-FREE transforms (ToTensor + Normalize)."""
    if dataset_name not in _NORM:
        supported = ", ".join(sorted(_NORM))
        raise ValueError(f"feature caching supports {supported}, not {dataset_name!r}")
    mean, std = _NORM[dataset_name]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset_name == "tinyimagenet":
        return _tinyimagenet_dataset(data_dir, split, tf)
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
    preprocessing_id = getattr(backbone, "preprocessing_id", "unknown_preprocessing")
    path = cache_dir / f"{dataset_name}_{split}_imagenet_resnet18_{preprocessing_id}.pt"
    base = _augfree_base_dataset(dataset_name, data_dir, split)
    n = len(base)
    feature_dim = _backbone_feature_dim(backbone)
    if path.is_file():
        feats = torch.load(path, map_location="cpu")
        if torch.is_tensor(feats) and feats.ndim == 2 and feats.shape[0] == n and feats.shape[1] == feature_dim:
            print(f"[cache] loaded {split} features from {path} {tuple(feats.shape)}", flush=True)
            return feats
        have = tuple(feats.shape) if torch.is_tensor(feats) else type(feats).__name__
        print(f"[cache] stale cache {path} (have {have}, need ({n}, {feature_dim})); recomputing", flush=True)
    print(f"[cache] computing {split} features for {dataset_name} ({n} images) ...", flush=True)
    feats = compute_features(base, backbone, device)
    if feats.ndim != 2 or feats.shape[1] != feature_dim:
        raise RuntimeError(
            f"feature cache expected backbone output shape (N, {feature_dim}), got {tuple(feats.shape)}"
        )
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
        self._label_to_task_label = {int(label): idx for idx, label in enumerate(self.permutation)}
        self.features = features
        if self.sub_indices and max(self.sub_indices) >= len(self.features):
            raise ValueError(
                f"cached feature count ({len(self.features)}) does not cover dataset index "
                f"{max(self.sub_indices)}"
            )

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        under = self.sub_indices[index]
        label = self._targets[under]
        label = int(label.item()) if torch.is_tensor(label) else int(label)
        return self.features[under], self._label_to_task_label[label]


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
    eval_split = "val" if args.dataset == "tinyimagenet" else "test"
    test_feats = build_or_load_cache(args.dataset, eval_split, args.data_dir, backbone, device, cache_dir)
    train_datasets = [CachedFeatureDataset(d, train_feats) for d in train_datasets]
    test_datasets = [CachedFeatureDataset(d, test_feats) for d in test_datasets]
    # The backbone is no longer needed by the feature-space model: drop it (memory)
    # and flip the net into precomputed-feature mode.
    model.net.frozen_backbone = None
    model.net.features_are_precomputed = True
    print("[cache] feature caching active: adapters/classifier now train on cached "
          "512-d features (backbone freed).", flush=True)
    return train_datasets, test_datasets
