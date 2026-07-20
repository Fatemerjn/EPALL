import os
import copy
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


_CIFAR10_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
]

_CIFAR10_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
]

_CIFAR100_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]

_CIFAR100_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]

_TINYIMAGENET_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

_TINYIMAGENET_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

_CIFAR100_SUPERCLASSES = [
    ("aquatic_mammals", ["beaver", "dolphin", "otter", "seal", "whale"]),
    ("fish", ["aquarium_fish", "flatfish", "ray", "shark", "trout"]),
    ("flowers", ["orchid", "poppy", "rose", "sunflower", "tulip"]),
    ("food_containers", ["bottle", "bowl", "can", "cup", "plate"]),
    ("fruit_and_vegetables", ["apple", "mushroom", "orange", "pear", "sweet_pepper"]),
    ("household_electrical_devices", ["clock", "keyboard", "lamp", "telephone", "television"]),
    ("household_furniture", ["bed", "chair", "couch", "table", "wardrobe"]),
    ("insects", ["bee", "beetle", "butterfly", "caterpillar", "cockroach"]),
    ("large_carnivores", ["bear", "leopard", "lion", "tiger", "wolf"]),
    ("large_man_made_outdoor_things", ["bridge", "castle", "house", "road", "skyscraper"]),
    ("large_natural_outdoor_scenes", ["cloud", "forest", "mountain", "plain", "sea"]),
    ("large_omnivores_and_herbivores", ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"]),
    ("medium_mammals", ["fox", "porcupine", "possum", "raccoon", "skunk"]),
    ("non_insect_invertebrates", ["crab", "lobster", "snail", "spider", "worm"]),
    ("people", ["baby", "boy", "girl", "man", "woman"]),
    ("reptiles", ["crocodile", "dinosaur", "lizard", "snake", "turtle"]),
    ("small_mammals", ["hamster", "mouse", "rabbit", "shrew", "squirrel"]),
    ("trees", ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"]),
    ("vehicles_1", ["bicycle", "bus", "motorcycle", "pickup_truck", "train"]),
    ("vehicles_2", ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]),
]

_DATASET_METADATA = {
    "cifar10": {"num_classes": 10, "image_size": 32, "channels": 3},
    "cifar100": {"num_classes": 100, "image_size": 32, "channels": 3},
    "tinyimagenet": {"num_classes": 200, "image_size": 64, "channels": 3},
}


def get_dataset_metadata(dataset_name):
    try:
        return dict(_DATASET_METADATA[dataset_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def get_evaluation_transform(dataset_name):
    """Return the deterministic test-time transform for a supported dataset."""
    transform_lists = {
        "cifar10": _CIFAR10_TEST_TRANSFORMS,
        "cifar100": _CIFAR100_TEST_TRANSFORMS,
        "tinyimagenet": _TINYIMAGENET_TEST_TRANSFORMS,
    }
    try:
        return transforms.Compose(transform_lists[dataset_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def _resolve_tinyimagenet_root(data_dir):
    candidates = [
        Path(data_dir) / "tiny-imagenet-200",
        Path(data_dir),
    ]
    for candidate in candidates:
        if (
            (candidate / "train").is_dir()
            and (candidate / "val").is_dir()
            and (candidate / "wnids.txt").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "TinyImageNet not found. Expected data/tiny-imagenet-200/ with train/, val/, and wnids.txt."
    )


def _load_tinyimagenet_wnids(root):
    with open(root / "wnids.txt", "r", encoding="utf-8") as handle:
        wnids = [line.strip() for line in handle if line.strip()]
    expected_classes = _DATASET_METADATA["tinyimagenet"]["num_classes"]
    if len(wnids) != expected_classes:
        raise ValueError(f"Expected {expected_classes} TinyImageNet classes in wnids.txt, found {len(wnids)}.")
    return wnids


class TinyImageNetTrainDataset(datasets.ImageFolder):
    def __init__(self, root, wnids, transform=None):
        self.wnids = list(wnids)
        self._wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        super().__init__(str(root), transform=transform)
        self.targets = [target for _, target in self.samples]

    def find_classes(self, directory):
        missing = [wnid for wnid in self.wnids if not os.path.isdir(os.path.join(directory, wnid))]
        if missing:
            raise FileNotFoundError(
                f"Missing TinyImageNet train class directories under {directory}: {missing[:5]}"
            )
        return list(self.wnids), dict(self._wnid_to_idx)


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, wnids, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.loader = default_loader
        self.classes = list(wnids)
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.classes)}

        annotations_path = self.root / "val_annotations.txt"
        images_dir = self.root / "images"
        if not annotations_path.is_file():
            raise FileNotFoundError(f"TinyImageNet validation annotations not found: {annotations_path}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"TinyImageNet validation image directory not found: {images_dir}")

        self.samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    raise ValueError(f"Unknown TinyImageNet wnid '{wnid}' in {annotations_path}.")
                image_path = images_dir / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(f"TinyImageNet validation image not found: {image_path}")
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


def get_cifar100_superclass_tasks(dataset):
    label_to_index = {label: idx for idx, label in enumerate(dataset.classes)}
    labels_per_task = []
    for _, class_names in _CIFAR100_SUPERCLASSES:
        labels_per_task.append([label_to_index[name] for name in class_names])
    permutation = [label for task_labels in labels_per_task for label in task_labels]
    return labels_per_task, permutation


class SubDataset(Dataset):
    def __init__(self, original_dataset, sub_labels, permutation):
        super().__init__()
        self.dataset = original_dataset
        self.permutation = permutation
        self.sub_indices = []
        for index in range(len(self.dataset)):
            label = self.dataset.targets[index]
            if label in sub_labels:
                self.sub_indices.append(index)

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        return sample[0], self.permutation.index(sample[1])


def make_augmentation_free_evaluation_view(dataset, dataset_name):
    """Clone a task dataset with training augmentation disabled for audits.

    Cached feature datasets are already deterministic and are returned unchanged.
    """
    if not isinstance(dataset, SubDataset):
        return dataset
    base = copy.copy(dataset.dataset)
    if not hasattr(base, "transform"):
        return dataset
    view = copy.copy(dataset)
    base.transform = get_evaluation_transform(dataset_name)
    view.dataset = base
    return view


def get_task_datasets(args):
    T = args.n_tasks
    CPT = args.class_per_task
    dataset_metadata = get_dataset_metadata(args.dataset)

    data = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
    }
    train_transform = {
        'cifar10': _CIFAR10_TRAIN_TRANSFORMS,
        'cifar100': _CIFAR100_TRAIN_TRANSFORMS,
        'tinyimagenet': _TINYIMAGENET_TRAIN_TRANSFORMS,
    }
    test_transform = {
        'cifar10': _CIFAR10_TEST_TRANSFORMS,
        'cifar100': _CIFAR100_TEST_TRANSFORMS,
        'tinyimagenet': _TINYIMAGENET_TEST_TRANSFORMS,
    }

    if args.dataset == 'tinyimagenet':
        if T * CPT > dataset_metadata["num_classes"]:
            raise ValueError("TinyImageNet requires --class_per_task * --n_tasks <= 200.")
        tinyimagenet_root = _resolve_tinyimagenet_root(args.data_dir)
        wnids = _load_tinyimagenet_wnids(tinyimagenet_root)
        train = TinyImageNetTrainDataset(
            tinyimagenet_root / 'train',
            wnids=wnids,
            transform=transforms.Compose(train_transform[args.dataset]),
        )
        test = TinyImageNetValDataset(
            tinyimagenet_root / 'val',
            wnids=wnids,
            transform=transforms.Compose(test_transform[args.dataset]),
        )
        print(
            f"[INFO] Loaded TinyImageNet: num_classes={dataset_metadata['num_classes']} "
            f"train_size={len(train)} val_size={len(test)} tasks={T} classes_per_task={CPT}"
        )
    else:
        train = data[args.dataset](args.data_dir, train=True, download=True,
                                   transform=transforms.Compose(train_transform[args.dataset]))
        test = data[args.dataset](args.data_dir, train=False, download=True,
                                  transform=transforms.Compose(test_transform[args.dataset]))

    cifar100_split = getattr(args, "cifar100_split", "superclass")
    if args.dataset == "cifar100" and cifar100_split == "superclass":
        # Overlap-heavy setting: semantically coherent superclass tasks (5 fine
        # classes per superclass). See docs/standard_vs_overlap.md.
        labels_per_task, permutation = get_cifar100_superclass_tasks(train)
        if CPT != 5:
            raise ValueError("CIFAR-100 superclass tasks require --class_per_task 5.")
        if T > len(labels_per_task):
            raise ValueError("CIFAR-100 superclass tasks require --n_tasks <= 20.")
        labels_per_task = labels_per_task[:T]
        permutation = [label for task_labels in labels_per_task for label in task_labels]
    elif args.dataset == "tinyimagenet":
        permutation = np.random.permutation(np.arange(dataset_metadata["num_classes"]))[:T * CPT]
        labels_per_task = [list(permutation[task_id * CPT:(task_id + 1) * CPT]) for task_id in range(T)]
    else:
        # cifar10, OR standard Split-CIFAR-100 (--cifar100_split standard):
        # random disjoint class splits over T*CPT classes (literature-comparable).
        permutation = np.random.permutation(np.arange(T * CPT))
        labels_per_task = [list(permutation[task_id * CPT:(task_id + 1) * CPT]) for task_id in range(T)]

    print("Labels per task: ", labels_per_task)

    train_datasets, test_datasets = [], []
    for task_id, labels in enumerate(labels_per_task):
        train_datasets.append(SubDataset(train, labels, list(permutation)))
        test_datasets.append(SubDataset(test, labels, list(permutation)))

    return train_datasets, test_datasets
