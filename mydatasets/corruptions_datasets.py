
import os
import json
import torch
import logging
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder

try:
    from robustbench.loaders import CustomCifarDataset
except ImportError:
    # Newer robustbench releases may not export CustomCifarDataset.
    # Keep a minimal local fallback for CIFAR-C evaluation compatibility.
    import numpy as np
    from PIL import Image

    class CustomCifarDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform=None):
            super().__init__()
            self.samples = samples
            self.transform = transform

        def __getitem__(self, index):
            img, label, domain = self.samples[index]
            if self.transform is not None:
                img = Image.fromarray(np.uint8(img * 255.0)).convert("RGB")
                img = self.transform(img)
            else:
                img = torch.tensor(img.transpose((2, 0, 1)))
            return img, torch.tensor(label), domain

        def __len__(self):
            return len(self.samples)

logger = logging.getLogger(__name__)


def _get_submission_root() -> str:
    # classification_submission_anonymous root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_canonical_imagenet_id_list_path() -> str:
    """Resolve the canonical ImageNet val 50k id list from local submission tree.

    This avoids dependence on the process working directory so all methods
    always use the same protocol file.
    """
    return os.path.join(_get_submission_root(), "mydatasets", "imagenet_list", "imagenet_val_ids_50k.txt")


def _resolve_robustbench_data_path(filename: str) -> str:
    return os.path.join(_get_submission_root(), "mydatasets", "assets", filename)


def _resolve_imagenetc_severity_dir(data_dir: str, corruption: str, severity: int) -> str:
    """Resolve ImageNet-C path for a corruption/severity pair.

    Supports both layouts:
    1) <data_dir>/<corruption>/<severity>
    2) <data_dir>/extracted/<corruption>/<severity>
    """
    candidates = [
        os.path.join(data_dir, corruption, str(severity)),
        os.path.join(data_dir, "extracted", corruption, str(severity)),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Could not find ImageNet-C corruption directory for corruption='{corruption}', severity={severity}. "
        f"Checked: {candidates}"
    )


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = _resolve_imagenetc_severity_dir(
        data_dir=data_dir,
        corruption=corruptions_seq[0],
        severity=severity,
    )
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(_resolve_robustbench_data_path("imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # prefer the full 50k validation id list for sampling when requested
            file_path = _resolve_canonical_imagenet_id_list_path()
            # require the dataset-specific 50k validation id list to be present
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Required file not found: {file_path}. Please provide the canonical "
                    f"imagenet validation id list at this path so all methods use the same protocol."
                )
        else:
            # use robustbench default test list when not requesting a subsample
            file_path = _resolve_robustbench_data_path("imagenet_test_image_ids.txt")

        # load file containing file ids (strip newlines)
        with open(file_path, 'r') as f:
            fnames = [ln.strip() for ln in f.readlines() if ln.strip()]

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = _resolve_imagenetc_severity_dir(
                data_dir=data_dir,
                corruption=cor,
                severity=severity,
            )
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test
