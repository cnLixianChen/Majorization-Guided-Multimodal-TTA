# THIRD_PARTY_NOTICES

This package contains a minimal anonymous reproduction bundle and references third-party software dependencies required at runtime.

## 1) Package Scope

- This archive keeps only the minimal runtime path required for the proposed method under review.
- Unrelated experiment artifacts, logs, caches, datasets, checkpoints, and auxiliary exports are not included.

## 2) Included License File

- `LICENSE` is preserved from the source project and included in this archive.

## 3) Retained In-Tree Components with Adaptation Notes

The following files are retained because they are runtime-critical for the submitted method and contain adapted implementation details:

- `models/custom_clip.py`
- `models/resnet26.py`
- `utils/registry.py`
- `mydatasets/data_loading.py`
- `mydatasets/cls_names.py`
- `mydatasets/prompts.py`
- `mydatasets/imagenet_subsets.py`
- `mydatasets/imagenet_d_utils.py`
- `mydatasets/imagenet_dict.py`

## 4) External Runtime Dependencies (Not Vendored)

The following packages are required at runtime and are declared in `requirements.txt`:

- `torch`, `torchvision`
- `numpy`, `packaging`, `Pillow`, `requests`
- `yacs`, `iopath`
- `timm`, `open_clip_torch`
- `robustbench`, `webdataset`
- `autoattack`

Tested versions include:

- `timm==1.0.9`
- `open_clip_torch==2.32.0`
- `robustbench==1.1.1`
- `webdataset==1.0.2`
- `yacs==0.1.8`
- `iopath==0.1.10`

Compatibility note:

- `robustbench` imports may rely on `pkg_resources`; accordingly, this package pins `setuptools<81` in `requirements.txt`.

## 5) Explicitly Not Included

- A local in-tree copy of `robustbench` is not included in this archive.
- External datasets, checkpoints, logs, caches, and generated outputs are not included.
- Only the following small runtime data artifacts are retained under `mydatasets/assets/`:
  - `imagenet_class_to_id_map.json`
  - `imagenet_test_image_ids.txt`

## 6) Third-Party Delivery Mode

Third-party libraries such as `robustbench`, `autoattack`, and other external dependencies are not vendored in this archive and are instead installed via `requirements.txt`.