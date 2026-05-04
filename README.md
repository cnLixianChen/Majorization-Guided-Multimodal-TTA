# MG-MTTA Anonymous Reproduction Package

This package contains the minimal code required to run the MG-MTTA ImageNet-C experiment in anonymous review mode.

## Main Entry

- Python entry: `test_time.py`
- Primary config: `cfgs/imagenet_c/mg_mtta_imagenet_c.yaml`
- Optional launcher: `scripts/run_mg_mtta_multi_seed.sh`

## Included Scope

Included runtime modules only:

- `methods/` (MG-MTTA path only)
- `models/`
- `mydatasets/`
- `prompts/`
- `augmentations/`
- `utils/`
- `conf.py`, `test_time.py`, `cfgs/imagenet_c/mg_mtta_imagenet_c.yaml`
- `LICENSE`, `requirements_anonymous.txt`

Not included:

- datasets, checkpoints, logs, outputs, figures, cache, temporary files
- unrelated experiment scripts and non-MG-MTTA configs

## Setup

```bash
pip install -r requirements_anonymous.txt
```

Tested runtime dependency versions are pinned in `requirements_anonymous.txt`.

Important environment note:

- This package was tested in an existing Python environment with preinstalled CUDA-enabled PyTorch.
- If your environment already provides `torch`/`torchvision`, keep them compatible with your local CUDA stack.

## Required External Assets

Place external assets under these paths (or override from CLI):

- data root: `./data`
  - expected ImageNet-C path: `./data/ImageNet-C/...`
- model weights: `./checkpoints/open_clip/open_clip_model.safetensors`

Default path behavior:

- `cfgs/imagenet_c/mg_mtta_imagenet_c.yaml` defaults to:
  - `DATA_DIR: ./data`
  - `MODEL.WEIGHTS: ./checkpoints/open_clip/open_clip_model.safetensors`

## Run

Single run:

```bash
python test_time.py --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml
```

Override paths and runtime options:

```bash
python test_time.py --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  DATA_DIR ./data \
  MODEL.WEIGHTS ./checkpoints/open_clip/open_clip_model.safetensors \
  TEST.BATCH_SIZE 64 TEST.NUM_WORKERS 4
```

Or use the launcher:

```bash
bash scripts/run_mg_mtta_multi_seed.sh
```

## Quick Sanity Checks

Parser/import sanity check:

```bash
python test_time.py --help
```

Config loading sanity check:

```bash
python -c "from conf import reset_cfg, merge_from_file, cfg; reset_cfg(); merge_from_file('cfgs/imagenet_c/mg_mtta_imagenet_c.yaml'); print(cfg.MODEL.ADAPTATION, cfg.CORRUPTION.DATASET, cfg.MODEL.USE_CLIP)"
```

Tiny end-to-end smoke run (requires external data and checkpoint to be available):

```bash
CUDA_VISIBLE_DEVICES=0 BATCLIP_CUDA_INDEX=0 \
python test_time.py --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  CORRUPTION.NUM_EX 8 TEST.BATCH_SIZE 8 TEST.NUM_WORKERS 2 OPTIM.STEPS 1
```

If assets are not under default relative paths, override explicitly:

```bash
python test_time.py --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  DATA_DIR /abs/path/to/data \
  MODEL.WEIGHTS /abs/path/to/open_clip_model.safetensors \
  CORRUPTION.NUM_EX 8 TEST.BATCH_SIZE 8 TEST.NUM_WORKERS 2 OPTIM.STEPS 1
```

## Anonymous Handling

- Absolute local paths were replaced with relative placeholders.
- In-file GitHub/Gitee URLs were redacted.
- Author/project-identifying README and export artifacts were not copied.

For third-party and license notes, see `THIRD_PARTY_NOTICES.md`.
