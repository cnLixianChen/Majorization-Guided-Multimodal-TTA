# MG-MTTA: Majorization-Guided Multimodal Test-Time Adaptation

This repository contains the research code for **Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift**.

MG-MTTA studies test-time adaptation for vision-language models under modality-specific distribution shifts, including visual corruption, semantics-preserving textual shift, and joint visual-textual shift. Instead of blindly minimizing fused-posterior entropy, MG-MTTA uses reliability-aware gate regularization to reduce over-confident but incorrect adaptation caused by unreliable modality dominance.

> Paper status: manuscript under submission / under review.

## Highlights

- Vision-language model test-time adaptation
- Modality-specific distribution shift
- Reliability-aware multimodal fusion
- Majorization-guided posterior de-mixing
- Lightweight gate / adapter update with frozen CLIP backbone
- ImageNet-C, CIFAR-100C, textual shift, and joint-shift evaluation

## Method Overview

MG-MTTA keeps the pretrained vision-language backbone frozen and updates only a lightweight gate or adapter during test-time adaptation.

The adaptation objective combines:

- fused-posterior entropy minimization
- reliability-aware gate regularization
- batch-level diversity regularization

The reliability prior is built from:

- anchor-based modality consistency
- cross-modality conflict

This design aims to prevent entropy minimization from amplifying an unreliable modality under asymmetric multimodal shift.

## Project Structure

```text
MG-MTTA/
├── test_time.py
├── conf.py
├── cfgs/
│   └── imagenet_c/
│       └── mg_mtta_imagenet_c.yaml
├── methods/
├── models/
├── mydatasets/
├── prompts/
├── augmentations/
├── utils/
├── scripts/
│   └── run_mg_mtta_multi_seed.sh
├── requirements.txt
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Installation

```bash
conda create -n mgmtta python=3.10 -y
conda activate mgmtta
pip install -r requirements.txt
```

Please make sure CUDA-enabled PyTorch is correctly installed in your environment.

## Required Assets

Datasets and checkpoints are not included in this repository.

Please prepare the following files:

```text
data/
└── ImageNet-C/
    └── ...

checkpoints/
└── open_clip/
    └── open_clip_model.safetensors
```

The default config uses:

```yaml
DATA_DIR: ./data
MODEL.WEIGHTS: ./checkpoints/open_clip/open_clip_model.safetensors
```

You can override these paths from the command line.

## Quick Start

Single run:

```bash
CUDA_VISIBLE_DEVICES=0 python test_time.py \
  --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml
```

Run with custom paths:

```bash
CUDA_VISIBLE_DEVICES=0 python test_time.py \
  --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  DATA_DIR /abs/path/to/data \
  MODEL.WEIGHTS /abs/path/to/open_clip_model.safetensors \
  TEST.BATCH_SIZE 64 \
  TEST.NUM_WORKERS 4
```

Tiny smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python test_time.py \
  --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  CORRUPTION.NUM_EX 8 \
  TEST.BATCH_SIZE 8 \
  TEST.NUM_WORKERS 2 \
  OPTIM.STEPS 1
```

Multi-seed evaluation:

```bash
bash scripts/run_mg_mtta_multi_seed.sh
```

## Main Results

### Visual-Shift Anchoring Results

Mean top-1 accuracy (%) over 15 corruption types at severity level 5.

| Method | CIFAR-100C | ImageNet-C |
|---|---:|---:|
| Source-only | 35.80 | 25.45 |
| TENT-Visual | 37.96 | 25.64 |
| SAR | 41.42 | 29.70 |
| TPT | 36.20 | 24.90 |
| VTE | 35.00 | 25.60 |
| BATCLIP | 42.21 | 30.70 |
| Entropy-only | 37.96 | 26.39 |
| Entropy + Diversity | 38.10 | 26.29 |
| MG-MTTA | 38.11 | 26.32 |

MG-MTTA is mainly designed for modality-specific multimodal shift. On visual-only corruption, it remains competitive with matched entropy-based internal baselines under the same lightweight adaptation protocol.

### Main Multimodal Results

Top-1 accuracy (%) on the unified ImageNet-based benchmark. Textual and Joint are averaged over textual stress levels L1-L4. Joint additionally uses visual corruption at severity level 5.

| Method | Textual Shift Acc. | Textual Δ | Joint Shift Acc. | Joint Δ |
|---|---:|---:|---:|---:|
| Source-only | 57.97 | 0.00 | 21.68 | 0.00 |
| Entropy-only | 55.40 | -2.57 | 21.68 | 0.00 |
| Entropy + Diversity | 55.38 | -2.59 | 21.76 | +0.08 |
| MG-MTTA | 66.51 | +8.54 | 26.27 | +4.59 |

The main gain appears under textual and joint shift, where asymmetric modality perturbations directly affect multimodal fusion.

### Strongest Textual Stress Probe

Top-1 accuracy (%) under the L5 strongest probe.

| Method | Textual L5 Acc. | Textual Δ | Joint L5 Acc. | Joint Δ |
|---|---:|---:|---:|---:|
| Source-only | 25.48 | 0.00 | 9.66 | 0.00 |
| Entropy-only | 22.11 | -3.37 | 9.40 | -0.26 |
| Entropy + Diversity | 22.14 | -3.34 | 9.39 | -0.27 |
| MG-MTTA | 65.88 | +40.40 | 26.13 | +16.47 |

This stress test shows that entropy minimization alone can fail under severe modality-specific shift, while reliability-aware adaptation improves recovery.

### Additional Backbone Results

Top-1 accuracy (%) using CLIP ViT-B/32.

| Method | Visual | Textual | Joint |
|---|---:|---:|---:|
| Source-only | 24.29 | 52.26 | 37.61 |
| Entropy-only | 22.31 | 49.92 | 36.81 |
| Entropy + Diversity | 22.41 | 49.97 | 36.87 |
| MG-MTTA | 22.41 | 65.61 | 46.16 |

The multimodal advantage transfers to another CLIP backbone, especially under textual and joint shift.

## Notes on Reproducibility

The following files are intentionally excluded from this repository:

- datasets
- checkpoints
- logs
- outputs
- cache files
- temporary files
- unrelated experiment scripts
- unrelated configs

For third-party dependency notes, please see `THIRD_PARTY_NOTICES.md`.


## Contact

Lixian Chen  
Email: 3123003175@mail2.gdut.edu.cn

## License

This project is released under the license provided in `LICENSE`.
