# MG-MTTA: Majorization-Guided Multimodal Test-Time Adaptation

Official PyTorch implementation of:

**Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift**

**Accepted by ACM Multimedia 2026 (MM '26).**

MG-MTTA addresses the **confidence–reliability mismatch** in multimodal test-time adaptation, where an unreliable modality may dominate fusion and produce confident but incorrect predictions.

## Highlights

- Reliability-aware multimodal fusion
- Majorization-guided posterior analysis
- Reduced wrong-more-confident failures under modality-specific shifts

## Implementation Note

MG-MTTA does not use two independent CLIP encoders. Instead, the two prediction streams are constructed from the same CLIP image representation.

The visual-reference stream uses the clean text bank, while the text-side stream uses the shifted text bank when textual perturbation is enabled. Visual shift is introduced through corrupted image inputs, while textual shift is introduced through the prompt bank.

This gives the following settings:

- **Textual shift:** clean image + clean/shifted text banks
- **Visual shift:** corrupted image + clean text bank
- **Joint shift:** corrupted image + clean/shifted text banks

Under visual-only shift, textual perturbation is disabled and the clean text bank is retained as the textual reference.

## Main Results

Top-1 accuracy (%) under modality-specific shifts.

| Method | Textual Shift | Joint Shift |
|---|---:|---:|
| Source-only | 57.97 | 21.68 |
| Entropy-only | 55.40 | 21.68 |
| Entropy + Diversity | 55.38 | 21.76 |
| **MG-MTTA** | **66.51** | **26.27** |

### Strongest Textual Stress Probe

| Method | Textual L5 | Joint L5 |
|---|---:|---:|
| Source-only | 25.48 | 9.66 |
| Entropy-only | 22.11 | 9.40 |
| Entropy + Diversity | 22.14 | 9.39 |
| **MG-MTTA** | **65.88** | **26.13** |

## Installation

```bash
conda create -n mgmtta python=3.10 -y
conda activate mgmtta
pip install -r requirements.txt
```

Prepare ImageNet-C and the OpenCLIP checkpoint:

```text
data/
└── ImageNet-C/

checkpoints/
└── open_clip/
    └── open_clip_model.safetensors
```

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python test_time.py \
  --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml
```

Custom paths:

```bash
CUDA_VISIBLE_DEVICES=0 python test_time.py \
  --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  DATA_DIR /path/to/data \
  MODEL.WEIGHTS /path/to/open_clip_model.safetensors
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python test_time.py \
  --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
  CORRUPTION.NUM_EX 8 \
  TEST.BATCH_SIZE 8 \
  OPTIM.STEPS 1
```

Multi-seed evaluation:

```bash
bash scripts/run_mg_mtta_multi_seed.sh
```

## Citation

```bibtex
@inproceedings{chen2026mgmtta,
  title     = {Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift},
  author    = {Chen, Lixian and others},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836429}
}
```

## License

This project is released under the license provided in [LICENSE](LICENSE).
