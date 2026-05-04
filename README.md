# MG-MTTA: Majorization-Guided Multimodal Test-Time Adaptation

This repository contains the research code for **Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift**.

MG-MTTA studies how vision-language models behave under modality-specific distribution shifts, including visual corruption, text prompt shift, and joint modality shift. The method performs lightweight test-time adaptation while keeping the backbone model frozen.

> Paper status: manuscript under submission / under review.

## Highlights

- Vision-Language Model test-time adaptation
- Modality-specific distribution shift
- ImageNet-C evaluation
- CLIP / OpenCLIP-style backbone
- Lightweight adaptation with frozen backbone
- Single-run and multi-seed evaluation scripts

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

You can also override these paths from the command line.

## Run

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

## Results

Experimental results will be updated after the paper review process.

| Setting | Dataset | Backbone | Method | Metric |
|---|---|---|---|---|
| Visual Shift | ImageNet-C | CLIP / OpenCLIP | Source | TODO |
| Visual Shift | ImageNet-C | CLIP / OpenCLIP | MG-MTTA | TODO |
| Text Shift | ImageNet-style evaluation | CLIP / OpenCLIP | MG-MTTA | TODO |
| Joint Shift | ImageNet-C + Text Shift | CLIP / OpenCLIP | MG-MTTA | TODO |

## Notes

The following files are intentionally excluded:

- datasets
- checkpoints
- logs
- outputs
- cache files
- temporary files
- unrelated experiment scripts

For third-party dependency notes, please see `THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@misc{chen2026mgmtta,
  title  = {Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift},
  author = {Chen, Lixian and others},
  year   = {2026},
  note   = {Manuscript under submission}
}
```

## Contact

Lixian Chen  
Email: lix.chen41@gmail.com

## License

This project is released under the license provided in `LICENSE`.
