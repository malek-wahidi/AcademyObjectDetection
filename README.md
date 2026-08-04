# Academy Object Detection

A minimal PyTorch baseline for detecting warehouse objects in the
[LOCO dataset](https://github.com/tum-fml/loco).

## Project objective

Improve the baseline detector while balancing three goals:

- increase mAP@0.5;
- reduce the total number of model parameters;
- reduce GFLOPs per image.

Your final model should be Pareto-optimal: it is dominated when another submission has no lower
mAP@0.5, no higher total parameter count or GFLOPs, and is strictly better on at least one axis.

## Rules

- Train and fine-tune only on subsets 2, 3, and 5. Make a fixed validation holdout from them for tuning and checkpoint selection.
- Reserve subsets 1 and 4 together for final evaluation. Never train, fine-tune, or tune on them.
- Pretrained models are allowed only when fine-tuned.
- Your submitted repository must reproduce your final reported metrics when cloned and rerun.
- The submitted code must pass the repository's unit tests and Ruff checks.
- Do not modify, bypass, or otherwise tamper with the existing evaluation metrics.

## Setup

```bash
git clone https://github.com/malek-wahidi/AcademyObjectDetection.git
cd AcademyObjectDetection
uv sync
bash scripts/download_loco.sh
```

The download script stores the dataset in `dataset/`, which is the default location in
`config.yaml`. To use another location, pass it to the script and update `data.raw_dir`:

```bash
bash scripts/download_loco.sh /path/to/data
```

The data directory must contain the `rgb/` annotations and `subset-*/` image directories.

For a hosted GPU, upload `AcademyObjectDetection.ipynb` to Colab or Kaggle and follow the
notebook setup.

## Train

```bash
uv run train.py
```

Choose your final checkpoint using the validation holdout. The baseline reserves one image in every five
from each training subset for validation and saves its best checkpoint to `runs/baseline/weights/best.pt`.

## Evaluate

```bash
uv run eval.py --weights runs/baseline/weights/best.pt
```

After selecting a checkpoint, run evaluation once on subsets 1 and 4. It reports mAP@0.5, parameter counts, GFLOPs per image, and batch-one latency. Submissions
are compared using mAP@0.5, total parameters, and GFLOPs. Latency is informational because it
depends on the hardware.

## Inspect predictions

```bash
uv run predict.py --weights runs/baseline/weights/best.pt
```

Prediction images are saved to `runs/predictions/`.

## Baseline results

Stock `fasterrcnn_mobilenet_v3_large_fpn` with LOCO's five classes plus background:

| mAP@0.5 | Parameters | GFLOPs |
| ------- | ---------- | ------ |
| 0.2547  | 18,950,729 | 23.825 |

Pareto comparison requires final-evaluation mAP@0.5 at least as high as the recorded baseline mAP.

## Checks

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
```
