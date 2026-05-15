# Thesis Multi-Task Fire Detection

Multi-task YOLOv8 training pipeline for **fire detection** (bounding boxes) plus **environment classification** (indoor vs outdoor).

## Project structure (key files)

- `train.py` — custom `MultiTaskTrainer` and training entrypoint.
- `dataset.py` — `MultiTaskFireDataset` + `collate_fn`.
- `download_data.py` — Roboflow download + folder routing.
- `test_model.py` — inference/visualization script.
- `fire_data.yaml` — dataset config used by Ultralytics.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `download_data.py` uses Roboflow. Replace the API key in `download_data.py` with your own key before downloading datasets.

## Data download

Run the downloader to fetch and organize the datasets into the expected multi-task folder structure:

```bash
python download_data.py
```

Expected data layout after download:

```text
fire_multitask_data/
  train/
    indoor/
      images/
      labels/
    outdoor/
      images/
      labels/
  val/
    indoor/
      images/
      labels/
    outdoor/
      images/
      labels/
```

## Training

The default training script **expects an existing checkpoint**:

- `train.py` loads `runs/detect/train-69/weights/last.pt` by default.
- If you don’t have that file, update `ckpt_path` and the `overrides["model"]` value in `train.py` to point to a valid checkpoint or pretrained weights.

Run training:

```bash
python train.py
```

## Inference / demo

`test_model.py` loads a weights file and runs a prediction on a single image, then displays results with OpenCV.

Update these values in `test_model.py` as needed:

- `source_path` (default: `outdoor.jpg`)
- `weights_path` (default: `runs/detect/train-45/weights/best.pt`)

Then run:

```bash
python test_model.py
```

## Dataset loader sanity check (optional)

You can quickly validate that the dataset is loading correctly:

```bash
python dataset.py
```

This prints tensor shapes and bounding box batches for one mini-batch.
