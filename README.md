---
title: Dental Teeth Segmentation
emoji: 🦷
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.0.0"
app_file: app/gradio_demo.py
pinned: true
---

# Dental Teeth Segmentation — Mask R-CNN

Instance segmentation of individual teeth in dental panoramic X-ray images using a fine-tuned **Mask R-CNN** (ResNet-50+FPN, torchvision,  COCO pre-trained weights).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Configuration](#6-configuration)
7. [Setup & Installation](#7-setup--installation)
8. [Usage](#8-usage)
9. [API](#9-api)
10. [Docker](#10-docker)
11. [Results](#11-results)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Design Decisions](#13-design-decisions)
14. [References](#14-references)

---

## 1. Project Overview

This project uses torchvision Mask RCNN for **dental tooth instance segmentation**. Each tooth in a panoramic X-ray is detected and segmented independently, producing a per-tooth binary mask, bounding box, confidence score and FDI class label.

**What was done:**
- Implemented a custom `TeethDataset` loader that parses COCO JSON annotations
- Applied **CLAHE** contrast enhancement as a preprocessing step to improve tooth boundary visibility
- Fine-tuned the whole network (starting from COCO weights) on the AKUDENTAL dataset (333 panoramic X-ray images)
- Evaluated with pycocotools - COCO style mAP@50 and mAP@50-95
- Exposed predictions via a **FastAPI** REST endpoint
- Built a **Gradio** demo with sample images from validation set and unseen test images
- Containerised the full stack with Docker
- Weights hosted on Huggig Face Hub, demo deployed on Hugging Face Spaces

---

## 2. Project Structure

```
dental-segmentation/
│
├── app/
│   └── main.py                  # FastAPI inference server
|   └── gradio_demo.py           # Gradio application
│
├── configs/
│   └── model_config.py          # Model configurations
│
├── models/
│   └── teeth_segmentation.py    # TeethDataset, train(), build_model,
predict(), evaluate(), load_inference_model()
│
├── utils/
│   ├── preprocessing.py         # Image loading, CLAHE
│   ├── visualization.py         # Mask overlays, prediction plots, training curves
│   └── metrics.py               # IoU, mAP@50/75, MAE on tooth counts
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Dataset exploration and annotation analysis
│   ├── 02_Training.ipynb        # Interactive training with progress display
│   └── 03_Evaluation.ipynb      # Validation metrics and prediction visualisations
│
├── data/
│   ├── images/                  # Original images (gitignored)
│   ├── processed/               # Contrast-enhanced, resized images (gitignored)   
│   └── annotations/             # Raw VIA JSON exports
│   │   │   
│   │   └── akudental_instances.json # Original .json file of the images
│   │   └── train.json           Genearted in Preprocessing notebook, after splitting the original .json for purposes of training and val
|   |   └── val.json
│   │  
|   └── test/                    # Contains never before seen images for testing
│
├── outputs/
│   ├── logs/                    # Training logs (gitignored)
│   ├── results/ 
|   |     └──maskrcnn_torch                # best.pth and last.pth, training_history.csv - loss data csv
│   └── visualizations/          # Saved prediction overlays
│
├── tests/
│   └── test_utils.py            # Unit tests for preprocessing and metrics
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Dataset

### Source
Dental panoramic X-ray images (OPG — Orthopantomogram). Used the AKUDENTAL dataset, collected from **https://github.com/melihoz/AKUDENTAL/tree/main/AKUDENTAL** 

**333 panoramic X-rays** annotated with per-tooth polygon segmentation masks in COCO JSON format.



### Annotation format (COCO JSON)

```json
{
  "images": [{"id": 1, "file_name": "image_001.jpg", "width": 1976, "height": 976}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 11,
      "segmentation": [[120, 80, 145, 82, 148, 110, 122, 108]],
      "bbox": [120, 80, 28, 30],
      "area": 720
    }
  ],
  "categories": [{"id": 11, "name": "11"}, {"id": 12, "name": "12"}, ...]
}
```
Categories follow the **FDI numbering system**:
11-18 Upper right
21-28 Upper left
31-38 Lower left
41-48 Lower right
Additional: Bridge, Filling-Crown, Implant

### Split
- **Train:** 80% (~266 images)
- **Validation:** 20% (~67 images)
- Fixed random seed (42) for reproducibility

### Preprocessing
1. **Load** — grayscale X-rays are converted to 3-channel RGB (required by ResNet backbone)
2. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation on the L channel (LAB colour space) to improve local contrast at tooth boundaries
3. **Resize** — handled by torchvision

---

## 4. Model Architecture

**Mask R-CNN** with:
- **Backbone:** ResNet-50 + Feature Pyramid Network (FPN)
- **Region Proposal Network (RPN):** generates tooth candidate regions
- **ROI Align:** extracts fixed-size features per proposal
- **Heads:** classification, bounding-box regression, and mask prediction heads

Pre-trained on **MS-COCO** (80 classes). The final heads are replaced and retrained for 1(binary) or 35 FDI tooth classes.

### Key Configuration (**model_config.py**)

| Parameter | Value | Reason |
|---|---|---|
| `NUM_CLASSES` | 2 or 36 | background + 1 tooth(binary) or 35 FDI classes |
| `IMAGE_MIN_DIM` | 800 | torchvision default |
| `IMAGE_MAX_DIM` | 1333 | torchvision default |
| `RPN_ANCHOR_SCALES` | (32,64,128,256,512) | standard FPN anchors |
| `CONF_THRESHOLD` | 0.3 | lower than default to retain partially occluded teeth |
| `NMS_THRESHOLD` | 0.5 | higher than default, because adjescent teeth naturally overlap |
| `IMAGES_PER_GPU` | 2 | uses NVIDIA RTX 4090 for training |

---

## 5. Training Strategy

Single stage fine-tuning, using SGD.

**Optimizer**: SGD with lr = 0.005, momentum = 0.9 and weight_decay=0.0005
**Scheduler**: StepLR with step_size=10, gamma = 0.1 (LR drops 10x every 10 epochs)
**Epochs**: 30 (optimal for 200-300 images)
**Batch size**: 2

All layers are trained from the first epoch. Weights are initialized by COCO-pretrained weights. COCO-pretrained backbone provides strong low-level features (edges,texture), so a two-stage freeze/unfreeze is not necessary with torchvsion's implementation.

Best checkpoint is chosen based on the lowest validation loss, and it is saved inside **outputs/results/maskrcnn_torch/best.pth**
---

## 6. Configuration

All model parameters are in `configs/model_config.py`:
Switch between binary and FDI mode by changing flag **BINARY** to **True** (if 1 class: "tooth") and **False** for 35 FDI classes.

---

## 7. Setup & Installation

### Requirements
- Python 3.10
- CUDA 11.7+ / cuDNN 8 (for GPU training)
- 8 GB+ GPU VRAM recommended

### Steps

```bash
# 1. Clone this repo
git clone 
cd dental-teeth-segmentation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Preprare data by running 01_Preprocessing.ipynb
# Places images into data/processed
```

---

## 8. Usage

### Training (CLI) - from your local terminal

```bash
python models/teeth_segmentation.py train
```

### Evaluate(COCO mAP on validation set)

```bash
python models/teeth_segmentation.py evaluate
```

### Inference on a Single Image

```bash
python models/teeth_segmentation.py predict \
    --image data/test/012.jpg
```
Output is saved to **outputs/visualizations/**

### Batch Evaluation

```bash
jupyter notebook notebooks/03_Evaluation.ipynb
```

---

## 9. API

Start the inference server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Upload an image, receive segmentation results |


**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@xray.jpg" \
    | python -m json.tool
```

**Example response:**
```json
{
  "n_teeth": 28,
  "boxes": [[y1, x1, y2, x2], ...],
  "scores": [0.97, 0.95, 0.92, ...],
  "masks_b64": ["iVBORw0KGgo...", ...],
  "overlay_b64": "/9j/4AAQSkZ..."
}
```

Interactive docs at: `http://localhost:8000/docs`

---

## 10. Docker

```bash
# Build
docker build -t dental-maskrcnn-torch:latest .

# Run API (GPU, with mounted weights and data)
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/data:/app/data \
    dental-teeth-seg:latest

#Run Gradio demo
docker run --gpus all -p 7860:7860 \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/data:/app/data \
    dental-teeth-seg:latest \
    python app/gradio_demo.py


# Train
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    --shm-size=4gb \
    --name maskrcnn_training \
    dental-teeth-seg:latest \
    python models/teeth_segmentation.py train 

# Evaluate
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    dental-maskrcnn-torch:latest \
    python models/teeth_segmentation.py evaluate
```

---

## 11. Results

Training and validation outputs are saved to `outputs/`:

---

## 12. Evaluation Metrics

Evaluated with **pycocotools** (COCO standard):

| Metric | Description |
|---|---|
| **mAP@50** | Mean AP at IoU threshold 0.50 (COCO-style) |
| **mAP@50-95** | Mean AP averaged over IoU 0.50:0.05:0.95|
| **MAE** | Mean absolute error on tooth count per image (how off is the number of predicted teeth from ground truth)|

Run evaluation:

```bash
python models/teeth_segmentation_torch.py evaluate
```

---

## 13. Design Decisions

| Decision | Rationale |
|---|---|
| **torchvision Mask R-CNN** | No external forks or patches, actively maintained, not depracted
| **CLAHE over global histogram equalisation** | CLAHE enhances local contrast without over-amplifying noise in uniform regions, which is critical for tooth boundaries in dense bone areas |
| **COCO JSON annotations** | Standard format directly compatible with pycocotools for evaluation |
| **COCO pre-training** | COCO-trained weights encode robust low-level features (edges, textures) that transfer well to X-ray imagery |
| **HF Hub for weights** | Keeping the git repo lightweight |

---

## 14. References

- Oz M, Sengul A, Hatipoglu M, Danisman T. AKUDENTAL teeth instance segmentation dataset: a cross-dataset analysis. BMC Oral Health. 2026 Jan 12;26(1):247. doi: 10.1186/s12903-025-07645-0. PMID: 41527067; PMCID: PMC12874774.
- Lin, T-Y. et al. (2014). *Microsoft COCO: Common Objects in Context*. https://arxiv.org/abs/1405.0312
- He, K. et al. (2017). Mask R-CNN. https://arxiv.org/abs/1703.06870
