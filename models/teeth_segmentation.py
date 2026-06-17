"""
models/teeth_segmentation_torch.py
====================================
Torchvision Mask R-CNN for dental panoramic X-ray tooth segmentation.

COCO pretrained weights loaded automatically.

Dataset:  AKUDENTAL — 333 panoramic X-rays, COCO JSON format
Strategy: Binary first (1 class), then 35 FDI classes

Usage:
    python models/teeth_segmentation_torch.py train
    python models/teeth_segmentation_torch.py evaluate
    python models/teeth_segmentation_torch.py predict --image data/images/1.jpg
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import skimage.io
import skimage.draw
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.preprocessing import enhance_contrast, load_image
from configs.model_config import (
    BINARY, NUM_CLASSES, IMAGE_MIN_SIZE, IMAGE_MAX_SIZE,
    EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, MOMENTUM, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, CONF_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS,
    ANCHOR_SIZES, ANCHOR_RATIOS, BINARY_CLASSES, FDI_CLASSES,
)

DATA_DIR    = PROJECT_ROOT / 'data'
IMG_DIR     = DATA_DIR / 'processed'
ANN_DIR     = DATA_DIR / 'annotations'
RESULTS_DIR = PROJECT_ROOT / 'outputs' / 'results' / 'maskrcnn_torch'
VIZ_DIR     = PROJECT_ROOT / 'outputs' / 'visualizations'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

QUADRANT_COLORS = {
    'UR': (74,  144, 217),
    'UL': (232, 112, 64),
    'LL': (46,  204, 113),
    'LR': (155, 89,  182),
    'XX': (170, 170, 170),
}
FDI_TO_QUADRANT = {
    **{str(fdi): 'UR' for fdi in range(11, 19)},
    **{str(fdi): 'UL' for fdi in range(21, 29)},
    **{str(fdi): 'LL' for fdi in range(31, 39)},
    **{str(fdi): 'LR' for fdi in range(41, 49)},
}


def get_color(cls_name):
    if cls_name in ('tooth', '__background__'):
        return QUADRANT_COLORS['UR']
    return QUADRANT_COLORS.get(FDI_TO_QUADRANT.get(cls_name, 'XX'), (170, 170, 170))


# ── Dataset ───────────────────────────────────────────────────────────────────

class TeethDataset(Dataset):

    def __init__(self, annotation_file, img_dir, binary=True, transforms=None):
        with open(annotation_file) as f:
            self.coco = json.load(f)

        self.img_dir    = Path(img_dir)
        self.binary     = binary
        self.transforms = transforms

        # Build category map
        if binary:
            self.cat_map    = {cat['id']: 1 for cat in self.coco['categories']}
            self.class_names = BINARY_CLASSES
        else:
            sorted_cats     = sorted(self.coco['categories'], key=lambda x: x['id'])
            self.cat_map    = {cat['id']: i+1 for i, cat in enumerate(sorted_cats)}
            self.class_names = FDI_CLASSES

        # Index annotations by image_id
        self.anns_by_image = {}
        for ann in self.coco['annotations']:
            self.anns_by_image.setdefault(ann['image_id'], []).append(ann)

        # Filter to images with annotations that exist on disk
        self.images = []
        for img in self.coco['images']:
            path = self.img_dir / img['file_name']
            if path.exists() and self.anns_by_image.get(img['id']):
                self.images.append(img)

        print(f'  Loaded {len(self.images)} images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info['file_name']

        # Load image
        image = load_image(str(img_path))

        H, W = image.shape[:2]
        anns = self.anns_by_image.get(img_info['id'], [])

        masks    = []
        boxes    = []
        labels   = []

        for ann in anns:
            seg = ann.get('segmentation', [])
            if not seg or not isinstance(seg[0], list):
                continue
            flat = seg[0]
            if len(flat) < 6:
                continue

            xs = np.array(flat[0::2])
            ys = np.array(flat[1::2])

            # Draw polygon mask
            mask = np.zeros((H, W), dtype=np.uint8)
            rr, cc = skimage.draw.polygon(ys, xs)
            rr = np.clip(rr, 0, H-1)
            cc = np.clip(cc, 0, W-1)
            mask[rr, cc] = 1

            if mask.sum() == 0:
                continue

            # Bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            if x2 <= x1 or y2 <= y1:
                continue

            masks.append(mask)
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_map.get(ann['category_id'], 1))

        if not masks:
            # Return empty target if no valid annotations
            target = {
                'boxes':   torch.zeros((0, 4), dtype=torch.float32),
                'labels':  torch.zeros(0, dtype=torch.int64),
                'masks':   torch.zeros((0, H, W), dtype=torch.uint8),
                'image_id': torch.tensor([img_info['id']]),
                'area':    torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64),
            }
        else:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t  = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            area     = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

            target = {
                'boxes':    boxes_t,
                'labels':   labels_t,
                'masks':    masks_t,
                'image_id': torch.tensor([img_info['id']]),
                'area':     area,
                'iscrowd':  torch.zeros(len(masks), dtype=torch.int64),
            }

        # Convert image to tensor
        image_t = F.to_tensor(image)

        if self.transforms:
            image_t, target = self.transforms(image_t, target)

        return image_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    """
    Build Mask R-CNN with ResNet50+FPN backbone.
    Loads COCO pretrained weights and replaces heads for num_classes.
    """
    # Load pretrained model
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_nms_thresh = NMS_THRESHOLD,
        box_score_thresh = CONF_THRESHOLD,
        box_detections_per_img = MAX_DETECTIONS,
    )

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print(f'Device: {DEVICE}')
    print(f'Binary: {BINARY}')

    num_classes = 2 if BINARY else 36  # bg + classes

    # Datasets
    dataset_train = TeethDataset(ANN_DIR / 'train.json', IMG_DIR, binary=BINARY)
    dataset_val   = TeethDataset(ANN_DIR / 'val.json',   IMG_DIR, binary=BINARY)

    loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=1,
        shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Model
    model = build_model(num_classes)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA # step_size reduces LR every LR_STEP_SIZE
    )

    # CSV logger
    import csv
    csv_path = RESULTS_DIR / 'training_history.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'loss_classifier',
                         'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])

    best_loss = float('inf')

    print(f'\nTraining for {EPOCHS} epochs...')
    print(f'Train: {len(dataset_train)} images')
    print(f'Val:   {len(dataset_val)} images\n')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for i, (images, targets) in enumerate(loader_train):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_losses.append(losses.item())

            if (i + 1) % 20 == 0:
                print(f'  Epoch {epoch}/{EPOCHS} step {i+1}/{len(loader_train)} '
                      f'loss={losses.item():.4f}')
                
        #Validationn phase
        model.train()
        val_losses = []

        with torch.no_grad():
            for images, targets in loader_val:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
                loss_dict_val = model(images,targets)
                val_loss = sum(loss for loss in loss_dict_val.values())    
                val_losses.append(val_loss.item())

        lr_scheduler.step()

        avg_train = np.mean(epoch_losses)
        avg_val = np.mean(val_losses)
        ld       = {k: v.item() for k, v in loss_dict.items()}

        print(f'Epoch {epoch}/{EPOCHS}  '
              f'train={avg_train:.4f}  '
              f'val={avg_val:.4f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.6f}')

        csv_writer.writerow([
            epoch, avg_train, avg_val,
            ld.get('loss_classifier', 0),
            ld.get('loss_box_reg', 0),
            ld.get('loss_mask', 0),
            ld.get('loss_objectness', 0),
            ld.get('loss_rpn_box_reg', 0),
        ])
        csv_file.flush()

        # Save best model
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), RESULTS_DIR / 'best.pth')
            print(f'  → Saved best model (val_loss={best_loss:.4f})')

        # Save latest
        torch.save(model.state_dict(), RESULTS_DIR / 'last.pth')

    csv_file.close()
    print(f'\nTraining complete!')
    print(f'Best weights: {RESULTS_DIR}/best.pth')


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate():
    """Run COCO-style evaluation using pycocotools"""
    #pycocotools reads coco objects and calculates IoU,precision,recall,mAP
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as maskUtils
    import copy, tempfile, os

    num_classes = 2 if BINARY else 36
    weights_path = RESULTS_DIR / 'best.pth'
    assert weights_path.exists(), f'Weights not found: {weights_path}'

    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    class_names = BINARY_CLASSES if BINARY else FDI_CLASSES

    dataset_val = TeethDataset(ANN_DIR / 'val.json', IMG_DIR, binary=BINARY)
    loader_val  = DataLoader(dataset_val, batch_size=1,
                             shuffle=False, collate_fn=collate_fn)

    coco_gt      = COCO(str(ANN_DIR / 'val.json'))
    coco_results = []

    print(f'Evaluating {len(dataset_val)} val images...')
    with torch.no_grad():
        for images, targets in loader_val:
            images  = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target['image_id'].item()
                boxes    = output['boxes'].cpu().numpy()
                scores   = output['scores'].cpu().numpy()
                labels   = output['labels'].cpu().numpy()
                masks    = output['masks'].cpu().numpy()  # (N, 1, H, W)

                for i in range(len(boxes)):
                    if scores[i] < CONF_THRESHOLD:
                        continue
                    #i-th tooth, 0-drops channel dimension
                    #as model outputs probabilites, >0.5 converts them into yes/no decisions
                    # turn into np.uint8 as pycocotools expects ints not bools
                    mask = (masks[i, 0] > 0.5).astype(np.uint8)
                    rle  = maskUtils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    x1, y1, x2, y2 = boxes[i]
                    coco_results.append({
                        'image_id':    image_id,
                        'category_id': int(labels[i]),
                        'segmentation': rle,
                        'bbox':        [float(x1), float(y1),
                                        float(x2-x1), float(y2-y1)],
                        'score':       float(scores[i]),
                    })

    if not coco_results:
        print('No detections above threshold.')
        return

 
    # Binary mode: remap all GT categories to 1
    if BINARY:
        val_gt = copy.deepcopy(coco_gt.dataset)
        for ann in val_gt['annotations']:
            ann['category_id'] = 1
        val_gt['categories'] = [{'id': 1, 'name': 'tooth'}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(val_gt, f)
            tmp_path = f.name
        coco_gt = COCO(tmp_path)
        os.unlink(tmp_path)


    # converrts lists of prediction dicts into COCO obejct that
    # pycocotools can compare against ground truth
    coco_dt  = coco_gt.loadRes(coco_results) 
    # creates evaluator: coco_gt - gt annotations, coco_dt -model prediction
    # 'segm' evaluate mask IoU
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    #for every detection computes of matches a gt annotation
    coco_eval.evaluate()
    #aggregates per image results into PR curves across all images and IoU thresholds
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP50 = coco_eval.stats[1]
    mAP   = coco_eval.stats[0]

    print('RESULTS SUMMARY')
    print('='*55)
    print(f'Model:     Mask R-CNN ResNet50+FPN (torchvision)')
    print(f'Dataset:   AKUDENTAL (333 panoramic X-rays)')
    print(f'Classes:   {"1 (binary tooth)" if BINARY else "35 FDI"}')
    print(f'mAP@50:    {mAP50*100:.1f}%')
    print(f'mAP@50-95: {mAP*100:.1f}%')


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image_path):
    num_classes  = 2 if BINARY else 36
    weights_path = RESULTS_DIR / 'best.pth'
    assert weights_path.exists(), f'Weights not found: {weights_path}'

    class_names = BINARY_CLASSES if BINARY else FDI_CLASSES

    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    image    = load_image(str(image_path))
    enhanced = enhance_contrast(image, method='clahe')
    image_t  = F.to_tensor(enhanced).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_t)

    output = outputs[0]
    boxes  = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    masks  = output['masks'].cpu().numpy()  # (N, 1, H, W)

    keep = scores >= CONF_THRESHOLD
    boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
    n = len(boxes)
    print(f'Detected: {n} instances')
    if n == 0:
        return enhanced

    output_img = enhanced.copy()
    for i in range(n):
        cls_name  = class_names[labels[i]] if labels[i] < len(class_names) else 'tooth'
        color     = get_color(cls_name)
        mask      = (masks[i, 0] > 0.5)

        overlay   = output_img.copy()
        overlay[mask] = color
        output_img = cv2.addWeighted(output_img, 0.55, overlay, 0.45, 0)

        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        # FDI label at centroid
        mask_u8 = mask.astype(np.uint8)
        M = cv2.moments(mask_u8)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.putText(output_img, cls_name, (cx-10, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    out_path = VIZ_DIR / f'torch_pred_{Path(image_path).name}'
    cv2.imwrite(str(out_path), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f'Saved: {out_path}')
    return output_img

# used in Gradio, loads once at startup then reuse for every request
def load_inference_model():
    """Load model for use in Gradio/FastAPI."""
    weights_path = RESULTS_DIR / 'best.pth'

    # Downloads model weights stored on huggingface
    if not weights_path.exists():
        from huggingface_hub import hf_hub_download
        print("Downloading weights from Hugging Face Hub ...")
        hf_hub_download(
            repo_id = "chocodo/dental-segmentation-maskrcnn-torch",
            filename = "best.pth",
            local_dir = str(RESULTS_DIR),
        )
        print("Weights downloaded.")

    num_classes  = 2 if BINARY else 36
    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask R-CNN dental tooth segmentation')
    parser.add_argument('command', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--image', help='Image path for predict command')
    args = parser.parse_args()

    if args.command == 'train':
        train()
    elif args.command == 'evaluate':
        evaluate()
    elif args.command == 'predict':
        assert args.image, '--image required'
        predict(args.image)