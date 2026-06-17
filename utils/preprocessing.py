"""
Preprocessing
=======================
Image preprocessing pipeline for dental panoramic X-ray images.

"""

import os
import json
import numpy as np
import cv2
import skimage.io
import skimage.draw
import random
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Optional


# Load DICOM image (if the case)

def load_dicom(image_path):
    """
    Loads DICOM files and returns RGB uint8.

    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "requires pydicom library." \
            "Install it using: pip install pydicom"
        )
    
    dicom_img = pydicom.dcmread(image_path)
    array = dicom_img.pixel_array.astype(np.float32)
    a_min, a_max = array.min(), array.max()
    if a_max > a_min:
        array = (array - a_min)/ (a_max - a_min) * 255.0
    array = array.astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis = -1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.concatenate([array]*3, axis = -1)

    return array


#  Image loading 

def load_image(path):
    """
    Load a dental image and return as RGB uint8.
    Handles grayscale X-rays (→ 3-channel), RGBA (alpha dropped), TIF, PNG, JPEG and DICOM
    
    JPEG, PNG, TIFF are handled via skimage
    DICOM via pydicom
    Greyscale is converted to 3-channel (acting like RGB)
    RGBA - alpha channel dropped

    """
    suffix = Path(path).suffix.lower()
    if suffix == ".dcm":
        return load_dicom(path)
    image = skimage.io.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[:,:,:3]
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image]*3, axis = -1)
    return image.astype(np.uint8)


#  Contrast enhancement 

def enhance_contrast(image, method = "clahe"):
    """
    Enhance contrast of a dental X-ray.

    Args:
        image:  RGB uint8 image.
        method: 'clahe' (default) or 'histogram_eq' — global.

    CLAHE is preferred for panoramic X-rays, it sharpens tooth
    boundaries without over-amplifying noise in the dense bone background.
    """
    if method == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0]) # applied only on the L channel
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif method == "histogram_eq":
        channels = [cv2.equalizeHist(image[:, :, c]) for c in range(3)]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'clahe' or 'histogram_eq'.")


# Resize 

def resize_image(image, min_dim = 512, max_dim = 1024):
    """
    Resize preserving aspect ratio, then zero-pad to a multiple of 64.
    Panoramic images (1092x512) fit naturally into 512-1024 range.

    Returns:
        (resized_image, scale, (top, bottom, left, right) padding size)
    """
    h, w = image.shape[:2]
    scale = max(min_dim / min(h, w), 1.0)
    if round(max(h, w) * scale) > max_dim:
        scale = max_dim / max(h, w)

    new_h, new_w = round(h * scale), round(w * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


    # ResNet halves image dimensions through layer pooling.
    # Avoiding image mismatch by making the image divisible by 64 - padding to the nearest multiple of 64
    pad_h = (64 - new_h % 64) % 64
    pad_w = (64 - new_w % 64) % 64
    top,  bottom = pad_h // 2, pad_h - pad_h // 2 # splits padding equally
    left, right  = pad_w // 2, pad_w - pad_w // 2
    image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="constant")
    return image, scale, (top, bottom, left, right)


def normalize_image(image):
    """Normalize pixel values to [0, 1] float32."""
    return image.astype(np.float32) / 255.0


#  COCO annotation parsing 

def load_coco_annotations(json_path):
    """
    Load a COCO-format annotation file.

    Returns the full dict with keys: info, images, annotations, categories.
    """
    with open(json_path) as f:
        return json.load(f)


def get_image_annotations(coco, filename):
    """
    Return all annotation dicts for a given image filename.

    Args:
        coco:     Loaded COCO dict. (coco annotation json file)
        filename: Image filename, e.g. '001.jpg'.

    Returns:
        a list that contains annotation dictionaries
    """
    img_map = {img["file_name"]: img["id"] for img in coco["images"]}
    if filename not in img_map:
        return []
    image_id = img_map[filename]
    return [a for a in coco["annotations"] if a["image_id"] == image_id]


def coco_seg_to_mask(segmentation, height, width):
    """
    Convert a COCO segmentation polygon to a binary mask.

    Args:
        segmentation: COCO segmentation — list of flat [x1,y1,x2,y2,...] arrays.
        call get_image_annotations to get annots.
        segmentation = annots[i]['segmentation'][0], where i is the
        mask of the i-th tooth.
        height, width: Image dimensions.

    Returns:
        Boolean mask [H, W] showing just one tooth.
        Call it in a loop to get all the masks
    """
    mask = np.zeros((height, width), dtype=bool)
    for poly in segmentation:
        xs = np.array(poly[0::2])
        ys = np.array(poly[1::2])
        rr, cc = skimage.draw.polygon(ys, xs)
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        mask[rr, cc] = True
    return mask


def build_masks(coco, image_filename, height, width):
    """
    (H, W, N) boolean mask array for all annotated teeth in one image.

    Args:
        coco:           Loaded COCO annotation dict.
        image_filename: Image filename e.g. '001.jpg'
        height:         Image height in pixels.
        width:          Image width in pixels.

    Returns:
        masks: Boolean array of shape (H, W, N) where N = number of teeth.
        class_ids - list of N category_ids(FDI numbers)
    """
    # get all annotations for this image
    anns = get_image_annotations(coco, image_filename)

    if not anns:
        #return empty mask if no annptations
        return np.zeros((height, width, 0), dtype=bool)

    masks = []
    class_ids = []
    for ann in anns:
        seg = ann.get("segmentation", [])
        if not seg:
            continue
        mask = coco_seg_to_mask(seg, height, width)
        masks.append(mask)
        class_ids.append(ann['category_id'])
    
    if not masks:
        return np.zeros((height,width,0),dtype=bool), []

    return np.stack(masks, axis=-1), class_ids   # → (H, W, N)


def count_teeth_per_image(coco):
    """
    Return a dict mapping filename → number of annotated teeth.
    
    Args:
        coco - annotaiton.json file
    """
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    counts = {}
    for ann in coco["annotations"]:
        fname = img_map.get(ann["image_id"], "unknown")
        counts[fname] = counts.get(fname, 0) + 1
    return counts


def class_frequency(coco):
    """
    Return a dict mapping category_id → annotation count.
    
    Args:
        annotation json file 
    
    Returns:
        Dictionary containing the tooth category and the number of
        time that tooth mask appears (Dict[int,int])

    """
    freq= {}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        freq[cat_id] = freq.get(cat_id, 0) + 1
    return freq


#  Dataset split utility 

def split_coco_dataset(coco, train_ratio = 0.8, seed = 42):
    """
    Split a COCO dataset dict into train/val subsets.

    Args:
        coco:        Full COCO dataset dict.
        train_ratio: Fraction for training.
        seed:        Random seed.

    Returns:
        (train_coco, val_coco) — tuple with 2 dicts, each a valid COCO dict.
    """
    random.seed(seed)
    images = coco["images"][:]
    random.shuffle(images)
    n = int(len(images) * train_ratio)
    train_imgs, val_imgs = images[:n], images[n:]
    train_ids = {img["id"] for img in train_imgs}
    val_ids   = {img["id"] for img in val_imgs}

    def subset(imgs, ids):
        return {
            **coco,
            "images": imgs,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        }

    return subset(train_imgs, train_ids), subset(val_imgs, val_ids)


#  Batch preprocessing 

def preprocess_directory(input_dir,output_dir,
                         contrast_method = "clahe",
                         min_dim = 512,
                         max_dim = 1024):
    """
    Preprocess all images in a directory (enhance contrast, resize).
    TIF images are converted to PNG on output.
    """
    os.makedirs(output_dir, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}
    paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]
    print(f"Preprocessing {len(paths)} images → {output_dir}")

    for p in paths:
        img = load_image(str(p))
        img = enhance_contrast(img, method=contrast_method)
        #img, _, _ = resize_image(img, min_dim=min_dim, max_dim=max_dim)

        # Save as PNG (handles TIF sources cleanly)
        out_name = p.stem + ".jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("Done.")
