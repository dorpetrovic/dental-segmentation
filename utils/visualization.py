# """
# Visualization Utilities
# =======================
# Plotting and overlay functions for dental teeth segmentation results.
# Includes FDI-aware labelling, per-quadrant colour coding, and training curves.
# """

# import os
# import numpy as np
# import cv2
# import matplotlib
# matplotlib.use("Agg") #sets the backend (for Docker use)
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.cm as cm
# from typing import Optional, List, Dict
# from pathlib import Path
# import pandas as pd
# parent_dir = os.path.abspath("..")
# import sys
# sys.path.append(parent_dir)
# import utils.preprocessing
# import importlib
# importlib.reload(utils.preprocessing)
# from utils.preprocessing import (
#     count_teeth_per_image,
#     class_frequency
# )


# # FDI quadrant colours for intuitive display
# QUADRANT_COLORS = {
#     "UR": "#4A90D9",   # upper right — blue
#     "UL": "#E87040",   # upper left  — orange
#     "LL": "#2ECC71",   # lower left  — green
#     "LR": "#9B59B6",   # lower right — purple
# }

# FDI_TO_QUADRANT = {
#     **{fdi: "UR" for fdi in range(11, 19)},
#     **{fdi: "UL" for fdi in range(21, 29)},
#     **{fdi: "LL" for fdi in range(31, 39)},
#     **{fdi: "LR" for fdi in range(41, 49)},
# }


# def _quadrant_color(class_name):
#     """
#     Pick a colour for a class name based on FDI quadrant.

#     AKUDENTAL categories:
#       "11 - Central Incisor" → extract FDI number → quadrant color
#       "Bridge" / "Filling-Crown" / "Implant" → gray
#     """
#     # Try to extract FDI number from name e.g. "11 - Central Incisor"
#     try:
#         fdi = int(class_name.split(" ")[0])
#         if 11 <= fdi <= 18: return QUADRANT_COLORS["UR"]
#         if 21 <= fdi <= 28: return QUADRANT_COLORS["UL"]
#         if 31 <= fdi <= 38: return QUADRANT_COLORS["LL"]
#         if 41 <= fdi <= 48: return QUADRANT_COLORS["LR"]
#     except (ValueError, IndexError):
#         pass

#     # Non-FDI categories: Bridge, Filling-Crown, Implant
#     return "#AAAAAA"  # gray

# def apply_masks(image, masks,class_names=None, alpha = 0.45):
#     """
#     Draw semi-transparent tooth masks(model predictions) on the
#     original jpg.
#     Args:
#         image: RGB image [H, W, 3] uint8.
#         masks: Bool masks [H, W, N].
#         class_names: Class name per mask (used for color).
#         alpha: Mask opacity.
#     """
#     output = image.copy().astype(np.float32)
#     for i in range(masks.shape[-1]):
#         if class_names and i < len(class_names):
#             hex_col = _quadrant_color(class_names[i])
#             r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
#             colour = np.array([r, g, b], dtype=np.float32)
#         else:
#             cmap = cm.get_cmap("tab20", max(1, masks.shape[-1]))
#             colour = np.array(cmap(i)[:3]) * 255

#         for c in range(3):
#             output[:, :, c] = np.where(
#                 masks[:, :, i],
#                 output[:, :, c] * (1 - alpha) + colour[c] * alpha,
#                 output[:, :, c],
#             )
#     return output.astype(np.uint8)


# def draw_bounding_boxes(image,rois,class_ids,scores,class_names):
#     """
#     Draw bounding boxes with label and confidence score.
    
#     Args:
#         image: (H,W,3) uint8 .jpg image
#         roi: bounding boxes (one 4-element array for each N teeth)
#         class_ids: array of class indices(one per tooth)
#         scores: confidence score (N,)
#         class_names: List of class names ["BG","tooth"]
#     """
#     out = image.copy()
#     for i, roi in enumerate(rois):
#         y1, x1, y2, x2 = roi
#         name = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "unknown"
#         hex_col = _quadrant_color(name)
#         color = (int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16))
#         label = f"{name} {scores[i]:.0%}"
#         cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(out, label, (x1, max(y1 - 5, 12)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
#     return out


# def visualize_prediction(image, result, class_names, save_path=None, show=False):
#     """
#     Visualizes both colored masks and bounding boxes

#     Args:
#         image: original Xray uint8
#         result: dict from model.detect (prediction dictionary)
#         class_names: ['bg','11-central incisor']
#         save_path: optional path to where to save the image
#         show: call plt.show() or not
    
#     """
#     masks = result.get("masks", np.zeros((*image.shape[:2], 0), dtype=bool))
#     rois = result.get("rois", np.zeros((0, 4), dtype=int))
#     class_ids = result.get("class_ids", np.array([], dtype=int))
#     scores = result.get("scores", np.array([], dtype=float))

#     det_names = [class_names[cid] for cid in class_ids if cid < len(class_names)]
#     annotated = apply_masks(image, masks, det_names)
#     annotated = draw_bounding_boxes(annotated, rois, class_ids, scores, class_names)

#     fig, axes = plt.subplots(1, 2, figsize=(16, 5))
#     axes[0].imshow(image)
#     axes[0].set_title("Original Panoramic X-ray")
#     axes[0].axis("off")
#     axes[1].imshow(annotated)
#     axes[1].set_title(f"Segmentation — {masks.shape[-1]} teeth detected")
#     axes[1].axis("off")


#     # Legend — quadrant colors + gray for restorations
#     legend_elements = ([patches.Patch(facecolor=c, label=q) for q, c in QUADRANT_COLORS.items()]
#     + [patches.Patch(facecolor="#AAAAAA", label="Bridge/Implant/Crown")])
 
#     axes[1].legend(handles=legend_elements, loc="lower right",
#                    fontsize=8, title="Category", framealpha=0.8)
    
#     plt.tight_layout()
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
#     if show:
#         plt.show()
    
#     plt.close(fig)
    
#     return annotated

# def plot_class_distribution(coco, save_path = None):
#     """
#     Bar chart showing how many times each tooth was annotated 
#     across all images we have.
#     Grouped by quadrant with quadrant colours.

#     Args:
#         coco: loaded coco dict
#         save_path: optional path for where to save the chart

#     """
#     freq = class_frequency(coco)

#     cat_map = {c['id']: c['name'] for c in coco['categories']}

#     cat_ids = sorted(freq.keys())
#     names = [cat_map.get(i,str(i)) for i in cat_ids]
#     counts = [freq[i] for i in cat_ids]
#     colors = [_quadrant_color(n) for n in names]

#     fig, ax = plt.subplots(figsize=(20, 5))
#     ax.bar(range(len(cat_ids)), counts, color=colors, edgecolor="white")
#     ax.set_xticks(range(len(cat_ids)))
#     ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
#     ax.set_ylabel("Annotation count")
#     ax.set_title("Annotation Frequency per Category (AKUDENTAL)")
#     ax.grid(axis="y", alpha=0.3)
 
#     legend_elements = [
#         patches.Patch(facecolor=c, label=q)
#         for q, c in QUADRANT_COLORS.items()
#     ] + [patches.Patch(facecolor="#AAAAAA", label="Bridge/Implant/Crown")]
#     ax.legend(handles=legend_elements, fontsize=8)
 
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)


# def plot_teeth_per_image(coco, save_path = None):
#     """
#     Bar chart of tooth count per image.

#     Args:
#         coco - annotation directory
#         save_path - full path where to save the image
    
#     """
#     counts = count_teeth_per_image(coco)
#     names  = sorted(counts.keys())
#     values = [counts[n] for n in names]

#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.bar(range(len(names)), values, color="#185FA5", edgecolor="white")
#     ax.axhline(np.mean(values), color="#D85A30", linestyle="--", linewidth=1.5,
#                label=f"Mean = {np.mean(values):.1f}")
#     ax.set_xlabel("Image")
#     ax.set_ylabel("Number of annotated teeth")
#     ax.set_title("Annotation count per image")
#     ax.legend()
#     ax.grid(axis="y", alpha=0.3)
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)

"""
Visualization Utilities
=======================
Plotting and overlay functions for dental teeth segmentation results.
Includes FDI-aware labelling, per-quadrant colour coding, and training curves.
Supports both binary mode ('tooth') and FDI multi-class mode (35 categories).
"""
 
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # sets the backend (for Docker use)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from pathlib import Path
import pandas as pd
parent_dir = os.path.abspath("..")
import sys
sys.path.append(parent_dir)
import utils.preprocessing
import importlib
importlib.reload(utils.preprocessing)
from utils.preprocessing import (
    count_teeth_per_image,
    class_frequency,
)
 
 
# FDI quadrant colours for intuitive display
QUADRANT_COLORS = {
    "UR": "#4A90D9",   # upper right — blue
    "UL": "#E87040",   # upper left  — orange
    "LL": "#2ECC71",   # lower left  — green
    "LR": "#9B59B6",   # lower right — purple
}
 
FDI_TO_QUADRANT = {
    **{fdi: "UR" for fdi in range(11, 19)},
    **{fdi: "UL" for fdi in range(21, 29)},
    **{fdi: "LL" for fdi in range(31, 39)},
    **{fdi: "LR" for fdi in range(41, 49)},
}
 
 
def _quadrant_color(class_name):
    """
    Pick a colour for a class name based on FDI quadrant.
 
    Binary mode:  'tooth'                  → blue
    FDI mode:     '11 - Central Incisor'   → quadrant color
    Other:        'Bridge'/'Implant'/etc.  → gray
    """
    # Binary mode — single tooth class
    if class_name == 'tooth':
        return "#4A90D9"  # blue
 
    # FDI mode — "11 - Central Incisor" → extract FDI number → quadrant color
    try:
        fdi = int(class_name.split(" ")[0])
        if 11 <= fdi <= 18: return QUADRANT_COLORS["UR"]
        if 21 <= fdi <= 28: return QUADRANT_COLORS["UL"]
        if 31 <= fdi <= 38: return QUADRANT_COLORS["LL"]
        if 41 <= fdi <= 48: return QUADRANT_COLORS["LR"]
    except (ValueError, IndexError):
        pass
 
    # Non-FDI categories: Bridge, Filling-Crown, Implant
    return "#AAAAAA"
 
 
def apply_masks(image, masks, class_names=None, alpha=0.45):
    """
    Draw semi-transparent tooth masks on the original image.
 
    Args:
        image:       RGB image [H, W, 3] uint8.
        masks:       Bool masks [H, W, N].
        class_names: Class name per mask (used for color).
        alpha:       Mask opacity (0=transparent, 1=opaque).
    """
    output = image.copy().astype(np.float32)
    for i in range(masks.shape[-1]):
        if class_names and i < len(class_names):
            hex_col = _quadrant_color(class_names[i])
            r = int(hex_col[1:3], 16)
            g = int(hex_col[3:5], 16)
            b = int(hex_col[5:7], 16)
            colour = np.array([r, g, b], dtype=np.float32)
        else:
            cmap   = cm.get_cmap("tab20", max(1, masks.shape[-1]))
            colour = np.array(cmap(i)[:3]) * 255
 
        for c in range(3):
            output[:, :, c] = np.where(
                masks[:, :, i],
                output[:, :, c] * (1 - alpha) + colour[c] * alpha,
                output[:, :, c],
            )
    return output.astype(np.uint8)
 
 
def draw_bounding_boxes(image, rois, class_ids, scores, class_names):
    """
    Draw bounding boxes with label and confidence score.
 
    Args:
        image:       (H, W, 3) uint8 image.
        rois:        (N, 4) bounding boxes [y1, x1, y2, x2].
        class_ids:   (N,) class indices.
        scores:      (N,) confidence scores.
        class_names: List of class names e.g. ['BG', 'tooth'].
    """
    out = image.copy()
    for i, roi in enumerate(rois):
        y1, x1, y2, x2 = roi
        name    = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "unknown"
        hex_col = _quadrant_color(name)
        color   = (int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16))
        label   = f"{name} {scores[i]:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out
 
 
def visualize_prediction(image, result, class_names, save_path=None, show=False):
    """
    Visualizes both colored masks and bounding boxes.
 
    Supports binary mode (['BG', 'tooth']) and FDI mode (['BG', '11 - Central Incisor', ...]).
 
    Args:
        image:       Original X-ray uint8 (H, W, 3).
        result:      Dict from model.detect() {masks, rois, class_ids, scores}.
        class_names: ['BG', 'tooth'] for binary or full FDI list for multi-class.
        save_path:   Optional path to save figure.
        show:        Whether to call plt.show().
    """
    masks     = result.get("masks",     np.zeros((*image.shape[:2], 0), dtype=bool))
    rois      = result.get("rois",      np.zeros((0, 4), dtype=int))
    class_ids = result.get("class_ids", np.array([], dtype=int))
    scores    = result.get("scores",    np.array([], dtype=float))
 
    det_names = [class_names[cid] for cid in class_ids if cid < len(class_names)]
    annotated = apply_masks(image, masks, det_names)
    annotated = draw_bounding_boxes(annotated, rois, class_ids, scores, class_names)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Panoramic X-ray")
    axes[0].axis("off")
    axes[1].imshow(annotated)
    axes[1].set_title(f"Segmentation — {masks.shape[-1]} teeth detected")
    axes[1].axis("off")
 
    # Legend — adapt to binary vs FDI multi-class mode
    if 'tooth' in class_names:
        # Binary mode — single color
        legend_elements = [
            patches.Patch(facecolor="#4A90D9", label="tooth")
        ]
    else:
        # FDI multi-class mode — quadrant colors + gray for restorations
        legend_elements = (
            [patches.Patch(facecolor=c, label=q) for q, c in QUADRANT_COLORS.items()]
            + [patches.Patch(facecolor="#AAAAAA", label="Bridge/Implant/Crown")]
        )
 
    axes[1].legend(handles=legend_elements, loc="lower right",
                   fontsize=8, title="Category", framealpha=0.8)
 
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
 
    if show:
        plt.show()
 
    plt.close(fig)
    return annotated
 
 
def plot_class_distribution(coco, save_path=None):
    """
    Bar chart showing annotation frequency per category.
    FDI teeth colored by quadrant, Bridge/Implant/Crown in gray.
 
    Args:
        coco:      Loaded COCO annotation dict.
        save_path: Optional path to save the chart.
    """
    freq    = class_frequency(coco)
    cat_map = {c['id']: c['name'] for c in coco['categories']}
 
    cat_ids = sorted(freq.keys())
    names   = [cat_map.get(i, str(i)) for i in cat_ids]
    counts  = [freq[i] for i in cat_ids]
    colors  = [_quadrant_color(n) for n in names]
 
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(len(cat_ids)), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(cat_ids)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Annotation count")
    ax.set_title("Annotation Frequency per Category (AKUDENTAL)")
    ax.grid(axis="y", alpha=0.3)
 
    legend_elements = (
        [patches.Patch(facecolor=c, label=q) for q, c in QUADRANT_COLORS.items()]
        + [patches.Patch(facecolor="#AAAAAA", label="Bridge/Implant/Crown")]
    )
    ax.legend(handles=legend_elements, fontsize=8)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_teeth_per_image(coco, save_path=None):
    """
    Histogram of annotation count per image.
 
    Args:
        coco:      Loaded COCO annotation dict.
        save_path: Optional path to save the chart.
    """
    counts = count_teeth_per_image(coco)
    values = list(counts.values())
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(values, bins=20, color="#185FA5", edgecolor="white")
    ax.axvline(np.mean(values), color="#D85A30", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(values):.1f}")
    ax.set_xlabel("Number of annotated instances")
    ax.set_ylabel("Number of images")
    ax.set_title("Annotation count per image (AKUDENTAL)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)