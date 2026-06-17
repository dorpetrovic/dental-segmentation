"""
Evaluation Metrics
==================
Metrics for evaluating instance segmentation quality on dental images.
Includes IoU, mAP, and per-image tooth count accuracy.
"""

import numpy as np
from typing import List, Tuple, Dict


def compute_iou(mask_a, mask_b):
    """
    Compute Intersection over Union between two binary masks.

    Args:
        mask_a: Boolean mask [H, W].
        mask_b: Boolean mask [H, W].

    Returns:
        IoU score is between 0 and 1.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_instance_iou_matrix(pred_masks, gt_masks):
    """
    Compute a matrix that calculates IoU between predicted and ground-truth masks.

    Args:
        pred_masks: Predicted masks [H, W, N_pred].
        gt_masks:   Ground-truth masks [H, W, N_gt].

    Returns:
        IoU matrix [N_pred, N_gt].
    """
    n_pred = pred_masks.shape[-1] #number of predicted masks
    n_gt = gt_masks.shape[-1] # number of ground-truth masks
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)

    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = compute_iou(pred_masks[:, :, i], gt_masks[:, :, j])

    return iou_matrix


def compute_precision_recall(iou_matrix, iou_threshold = 0.5):
    """
    Compute precision and recall at a given IoU threshold.

    A predicted mask is a True Positive if it matches a GT mask
    with IoU >= threshold (greedy matching, each GT matched once).

    Args:
        iou_matrix:    [N_pred, N_gt] pairwise IoU matrix.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        (precision, recall) floats.
    """
    if iou_matrix.size == 0:
        return 0.0, 0.0

    n_pred, n_gt = iou_matrix.shape
    matched_gt = set()
    tp = 0

    # For each prediction, find best-matching unmatched GT
    for i in range(n_pred):
        best_iou = iou_threshold - 1e-9 #=0.4999999 just below threshold, so that the best match is above this!
        best_j = -1 # no match found yet! 
        for j in range(n_gt):
            if j not in matched_gt and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_j >= 0: #the match only happens when if-statement is true, otherwise best_j stays -1 and we dont have a TP
            tp += 1
            matched_gt.add(best_j)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    return precision, recall


def compute_map(predictions,ground_truths,iou_thresholds = [0.5, 0.75]):
    """
    Compute mean Average Precision (mAP) over a set of images.

    Args:
        predictions:   List of dicts with keys 'masks' [H,W,N] and 'scores' [N].
        ground_truths: List of dicts with key 'masks' [H,W,N].
        iou_thresholds: IoU thresholds to evaluate. Defaults to [0.5, 0.75].

    Returns:
        Dict with 'mAP', 'mAP@50', 'mAP@75'.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.75]

    aps_per_threshold = {t: [] for t in iou_thresholds}

    for pred, gt in zip(predictions, ground_truths):
        pred_masks = pred.get("masks", np.zeros((*gt["masks"].shape[:2], 0), dtype=bool))
        gt_masks = gt["masks"]

        if pred_masks.shape[-1] == 0 or gt_masks.shape[-1] == 0:
            for t in iou_thresholds:
                aps_per_threshold[t].append(0.0)
            continue

        iou_mat = compute_instance_iou_matrix(pred_masks, gt_masks)
        for t in iou_thresholds:
            p, r = compute_precision_recall(iou_mat, iou_threshold=t)
            # AP approximation: precision at the recall point
            aps_per_threshold[t].append(p)

    results = {}
    for t, aps in aps_per_threshold.items():
        key = f"mAP@{int(t*100)}"
        results[key] = float(np.mean(aps))

    results["mAP"] = float(np.mean(list(results.values())))
    return results


def count_accuracy(pred_counts, gt_counts):
    """
    Evaluate how accurately the model counts teeth per image.

    Args:
        pred_counts: List of predicted tooth counts per image (number of tooth that the model counted in every image)
        gt_counts:   List of ground-truth tooth counts per image.

    Returns:
        Dict with 'mae' (mean absolute error) and 'exact_match_rate'.
    """
    diffs = [abs(p - g) for p, g in zip(pred_counts, gt_counts)] #number of teeth in the mouth - number detected by the model
    exact = sum(1 for d in diffs if d == 0)
    return {
        "mae": float(np.mean(diffs)), # shows by how much the model is off. So somewhat as the variation of the error of the model. so if error is 2, that mwans the model is off by on average 2 teeth!
        "exact_match_rate": exact / len(diffs) if diffs else 0.0, #the number of images that had perfect count!
    #mae = 2
      #on average model is off by 2 teeth
      #for a full mouth (32 teeth) that's about 5.6% error
      #acceptable for first model with 17 training images

    #exact_match_rate = 0.40
      #40% of images had perfectly correct tooth count
      #means 60% had at least 1 tooth count error
      #room for improvement with more training data
    }
