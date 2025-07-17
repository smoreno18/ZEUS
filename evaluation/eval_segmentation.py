import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# Allow handling of very large images
Image.MAX_IMAGE_PIXELS = None

def dice_coefficient(tissue: np.ndarray,
                     mask_true: np.ndarray,
                     mask_pred: np.ndarray,
                     smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient for overlapping pixels between ground truth and prediction
    within the tissue region only.

    Args:
        tissue (np.ndarray): Boolean tissue mask.
        mask_true (np.ndarray): Ground truth binary mask.
        mask_pred (np.ndarray): Predicted binary mask.
        smooth (float): Small value to avoid division by zero.

    Returns:
        float: Dice score between 0 and 1.
    """
    mask_pred = mask_pred[tissue]
    mask_true = mask_true[tissue]
    mask_pred = mask_pred.astype(bool)
    intersection = np.logical_and(mask_true, mask_pred).sum()
    total = mask_true.sum() + mask_pred.sum()
    return (2.0 * intersection + smooth) / (total + smooth)

def precision_score(tissue: np.ndarray,
                    mask_true: np.ndarray,
                    mask_pred: np.ndarray,
                    smooth: float = 1e-6) -> float:
    """
    Compute precision = TP / (TP + FP), only within tissue area.

    Args:
        tissue (np.ndarray): Boolean tissue mask.
        mask_true (np.ndarray): Ground truth binary mask.
        mask_pred (np.ndarray): Predicted binary mask.
        smooth (float): Small value to avoid division by zero.

    Returns:
        float: Precision score.
    """
    mask_pred = mask_pred[tissue]
    mask_true = mask_true[tissue]
    tp = np.logical_and(mask_pred, mask_true).sum()
    fp = np.logical_and(mask_pred, ~mask_true.astype(bool)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(tissue: np.ndarray,
                 mask_true: np.ndarray,
                 mask_pred: np.ndarray,
                 smooth: float = 1e-6) -> float:
    """
    Compute recall = TP / (TP + FN), only within tissue area.

    Args:
        tissue (np.ndarray): Boolean tissue mask.
        mask_true (np.ndarray): Ground truth binary mask.
        mask_pred (np.ndarray): Predicted binary mask.
        smooth (float): Small value to avoid division by zero.

    Returns:
        float: Recall score.
    """
    mask_pred = mask_pred[tissue]
    mask_true = mask_true[tissue]
    tp = np.logical_and(mask_pred, mask_true).sum()
    fn = np.logical_and(~mask_pred.astype(bool), mask_true).sum()
    return (tp + smooth) / (tp + fn + smooth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute Dice, Precision, and Recall scores from binarized predictions using slide list.'
    )
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset (should contain "images" and "masks").')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment folder containing predictions and tissue masks.')
    parser.add_argument('--template_name', type=str, required=True, help='Name of experiment template.')
    parser.add_argument('--model', type=str, required=True, help='Model name used in the prediction.')

    args = parser.parse_args()

    metrics_dir = os.path.join(args.exp, "metrics", args.template_name)
    os.makedirs(metrics_dir, exist_ok=True)

    base_out = os.path.join(args.exp, "pred_masks", args.template_name, args.model)
    save_binary = os.path.join(base_out, "prediction")

    # Get list of all WSI files
    slides = os.listdir(os.path.join(args.data, "images"))

    metrics_dict = {}
    missing_slides = []

    for wsi in tqdm(slides, desc="[INFO] Computing metrics from binarized predictions", unit="slide"):
        slide = os.path.splitext(wsi)[0]
        out_b = os.path.join(save_binary, slide + "_pred_mask.png")

        # Check if prediction exists
        if not os.path.exists(out_b):
            print(f"[WARNING] Binary prediction not found for {slide}")
            missing_slides.append(slide)
            continue

        # Load predicted binary mask
        binary_pred = Image.open(out_b).convert("L")
        binary_pred = np.array(binary_pred)
        binary_pred = (binary_pred > 0).astype(int)

        # Load tissue mask
        tissue_path = os.path.join(args.exp, "tissue", slide + ".png")
        if not os.path.exists(tissue_path):
            print(f"[WARNING] Tissue mask not found for {slide}")
            continue
        tissue_img = Image.open(tissue_path).convert("L")
        tissue = np.array(tissue_img.resize(binary_pred.shape[::-1], resample=Image.Resampling.NEAREST)) > 0

        # Load ground truth mask
        gt_path = os.path.join(args.data, "masks", f"{slide}.png")
        if not os.path.exists(gt_path):
            print(f"[WARNING] Ground truth mask not found for {slide}.")
            continue
        mask_true = Image.open(gt_path).convert("L")
        mask_true = np.array(mask_true.resize(binary_pred.shape[::-1], resample=Image.Resampling.NEAREST))
        mask_true = (mask_true > 0).astype(int)

        # Compute metrics
        dice = dice_coefficient(tissue, mask_true, binary_pred)
        precision = precision_score(tissue, mask_true, binary_pred)
        recall = recall_score(tissue, mask_true, binary_pred)

        metrics_dict[slide] = {
            "dice": dice,
            "precision": precision,
            "recall": recall
        }

    # Save metrics as JSON
    output_path = os.path.join(metrics_dir, f"{args.model}_metrics.json")
    with open(output_path, 'w') as fp:
        json.dump(metrics_dict, fp, indent=4)

    print(f"[INFO] Metrics saved to: {output_path}")
