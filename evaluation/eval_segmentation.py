import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

# Permitir imágenes grandes
Image.MAX_IMAGE_PIXELS = None

def dice_coefficient(tissue: np.ndarray,
                     mask_true: np.ndarray,
                     mask_pred: np.ndarray,
                     smooth: float = 1e-6) -> float:

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

    mask_pred = mask_pred[tissue]
    mask_true = mask_true[tissue]
    tp = np.logical_and(mask_pred, mask_true).sum()
    fp = np.logical_and(mask_pred, ~mask_true.astype(bool)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(tissue: np.ndarray,
                 mask_true: np.ndarray,
                 mask_pred: np.ndarray,
                 smooth: float = 1e-6) -> float:

    mask_pred = mask_pred[tissue]
    mask_true = mask_true[tissue]
    tp = np.logical_and(mask_pred, mask_true).sum()
    fn = np.logical_and(~mask_pred.astype(bool), mask_true).sum()
    return (tp + smooth) / (tp + fn + smooth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcular DICE, Precision y Recall desde binarized_pred con lista de slides en CSV')
    parser.add_argument('--source', type=str, required=True, help='Ruta de la fuente de datos')
    parser.add_argument('--exp', type=str, required=True, help='Ruta del experimento')
    parser.add_argument('--template_name', type=str, required=True, help='Template name')
    parser.add_argument('--model', type=str, required=True, help='Modelo a usar')
    args = parser.parse_args()

    metrics_dir = os.path.join(args.exp, "metrics", args.template_name)
    os.makedirs(metrics_dir, exist_ok=True)

    base_out = os.path.join(args.exp, "masks_pred", args.template_name, args.model)
    save_binary = os.path.join(base_out, "binary")


    # slides = os.listdir(os.path.join(args.source, "regions"))
    slides = os.listdir(os.path.join(args.source, "images"))

    metrics_dict = {}
    missing_slides = []

    for wsi in tqdm(slides, desc="[INFO] Calculando métricas desde binarized_pred", unit="slide"):
        slide = os.path.splitext(wsi)[0]
        # breakpoint()
        out_b = os.path.join(save_binary, slide + "_binary_mask.png")
        # breakpoint()
        if not os.path.exists(out_b):
            print(f"[WARNING] Máscara binaria no encontrada para {slide}")
            missing_slides.append(slide)
            continue

        # Cargar predicción binaria
        binary_pred = Image.open(out_b).convert("L")
        binary_pred = np.array(binary_pred)
        binary_pred = (binary_pred > 0).astype(int)

        # Cargar máscara de tejido
        tissue_path = os.path.join(args.exp, "tissue", slide + ".png")
        if not os.path.exists(tissue_path):
            print(f"[WARNING] Tissue mask not found for {slide}")
            continue
        tissue_img = Image.open(tissue_path).convert("L")
        tissue = np.array(tissue_img.resize(binary_pred.shape[::-1], resample=Image.Resampling.NEAREST)) > 0

        # Cargar ground truth
        gt_path = os.path.join(args.source, "masks", f"{slide}.png")
        
        if not os.path.exists(gt_path):
            print(f"[WARNING] No existe la máscara de referencia para {slide}.")
            continue
        mask_true = Image.open(gt_path).convert("L")
        mask_true = np.array(mask_true.resize(binary_pred.shape[::-1], resample=Image.Resampling.NEAREST))
        mask_true = (mask_true > 0).astype(int)

        # Calcular métricas
        dice = dice_coefficient(tissue, mask_true, binary_pred)
        precision = precision_score(tissue, mask_true, binary_pred)
        recall = recall_score(tissue, mask_true, binary_pred)
        
        metrics_dict[slide] = {
            "dice": dice,
            "precision": precision,
            "recall": recall
        }

    # Guardar métricas en JSON
    output_path = os.path.join(metrics_dir, f"{args.model}_metrics.json")
    with open(output_path, 'w') as fp:
        json.dump(metrics_dict, fp, indent=4)

    print(f"[INFO] Métricas guardadas en: {output_path}")

