from ZeroShot_inference import Processor
import gc
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\wsi_core"
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# Allow very large images to be processed
Image.MAX_IMAGE_PIXELS = None


def mask_from_predictions(coords: np.ndarray,
                           scores: np.ndarray,
                           width: int,
                           height: int,
                           patch_size: int) -> np.ndarray:
    """
    Generate a full-size score mask from patch-level predictions.

    Args:
        coords: List of (x, y) coordinates for patches.
        scores: Corresponding predicted scores for patches.
        width, height: Dimensions of the full WSI.
        patch_size: Patch size in pixels.

    Returns:
        Float32 mask with aggregated scores.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for (x, y), score in zip(coords, scores):
        x_end = min(x + patch_size, width)
        y_end = min(y + patch_size, height)
        mask[y:y_end, x:x_end] += score
    return mask


def to_uint8(array):
    """
    Convert float array in [0, 1] to uint8 [0, 255].

    Args:
        array: Normalized float image.

    Returns:
        Converted uint8 image.
    """
    array = np.clip(array, 0, 1)
    return (array * 255).astype(np.uint8)


def generate_and_save_masks(coords, scores_normal, scores_tumor, tissue,
                            width, height, patch_size,
                            out_normal, out_tumor, out_binary,
                            resize_factor=1,
                            save_sim_maps=False):
    """
    Generate and optionally save similarity maps and binary tumor prediction.

    Args:
        coords: Patch coordinates.
        scores_normal: Similarity scores for normal class.
        scores_tumor: Similarity scores for tumor class.
        tissue: Optional tissue mask (PIL Image or None).
        width, height: WSI dimensions.
        patch_size: Patch size in pixels.
        out_normal, out_tumor, out_binary: Output paths.
        resize_factor: Optional resizing factor.
        save_sim_maps: If True, save normal and tumor score maps.
    """
    # Generate similarity masks
    mask_n = mask_from_predictions(coords, scores_normal, width, height, patch_size)
    mask_t = mask_from_predictions(coords, scores_tumor, width, height, patch_size)

    # Convert to uint8 for saving
    mask_n_u8 = to_uint8(mask_n)
    mask_t_u8 = to_uint8(mask_t)

    # Binary classification: 0 = normal, 1 = tumor
    pred = np.argmax(np.stack([mask_n, mask_t]), axis=0).astype(np.uint8) * 255

    # Smooth binary mask
    patch_size_small = max(int(2 * patch_size + 1), 3)
    if patch_size_small % 2 == 0:
        patch_size_small += 1
    pred = cv2.GaussianBlur(pred, (patch_size_small, patch_size_small), 0)
    pred = pred > 255 * 0.1

    # Resize and save binary mask
    os.makedirs(os.path.dirname(out_binary), exist_ok=True)
    image = Image.fromarray(pred)
    image = image.resize((width // resize_factor, height // resize_factor), resample=Image.Resampling.BILINEAR)
    image.save(out_binary)

    # Apply tissue mask (optional)
    if isinstance(tissue, Image.Image):
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        tissue_resized = tissue.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        tissue_np = np.array(tissue_resized) > 0
        pred[~tissue_np] = 0

    # Save similarity maps if requested
    if save_sim_maps:
        os.makedirs(os.path.dirname(out_normal), exist_ok=True)
        os.makedirs(os.path.dirname(out_tumor), exist_ok=True)
        Image.fromarray(mask_n_u8).save(out_normal)
        Image.fromarray(mask_t_u8).save(out_tumor)

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute similarities and generate segmentation masks.')
    parser.add_argument('--data', type=str, help='Path to WSI images')
    parser.add_argument('--source', type=str, help='Path to experiment output folder')
    parser.add_argument('--pred', action='store_true', help='Generate binary prediction masks')
    parser.add_argument('--tissue', action='store_true', help='Use tissue masks for filtering')
    parser.add_argument('--sim_maps', action='store_true', help='Save similarity maps (normal/tumor)')
    parser.add_argument('--template_name', type=str, help='Experiment template name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--resize_factor', type=int, help='Resize factor for output masks')
    args = parser.parse_args()

    # Initialize model processor
    processor = Processor(args.template_name, args.source, args.model)

    # Create ensemble prompts and compute similarities
    ensemble = processor.prompt_emsmble(save=True)
    processor.similarities(ensemble, save=True)

    if args.pred:
        # Define output directories
        base_out = os.path.join(args.source, "pred_masks", args.template_name, args.model)
        save_normal = os.path.join(base_out, "normal_maps")
        save_tumor = os.path.join(base_out, "tumor_maps")
        save_binary = os.path.join(base_out, "prediction")

        for wsi_name in tqdm(sorted(os.listdir(args.data)), desc="[INFO] Processing slides", unit="slide"):
            slide, _ = os.path.splitext(wsi_name)
            out_n = os.path.join(save_normal, slide + "_normal_map.png")
            out_t = os.path.join(save_tumor, slide + "_tumor_map.png")
            out_b = os.path.join(save_binary, slide + "_pred_mask.png")

            if os.path.exists(out_b):
                print(f"[INFO] Prediction already exists for: {slide}")
                continue

            wsi_path = os.path.join(args.data, wsi_name)
            t_path = os.path.join(args.source, "tissue", slide + ".png")

            try:
                # Read WSI image
                if wsi_path.endswith(".tif") and os.path.exists(wsi_path):
                    wsi = openslide.OpenSlide(wsi_path)
                    w, h = wsi.dimensions
                else:
                    wsi = Image.open(wsi_path).convert("RGB")
                    w, h = wsi.size

                # Load tissue mask if enabled
                if args.tissue and os.path.exists(t_path):
                    tissue = Image.open(t_path).convert("L")
                else:
                    tissue = None

                # Load coordinates and similarity scores
                coords = np.load(os.path.join(args.source, "coords", slide + ".npy"))
                scores = np.load(os.path.join(args.source, "similarities", args.template_name, args.model, slide + ".npy"))
                scores_n = scores[:, 0]
                scores_t = scores[:, 1]

                # Generate masks
                binary_pred = generate_and_save_masks(coords, scores_n, scores_t,
                                                      tissue,
                                                      w, h, patch_size=448,
                                                      out_normal=out_n,
                                                      out_tumor=out_t,
                                                      out_binary=out_b,
                                                      resize_factor=1,
                                                      save_sim_maps=args.sim_maps)

                del binary_pred, coords, scores, scores_n, scores_t
                gc.collect()

            except Exception as e:
                print(f"[ERROR] Failed to process {slide}: {e}")

    print("✅ Inference process finished successfully!")
