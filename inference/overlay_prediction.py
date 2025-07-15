import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import glob

Image.MAX_IMAGE_PIXELS = None

def load_wsi_and_masks(wsi_path: str, gt_mask_path: str, tissue_path: str, binary_path_dict: dict, resize_factor: float):
    """
    Load and resize the WSI, GT mask, and prediction masks to match the size of the binary prediction mask.
    If resize_factor is provided, apply it to scale the reference size.
    Applies tissue mask to predictions.
    """
    sample_binary_mask_path = next(iter(binary_path_dict.values()))
    binary_mask = Image.open(sample_binary_mask_path).convert("L")
    mask_size = binary_mask.size

    if resize_factor is None:
        resize_factor = 1.0

    target_size = (int(mask_size[0] * resize_factor), int(mask_size[1] * resize_factor))

    wsi = Image.open(wsi_path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    gt_mask = Image.open(gt_mask_path).convert("L").resize(target_size, Image.Resampling.NEAREST)
    tissue = Image.open(tissue_path).convert("L").resize(target_size, Image.Resampling.LANCZOS)
    tissue_np = np.array(tissue) > 0

    binary_masks = {}
    for model, path in binary_path_dict.items():
        mask_image = Image.open(path).convert("L")
        mask_arr = np.array(mask_image)
        mask_arr[~tissue_np] = 0
        binary_masks[model] = mask_arr

    return wsi, gt_mask, binary_masks


def overlay_tumor_contours(wsi_img: Image.Image, masks_dict: dict, colors: dict, thickness: int = 2) -> Image.Image:
    """
    Overlay contours of binary masks onto the WSI image using semi-transparent colors.
    Supports NumPy arrays as mask inputs and applies alpha blending.
    """
    wsi_cv = cv2.cvtColor(np.array(wsi_img), cv2.COLOR_RGB2BGR)
    overlay = wsi_cv.copy()

    for model_name, mask_image in masks_dict.items():
        # Ensure input is a NumPy array
        mask_arr = mask_image if isinstance(mask_image, np.ndarray) else np.array(mask_image)

        if not isinstance(mask_arr, np.ndarray):
            raise TypeError(f"Expected NumPy array for mask, got {type(mask_arr)}")

        # Binarize the mask
        _, bw = cv2.threshold(mask_arr.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rgba = colors.get(model_name, (0, 0, 0, 255))
        color_bgr = (rgba[2], rgba[1], rgba[0])  # Convert RGBA to BGR
        alpha = rgba[3] / 255.0

        cv2.drawContours(overlay, contours, -1, color_bgr, thickness)
        wsi_cv = cv2.addWeighted(overlay, alpha, wsi_cv, 1 - alpha, 0)

    return Image.fromarray(cv2.cvtColor(wsi_cv, cv2.COLOR_BGR2RGB))


def compose_and_save(ov1: Image.Image, ov2: Image.Image, out_path: str, dice_dict: dict, colors: dict):
    """
    Combine two overlay images side-by-side and draw a legend with Dice scores and model colors.
    Saves the composed image to the specified output path.
    """
    legend_scale = 1.8
    square_scale = 0.4

    w1, h1 = ov1.size
    w2, h2 = ov2.size
    W = w1 + w2
    H = max(h1, h2)

    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(ov1, (0, 0))
    canvas.paste(ov2, (w1, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")

    base_font_size = int(24 * legend_scale)
    base_square_size = int(25 * legend_scale * square_scale)
    base_line_height = int(35 * legend_scale)

    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)
    except:
        font = ImageFont.load_default()

    legend_texts = [f"{model} (Dice = {dice_dict.get(model, 0.0):.3f})" for model in dice_dict.keys()]
    text_widths = [draw.textlength(t, font=font) for t in legend_texts]
    max_text_width = max(text_widths)

    legend_width = int(max_text_width + base_square_size + 40)
    legend_height = int(len(legend_texts) * base_line_height + 20)

    x_legend = int(W - legend_width - int(25 * legend_scale))
    y_legend = int(H - legend_height - int(25 * legend_scale))

    radius = int(15 * legend_scale)
    legend_box = Image.new("RGBA", (legend_width, legend_height), (255, 255, 255, 0))
    legend_draw = ImageDraw.Draw(legend_box, "RGBA")
    legend_draw.rounded_rectangle(
        [(0, 0), (legend_width, legend_height)],
        radius=radius,
        fill=(255, 255, 255, 180),
        outline=(180, 180, 180, 255),
        width=2
    )

    y_offset = 10
    x_content = 15

    for model_name in dice_dict.keys():
        color = colors.get(model_name, (0, 0, 0))
        color_rgb = tuple(color[:3])

        text = f"{model_name} (Dice = {dice_dict.get(model_name, 0.0):.3f})"
        text_bbox = legend_draw.textbbox((0, 0), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = y_offset + (base_square_size - text_height) // 2

        legend_draw.rectangle(
            [x_content, y_offset, x_content + base_square_size, y_offset + base_square_size],
            fill=color_rgb,
            outline="black"
        )

        legend_draw.text((x_content + base_square_size + 10, text_y), text, fill="black", font=font)
        y_offset += base_line_height

    canvas.paste(legend_box, (x_legend, y_legend), legend_box)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay GT and tumor predictions with transparent contours and legend.')
    parser.add_argument('--data', type=str, required=True, help='Directory containing WSI data.')
    parser.add_argument('--exp', type=str, required=True, help='Directory to save outputs.')
    parser.add_argument('--template_name', type=str, required=True, help='Experiment template name.')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated model names.')
    parser.add_argument('--resize_factor', type=float, required=False, default=None, help='Optional resize factor.')

    args = parser.parse_args()

    # RGBA colors (with transparency)
    colors = {
        "CONCH": (0, 0, 255, 127),   # Blue
        "KEEP": (255, 0, 0, 127)     # Red
    }

    # Load Dice scores from metrics
    dice_data = {}
    for model in args.models.split(","):
        metrics_files = glob.glob(os.path.join(args.exp, 'metrics', args.template_name, f"*{model}_metrics.json"))
        if not metrics_files:
            print(f"[WARNING] Metrics not found for {model}")
            continue

        with open(metrics_files[0], 'r') as f:
            model_metrics = json.load(f)

        for slide_name, metrics in model_metrics.items():
            if slide_name not in dice_data:
                dice_data[slide_name] = {}
            dice_data[slide_name][model] = metrics.get("dice", 0.0)

    wsi_dir = os.path.join(args.data, "images")
    gt_dir = os.path.join(args.data, "masks")
    tissue_dir = os.path.join(args.exp, "tissue")

    for wsi_file in tqdm(os.listdir(wsi_dir), desc="[INFO] Processing slides", unit="slide"):
        slide_name, _ = os.path.splitext(wsi_file)
        wsi_p = os.path.join(wsi_dir, wsi_file)
        gt_p = os.path.join(gt_dir, f"{slide_name}.png")
        t_p = os.path.join(tissue_dir, f"{slide_name}.png")

        models_p_dict = {
            model: os.path.join(args.exp, "pred_masks", args.template_name, model, "prediction", f"{slide_name}_pred_mask.png")
            for model in args.models.split(',')
        }

        try:
            wsi_img, gt_mask_img, binary_masks = load_wsi_and_masks(wsi_p, gt_p, t_p, models_p_dict, args.resize_factor)

            ov_gt = overlay_tumor_contours(wsi_img, {"GT": gt_mask_img}, colors={"GT": (0, 255, 0, 127)})
            ov_pred = overlay_tumor_contours(wsi_img, binary_masks, colors)

            slide_dice_dict = {
                model: dice_data.get(slide_name, {}).get(model, 0.0)
                for model in args.models.split(',')
            }

            out_file = os.path.join(args.exp, "overlay_prediction", f"{slide_name}.png")
            compose_and_save(ov_gt, ov_pred, out_file, slide_dice_dict, colors)

        except Exception as e:
            print(f"[ERROR] Failed to process {wsi_file}: {e}")
            continue

    print("[INFO] Overlay images generated successfully.")
