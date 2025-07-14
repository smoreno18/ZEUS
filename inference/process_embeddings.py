from ZeroShot_inference import Processor
import gc
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import openslide

# Permitir imágenes grandes
Image.MAX_IMAGE_PIXELS = None

def mask_from_predictions(coords: np.ndarray,
                           scores: np.ndarray,
                           width: int,
                           height: int,
                           patch_size: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for (x, y), score in zip(coords, scores):
        x_end = min(x + patch_size, width)
        y_end = min(y + patch_size, height)
        mask[y:y_end, x:x_end] += score
    return mask

def to_uint8(array):
    array = np.clip(array, 0, 1)
    return (array * 255).astype(np.uint8)

def generate_and_save_masks(coords, scores_normal, scores_tumor, tissue,
                            width, height, patch_size,
                            out_normal, out_tumor, out_binary,
                            resize_factor=1):
    # Crear máscaras
    mask_n = mask_from_predictions(coords, scores_normal, width, height, patch_size)
    mask_t = mask_from_predictions(coords, scores_tumor, width, height, patch_size)

    # Convertir a uint8
    mask_n_u8 = to_uint8(mask_n)
    mask_t_u8 = to_uint8(mask_t)

    # Máscara binaria
    pred = np.argmax(np.stack([mask_n, mask_t]), axis=0).astype(np.uint8) * 255
    # Blur contours
    patch_size_small = max(int(2 * patch_size + 1), 3)
    if patch_size_small % 2 == 0:
        patch_size_small += 1
    pred = cv2.GaussianBlur(pred, (patch_size_small, patch_size_small), 0)
    pred = pred > 255*0.1
   
    os.makedirs(os.path.dirname(out_binary), exist_ok=True)
    image = Image.fromarray(pred)
    
    if resize_factor != 1:
        image.resize((width*resize_factor, height*resize_factor),  resample=Image.Resampling.BILINEAR).save(out_binary)
    else:
        image.save(out_binary)
   
    if args.save_maps:
        # Crear carpetas
        os.makedirs(os.path.dirname(out_normal), exist_ok=True)
        os.makedirs(os.path.dirname(out_tumor), exist_ok=True)

        # Guardar
        mask_n = Image.fromarray(mask_n_u8)
        mask_t = Image.fromarray(mask_t_u8)
        
        if resize_factor != 1:
            mask_n.resize((width*resize_factor, height*resize_factor),  resample=Image.Resampling.BILINEAR).save(out_normal)
            mask_t.resize((width*resize_factor, height*resize_factor),  resample=Image.Resampling.BILINEAR).save(out_tumor)
        else:
            mask_n.save(out_normal)
            mask_t.save(out_tumor)
            
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute similarities and segmentation')
    parser.add_argument('--data', type=str, help='Path to data folder')
    parser.add_argument('--source', type=str, help='Path to project folder')
    parser.add_argument('--pred', action='store_true', help='Gerate masks')
    parser.add_argument('--tissue', action='store_true', help='Use tissue')
    parser.add_argument('--sim_maps', action='store_true', help='Save similarity maps')
    parser.add_argument('--template_name', type=str, help='Template name')
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--resize_factor', type=int, help='Resize factor for save masks')
    args = parser.parse_args()

    processor = Processor(args.template_name, args.source, args.model)
    ensemble = processor.prompt_emsmble(save=True)
    processor.similarities(ensemble, save=True)

    emb_dir = os.path.join(args.source, "embeddings", args.model)
    emb_files = [f[:-4] for f in os.listdir(emb_dir) if f.endswith(".npy")]

    if args.pred:
        base_out = os.path.join(args.source, "pred_masks", args.template_name, args.model)
        save_normal = os.path.join(base_out, "normal")
        save_tumor  = os.path.join(base_out, "tumor")
        save_binary = os.path.join(base_out, "prediction")

        for wsi_name in tqdm(sorted(emb_files), desc="[INFO] Procesando slides", unit="slide"):
            slide = wsi_name
            
            out_n = os.path.join(save_normal, slide + "_normal_map.png")
            out_t = os.path.join(save_tumor,  slide + "_tumor_map.png")
            out_b = os.path.join(save_binary, slide + "_pred_mask.png")
            
            if os.path.exists(out_b):
                print(f"Ya existe la mascara para: {slide}")
            
            wsi_path = os.path.join(args.data,"images", slide)
            
            t_path = os.path.join(args.source,"tissue", slide + ".png")
            
            if args.pred:
                try:
                    if wsi_path.endswith(".tif") and os.path.exists(wsi_path):
                        wsi = openslide.OpenSlide(wsi_path)
                        w, h = wsi.dimensions
                    else:
                        wsi = Image.open(wsi_path).convert("RGB")
                        w, h = wsi.size
                    
                    # Verificar tissue
                    if args.tissue and os.path.exists(t_path):
                        tissue = Image.open(t_path).convert("L")
                    else:
                        tissue = None

                    coords = np.load(os.path.join(args.source, "coords", slide + ".npy"))
                    scores = np.load(os.path.join(args.source, "similarities", args.template_name, args.model, slide + ".npy"))
                    scores_n = scores[:, 0]
                    scores_t = scores[:, 1]

                    binary_pred = generate_and_save_masks(coords, scores_n, scores_t, 
                                                        tissue,
                                                        w, h, patch_size=448,
                                                        out_normal=out_n,
                                                        out_tumor=out_t,
                                                        out_binary=out_b,
                                                        resize_factor=args.resize_factor)

                    del binary_pred, coords, scores, scores_n, scores_t
                    gc.collect()

                except Exception as e:
                    print(f"[ERROR] {slide}: {e}")

    print("✅ ¡Terminó correctamente sin consumir toda la memoria!")
