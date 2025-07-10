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

def mask_from_predictions(coords, scores, width, height, patch_size, resize_factor):
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)

    mask = np.zeros((new_height, new_width), dtype=np.float32)
    count = np.zeros((new_height, new_width), dtype=np.float32)

    for (x, y), score in zip(coords, scores):
        x_small = int(x * resize_factor)
        y_small = int(y * resize_factor)
        patch_size_small = int(patch_size * resize_factor)

        x_end = min(x_small + patch_size_small, new_width)
        y_end = min(y_small + patch_size_small, new_height)

        mask[y_small:y_end, x_small:x_end] += score
        count[y_small:y_end, x_small:x_end] += 1

    count[count == 0] = 1
    
    return mask /count

def to_uint8(array):
    array = np.clip(array, 0, 1)
    return (array * 255).astype(np.uint8)

def generate_and_save_masks(coords, scores_normal, scores_tumor, tissue,
                            width, height, patch_size,
                            out_normal, out_tumor, out_binary,
                            resize_factor=1):
    # Crear máscaras
    mask_n = mask_from_predictions(coords, scores_normal, width, height, patch_size, resize_factor)
    mask_t = mask_from_predictions(coords, scores_tumor, width, height, patch_size, resize_factor)


    # Convertir a uint8
    mask_n_u8 = to_uint8(mask_n)
    mask_t_u8 = to_uint8(mask_t)

    # Máscara binaria
    pred = np.argmax(np.stack([mask_n, mask_t]), axis=0).astype(np.uint8) * 255

    # Suavizado
    patch_size_small = max(int(2 * patch_size * resize_factor + 1), 3)
    if patch_size_small % 2 == 0:
        patch_size_small += 1
    pred = cv2.GaussianBlur(pred, (patch_size_small, patch_size_small), 0)
    pred = pred > 255*0.1
    
    if isinstance(tissue, Image.Image):
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        tissue_resized = tissue.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        tissue_np = np.array(tissue_resized) > 0
        
        pred[~tissue_np] = 0

    # Crear carpetas
    # os.makedirs(os.path.dirname(out_normal), exist_ok=True)
    # os.makedirs(os.path.dirname(out_tumor), exist_ok=True)
    os.makedirs(os.path.dirname(out_binary), exist_ok=True)

    # # Guardar
    # Image.fromarray(mask_n_u8).save(out_normal)
    # Image.fromarray(mask_t_u8).save(out_tumor)
    Image.fromarray(pred).save(out_binary)

    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute similarities and segmentation')
    parser.add_argument('--source', type=str, help='Ruta del experimento')
    parser.add_argument('--pred', action='store_true', help='Generar predicciones')
    parser.add_argument('--tissue', action='store_true', help='Usar tissue')
    parser.add_argument('--template_name', type=str, help='Template name')
    parser.add_argument('--model', type=str, help='Modelo a usar')
    parser.add_argument('--resize_factor', type=int, help='Resize factor')
    args = parser.parse_args()

    processor = Processor(args.template_name, args.source, args.model)
    ensemble = processor.prompt_emsmble(save=True)
    processor.similarities(ensemble, save=True)

    emb_dir = os.path.join(args.source, "embeddings", args.model)
    emb_files = [f[:-4] for f in os.listdir(emb_dir) if f.endswith(".npy") and f.split('_')[1][0] =='0']

    if args.pred:
        base_out = os.path.join(args.source, "masks_pred", args.template_name, args.model)
        save_normal = os.path.join(base_out, "normal")
        save_tumor  = os.path.join(base_out, "tumor")
        save_binary = os.path.join(base_out, "binary")

        for slide in tqdm(sorted(emb_files), desc="[INFO] Procesando slides", unit="slide"):
            out_n = os.path.join(save_normal, slide + "_normal_mask.png")
            out_t = os.path.join(save_tumor,  slide + "_tumor_mask.png")
            out_b = os.path.join(save_binary, slide + "_binary_mask.png")
            
            # if os.path.exists(out_b):
            #     print(f"Ya existe la mascara para: {slide}")
                    
            wsi_path = os.path.join(f"/BBDD/data/images", slide + ".jpg")
            t_path = os.path.join(f"{args.source}/tissue", slide + ".png")
                        
            try:
            # if args.pred:
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
                scores = np.load(os.path.join(args.source, "scores", args.template_name, args.model, slide + ".npy"))
                scores_n = scores[:, 0]
                scores_t = scores[:, 1]

                binary_pred = generate_and_save_masks(coords, scores_n, scores_t, 
                                                      tissue,
                                                      w, h, patch_size=448,
                                                      out_normal=out_n,
                                                      out_tumor=out_t,
                                                      out_binary=out_b,
                                                      resize_factor=1/8)

                del binary_pred, coords, scores, scores_n, scores_t
                gc.collect()

            except Exception as e:
                print(f"[ERROR] {slide}: {e}")


    print("✅ ¡Terminó correctamente sin consumir toda la memoria!")
