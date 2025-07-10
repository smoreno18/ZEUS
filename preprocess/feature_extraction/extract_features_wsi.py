import argparse
import fnmatch
import os

with os.add_dll_directory('C:/Users/pabmees/PycharmProjects/CLAM/venv/Lib/site-packages/openslide-win64-20230414/bin'): # Modifiy accordingly
    import openslide

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from huggingface_hub import login
import random
from PIL import Image

parser = argparse.ArgumentParser(description="Slide-level feature etxraction with TITAN")
parser.add_argument('--folder', type=str)
parser.add_argument('--patch_size', type=int, default=512, help = "Patch size in the cooridnates")
args = parser.parse_args()

folder_wsi = os.path.join(args.folder, "patching/patches")
folder_coords = os.path.join(args.folder, "patching/coords/")
folder_CONCH = os.path.join(args.folder, "embeddings/CONCHv1.5")
folder_TITAN = os.path.join(args.folder, "embeddings/TITAN")
os.makedirs(folder_CONCH, exist_ok=True)
os.makedirs(folder_TITAN, exist_ok=True)

login("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")

# TITAN model loading (164x larger than ABMIL, 8x larger than TransMIL, 8x shorter than CONCH v1.5)
titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
conch, eval_transform = titan.return_conch()
conch.cuda()

slides_fp, slides = [], [] # Files loop
for root, dirs, files in os.walk(folder_wsi):
    for file in files:
        if fnmatch.fnmatch(file, f"*.h5"):
            slides_fp.append(os.path.join(root, file))
            slides.append(os.path.basename(file))

for slide_fp, slide in zip(slides_fp, slides):
    print(slide)

    # Slide-level embedding w/ TITAN
    fn_save_TITAN = os.path.join(folder_TITAN, slide.replace(".h5",".npy"))
    if os.path.exists(fn_save_TITAN):
        continue

    # (1) --> Read patches and coordinates from .h5 file
    with h5py.File(os.path.join(slide_fp), 'r') as file:
        images = file['imgs'][:]
        coords = file['coords'][:]

    fn_save_CONCH = os.path.join(folder_CONCH, slide.replace(".h5", ".npy"))
    if not os.path.exists(fn_save_CONCH):
        patch_embeddings = []
        for img in tqdm(images):
            img = Image.fromarray(img)
            img = eval_transform(img).unsqueeze(dim=0).cuda()
            with torch.inference_mode():
                x = conch(img).squeeze().cpu().numpy()
            patch_embeddings.append(x)
        patch_embeddings = np.stack(patch_embeddings)
        np.save(fn_save_CONCH, patch_embeddings)
    else:
        patch_embeddings = np.load(fn_save_CONCH)

    coords = torch.from_numpy(coords).to(torch.int64)
    patch_size_lv0 = np.int64(args.patch_size)
    patch_embeddings = torch.from_numpy(patch_embeddings)
    try:
        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            slide_embedding = titan.encode_slide_from_patch_features(patch_embeddings, coords, patch_size_lv0)
            slide_embedding = slide_embedding.squeeze().cpu().numpy()
        np.save(fn_save_TITAN, slide_embedding)
    except Exception as e:
        print(f"Exception: {type(e).__name__} - {e}")
        continue