import argparse
import os
import sys
import warnings

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.CONCH.open_clip_custom import create_model_from_pretrained

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument('--models', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--model_path', type=str, default = "./models/")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Future support of simultaneous multimodel feature extratcion
    models = args.models.split(",")
    n_models = len(models)
    for model_it in models:
        os.makedirs((os.path.join(args.save_dir, model_it)), exist_ok=True)

    if 'CONCH' in models: # CONCH: Vision-language supervision
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_CONCH, preprocess_CONCH = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path = args.model_path + "CONCH.bin")
        model_CONCH.to(device)
        model_CONCH.eval()

    list_wsi = os.listdir(args.source)
    for name_wsi in list_wsi:
        name_wsi =  name_wsi[:-3] # Remove extension
        file_npy = os.path.join(args.save_dir, "CONCH", name_wsi + ".npy")
        if os.path.isfile(file_npy):
            continue
        print(name_wsi)
        sys.stdout.flush()

        # Data loading
        with h5py.File(os.path.join(args.source, name_wsi + ".h5"), 'r') as file:
            images = file['imgs'][:]
            coords = file['coords'][:]
        images = [Image.fromarray(patch) for patch in images]

        patch_emb_CONCH = []
        for img in tqdm(images):
            if "CONCH" in models:
                img_conch = preprocess_CONCH(img).unsqueeze(dim=0).to(device)
                with torch.inference_mode() and torch.no_grad():
                    x = model_CONCH.encode_image(img_conch, proj_contrast=False, normalize=False).squeeze().cpu().numpy()
                patch_emb_CONCH.append(x)

        # Saving features
        if "CONCH" in models:
            np.save(os.path.join(args.save_dir, "CONCH", name_wsi + ".npy"),  np.stack(patch_emb_CONCH))