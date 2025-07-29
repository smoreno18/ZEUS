import torch

from utils.utils import device


# Factory of histopathology foundation models



def load_model(model_name:str, checkpoint_path = None):
    checkpoint_path = f"models/checkpoints/{model_name}.bin"

    # CONCH: https://github.com/mahmoodlab/CONCH
    if model_name == "CONCH":
        from models.CONCH.open_clip_custom import create_model_from_pretrained
        model, transform = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        model.to(device)
        model.eval()

    # KEEP: https://github.com/MAGIC-AI4Med/KEEP
    elif model_name == "KEEP":
        from models.KEEP.KEEP import return_KEEP
        model, transform = return_KEEP(checkpoint_path=checkpoint_path)
        model.to(device)
        model.eval()
        
        # MUSK: https://github.com/lilab-stanford/MUSK
    elif model_name == "MUSK":
        from huggingface_hub import login
        from models.ELON.musk.modeling import return_MUSK
        login("<place_your_token_here>")  # Replace with your Hugging Face token
        model, transform = return_MUSK(checkpoint_path="musk_large_patch16_384")
        model.to(device= device, dtype=torch.float16)
        model.eval()

    # # UNI: https://github.com/mahmoodlab/UNI
    # elif model_name == "UNI2":
    #     from models.UNI import get_encoder
    #     model, transform = get_encoder(enc_name='uni2-h', device="cuda:0", assets_dir=checkpoint_path)
    #     model.to(device)
    #     model.eval()

    # # PLIP: https://github.com/PathologyFoundation/plip
    # elif model_name == "PLIP":
    #     from models.PLIP import PLIP
    #     model = PLIP('vinid/plip')
    #     transform = None

    else:
        print("[ERROR: Unknown model name]")
        exit()

    return model, transform

def extract_embedding(image, model_name:str, model):

    if model_name == "CONCH":
            embedding = model.encode_image(image, proj_contrast=False, normalize=False)

    elif model_name in ["KEEP"]:
            embedding = model.encode_image(image)
            
    elif model_name in ["MUSK"]:
        embedding = model(
        image=image.half(),
        with_head=True,
        out_norm=True,
        ms_aug=True,
        return_global=True  
        )[0]

    elif model_name in ["PLIP"]:
            embedding = model.encode_images([image], batch_size=1)

    elif model_name in ["UNI2"]:
            embedding = model(image)

    embedding = embedding.squeeze().cpu().detach().numpy()

    return embedding
