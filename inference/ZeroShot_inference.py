import json
import os
import numpy as np
import torch
from tqdm import tqdm
from models.ELON.musk import modeling
from zeroshot_path import zero_shot_classifier
from models.CONCH.open_clip_custom import create_model_from_pretrained
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer

class Processor:
    """
    Class to process whole-slide images (WSIs) using the CONCH framework for 
    computational pathology tasks. Handles model loading, embedding processing, 
    (cosine similarity calculation) and visualization of results (heatmaps and histograms).
    """

    def __init__(self, template_name, project_folder, model_name):
        """
        Initializes the CONCHProcessor with folder paths and project configuration.

        Args:
        - project_name (str): Name of the project used to identify prompts and results.
        - folder_wsi (str): Path to the folder containing WSI files.
        - folder_heatmaps (str): Path to save generated heatmaps.
        - folder_histograms (str): Path to save generated histograms.
        - folder_coords (str): Path to the folder containing coordinate files.
        - folder_scores (str): Path to save score files.
        - folder_emb (str): Path to the folder containing embedding files.
        """
        
        self.model_name = model_name
        self.project_name = template_name
        self.folder_coords = f"{project_folder}/coords"
        self.folder_scores = f"{project_folder}/similarities/{self.project_name}/{self.model_name}"
        self.folder_emb = f"{project_folder}/embeddings/{self.model_name}"
        
        # Ensure all required output folders exist
        os.makedirs(self.folder_scores, exist_ok=True)
        
        print("[INFO] Model loading")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        if self.model_name == "CONCH":
            # Load the pretrained model with its configuration
            checkpoint_path = 'models/checkpoints/CONCH.bin'
            model_cfg = 'conch_ViT-B-16'
            self.model_CONCH, _ = create_model_from_pretrained(model_cfg=model_cfg, checkpoint_path=checkpoint_path, device=self.device)
            self.model_CONCH.eval()
        
        if self.model_name == "KEEP":
            # KEEP update
            self.model_KEEP = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
            self.model_KEEP.eval()
            self.model_KEEP.cuda()
            
        if self.model_name == "MUSK":
            from models.ELON.musk import utils
            from timm.models import create_model
            from huggingface_hub import login
            login("<place_your_login_token_here>")
            self.model_MUSK = create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self.model_MUSK, 'model|module', '')
            self.tokenizer = XLMRobertaTokenizer("models/ELON/musk/models/tokenizer.spm")
            self.model_MUSK.to(device=self.device, dtype=torch.float32)
            self.model_MUSK.eval()

    def prompt_emsmble(self, save = False):
        """
        Loads the pretrained CONCH model and prepares zero-shot classifier prompts.
        Saves the computed embeddings for reuse.
        """
        print("[INFO] Prompt ensembling")
        # Load prompts for zero-shot classification
        prompt_file = os.path.join('inference/local_data/prompts/Templates', self.project_name + '.json')
        with open(prompt_file, 'r') as pf:
            prompt = json.load(pf)['0']
        self.classnames = prompt['classnames']
        self.templates = prompt['templates']
        
        # Convert classnames into a list of textual prompts
        classes_id = list(self.classnames.keys())
        classnames_text = [self.classnames[classes_id_it] for classes_id_it in classes_id]
        
        # Generate zero-shot classifier weights
        if self.model_name == "CONCH":
            # print(classnames_text)
            self.zs_weights, _ = zero_shot_classifier(self.model_CONCH, self.model_name, classnames_text, self.templates, device=self.device)

        elif self.model_name == "KEEP":
            self.zs_weights, _ = zero_shot_classifier(self.model_KEEP, self.model_name, classnames_text, self.templates, device=self.device, tokenizer=self.tokenizer) # KEEP update
        
        elif self.model_name == "MUSK":
            self.zs_weights, _ = zero_shot_classifier(self.model_MUSK, self.model_name, classnames_text, self.templates, device=self.device, tokenizer=self.tokenizer) # MUSK update
            
        self.zs_weights = self.zs_weights.cpu().detach().numpy()
        
        if save:
            # Save the zero-shot weights for later use
            np.save("inference/local_data/prompts/ensembles/" + self.project_name +f"_{self.model_name}", self.zs_weights)
        print("[INFO] Process completed")
        return self.zs_weights

    def similarities(self, ensemble, save = False):
        """
        Processes patch embeddings to compute similarity scores with the zero-shot weights.
        Saves the similarity scores for each WSI.
        """
        if len(os.listdir(self.folder_scores)) != 0:
            return print("[INFO] Similarities already computed. Skipping...")
        
        def conch_visual_head(patch_embd, model):
            """
            Projects and normalizes patch embeddings using the model's visual head.
            """
            model.eval()
            patch_embd_v = []
            for patch_embd_it in patch_embd:
                patch_embd_it = torch.tensor(patch_embd_it).cuda()
                # patch_embd_it = torch.tensor(patch_embd_it)
                patch_embd_it = model.visual.forward_project(patch_embd_it)
                patch_embd_it = torch.nn.functional.normalize(patch_embd_it, dim=-1)
                patch_embd_it = patch_embd_it.cpu().detach().numpy()
                patch_embd_v.append(patch_embd_it)
            return patch_embd_v

        # Load precomputed zero-shot weights
        prompt = ensemble.squeeze()
       # Iterate over embedding files
        list_wsi = os.listdir(self.folder_emb)
        results = {}
        similarities = []
        for slide in tqdm(list_wsi, desc="[INFO] Computing similarities"):
            slide_id = slide[:-4]
            embeddings = np.load(os.path.join(self.folder_emb, slide))
            if self.model_name == "CONCH":
                embeddings = conch_visual_head([embeddings], self.model_CONCH)[0]
            if prompt.ndim == 1:  # Single prompt
                sims = embeddings @ prompt.T
                # sims = embeddings @ prompt.T
                similarities.append(sims)
            else:  # Múltiples clases
                sims = np.stack([embeddings @ prompt[:, i].T for i in range(prompt.shape[1])], axis=-1)

            # Guardar si se solicita
            if save:
                np.save(os.path.join(self.folder_scores, slide_id + ".npy"), sims)

            results[slide_id] = sims
            
        return results
                
                
        


