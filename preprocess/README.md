# Pipepile for WSI processing in computational pathology

### Framework

This repository aims to integrate the entire pipeline to process whole-slide images in computational pathology (CPath) including (1) WSI pyramide reconstruction, (2) segmentation and patch extraction and (3) patch-level feature extraction. 

![Alt text](figure.avif)

## Usage notes
### (0) From MRXS to TIF
New WSIs from ASSIST are transfered from the hospitakl in MRXS fromart. 
An MRXS file is an image file created by MIRAX-compatible microscope digital slide scanners. 
Use SlideViewer fro WSI visualization and SlideMaster for WSI conversion to TIF. Softawre available to download at https://www.3dhistech.com/research/software-downloads/.

### (1) WSI pyramide reconstruction
WSI pyramid reconstruction is necessary to efficiently handle large WSIs for tissue detection.
This function it is also useful to get a particular magnification level within the slide. 
**¡IMPORTANT!** Go to [README.txt](reconstruction/README.md) for the installation of the pyvips library. 

```
python reconstruction/reconstructor.py --p1 <path_slides> --p2 <path_tif> --level <0,1,2> --format <tif,svs>
```

### (2.1) WSI patching
The gigapixel size of WSI makes them impossible to handle by current hardware, thus requiring the implementation of Multiple Instance Learning (MIL) paradigms. MIL frameworks requires the extraction of equal-sized patches/tiles from the WSI. Segmentation is applied to discard white patches from the background.

CLAM software [1] is used to segment and extract the tiles. All patches within WSI are stacked into a matrix of shape (N,3,S,S), being N the number of patches and S their size. Masks and stitches are provided to visualize the segmentation. A large variability of parametres can be modified to adjust the segmentation to the WSI at hand. 

Considering initial 40x magnification, run the following line to extract 512x512 patches at 10x magnification without overlap.

```
python patching/extract_patches.py --source <path_tif> --save_dir <path_save> --seg --patch --stitch --patch_size 512 --step_size 512 --custom_downsample 4
```

### (2.2) WSI patching + feature extraction
Focusing on efficiency and storage saving, we can run the patching of the slide and the feature extraction simultaneously.
Use the argument --models to specify the names of the models separated by commas. The rest of the usage is identical, except for the --stich flag that should be avoided. 

Currently avilable foundation models: CONCH [2], KEEP [4] and UNI2 [6]. 
Place the chekcpoints of the models in the `feature_extraction/checkpoints` directory with the file named as the model. 

```
python patching/extract_patches_features.py --source <path_tif> --save_dir <path_patching> --seg --patch --patch_size 512 --step_size 512 --custom_downsample 4 --models CONCH,KEEP
```

### (3.1) Patch-level feature extraction
Current trend in computational pathology is to use patch-level feature extractor pretrained on large datasets of in-domain histopathology images either with unsupervised learning or vision-language supervision.

These script takes as input a H5 file containing a matrix with all the patches within a WSI and performs patch-level feature extraction. The output of the script is a numpy array of shape (N,L), being N the number of patches and L the length of the latent space. The available model is the image encoder of CONCH [2]. 

```
python feature_extraction/extract_features.py --source <path_patching/patches> --save_dir <path_embeddings> --models CONCH
```

### (3.2) Slide-level feature extraction
Slide foundation models automatically extract task-agnostic WSI-level representations.
The available model is the slide encoder of TITAN [5] which uses the patch features of CONCH v1.5.

These script takes as input an H5 file containing the patches within the slide and save the slide-level embedding extracted by TITAN.

```
python feature_extraction/extract_features_wsi.py --source <path_patching/patches> --save_dir <path_embeddings> --models CONCH
```

### Resources
[1] Lu, M. Y., Williamson, D. F., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6), 555-570.

[2] Lu, M. Y., Chen, B., Williamson, D. F., Chen, R. J., Liang, I., Ding, T., ... & Mahmood, F. (2024). A visual-language foundation model for computational pathology. Nature Medicine, 30(3), 863-874.

[3] Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F., Jaume, G., Song, A. H., ... & Mahmood, F. (2024). Towards a general-purpose foundation model for computational pathology. Nature Medicine, 30(3), 850-862.

[4] Zhou, X., Sun, L., He, D., Guan, W., Wang, R., Wang, L., ... & Xie, W. (2024). A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis. arXiv preprint arXiv:2412.13126.

[5] Ding, T., Wagner, S. J., Song, A. H., Chen, R. J., Lu, M. Y., Zhang, A., ... & Mahmood, F. (2024). Multimodal whole slide foundation model for pathology. arXiv preprint arXiv:2411.19666.

[6] Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F., Jaume, G., Song, A. H., ... & Mahmood, F. (2024). Towards a general-purpose foundation model for computational pathology. Nature Medicine, 30(3), 850-862.