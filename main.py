import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute similarities and segmentation')
    parser.add_argument('--data', type=str, help='Ruta del experimento')
    parser.add_argument('--save_dir', type=str, help='Generar predicciones')
    parser.add_argument('--template_name', type=str, help='Template name')
    parser.add_argument('--models', type=str, help='Modelo a usar')
    args = parser.parse_args()

    # print("\n🚀 Step 1: Performing tissue segmentation and extracting features from patches...\n")
    # subprocess.run([
    #     "python", "/workspace/Projects/ZEUS/preprocess/patching/extract_patches_features.py",
    #     "--source", args.data,
    #     "--save_dir", args.save_dir,
    #     "--seg",
    #     "--patch", 
    #     "--patch_size", "448",
    #     "--step_size", "112",   
    #     "--models", args.models,
    # ], check=True)

    for model in args.models.split(','):
        print(f"Step 2: Performing segmentation task...\n")
        subprocess.run([
            "python", "inference/process_embeddings.py",
            "--source", args.save_dir,
            "--pred",
            "--tissue",
            "--template_name", args.template_name,
            "--model", model
        ], check=True)

        # Puedes descomentar estos pasos si quieres
        print("Step 3: Computing segmentation metrics...\n")
        subprocess.run([
            "python", "/workspace/Projects/ZEUS/evaluation/eval_segmentation.py",
            "--source", args.data,
            "--exp", args.save_dir,
            "--template_name", args.template_name,
            "--model", model
        ], check=True)

    # print("Step 4: Ejecutando overlay_prediction_v3.py...\n")
    # subprocess.run([
    #     "python", "/workspace/Projects/scripts/overlay_prediction_v3.py",
    #     "--exp", args.save_dir,
    #     "--wsi_path", args.data,
    #     "--gt",
    #     "--template_name", args.template_name,
    #     "--models", args.models
    # ], check=True)

    # print("✅ Pipeline completed.")
