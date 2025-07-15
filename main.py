import subprocess
import argparse
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute similarities and segmentation')
    parser.add_argument('--data', type=str, help='Ruta del experimento')
    parser.add_argument('--save_dir', type=str, help='Generar predicciones')
    parser.add_argument('--template_name', type=str, help='Template name')
    parser.add_argument('--models', type=str, help='Modelo a usar')
    args = parser.parse_args()

    print("\n🚀 Step 1: Performing tissue segmentation and extracting features from patches...\n")
    subprocess.run([
        "python", "preprocess/patching/extract_patches_features.py",
        "--source", args.data,
        "--save_dir", args.save_dir,
        "--seg",
        "--patch", 
        "--patch_size", "448",
        "--step_size", "112",   
        "--models", args.models,
    ], check=True)

    for model in args.models.split(','):

        os.chdir(script_dir)

        print(f"\nStep 2: Performing segmentation task...\n")
        subprocess.run([
            "python", "inference/process_embeddings.py",
            "--data", os.path.join(args.data,"images"),
            "--source", args.save_dir,
            "--pred",
            # "--tissue",
            # "--sim_maps",
            "--template_name", args.template_name,
            "--model", model
        ], check=True)

        os.chdir(script_dir)

        # Puedes descomentar estos pasos si quieres
        print("\nStep 3: Computing segmentation metrics...\n")
        subprocess.run([
            "python", "evaluation/eval_segmentation.py",
            "--source", args.data,
            "--exp", args.save_dir,
            "--template_name", args.template_name,
            "--model", model
        ], check=True)
        
    os.chdir(script_dir)

    print("\nStep 4: Ejecutando overlay_prediction_v3.py...")
    subprocess.run([
        "python", "inference/overlay_prediction.py",
        "--data", args.data,        
        "--exp", args.save_dir,
        "--template_name", args.template_name,
        "--models", args.models
        # "--resize_factor", "8"
    ], check=True)

    print("✅ Pipeline completed.")

    os.chdir(script_dir)