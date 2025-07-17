import subprocess
import argparse
import os

# Get the absolute path to the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Compute similarities and segmentation')
    parser.add_argument('--data', type=str, help='Ruta del experimento')  # Path to dataset
    parser.add_argument('--save_dir', type=str, help='Generar predicciones')  # Where outputs will be saved
    parser.add_argument('--template_name', type=str, help='Template name')  # Experiment/template ID
    parser.add_argument('--models', type=str, help='Modelo a usar')  # Comma-separated model names
    args = parser.parse_args()

    # Step 1: Run tissue segmentation and extract patch features
    print("\n🚀 Step 1: Performing tissue segmentation and extracting features from patches...\n")
    subprocess.run([
        "python", "preprocess/patching/extract_patches_features.py",
        "--source", args.data,
        "--save_dir", args.save_dir,
        "--seg",           # Tissue segmentation
        "--patch",         # Patch extraction
        "--patch_size", "448",  # Patch size
        "--step_size", "112",   # Stride between patches
        "--models", args.models,
    ], check=True)

    # Step 2 & 3: For each model, run inference and evaluation
    for model in args.models.split(','):

        os.chdir(script_dir)

        print(f"\nStep 2: Performing segmentation task...\n")
        subprocess.run([
            "python", "inference/process_embeddings.py",
            "--data", os.path.join(args.data,"regions"),
            "--source", args.save_dir,
            "--pred",  # Perform prediction
            # "--tissue",       # Optional: generate tissue map again if needed
            # "--sim_maps",     # Optional: save similarity maps
            "--template_name", args.template_name,
            "--model", model
        ], check=True)

        os.chdir(script_dir)

        # Puedes descomentar estos pasos si quieres
        print("\nStep 3: Computing segmentation metrics...\n")
        subprocess.run([
            "python", "evaluation/eval_segmentation.py",
            "--data", args.data,
            "--exp", args.save_dir,
            "--template_name", args.template_name,
            "--model", model
        ], check=True)

    os.chdir(script_dir)

    # Step 4: Generate overlay with GT vs prediction
    print("\nStep 4: Ejecutando overlay_prediction_v3.py...")
    subprocess.run([
        "python", "inference/overlay_prediction.py",
        "--data", args.data,        
        "--exp", args.save_dir,
        "--template_name", args.template_name,
        "--models", args.models
        # "--resize_factor", "8"  # Optional resizing of input data
    ], check=True)

    print("✅ Pipeline completed.")

    os.chdir(script_dir)
