#################################################################
###### GUARDAR PARCHES EN FORMATO PNG DESDE UN .H5 DE CLAM ######
#################################################################
import os
import time
import argparse
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument('--folder', type=str)
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    h5file = os.path.join(args.folder , "patches", args.file_name + ".h5")
    elapsed = time.time()
    with h5py.File(h5file, 'r') as file:
        patches = file['imgs'][:]
        coords = file['coords'][:]
    elapsed = time.time() - elapsed
    print("H5 loading = " + str(elapsed) + " seconds")

    for ct, patch in enumerate(tqdm(patches)):
        path_save = os.path.join(args.folder , "patches_png")
        fn = args.file_name + '_x' + str(coords[ct][0]) + '_y' + str(coords[ct][1]) + '.png'
        plt.imsave(os.path.join(path_save, fn), patch)