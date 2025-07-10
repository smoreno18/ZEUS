import fnmatch
import os
from optparse import OptionParser
from pathlib import Path

import pyvips

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--p1", "--path1", dest="path1",
                      help="Path with the images to reconstruct")
    parser.add_option("--p2", "--path2", dest="path2",
                      help="Output path for reconstructed images")
    parser.add_option("--l", "--level", dest="level", type="int", default=0,
                      help="Magnification level to save the WSI")
    parser.add_option("--f", "--format", dest="format", type="str", default="tif",
                      help="File format of the slides")

    options, _ = parser.parse_args()
    assert options.path1 is not None, "--path1 is an required option"
    assert options.path2 is not None, "--path2 is an required option"

    os.makedirs(options.path2, exist_ok=True) # Create saving directory

    # Read list of slides to reconstruct (support multifolder)
    slides, slides_fp = [], []
    for root, dirs, files in os.walk(options.path1):
        for file in files:
            if fnmatch.fnmatch(file, f"*.{options.format}"):
                slides.append(os.path.basename(file))
                slides_fp.append(os.path.join(root, file))

    # Loop for reconstruct all images in options.path1
    for file, file_fp in zip(slides, slides_fp):
        path_rec = os.path.join(options.path2, Path(file).stem + '.tif')
        if os.path.isfile(path_rec):
            print(f'{file} already exists')
            continue

        try:
            print(str(file))
            if options.level == 0:
                image = pyvips.Image.new_from_file(path_tif) # WSI loading
            else:
                if options.format == ".tif":
                    image = pyvips.Image.new_from_file(path_tif, page=options.level)
                elif options.format == ".svs":
                    image = pyvips.Image.new_from_file(path_tif, level=options.level)
            image.write_to_file(path_rec, pyramid=True, tile=True, compression="jpeg") # WSI reconstruction

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    print('Reconstruction finished')