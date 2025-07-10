# MISSING LEVELS RECONSTRUCTION

This module allows you to rebuild slides that lack resolution levels required for patching. **Please read the requirements 
section if you don't have the library *pyvips* installed**.

This is an example script call:

`python recontructor.py --p1 Directory_with_incomplete_slides --p2 Output_directory`.

| Arguments | Description |
|-----------|-------------|
|       --p1    |       Path containing all the slides to reconstruct      |
|    --p2       |     Output directory for the recontructed slides        |


## Requirements 

If you want to use this script, you need to install **_pyvips_**. These are the following steps:
1. Execute on cmd: pip install pyvips

2. Download from this link the version used when this script was created: `https://github.com/libvips/libvips/releases/download/v8.10.0/vips-dev-w64-all-8.10.0.zip`.
You can find other versions for this library in: `https://github.com/libvips/libvips/releases`

3. Add the bin folder of the previous download to the path of your system. A quick guide to add folders to the path can be 
found here on Stack Overflow: `https://stackoverflow.com/questions/44272416/how-to-add-a-folder-to-path-environment-variable-in-windows-10-with-screensho`
For example,
if you place that content in your *C:* unit, the folder to add to the path will be like this: `C:\vips-dev-8.10\bin`

4. Restart the computer

Now you can execute the script. 
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

