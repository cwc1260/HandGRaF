# Preprocess MSRA dataset

1. Install the Point Cloud Library (PCL)

    navigate to "./python-pcl" directory
    execute ``` sh build.sh``` (under Linux environment)

2. Compile dataset generation code

    execute ```python setup.py --build_ext -i```
 
3. Dataset generation

    set the ```dataset_dir``` (input path of downloaded raw data) and ```save_dir``` (output path of generated dataset) in the ```preprocess.py``` source file properly
    execute ```python preprocess.py```
