# HandGRaF-Net: Hand Graph Reasoning and Folding
Wencan Cheng, Jong Hwan Ko

This is the implementation of the manuscript

1. Prepare dataset 

    please download the MSRA dataset

    follow the instructions in the './preprocess_msra/' for datasets preprocessing 

2. Install PointNet++ CUDA operations

    follow the instructions in the './train_eval/pointnet2' for installation 

3. Evaluate

    go to "train_eval" directory

    execute ``` python3 eval_msra.py --model [saved model name] --test_path [testing set path]```

    for example 
    ```python3 eval_msra.py --model best_model.pth --test_path ../data/msra_preprocess/```

    we provided the pre-trained models ('./results/msra_handgraf_adam_rotaug/P0/best_model.pth') for MSRA

4. If a new training process is needed, please execute the following instructions after step 1 and 2 are completed

   go to "train_eval" directory

   . for training MSRA
    execute ``` python3 train_msra_adamw_rotaug.py --dataset_path [MSAR dataset path]```
    example ``` python3 train_msra_adamw_rotaug.py --dataset_path ../data/msra_preprocess/```