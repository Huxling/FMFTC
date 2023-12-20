# FMFTC

This repository contains the code used in our paper: FMFTC: Federated Multi Feature Trajectory Clustering

## Requirements

- Ubuntu OS
- Python >= 3.7 (Anaconda3 is recommended)
- PyTorch 1.12+
- A Nvidia GPU with cuda 10.2+

Please refer to the source code to install all required packages in Python.

## Data

* Our qstaxi trajectory clustering datasets are stored in `data` according to our **Ground Truth Generation algorithm**.
* We provide raw trajectory data for training.

## Preprocessing

The preprocessing step will generate all data required in the training stage.

For the qstaxi dataset, you can do as follows.
    ```shell
    $ cd Preprocess
    $ python preprocess.py
    $ python spatial_similarity.py
    $ python speed_similarity.py
    $ python temporal_similarity.py
    $ python merge_STD_similarity.py
    $ cd ..
    ```

## Train

1. Training with parameters

```shell
python main.py
```
2. The training produces two model `coordinator_checkpoint.pkl`, `participant_checkpoint.pkl` and `coordinator_NMI_BEST.pkl`, `participant_NMI_BEST.pkl`. `checkpoint` contains the latest trained model and `NMI_BEST` saves the model which has the best performance on the validation data. 

Some code comes from [ST2vec](https://github.com/zealscott/ST2Vec).
