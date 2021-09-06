# Roto-translation equivariant CNNs for hippocampus segmentation on 3D MRI

This code contains a Pytorch implementation of discrete 3D roto-translational equivariant convolutions, developed during my master thesis at the [NeuroSpin](https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/NeuroSpin.aspx) lab of [CEA](https://www.cea.fr/).

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Table of environment variables](#table-of-environment-variables)
- [Repository structure](#repository-structure)
- [References](#references)
- [License](#license)

# Setup

## Requirements

The following packages are prerequisites:
- Python 3
- [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

## Setting up the environment

```sh
source PATH_TO_CONDA/bin/activate
```

You can edit the `name` field to name the conda environment about to be created to the name of your choice. Then execute the following command to create the environment with the required packages installed

```sh
conda env create -f environment.yml
```

## Using the environment

To activate the environment, do:

```sh
conda activate CUNet # You should use the name you put in environment.yml if you changed it
```

## Setting up the logs directory

In the root of the repo, do:

```sh
mkdir logs
```

## Setting up the configuration file

A `.env` file is used for the configuration, and a template of it can be found in the `.env.template` file. Make a copy of this file and rename it `.env` with:

```sh
cp .env.template .env
```

Then fill out the fields with the values corresponding to your use case.

> :warning: **Note on the GROUP field**: It should be removed completely from the file if you intend to use the CNN model and not the G-CNN one.

## Setting up the data structure

In order to be able to use the main script of this repository (in `main.py`), the data needs to have the following structure : 

```sh
.
├── imagesTr # Folder with the training and validation images.
├── labelsTr # Folder with the ground truth segmentations for the training and validation subjects.
├── imagesTs # Folder with the test images.
├── labelsTs # Folder with the ground truth segmentations for the test subjects, is any.
```

Please note that each subject's image and segmentation should have the same dimensions, and should have the same name (or at least should have names respecting the same alphabetical orders with regards to the other images).

> :warning: **Size of the images**: The images size should be as low as possible, so avoid passing whole brain images as input. Do not hesitate to use [ROI Loc](https://pypi.org/project/roiloc/) to automatically localize and crop the images and their segmentation around the hippocampus.

> :warning: **Minimal number of images per folder**: Due to a limitation in the implementation of the Datamodule, at least two images and two segmentations should be put in `imagesTr` and `labelsTr` respectively, and at least one image in `imagesTs`, and so irrespectively of the chosen use case (see [Usage](#Usage)). Please note that if you no not intend to train the model, the segmentations can be blank ones.

# Usage

There are three different *use cases* possible of the model: **training without prior checkpoints**, **loading from checkpoints and not resuming training**, **loading from checkpoints and resuming training**. The use case can be chosen through the environment variables.

## Training without prior checkpoints

The following variables should be set as:

```sh
LOAD_FROM_CHECKPOINTS=False
SHOULD_TRAIN=True
```

## Loading from checkpoints and not resuming training

The following variables should be set as:


```sh
LOAD_FROM_CHECKPOINTS=True
CHECKPOINTS_PATH=/path/to/you/checkpoints
SHOULD_TRAIN=False
```

## Loading from checkpoints and resuming training

The following variables should be set as:

```sh
LOAD_FROM_CHECKPOINTS=True
CHECKPOINTS_PATH=/path/to/you/checkpoints
SHOULD_TRAIN=True
```

## Using the model

After setting the variables to the desired use case, to run the model, use inside the activated environment:

```sh
python main.py
```

# Outputs

## Logs
- Execution logs can be found in the `.\logs` folder creted during installation.
- Tensorboard logs can be found in the `.\logs_tf` folder, inside subfolders named with the pattern `LOG_NAME-nb_layers-learning_rate-clip_value`, with `LOG_NAME` specified as a variable.

## Results

The results can be found in the `.\results` folder, inside subfolders named with the pattern `LOG_NAME-nb_layers-learning_rate-clip_value`, with `LOG_NAME` specified as a variable. They are comprised of:
- For each input image, plots of some slices in coronal and sagital orientation.
- For each input image, the predicted segmentation, in Nifty1 compressed format.
- A `metrics_report.csv` csv with the metrics for each input image, for each subfield
- A `metrics_report_summary.csv` csv with the mean, standard deviation, max and min values of the metrics for each subfield

# Table of environment variables
| Variable Name | Description | Default |
| --- | --- | --- |
| LOAD_FROM_CHECKPOINTS | Boolean to load from checkpoints. | False|
| CHECKPOINTS_PATH|Path to file with the checkpoints to load| None |
| SHOULD_TRAIN | Whether training should be performed. | True |
| CLASSES_NAME | Names of the classes, separated by a comma, like: background, Ca1, Ca2, DG, Ca3, Tail, Sub|
| PATH_TO_DATA | Path to the folder containing the data.|
| SUBSET_NAME | Substring of the file names to consider. Leave empty to use all data contained in the data folder.|
| BATCH_SIZE | Batch size for the datamodule.|
| NUM_WORKERS | Number of workers of the datamodule.|
| TEST_HAS_LABELS | Boolean indicating whether or not the test dataset has labels (in `./labelsTs`)| False |
| SEED|Seed for the train and val split generator| 1 |
| GROUP|Name of the group. **Remove this field from this file if you want to use a regular CNN model**.| None |
| GROUP_DIM|Dimension of the group.|
| OUT_CHANNELS|Number of output channels (classes).|
| FINAL_ACTIVATION|Type of final activation, can be "softmax" or "sigmoid".| softmax
| NONLIN|Non linearity, can be "relu", "leaky-relu", or "elu".| leaky-relu |
| DIVIDER|An integer to divide the number of channels of each layer with, in order to reduce the total number of parameters.|
| MODEL_DEPTH|Depth of the U-Net.|
| DROPOUT|Magnitude of the dropout.|
| LOGS_DIR|Path to the folder where Tensorboard logs should be saved.|
| LOG_NAME|Prefix of the name this particular run will be known as in Tensorboard and the results folder.|
| EARLY_STOPPING|Boolean to indicate whether or not to start training early when needed.| False |
| LEARNING_RATE|Learning rate for the trainer.| 0.001 |
| HISTOGRAMS|Boolean to store the histograms of gradients of weigths in Tensorboard.| False |
| GPUS|Identifier or number of the gpu to use| 1 |
| PRECISION|GPU precision to use (16, 32 or 64)| 32 |
| MAX_EPOCHS|Number of epochs to train| 30 |
| LOG_STEPS|Interval of steps to choose to log between.| 5 |
| GRADIENT_CLIP|Value of the gradient clipping, used to stabilize the network.| 0.5 |
| CMAP|Color map for the plots.| Oranges |



# Repository structure

```sh
.
├── environment.yml.template
├── .gitignore
├── LICENSE
├── README.md
├── logs # logging directory created during install
└── c_unet
    ├── architectures # Models
    │   ├── __init__.py
    │   ├── decoder.py
    │   ├── dilated_dense.py
    │   ├── encoder.py
    │   └── unet.py
    ├── groups # Groups definition
    │   ├── __init__.py
    │   ├── S4_group.py
    │   ├── T4_group.py
    │   └── V_group.py
    ├── __init__.py
    ├── layers # (Group) Convolution layers definition
    │   ├── __init__.py
    │   ├── convs.py
    │   └── gconvs.py
    ├── training # Pytorch lightning models and structures definition
    │   ├── datamodule.py
    │   ├── HomogeniseLaterality.py
    │   ├── __init__.py
    │   ├── lightningUnet.py
    │   └── tverskyLosses.py
    └── utils
        ├── concatenation # Custom concatenation layers
        │   ├── __init__.py
        │   ├── OperationAndCat.py
        │   └── ReshapedCat.py
        ├── dropout # Custom dropout layers
        │   ├── GaussianDropout.py
        │   └── __init__.py
        ├── helpers # Helper functions
        │   ├── helpers.py
        │   └── __init__.py
        ├── interpolation # Interpolation (upsampling) layers
        │   ├── Interpolate.py
        │   ├── ReshapedInterpolate.py
        │   └── __init__.py
        ├── __init__.py
        ├── logging # Logging definition module
        │   ├── logging.py
        │   ├── loggingConfig.yml # Logging configuration
        │   └── __init__.py
        ├── normalization # Custom normalization layers
        │   ├── __init__.py
        │   ├── ReshapedBatchNorm.py
        │   ├── ReshapedSwitchNorm.py
        │   └── SwitchNorm3d.py
        ├── plots # Logging definition module
        │   ├── __init__.py
        │   └── plot.py
        └── pooling # Custom pooling layers
            ├── __init__.py
            └── GPool3d.py
```


# References

Part of this repository was taken from the [Cubenet repository](https://github.com/danielewworrall/cubenet), which implements some model examples described in this [ECCV18 article](https://arxiv.org/abs/1804.04458):

```
@inproceedings{Worrall18,
  title     = {CubeNet: Equivariance to 3D Rotation and Translation},
  author    = {Daniel E. Worrall and Gabriel J. Brostow},
  booktitle = {Computer Vision - {ECCV} 2018 - 15th European Conference, Munich,
               Germany, September 8-14, 2018, Proceedings, Part {V}},
  pages     = {585--602},
  year      = {2018},
  doi       = {10.1007/978-3-030-01228-1\_35},
}
```

The code in `./c_unet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py), which corresponds to:

```
@article{SwitchableNorm,
  title={Differentiable Learning-to-Normalize via Switchable Normalization},
  author={Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li},
  journal={International Conference on Learning Representation (ICLR)},
  year={2019}
}
```

Some of the code in `./c_unet/architectures` was inspired from this [3D U-Net repository](https://github.com/JielongZ/3D-UNet-PyTorch-Implementation), as well as from the structure described in [Dilated Dense U-Net for Infant Hippocampus Subfield Segmentation](https://www.frontiersin.org/articles/10.3389/fninf.2019.00030/full):

```
@article{zhu_dilated_2019,
	title = {Dilated Dense U-Net for Infant Hippocampus Subfield Segmentation},
	url = {https://www.frontiersin.org/article/10.3389/fninf.2019.00030/full},
	doi = {10.3389/fninf.2019.00030},
	journaltitle = {Front. Neuroinform.},
	author = {Zhu, Hancan and Shi, Feng and Wang, Li and Hung, Sheng-Che and Chen, Meng-Hsiang and Wang, Shuai and Lin, Weili and Shen, Dinggang},
	year = {2019},
}
```

Some of the code for the losses in `./c_unet/training` was taken from this [repository regrouping segmentation losses](https://github.com/JunMa11/SegLoss), which corresponds to:

```
@article{LossOdyssey,
title = {Loss Odyssey in Medical Image Segmentation},
journal = {Medical Image Analysis},
volume = {71},
pages = {102035},
year = {2021},
author = {Jun Ma and Jianan Chen and Matthew Ng and Rui Huang and Yu Li and Chen Li and Xiaoping Yang and Anne L. Martel}
doi = {https://doi.org/10.1016/j.media.2021.102035},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521000815}
}
```

# License

This repository is covered by the MIT license, but some exceptions apply, and are listed below:
- The file in `./c_unet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py) by Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li, and is covered by the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/), as mentionned also at the top of the file.
