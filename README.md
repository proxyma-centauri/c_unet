# Roto-translation equivariant CNNs for hippocampus segmentation on 3D MRI

This code contains a Pytorch implementation of discrete 3D roto-translation equivariant convolutions.

# Usage

**TODO**

# Installation

The following section describes the installation step to use the code in this repository ; they were only tested on computers running **Ubuntu 18.04**.

## Requirements

The following packages are prerequisites:
- Python 3
- [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

## Setup

### Setting up the environment

```sh
source PATH_TO_CONDA/bin/activate
```

Copy the `environment.yml.template` file, and nmae the copy `environment.yml`. You can edit `NAME_OF_ENV` to name the conda environment about to be created. The execute the following command to create the environment with the required packages installed

```sh
conda env create -f environment.yml
```

### Using the environment

To activate the environment, do:

```sh
conda activate NAME_OF_ENV # Replace NAME_OF_ENV by the name you put in environment.yml
```

### Setting up the logs directory

In the root of the repo, do:

```sh
mkdir logs
```

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
    └── utils
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
        └── pooling # Custom pooling layers
            ├── __init__.py
            ├── ReshapedAvgPool.py
            └── ReshapedMaxPool.py
```


# References

Part of this repository was taken from the [Cubenet repository](https://github.com/danielewworrall/cubenet), which implements some model examples described in this ECCV18 article:

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

Some of the code in `./c_unet/architectures` was inspired from this [3D U-Net repository](https://github.com/JielongZ/3D-UNet-PyTorch-Implementation).


# License

**TODO**
