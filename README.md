# FGS-NeRF: A Fast Glossy Surface Reconstruction Method Based on Voxel and Reflection Directions

## Installation
Please first install [Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter).
We are using PyTorch version 2.1.1+cu121 and torch_scatter version 2.1.2.
```bash
pip install -r requirements.txt
```

## Datasets
- DTU
- shiny-blender
- smart-car  
The smart_car dataset is sourced from [ref-dvgo](https://github.com/gkouros/ref-dvgo).

## Running
We have provided the scripts in the ssh folder. You can modify the dataset directory and output directory in the .sh file.