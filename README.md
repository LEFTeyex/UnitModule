<div align="center"> 

## UnitModule: A Lightweight Joint Image Enhancement Module for Underwater Object Detection

</div>

<p align="center">

<a href="https://doi.org/10.1016/j.patcog.2024.110435">
    <img src="https://img.shields.io/badge/DOI-10.1016/j.patcog.2024.110435-blue" /></a>

<a href="https://arxiv.org/pdf/2309.04708.pdf">
    <img src="https://img.shields.io/badge/arXiv-2309.04708-rgb(179,27,27)" /></a>

<a href="https://github.com/LEFTeyex/UnitModule/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/LEFTeyex/UnitModule" /></a>

</p>

## Introduction

The official implementation of **UnitModule: A Lightweight Joint Image Enhancement Module for Underwater Object
Detection**.

## Installation

This project is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

- Python 3.8
- Pytorch 1.11.0+cu113

**Step 1.** Create a conda virtual environment and activate it.

```bash
conda create -n unitmodule python=3.8 -y
conda activate unitmodule
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

Linux and Windows

```bash
# Wheel CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

```bash
# Conda CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

**Step 3.** Install MMDetection and dependent packages.

```bash
pip install -U openmim
mim install mmengine==0.7.4
mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmyolo==0.5.0
pip install -r requirements.txt
```

## Dataset

The data structure DUO looks like below:

```text
# DUO

data
├── DUO
│   ├── annotations
│   │   ├── instances_train.json
│   │   ├── instances_test.json
│   ├── images
│   │   ├── train
│   │   ├── test
```

## Usage

### Training

```bash
bash tools/dist_train.sh configs/yolox/yolox_s_100e_duo.py 2
```

### Test

```bash
bash tools/dist_test.sh configs/yolox/yolox_s_100e_duo.py yolox_s_100e_duo.pth 2
```

## Cite

```
@article{liu2024unitmodule,
  title={UnitModule: A Lightweight Joint Image Enhancement Module for Underwater Object Detection},
  author={Liu, Zhuoyan and Wang, Bo and Li, Ye and He, Jiaxian and Li, Yunfeng},
  journal={Pattern Recognition},
  volume={151},
  pages={110435},
  year={2024},
  doi={10.1016/j.patcog.2024.110435},
}
```