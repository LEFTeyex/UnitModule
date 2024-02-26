# <center> UnitModule

### Installation

This project is based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main).

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

### Dataset

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

### Training

```bash
bash tools/dist_train.sh configs/yolox/yolox_s_100e_duo.py 2
```

### Test

```bash
bash tools/dist_test.sh configs/yolox/yolox_s_100e_duo.py yolox_s_100e_duo.pth 2
```