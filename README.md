# athome23_detection
Datasets and training scripts to reproduce the object detection system used by team NimbRo at RoboCup@Home 2023.

## Datasets
The datasets contain 30742 annotated object instances in 3130 frames. Bounding boxes and segmentation masks are available to any dataloader using the COCO format. To download everything, just clone this repository and use the helper script.

```bash
git clone --recursive --depth 1 https://github.com/JanNogga/athome23_detection.git
```

```bash
cd athome23_detection
```

```bash
chmod +x helpers/download_data.sh
```

```bash
./helpers/download_data.sh
```

If you want to load and explore the data using the overview notebook, we recommend starting with a fresh conda environment using python 3.9 and default packages

```bash
conda create -n athome23_detection python=3.9 anaconda && conda activate athome23_detection
```

Then install a few required packages

```bash
pip install fiftyone pillow==9.5.0 pycocotools==2.0.6
```

Over all datasets, some objects might be inconsistenly or even incorrectly annotated. You can easily modify our datasets as you see fit in CVAT by using the CVAT backup files provided in *data/robocup_bordeaux_2023/robocup_data/cvat_backups* after running the download script.

## Object detectors

We use detectron2 to finetune MaskDINO pretrained on COCO, yielding a detector for a given RoboCup@Home task in ~30 minutes. To setup detectron2, it is important that the installed version of PyTorch matches your CUDA version. In our case, we are using CUDA 11.7 and thus use

```bash
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If this is tedious in your setup or you want to maximize performance, we recommend starting from a PyTorch container like [nvcr.io/nvidia/pytorch:23.04-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-04.html#rel-23-04).

