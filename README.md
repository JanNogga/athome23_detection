# athome23_detection
Datasets and training scripts to reproduce the object detection system used by team NimbRo at RoboCup@Home 2023.


![Model Predictions](https://github.com/JanNogga/athome23_detection/assets/31764993/fbb58e28-f278-4564-a751-23d316a1080e)


## Datasets
The 21 datasets contain 30742 annotated object instances in 3130 frames. Bounding boxes and segmentation masks are available to any dataloader using the COCO format. To download everything, just clone this repository and use the helper script:

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

To load and explore the data using the [fiftyone](https://docs.voxel51.com/) UI in the overview notebook *dataset_overview.ipynb*, we recommend starting with a fresh conda environment using python 3.9 and default packages:

```bash
conda create -n athome23_detection python=3.9 anaconda && conda activate athome23_detection
```

Then install a few required packages:

```bash
pip install fiftyone pillow==9.5.0 pycocotools==2.0.6
```

Across all datasets, some objects might be inconsistently or incorrectly annotated. Our datasets are readily modified as you see fit in [CVAT](https://www.cvat.ai/) by using the CVAT backup files provided in *data/robocup_bordeaux_2023/robocup_data/cvat_backups* after running the download script.

## Object Detectors

We use [detectron2](https://detectron2.readthedocs.io/en/latest/) to finetune [MaskDINO](https://github.com/IDEA-Research/MaskDINO) pretrained on COCO. To set up detectron2, it is important that the installed version of PyTorch matches the CUDA version. In our case, we are using CUDA 11.7 and thus install:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git' shapely timm
```

To maximize performance or if this is tedious in your setup, we recommend starting from a PyTorch container like [nvcr.io/nvidia/pytorch:23.04-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-04.html#rel-23-04) instead. Finally, compile modules required for MaskDINO's pixel decoder:

```bash
cd contrib/MaskDINO/maskdino/modeling/pixel_decoder/ops && sh make.sh
```

The training notebook *model_training.ipynb* is then usable to finetune task-specific detectors. This notebook also demonstrates how to manipulate object annotations and mix the provided datasets in a reasonable manner using the dataset utility functions in *helpers/\*.py*.
