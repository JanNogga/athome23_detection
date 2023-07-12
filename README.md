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
