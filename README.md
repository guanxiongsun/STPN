# Spatio-temporal Prompting Network for Robust Video Feature Extraction

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Guanxiong Sun](https://sunguanxiong.github.io).

This repo is an official implementation of "Spatio-temporal Prompting Network for Robust Video Feature Extraction", accepted in ICCV 2023. This repository contains a PyTorch implementation of STPN based on [mmdetection](https://github.com/open-mmlab/mmdetection).


## Main Results
                                          |

## Installation

### Requirements:

- python 3.7
- pytorch 1.8.1
- torchvision 0.9.1
- mmdet 2.19.1
- mmcv-full 1.4.0
- GCC 7.5.0
- CUDA 10.1

### Option 1: Step-by-step installation

```bash
# conda create --name tdvit -y python=3.7
# source activate tdvit

# install PyTorch 1.8 with CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# install mmcv-full 1.3.17
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

# install other requirements
pip install -r requirements.txt
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can compile mmcv from source if you need to develop both mmcv and mmdet. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

## Data preparation

### Download Datasets

Please download ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](http://image-net.org/challenges/LSVRC/2015/downloads). After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

    ./data/ILSVRC/
    ./data/ILSVRC/Annotations/DET
    ./data/ILSVRC/Annotations/VID
    ./data/ILSVRC/Data/DET
    ./data/ILSVRC/Data/VID
    ./data/ILSVRC/ImageSets

**Note**: List txt files under `ImageSets` folder can be obtained from
[here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

### Convert Annotations

We use [CocoVID](mmdet/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase. In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```bash
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

```

## Usage

### Inference

This section will show how to test existing models on supported datasets.
The following testing environments are supported:

- single GPU
- single node multiple GPU
- multiple nodes

During testing, different tasks share the same API and we only support `samples_per_gpu = 1`.

You can use the following commands for testing:

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Optional arguments:

- `CHECKPOINT_FILE`: Filename of the checkpoint. You do not need to define it when applying some MOT methods but specify the checkpoints in the config.
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `bbox` is available for ImageNet VID, `track` is available for LaSOT, `bbox` and `track` are both suitable for MOT17.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file
- `--eval-options`: If specified, the key-value pair optional eval cfg will be kwargs for dataset.evaluate() function, it’s only for evaluation
- `--format-only`: If specified, the results will be formatted to the official format.

#### Examples of testing VID model

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test DFF on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   python tools/test.py configs/vid/tdvit/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```

2. Test DFF with 8 GPUs on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   ./tools/dist_test.sh configs/vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py 8 \
       --checkpoint checkpoints/dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172720-ad732e17.pth \
       --out results.pkl \
       --eval bbox
   ```

### Training

MMTracking also provides out-of-the-box tools for training models.
This section will show how to train _predefined_ models (under [configs](https://github.com/open-mmlab/mmtracking/tree/master/configs)) on standard datasets.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.

```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

**Important**: The default learning rate in all config files is for 8 GPUs.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., `lr=0.01` for 8 GPUs \* 1 img/gpu and `lr=0.04` for 16 GPUs \* 2 imgs/gpu.

#### Training on a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file or via CLI argument `--work-dir`.

#### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

Optional arguments remain the same as stated above.

If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

#### Examples of training VID model

1. Train DFF on ImageNet VID and ImageNet DET, then evaluate the bbox mAP at the last epoch.

   ```shell
   ./tools/dist_train.sh configs/vid/time_swin_lite/faster_rcnn_time_swint_lite_fpn_0.000025_3x_tricks_stride3_train.py 8
   ```
