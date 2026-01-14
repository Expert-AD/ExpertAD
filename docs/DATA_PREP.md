# nuScenes
Download nuScenes V1.0 full dataset data, CAN bus and map(v1.3) extensions, then follow the steps below to prepare the data.

**Download nuScenes, CAN_bus and Map extensions**

```shell
cd data
# Download nuScenes V1.0 full dataset data directly to (or soft link to) 
# Download CAN_bus and Map(v1.3) extensions directly to (or soft link to)
```
**Generate ego pose data**

```shell
python ego_pose_plugin.py
```

**Prepare Expert-UniAD data info**

```shell
cd nuScenes/data
mkdir infos
bash ./tools/uniad_create_data.sh
```
**Prepare Expert-VAD data info**

```shell
cd nuScenes/data
mkdir infos
python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag vad_nuscenes --version v1.0 --canbus ./data
```
# Bench2Drive

Download the Bench2Drive dataset and required auxiliary files, then follow the steps below to prepare the data.

## Download Bench2Drive Dataset

Download the dataset from the official repository and place it under `Bench2Drive/Bench2DriveZoo/datasets`
(or create a symbolic link to it).

- Dataset repository: https://github.com/Thinklab-SJTU/Bench2Drive
- **Note:** Different releases may have slightly different folder structures.
  If needed, please use symbolic links (`ln -s`) and adjust the corresponding data paths in the code.

```
    Bench2DriveZoo
    ├── ...                   
    ├── datasets/
    |   ├── bench2drive/
    |   |   ├── v1/                                          # Bench2Drive base 
    |   |   |   ├── Accident_Town03_Route101_Weather23/
    |   |   |   ├── Accident_Town03_Route102_Weather20/
    |   |   |   └── ...
    |   |   └── maps/                                        # maps of Towns
    |   |       ├── Town01_HD_map.npz
    |   |       ├── Town02_HD_map.npz
    |   |       └── ...
    |   ├── others
    |   |       └── b2d_motion_anchor_infos_mode6.pkl        # motion anchors for UniAD
    |   └── splits
    |           └── bench2drive_base_train_val_split.json    # trainval_split of Bench2Drive base 

```

## Prepare Bench2Drive data info

Run the following command:

```shell
cd mmcv/datasets
python prepare_B2D.py --workers 16   # workers used to prepare data
```

The command will generate `b2d_infos_train.pkl`, `b2d_infos_val.pkl`, `b2d_map_infos.pkl` under `data/infos`.

*Note: This command will be by default use all routes except those in data/splits/bench2drive_base_train_val_split.json as the training set.  It will take about 1 hour to generate all the data with 16 workers for Base set (1000 clips).*

## Structure of code

After installing and data preparing, the structure of our code will be as follows:

```
    Bench2DriveZoo
    ├── adzoo/
    |   ├── bevformer/
    |   ├── uniad/
    |   └── vad/                   
    ├── ckpts/
    |   ├── r101_dcn_fcos3d_pretrain.pth                   # pretrain weights for bevformer
    |   ├── resnet50-19c8e357.pth                          # image backbone pretrain weights for vad
    |   ├── bevformer_base_b2d.pth                         # download weights you need
    |   ├── uniad_base_b2d.pth                             # download weights you need
    |   └── ...
    ├── data/
    |   ├── bench2drive/
    |   |   ├── v1/                                        # Bench2Drive base 
    |   |   |   ├── Accident_Town03_Route101_Weather23/
    |   |   |   ├── Accident_Town03_Route102_Weather20/
    |   |   |   └── ...
    |   |   └── maps/                                      # maps of Towns
    |   |       ├── Town01_HD_map.npz
    |   |       ├── Town02_HD_map.npz
    |   |       └── ...
    │   ├── infos/
    │   │   ├── b2d_infos_train.pkl
    │   │   ├── b2d_infos_val.pkl
    |   |   └── b2d_map_infos.pkl
    |   ├── others
    |   |       └── b2d_motion_anchor_infos_mode6.pkl      # motion anchors for UniAD
    |   └── splits
    |           └── bench2drive_base_train_val_split.json  # trainval_split of Bench2Drive base 
    ├── docs/
    ├── mmcv/
    ├── team_code/  # for Closed-loop Evaluation in CARLA
```
---
<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval](./TRAIN_EVAL.md)