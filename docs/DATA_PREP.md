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
cd ExpertUniAD/data
mkdir infos
bash ./tools/uniad_create_data.sh
```
**Prepare Expert-VAD data info**

```shell
cd ExpertVAD/data
mkdir infos
python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag vad_nuscenes --version v1.0 --canbus ./data
```

---
<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval](./TRAIN_EVAL.md)