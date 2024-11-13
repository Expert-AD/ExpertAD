# Installation



```shell
# git clone  first
```

## Expert-UniAD
**a. Create a conda virtual environment.**

```shell
conda create -n expert-uniad python=3.8 -y
conda activate expert-uniad
```
**b. Torch: Install PyTorch and torchvision following the official instructions.**

```shell
conda install cudatoolkit=11.1.1 -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.
```

**c. Make sure gcc>=5 in conda env.**
```shell
# If gcc is not installed:
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
```

**d. Set up the CUDA_HOME.**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
```


**e. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**f. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**g. Install mmdet3d from source code.**

```shell
cd mmdetection3d
git checkout v0.17.1
pip install scipy==1.7.3
pip install scikit-image==0.20.0
pip install -v -e .
```
**h. Install Expert-UniAD Requirements.**

```shell
pip install -r requirements_Expert-UniAD.txt
```
**i. Prepare pretrained weights.**

```shell
Models will be made public upon publications.
```



## Expert-VAD

Detailed package versions can be found in [requirements-Expert-VAD.txt](../requirements-Expert-VAD.txt).

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n expert-vad python=3.8 -y
conda activate expert-vad
```

> for step b~g you could reuse env Expert-UniAD

**b. Install PyTorch and torchvision following the.**

```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 
```

**c. Install gcc>=5 in conda env (optional).**

```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install timm.**

```shell
pip install timm
```

**f. Install mmdet3d.**

```shell
cd /path/to/mmdetection3d
git checkout -f v0.17.1
python setup.py develop
```

**g. Install nuscenes-devkit.**

```shell
pip install nuscenes-devkit==1.1.9
```
**h. Install Expert-VAD Requirements.**

```shell
pip install -r requirements_Expert-VAD.txt
```

**i. Prepare pretrained weights.**

```shell
Models will be made public upon publications.
```

-> Next Page: [Prepare Dataset](./DATA_PREP.md)