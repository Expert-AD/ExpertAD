# Expert-UniAD

###  Train <a name="ExpertUniAD-train"></a>

**Training Command**

```shell
# N_GPUS is the number of GPUs used. Recommended >=8.
cd /path/to/ExpertUniAD
conda activate ExpertUniAD
./tools/expertuniad_dist_train.sh ./projects/configs/stage2_e2e/base_e2e.py N_GPUS
```

### Evaluation <a name="ExpertUniAD-eval"></a>

**Eval Command**

```shell
# N_GPUS is the number of GPUs used.  Recommended =8.
cd /path/to/ExpertUniAD
conda activate ExpertUniAD
./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py /PATH/TO/YOUR/CKPT.pth N_GPUS
```

### Visualization <a name="vis"></a>

**visualization Command**


```shell
# please refer to  ./tools/uniad_vis_result.sh
python ./tools/analysis_tools/visualize/run.py \
    --predroot /PATH/TO/YOUR/RESULTS.pkl \
    --out_folder /PATH/TO/YOUR/OUTPUT \
    --demo_video test_demo.avi \
    --project_to_cam True
```



# Expert-VAD
### Train <a name="ExpertVAD-train"></a>
**Training Command**
```shell
# N_GPUS is the number of GPUs used. Recommended >=8.
cd /path/to/ExpertVAD
conda activate Expertvad
python -m torch.distributed.run --nproc_per_node=N_GPUS --master_port=2333 tools/train.py projects/configs/VAD/VAD_base.py --launcher pytorch --deterministic --work-dir path/to/save/outputs
```

### Eval<a name="ExpertVAD-eval"></a>
**Eval  Command(with 1 GPU)**

```shell
cd /path/to/VAD
conda activate vad
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_base.py /path/to/ckpt.pth --launcher none --eval bbox --tmpdir tmp
```
### Visualization

**Visualization Command**

```shell
cd /path/to/VAD/
conda activate vad
python tools/analysis_tools/visualization.py --result-path /path/to/inference/results --save-path /path/to/save/visualization/results
```



<- Last Page: [Prepare The Dataset](./DATA_PREP.md)























