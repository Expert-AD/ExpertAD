<div align="center">   

# ExpertAD:Enhancing Autonomous Driving Systems with Mixture of Experts

</div>

## Intro

<img src="sources/Overview.png" alt="teaser" style="zoom: 67%;" />

- We propose a **Perception Adapter (PA)** to amplify task-critical features and a **Mixture of Sparse Experts (MoSE)** to reduce the driving task interference.
- We propose **ExpertAD**, which is a novel framework that integrates the lightweight efficiency of the Mixture of Experts (MoE) framework into ADSs.
- We integrate ExpertAD into three state-of-the-art vision-only ADSs, and conduct large scale experiments to show improved planning effectiveness and lower inference latency across both open-loop and closed-loop datasets.

## Getting Started
   - [Installation](docs/INSTALL.md)
   - [Prepare Dataset](docs/DATA_PREP.md)
   - [Train/Eval](docs/TRAIN_EVAL.md)

## Results

**ExpertAD** achieves ExpertAD reduces average collision rates by up to 20% and inference latency by 25% compared to prior methods.

| Approach        | Avg.col ↓ | Avg.L2 ↓ | DS ↑  | SR ↑  | RC ↑  | Latency ↓        | GFLOPs ↓ | Params ↓ |
|-----------------|-----------|----------|-------|-------|-------|------------------|----------|----------|
| UniAD           | 0.31      | 1.03     | 44.62 | 14.09 | 68.68 | 534 ± 18 ms      | ~856     | ~89M     |
| **Expert-UniAD** | **0.24**  | **0.89** | **55.49** | **20.63** | **81.04** | **445 ± 20 ms** | **~728** | ~125M    |
| VAD             | 0.43      | 1.21     | 43.31 | 17.27 | 61.60 | 225 ± 25 ms      | ~558     | ~58M     |
| **Expert-VAD**   | **0.34**  | **1.10** | **52.53** | **19.53** | **76.73** | **157 ± 23 ms** | **~461** | ~90M     |
| VADv2           | 0.12      | 0.33     | 75.90 | 55.01 | 90.08 | 330 ± 18 ms      | ~660     | ~76M     |
| **Expert-VADv2** | **0.10**  | **0.28** | **78.18** | **58.34** | 89.32 | **258 ± 22 ms** | **~573** | ~105M    |

## Catalog
- [x] Code Release
- [x] Initialization

