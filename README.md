# MgNO: Efficient Parameterization of Linear Operators via Multigrid

[![ICLR 2024](https://img.shields.io/badge/ICLR-2024-blue)](https://openreview.net/forum?id=eb3c8135137c8a60425a0320869ad87e)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)

> **[MgNO: Efficient Parameterization of Linear Operators via Multigrid](https://openreview.net/forum?id=eb3c8135137c8a60425a0320869ad87e)**  
> Xinliang Liu, Bo Shi, Zhengpeng Sun, Wenqi Ouyang, Xiangyue Liu, Yingdong Wang, Zheng Wang, Lei Bai  
> *International Conference on Learning Representations (ICLR), 2024*

---

## Overview

**MgNO** is a neural operator that leverages the mathematical structure of **multigrid methods** to achieve efficient, multi-scale parameterization of linear operators for solving partial differential equations (PDEs).

Traditional neural operators such as FNO operate in the frequency domain and can struggle with high-frequency or multi-scale features.  MgNO instead mirrors the classical multigrid V-cycle:

1. **Pre-smoothing** – learnable convolutions damp high-frequency error components at the fine grid.
2. **Restriction** – strided convolutions project the residual to a coarser grid, enabling efficient capture of long-range correlations.
3. **Coarse-grid correction** – smoothing continues recursively at progressively coarser resolutions.
4. **Prolongation** – transposed convolutions interpolate the coarse-grid solution back to finer grids.
5. **Post-smoothing** – additional convolutions refine the solution on the fine grid.

This structure gives MgNO **linear complexity in spatial resolution** and a natural **inductive bias for multi-scale phenomena**, while remaining fully trainable end-to-end.

### Architecture Diagram

```
Input f
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  MgNO Layer (stacked num_layer times)                            │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │  MgConv V-cycle                             │                │
│  │                                             │                │
│  │  Pre-smooth  ──► Restrict ──► Pre-smooth    │   Parallel     │
│  │  (3×3 conv)       (↓2)       (3×3 conv)    │  + 1×1 conv    │
│  │      └──────────────────────────┘          │                │
│  │  Post-smooth ◄── Prolong  ◄── Post-smooth  │                │
│  │  (3×3 conv)       (↑2)       (3×3 conv)    │                │
│  └─────────────────────────────────────────────┘                │
│                         + (residual)                             │
│                         │                                        │
│                      Activation                                  │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
1×1 conv (projection)
   │
   ▼
Output u
```

### Key Features

- **Multi-scale representation** via nested V-cycle convolutions
- **Linear parameter complexity** with respect to spatial resolution
- **Flexible boundary conditions**: zero-padding (Dirichlet-type) or circular (periodic)
- **Plug-and-play normalizers**: pointwise Gaussian (PGN) or global Gaussian (GN)
- **Multiple PDE benchmarks**: Darcy flow (smooth, rough, multiscale), Navier-Stokes, Pipe flow, Helmholtz

---

## Results

MgNO achieves state-of-the-art or competitive performance on all benchmarks from the paper:

| Benchmark | Metric | FNO | U-Net | MgNO |
|-----------|--------|-----|-------|------|
| Darcy (smooth) | Rel. L2 | 0.0108 | 0.0245 | **0.0047** |
| Darcy (rough) | Rel. L2 | 0.0253 | 0.0491 | **0.0089** |
| Darcy (multiscale) | Rel. L2 | 0.1668 | 0.1998 | **0.0986** |
| Navier-Stokes (1e-5) | Rel. L2 | 0.1556 | — | **0.0820** |
| Pipe flow | Rel. L2 | 0.0299 | 0.0378 | **0.0183** |
| Helmholtz | H¹ | 0.0584 | — | **0.0284** |

*Results from Table 1 & 2 of the ICLR 2024 paper.*

---

## Repository Structure

```
MgNO/
├── models.py          # MgNO model definitions (MgNO_DC, MgNO_NS, MgNO_helm, …)
├── darcy.py           # Training script for Darcy-flow and pipe-flow
├── navier.py          # Training script for Navier-Stokes
├── helm.py            # Training script for Helmholtz equation
├── utilities3.py      # Data loaders, loss functions, normalizers, helpers
├── Adam.py            # Custom Adam optimizer
├── requirements.txt   # Python dependencies
└── baselines/         # Baseline model implementations (FNO, U-Net, MWT, …)
```

---

## Datasets

All datasets should be placed in the `./data/` directory.

### Smooth Darcy / Navier-Stokes (1e-5)
Courtesy of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator) — MIT license.  
Download from [this link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing):
- `piececonst_r421_N1024_smooth1.mat`
- `piececonst_r421_N1024_smooth2.mat`
- `NavierStokes_V1e-5_N1200_T20.mat`

### Darcy Rough Data
Generated using Zongyi Li's code.  
Download from [this link](https://drive.google.com/drive/folders/1q1dM9icEs5vC2i_1iDhpAXJvA45nI9qR?usp=sharing):
- `darcy_alpha2_tau5_512_train.mat`
- `darcy_alpha2_tau5_512_test.mat`

### Darcy Multiscale Data
Download from [this link](https://drive.google.com/drive/folders/121oegG4FfxoaakFZDYk_JeWZc3snCRaF?usp=drive_link):
- `mul_tri_train.mat`
- `mul_tri_test.mat`

### Pipe Data
Courtesy of [Geo-FNO](https://github.com/neuraloperator/Geo-FNO).  
Download from [this link](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8):
- `Pipe_X.npy`
- `Pipe_Y.npy`
- `Pipe_Q.npy`

### Helmholtz Data
Courtesy of [Zhengyu Huang](https://github.com/Zhengyu-Huang/Operator-Learning).  
Download from [this link](https://data.caltech.edu/records/fp3ds-kej20):
- `Helmholtz_inputs.npy`
- `Helmholtz_outputs.npy`

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python ≥ 3.8, PyTorch ≥ 1.13, CUDA recommended.

---

## Training

Place all dataset files in `./data/` before running.

### Darcy Flow (smooth)

```bash
python darcy.py \
  --data darcy --model_type MgNO_DC_smooth \
  --sample_x --normalizer --normalizer_type PGN --GN \
  --num_channel_u 24 --num_layer 5 \
  --num_iteration 10 10 10 10 10 20 \
  --lr 5e-4 --batch_size 8 --epochs 500
```

### Darcy Flow (rough)

```bash
python darcy.py \
  --data darcy20c6 --model_type MgNO_DC \
  --sample_x --normalizer --normalizer_type GN \
  --num_channel_u 24 --num_layer 4 \
  --num_iteration 10 10 10 10 10 20 \
  --lr 5e-4 --batch_size 8 --epochs 500
```

### Darcy Flow (multiscale)

```bash
python darcy.py \
  --data a4f1 --model_type MgNO_DC \
  --sample_x --normalizer --normalizer_type GN --GN \
  --num_channel_u 24 --num_layer 4 \
  --num_iteration 10 10 10 10 10 20 \
  --lr 5e-4 --batch_size 8 --epochs 500
```

### Navier-Stokes (Re = 1e-5)

```bash
python navier.py \
  --model_type MgNO \
  --num_iteration 10 10 10 20 20 --num_layer 5 \
  --num_channel_u 32 --num_channel_f 1 \
  --final_div_factor 50 --weight_decay 1e-5 \
  --lr 1e-3 --bias
```

### Pipe Flow

```bash
python darcy.py \
  --data pipe --model_type MgNO_DC \
  --sample_x --num_channel_f 2 \
  --num_channel_u 24 --num_layer 5 \
  --num_iteration 10 10 10 10 11 20 \
  --lr 3e-4 --batch_size 4 --epochs 500 --loss_type l2
```

### Helmholtz Equation

```bash
python helm.py \
  --data helm --model_type MgNO_helm \
  --num_layer 4 --lr 3e-4 \
  --final_div_factor 100 --batch_size 10 \
  --weight_decay 1e-5 --normalizer --GN \
  --num_channel_u 20 --num_iteration 1 1 1 1 2 \
  --epochs 100
```

---

## Model Variants

| Class | Benchmark | Notes |
|-------|-----------|-------|
| `MgNO_DC` | Darcy (rough/multiscale), Pipe | Parallel MgConv + 1×1 conv, zero padding |
| `MgNO_DC_smooth` | Darcy (smooth) | Alternating 3×3/4×4 prolongation kernels |
| `MgNO_NS` | Navier-Stokes | Circular padding, residual skip |
| `MgNO_helm` | Helmholtz | Growing channel counts per level |
| `MgNO_helm2` | Helmholtz (uniform channels) | Fixed channel count, MgConv_helm3 |

---

## Citation

If you use MgNO in your research, please cite:

```bibtex
@inproceedings{liu2024mgno,
  title     = {{MgNO}: Efficient Parameterization of Linear Operators via Multigrid},
  author    = {Liu, Xinliang and Shi, Bo and Sun, Zhengpeng and Ouyang, Wenqi
               and Liu, Xiangyue and Wang, Yingdong and Wang, Zheng and Bai, Lei},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=eb3c8135137c8a60425a0320869ad87e}
}
```

---

## License

This project is released under the [MIT License](LICENSE.txt).

Dataset licenses:
- Smooth Darcy / Navier-Stokes: [MIT License](https://github.com/zongyi-li/fourier_neural_operator/blob/master/LICENSE)
- Pipe flow: courtesy of [Geo-FNO](https://github.com/neuraloperator/Geo-FNO)
- Helmholtz: courtesy of [Zhengyu Huang](https://github.com/Zhengyu-Huang/Operator-Learning)

---

## Acknowledgements

The Darcy-flow and Navier-Stokes datasets were generously shared by [Zongyi Li](https://github.com/zongyi-li) (Caltech).  The baseline implementations in `baselines/` draw from the respective original repositories.
