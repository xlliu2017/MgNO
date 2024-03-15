# MgNO: Efficient Linear Operators Parameterization through Multigrid

## Datasets
Darcy smooth datasets and navier stokes (1e-5) dataset have been made available courtesy of [Zongyi Li (Caltech)](https://github.com/zongyi-li/fourier_neural_operator) under the MIT license. 

### Smooth Data/Navier Stokes (1e-5)
Download from [this link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing):
- `piececonst_r421_N1024_smooth1.mat`
- `piececonst_r421_N1024_smooth2.mat`
- `NavierStokes_V1e-5_N1200_T20.mat`

### Darcy Rough Data
Generated using Zongyi Li's code. Download from [this link](https://drive.google.com/drive/folders/1q1dM9icEs5vC2i_1iDhpAXJvA45nI9qR?usp=sharing):
- `darcy_alpha2_tau5_512_train.mat`
- `darcy_alpha2_tau5_512_test.mat`

### Darcy Multiscale Data
Access [this link](https://drive.google.com/drive/folders/121oegG4FfxoaakFZDYk_JeWZc3snCRaF?usp=drive_link):
- `mul_tri_train.mat`
- `mul_tri_test.mat`

### Pipe Data
Made available courtesy of Geo-FNO. Download from [this link](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8).

### Helmholtz Data
Made available courtesy of [Huang](https://github.com/Zhengyu-Huang/Operator-Learning).
Download from [this link](https://data.caltech.edu/records/fp3ds-kej20)

## Requirements
To set up the required environment:

```bash
pip install -r requirements.txt
```

##  Training
Please put all the data into the ./data folder.

### darcy smooth experiment 
```train
python darcy.py  --data darcy  --model_type 'MgNO_DC_smooth' --sample_x  --normalizer --normalizer_type 'PGN' --GN  --num_channel_u 24 --num_layer 5 --num_iteration 10 10 10 10 10 20  --lr 5e-4 --batch_size 8 --epochs 500

```

### darcy rough experiments 
```train
python darcy.py  --data darcy20c6  --model_type 'MgNO_DC' --sample_x  --normalizer --normalizer_type 'GN'   --num_channel_u 24 --num_layer 4 --num_iteration 10 10 10 10 10 20  --lr 5e-4 --batch_size 8 --epochs 500

```
  
### darcy multiscale experiment 
```train
python darcy.py  --data a4f1  --model_type 'MgNO_DC' --sample_x  --normalizer --normalizer_type 'GN' --GN  --num_channel_u 24 --num_layer 4 --num_iteration 10 10 10 10 10 20  --lr 5e-4 --batch_size 8 --epochs 500

```
### Navier Stokes 2 (NS2) 
```train
python navier.py --model_type MgNO  --num_iteration 10 10 10 20 20 --num_layer 5 --num_channel_u 32 --num_channel_f 1   --final_div_factor 50 --weight_decay 1e-5 --lr 1e-3  --bias

```
### Pipe experiment
```train
python darcy.py  --data pipe  --model_type 'MgNO_DC' --sample_x  --num_channel_f 2  --num_channel_u 24 --num_layer 5 --num_iteration 10 10 10 10 11 20  --lr 3e-4 --batch_size 4 --epochs 500 --loss_type 'l2'

```

### Helmholtz experiment
```train
python helm.py --data helm  --epochs 100  --model_type 'MgNO_helm' --num_layer 4 --lr 3e-4 --final_div_factor 100 --batch_size 10 --weight_decay 1e-5 --normalizer  --GN --num_channel_u 20 --num_iteration 1 1 1 1 2 

```


