# PDnet
A PyTorch implementation of PDnet

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- tqdm
```
conda install tqdm
```
- opencv
```
conda install opencv
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [BSR_bsds500]
Train dataset has 200 images and Val dataset has 100 images.

## Usage

### Train
```
python train.py

optional arguments:
--crop                         training on cropped images 
--cropSize                   training images crop size [default value is 88]
--dataPath              select training dataset [default value is OASIS1700]
--epochs                  train epoch number [default value is 100]
--noiseLevel                select noise level on training [default value is 15]
```
The output val noisy and denoised images are on `logimg` directory. You may create on your own.

### Test Benchmark Datasets
```
python test.py

optional arguments:
--dataPath                      select training dataset [default value is OASIS1700]
--sweep                  generator model epoch name [default value is net.pth]
```