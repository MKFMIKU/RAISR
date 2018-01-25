# RAISR
> A unofficial Python implementation of  [Google Rapid and Accurate Image Super Resolution](https://arxiv.org/pdf/1606.01299.pdf)

[![forthebadge](http://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)

## Prepare

Dataset will use [291](http://cv.snu.ac.kr/research/VDSR/train_data.zip)

```bash
./getdataset.sh
python train.py

```

## Train
```bash
usage: train.py [-h] [--rate RATE] [--patch PATCH] [--Qangle QANGLE]
                [--Qstrength QSTRENGTH] [--Qcoherence QCOHERENCE]
                [--datasets DATASETS]

RAISR

optional arguments:
  -h, --help            show this help message and exit
  --rate RATE           upscale scale rate
  --patch PATCH         image patch size
  --Qangle QANGLE       Training Qangle size
  --Qstrength QSTRENGTH
                        Training Qstrength size
  --Qcoherence QCOHERENCE
                        Training Qcoherence size
  --datasets DATASETS   path save the train dataset
```

## Todo:
- left test.py implment to get PSNR/SSIM/Runtime