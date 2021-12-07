
# Input-Specific Robustness Certification for Randomized Smoothing

This repository is the official implementation of *Input-Specific Robustness Certification for Randomized Smoothing*. 

## Repository Overview

| File            | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `code/certify_iss.py`    | Certify with Input-specific Sampling|
| `code/model.py`       | Model architectures   |


To install requirements:

```setup
pip install -r requirements.txt
```

>📋  The code requires Python >=3.6. PyTorch may need to be [installed manually](https://pytorch.org/) because of different platforms and CUDA drivers.

## Certify with ISS
To evaluate predictions of base smoothed classifiers and get certified accuracy, run this command:

```certify with ISS
CUDA_VISIBLE_DEVICES=$gpu python code/certify_iss.py cifar10 <directory for checkpoint> $sigma $<output filename> --batch_size $batch_size --loss_type <absolute or relative> --max_size 0.05 --n0 $n0 --alpha 0.001
```
