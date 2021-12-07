#!/bin/bash

dataset='cifar10'
model="pretrained_models/macer/0.50.pth"
sigma=0.5
gpu=0
split='test'
skip=20
max=-1

# hyperparameters for ISS
alpha=0.001
max_loss=0.01
loss_type='absolute'
batch_size=200
max_size=100000
n0=2000

output="./outputs/ISS-"$dataset'-'$sigma'-'$loss_type'-'$n0'-'$batch_size'-'$max_size'-'$max_loss'.txt'

CUDA_VISIBLE_DEVICES=$gpu python certify_iss.py $dataset $model $sigma $output --batch_size $batch_size --loss_type $loss_type --max_loss $max_loss --max_size $max_size --n0 $n0 --alpha $alpha --skip $skip --max $max --split $split