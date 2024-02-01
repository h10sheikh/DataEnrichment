#!/bin/bash

python main_imagenet.py \
  --arch 'resnet18' \
  --exp_prefix 'baseline_lr_0.1_wd_1e-4' \
  --restart
  #--gpu 0 \
  #--workers 0 \
