#!/bin/bash

python main_cifar.py \
  --rand_fraction 0.4 \
  --exp_prefix 'filtered_with_epoch_80_40frac_corrupt' \
  --dataset_subset_ckpt 'weights_CL/cifar100/with_inst_params_lr_0.2_40frac_corrupt/epoch_80.pth.tar' \
  --subset_frac 0.60 \
  --restart