#!/bin/bash

python main_imagenet.py \
  --arch 'resnet18' \
  --init_inst_param 1.0 \
  --lr_inst_param 0.8 \
  --wd_inst_param 1e-8 \
  --learn_inst_parameters \
  --exp_prefix 'with_inst_params_lr_0.8_wd_1e-8' \
  --restart
  #--gpu 0 \
  #--workers 0 \