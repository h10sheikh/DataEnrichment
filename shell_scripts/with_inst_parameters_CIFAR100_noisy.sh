#!/bin/bash

python main_cifar.py \
  --rand_fraction 0.4 \
  --init_inst_param 1.0 \
  --lr_inst_param 0.2 \
  --wd_inst_param 0.0 \
  --learn_inst_parameters \
  --exp_prefix 'corrupt_label/with_inst_params_lr_0.2_40frac_corrupt_labels' \
  --restart