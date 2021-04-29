#!/usr/bin/env bash
nohup python main.py --mode train --num_domains 2 --w_hpf 1 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --train_img_dir data/celeba_hq/train --val_img_dir data/celeba_hq/val --img_size 128 --eval_every 5000  &> log.out &
