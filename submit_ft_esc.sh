#!/bin/bash

lr=5e-4

mask_t_prob=0.2
mask_f_prob=0.2

ckpt=/checkpoints/hoangmn/pretrain/366933/checkpoint-130.pth

dataset=esc50

python submitit_finetune.py \
    --nodes 4 \
    --model vit_b \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --data_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --weight_csv $audioset_train_all_weight \
    --finetune $ckpt \
    --lr $lr \
    --dist_eval \
    --batch_size 8 \
    --roll_mag_aug True \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 \
    --epochs 100 \
    --warmup_epochs 10 \
    --distributed_wrapper True \
    --mask_2d True \
    # --replacement\