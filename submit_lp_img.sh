#!/bin/bash

lr=8e-3

mask_t_prob=0.2
mask_f_prob=0.2

ckpt=/checkpoints/hoangmn/pretrain/377311/checkpoint-149.pth

python submitit_finetune.py \
    --nodes 4 \
    --model vit_b \
    --finetune $ckpt \
    --lr $lr \
    --dist_eval \
    --audio_exp \
    --batch_size 8 \
    --nb_classes 1000 \
    --linear_probe \
    --first_eval_ep 10 \
    --epochs 200 \
    --warmup_epochs 10 \
    --distributed_wrapper True \
    # --mask_t_prob $mask_t_prob \
    # --mask_f_prob $mask_f_prob \
    # --mask_2d True \
    # --precision fp32 \
    # --replacement\