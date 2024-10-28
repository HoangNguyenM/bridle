#!/bin/bash

lr=5e-4

mask_t_prob=0.2
mask_f_prob=0.2

ckpt=/checkpoints/hoangmn/pretrain/383686/checkpoint-199.pth

python submitit_finetune.py \
    --nodes 4 \
    --model vit_b \
    --finetune $ckpt \
    --lr $lr \
    --dist_eval \
    --audio_exp \
    --batch_size 32 \
    --nb_classes 1000 \
    --first_eval_ep 10 \
    --save_min_epoch 1000 \
    --epochs 200 \
    --warmup_epochs 5 \
    --distributed_wrapper True \
    # --mask_t_prob $mask_t_prob \
    # --mask_f_prob $mask_f_prob \
    # --mask_2d True \
    # --precision fp32 \
    # --replacement\