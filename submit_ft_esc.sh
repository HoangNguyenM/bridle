#!/bin/bash

lr=2e-3

mask_t_prob=0.2
mask_f_prob=0.2

ckpt=/checkpoints/hoangmn/pretrain/367618/checkpoint-130.pth

label_csv=/fsx/hoangmn/esc-50/esc_class_labels_indices.csv

dataset=esc50

for((fold=1;fold<=5;fold++));
do
    tr_data=/fsx/hoangmn/esc-50/esc_train_data_${fold}.json
    te_data=/fsx/hoangmn/esc-50/esc_eval_data_${fold}.json

    python submitit_finetune.py \
        --nodes 1 \
        --model vit_b \
        --dataset $dataset \
        --nb_classes 50 \
        --save_min_epoch 10000 \
        --data_train $tr_data \
        --data_eval $te_data \
        --label_csv $label_csv \
        --finetune $ckpt \
        --lr $lr \
        --dist_eval \
        --batch_size 8 \
        --roll_mag_aug True \
        --first_eval_ep 10 \
        --epochs 300 \
        --warmup_epochs 20 \
        --distributed_wrapper True \
        --mask_2d True \
        --mask_t_prob $mask_t_prob \
        --mask_f_prob $mask_f_prob \
        # --replacement\
done