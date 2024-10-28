#!/bin/bash

lr=2e-4

mask_t_prob=0.2
mask_f_prob=0.2

ckpt=/checkpoints/hoangmn/pretrain/382046/checkpoint-130.pth

audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_eval_json=/fsx/hoangmn/audioset/eval.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv

audioset_bal_train_weight=/fsx/hoangmn/audioset/weight_bal_train.csv
audioset_train_all_weight=/fsx/hoangmn/audioset/weight_train_all.csv
dataset=audioset

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
    --epochs 200 \
    --warmup_epochs 10 \
    --distributed_wrapper True \
    --mask_2d True \
    --weight_sampler \
    --precision fp32 \
    # --replacement\