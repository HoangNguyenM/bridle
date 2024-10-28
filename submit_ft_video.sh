#!/bin/bash

lr=7.5e-4

mask_t_prob=0.2
mask_f_prob=0.2

ckpt=/checkpoints/hoangmn/pretrain/383686/checkpoint-199.pth

dataset=k400

data_path=/datasets01/kinetics/092121/400
data_train=/fsx/hoangmn/kinetics400/train.csv
data_eval=/fsx/hoangmn/kinetics400/val.csv
label_csv=/fsx/hoangmn/kinetics400/kinetics_400_labels.csv

python submitit_finetune.py \
    --dataset $dataset \
    --data_path $data_path \
    --data_train $data_train \
    --data_eval $data_eval \
    --label_csv $label_csv \
    --nodes 8 \
    --model vvitb_224 \
    --finetune $ckpt \
    --lr $lr \
    --dist_eval \
    --audio_exp \
    --batch_size 6 \
    --nb_classes 400 \
    --num_sample 2 \
    --new_height 288 \
    --new_width 512 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --first_eval_ep 10 \
    --save_min_epoch 40 \
    --epochs 75 \
    --warmup_epochs 5 \
    --distributed_wrapper True \
    # --mask_t_prob $mask_t_prob \
    # --mask_f_prob $mask_f_prob \
    # --precision fp32 \