#!/bin/bash

if [ -z "$1" ]
then
	lr=1e-3
else
	lr=$1
fi

if [ -z "$2" ]
then
	ckpt=/checkpoints/hoangmn/pretrain/365837/checkpoint-120.pth
else
	ckpt=$2
fi

if [ -z "$3" ]
then
	model=vit_b
else
	model=$3
fi

audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_eval_json=/fsx/hoangmn/audioset/eval.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv

dataset=audioset

python submitit_finetune.py \
    --nodes 8 \
    --model $model \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --data_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --finetune $ckpt \
    --lr $lr \
    --epochs 30 \
    --warmup_epochs 2 \
    --first_eval_ep 2 \
    --dist_eval \
    --batch_size 8 \
    --roll_mag_aug True \