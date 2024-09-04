#!/bin/bash

if [ -z "$1" ]
then
	blr=1e-3
else
	blr=$1
fi

if [ -z "$2" ]
then
	ckpt=/checkpoints/hoangmn/pretrain
else
	ckpt=$2
fi

if [ -z "$3" ]
then
	model=vit_base_patch16
else
	model=$3
fi

audioset_bal_train_json=/fsx/hoangmn/audioset_resample/bal_train.json
audioset_unbal_train_json=/fsx/hoangmn/audioset_resample/unbal_train.json
audioset_eval_json=/fsx/hoangmn/audioset_resample/eval.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv
dataset=audioset

python submitit_finetune.py \
    --nodes 8 \
    --model $model \
    --dataset $dataset \
    --data_train $audioset_bal_train_json \
    --data_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --finetune $ckpt \
    --blr $blr \
    --epochs 30 \
    --warmup_epochs 2 \
    --first_eval_ep 2 \
    --dist_eval \
    --batch_size 8 \
    --roll_mag_aug True \