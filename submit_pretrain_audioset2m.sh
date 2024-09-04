#!/bin/bash
if [ -z "$1" ]
then
    blr=2e-4
else
    blr=$1
fi


audioset_bal_train_json=/fsx/hoangmn/audioset_resample/bal_train.json
audioset_unbal_train_json=/fsx/hoangmn/audioset_resample/unbal_train.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv


dataset=audioset

python submitit_pretrain.py \
--nodes 1 \
--batch_size 8 \
--model vit_b \
--mask_ratio 0.8 \
--epochs 20 \
--warmup_epochs 4 \
--save_every_epoch 10 \
--blr $blr --weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_unbal_train_json \
--label_csv $audioset_label \
--roll_mag_aug True \
--decoder_mode 1 \
--estimator_mode 2 \
--cold_start True \