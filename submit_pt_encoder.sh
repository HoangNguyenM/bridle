#!/bin/bash
if [ -z "$1" ]
then
    lr=5e-4
else
    lr=$1
fi


audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv


dataset=audioset

python submitit_pretrain.py \
--nodes 8 \
--batch_size 16 \
--model vit_b \
--mask_ratio 0.8 \
--epochs 200 \
--warmup_epochs 10 \
--save_every_epoch 10 \
--lr $lr \
--weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_train_all_json \
--label_csv $audioset_label \
--roll_mag_aug True \
--decoder_mode 1 \
--estimator_mode 2 \
--cold_start True \
# --codebook_type rq \
# --codebook_set 4 \
# --code_num 256 \