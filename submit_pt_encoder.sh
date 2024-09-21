lr=1e-3

# ckpt=/checkpoints/hoangmn/pretrain/367615/checkpoint-30.pth

audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv


dataset=audioset

python submitit_pretrain.py \
    --nodes 8 \
    --batch_size 32 \
    --model vit_b \
    --mask_ratio 0.8 \
    --epochs 160 \
    --warmup_epochs 10 \
    --save_every_epoch 10 \
    --lr $lr \
    --weight_decay 0.0001 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --label_csv $audioset_label \
    --train_encoder \
    --roll_mag_aug True \
    --decoder_mode 1 \
    --estimator_mode 2 \
    --codebook_type rq \
    --codebook_set 4 \
    --code_num 256 \
    --cold_start \
    # --prev_phase $ckpt \
