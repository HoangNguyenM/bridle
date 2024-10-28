lr=5e-4

# ckpt=/checkpoints/hoangmn/pretrain/381990/checkpoint-30.pth

audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv

dataset=audioset

python submitit_pretrain.py \
    --nodes 8 \
    --batch_size 16 \
    --model vit_b \
    --mask_ratio 0.8 \
    --epochs 140 \
    --warmup_epochs 10 \
    --save_every_epoch 10 \
    --save_min_epoch 50 \
    --lr $lr \
    --weight_decay 0.0001 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --label_csv $audioset_label \
    --train_encoder \
    --roll_mag_aug True \
    --decoder_mode 1 \
    --estimator_mode 2 \
    --ema \
    --codebook_type rq \
    --codebook_set 4 \
    --code_num 256 \
    --kmeans_init \
    --precision fp32 \
    --cold_start \
    --soft_code 2 \
    --restart_unused_codes \
    # --prev_phase $ckpt \
    # --init_weight_multiplier 0.1 \
