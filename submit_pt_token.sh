lr=2e-4

ckpt=/checkpoints/hoangmn/pretrain/374370/checkpoint-130.pth

audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv


dataset=audioset

python submitit_pretrain.py \
    --nodes 8 \
    --batch_size 16 \
    --model vit_b \
    --mask_ratio 0.8 \
    --epochs 31 \
    --warmup_epochs 2 \
    --save_every_epoch 10 \
    --save_min_epoch 19 \
    --lr $lr \
    --weight_decay 0.0001 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --label_csv $audioset_label \
    --roll_mag_aug True \
    --decoder_mode 1 \
    --estimator_mode 2 \
    --prev_phase $ckpt \
    --codebook_type rq \
    --codebook_set 4 \
    --code_num 256 \
    --precision fp32 \
    --ema \
    # --kmeans_init \
    # --soft_code 4 \
    # --restart_unused_codes \