lr=5e-4

ckpt=/checkpoints/hoangmn/pretrain/382368/checkpoint-49.pth

python submitit_pretrain.py \
    --nodes 8 \
    --batch_size 16 \
    --model vit_b \
    --mask_ratio 0.8 \
    --epochs 200 \
    --warmup_epochs 10 \
    --save_every_epoch 10 \
    --save_min_epoch 50 \
    --lr $lr \
    --weight_decay 0.0001 \
    --train_encoder \
    --decoder_mode 1 \
    --estimator_mode 2 \
    --ema \
    --audio_exp \
    --codebook_type rq \
    --codebook_set 4 \
    --code_num 256 \
    --kmeans_init \
    --prev_phase $ckpt \
    # --cold_start \
    # --precision fp32 \
    # --init_weight_multiplier 0.1 \
