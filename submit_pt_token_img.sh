lr=2e-4

ckpt=/checkpoints/hoangmn/pretrain/379382/checkpoint-199.pth

python submitit_pretrain.py \
    --nodes 8 \
    --batch_size 16 \
    --model vit_b \
    --mask_ratio 0.8 \
    --epochs 50 \
    --warmup_epochs 2 \
    --save_every_epoch 10 \
    --save_min_epoch 29 \
    --lr $lr \
    --weight_decay 0.0001 \
    --decoder_mode 1 \
    --estimator_mode 2 \
    --prev_phase $ckpt \
    --audio_exp \
    --codebook_type rq \
    --codebook_set 4 \
    --code_num 256 \
    --ema \
    --kmeans_init \
    # --precision fp32 \
    