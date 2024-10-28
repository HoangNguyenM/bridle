lr=1.2e-3

# ckpt=/checkpoints/hoangmn/pretrain/380211/checkpoint-49.pth

dataset=k400

data_path=/datasets01/kinetics/092121/400
data_train=/fsx/hoangmn/kinetics400/train.csv

python submitit_pretrain.py \
    --dataset $dataset \
    --data_path $data_path \
    --data_train $data_train \
    --nodes 8 \
    --batch_size 32 \
    --mask_type tube \
    --model videomae_b \
    --mask_ratio 0.9 \
    --epochs 801 \
    --warmup_epochs 40 \
    --save_every_epoch 20 \
    --save_min_epoch 100 \
    --lr $lr \
    --weight_decay 0.0001 \
    --decoder_depth 4 \
    --num_frames 16 \
    --sampling_rate 4 \
    --train_encoder \
    --ema \
    --audio_exp \
    --cold_start \
    --codebook_type rq \
    --codebook_set 4 \
    --code_num 256 \
    --kmeans_init \
    # --prev_phase $ckpt \
    # --precision fp32 \
