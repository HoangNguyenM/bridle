lr=5e-4

ckpt=/checkpoints/hoangmn/pretrain/367611/checkpoint-30.pth

audioset_bal_train_json=/fsx/hoangmn/audioset/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset/train_all.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv

dataset=audioset

python submitit_eval.py \
    --nodes 1 \
    --batch_size 128 \
    --model vit_b \
    --lr $lr \
    --weight_decay 0.0001 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --label_csv $audioset_label \
    --roll_mag_aug True \
    --decoder_mode 1 \
    --estimator_mode 2 \
    --prev_phase $ckpt \
    # --codebook_type rq \
    # --codebook_set 4 \
    # --code_num 256 \
    # --cold_start \
