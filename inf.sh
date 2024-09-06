#!/bin/bash
#SBATCH --job-name=audioset-ft
#SBATCH --partition=learnai4p
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=/checkpoints/%u/finetune/%A/%A.out
#SBATCH --error=/checkpoints/%u/finetune/%A/%A.err

audioset_bal_train_json=/fsx/hoangmn/audioset_resample/bal_train.json
audioset_train_all_json=/fsx/hoangmn/audioset_resample/train_all.json
audioset_eval_json=/fsx/hoangmn/audioset_resample/eval.json
audioset_label=/fsx/hoangmn/audioset/class_labels_indices.csv

dataset=audioset

if [ -z "$1" ]
then
    ckpt='/checkpoints/hoangmn/pretrain/365837/checkpoint-120.pth'
else
    ckpt=$1
fi


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_as.py \
--log_dir /checkpoints/hoangmn/finetune/$SLURM_JOB_ID \
--output_dir /checkpoints/hoangmn/finetune/$SLURM_JOB_ID \
--model vit_b \
--dataset $dataset \
--data_train $audioset_bal_train_json \
--data_eval $audioset_eval_json \
--label_csv $audioset_label \
--finetune $ckpt \
--batch_size 16 \
--eval \