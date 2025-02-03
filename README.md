# BRIDLE: Generalized Self-supervised Learning with Quantization

**Authors:**  
Hoang M. Nguyen¹², Satya N. Shukla², Qiang Zhang², Hanchao Yu²,  
Sreya D. Roy², Taipeng Tian², Lingjiong Zhu¹, Yuchen Liu²  

¹ Florida State University  
² Meta Platforms Inc.

## References

We want to thank the following amazing repositories, which serve as great references for our code:

- [BEATs](https://github.com/microsoft/unilm/tree/master/beats)
- [AST (Audio Spectrogram Transformer)](https://github.com/YuanGongND/ast/tree/master)
- [AudioMAE](https://github.com/facebookresearch/AudioMAE)
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)

## Environment Setup

To set up the environments required for this project, please follow the instructions below:

Use the `env_mae.yml` file to create the environment:

```bash
conda env create -f env_mae.yml
conda activate your_env_name
```

## Experiments

The default model for all of our experiments is ViT-B.

### Audio

For audio experiments, we utilize the AudioSet dataset. The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/agkphysics/AudioSet). Please note that due to Youtube video expiring, we are missing ~15% of the data, as shown in the table below:

| Data Segments     | Original Quantity | Obtained Quantity | Percentage |
|-------------------|-------------------|-------------------|------------|
| Balanced Train    | 22,160            | 18,685            | 84.3%      |
| Unbalanced Train  | 2,041,789         | 1,738,788         | 85.2%      |
| Evaluation        | 20,371            | 17,142            | 84.1%      |

Additionally, we use the ESC-50 dataset, which can be downloaded from [GitHub](https://github.com/karolpiczak/ESC-50.git). To prepare the data, we need to resample the frequency to 16k Hz, convert the audio to mono and create json files as in [AudioMAE](https://github.com/facebookresearch/AudioMAE). Additionally, during the finetuning phase, we find the weighted sampler in [AST](https://github.com/YuanGongND/ast/tree/master) beneficial. For data preparation process, please refer to `data/audio/` files, though you will need to modify the paths accordingly and run:
```bash
sbatch job.sh
```
For audio experiments with Residual quantization specifically, we notice that precision fp32 might be required to prevent numerical errors. To modify the hyperparameters, change the numbers in `submit_*.sh` files directly. For e.g., to pretrain the model on AudioSet2M, run:
```bash
bash submit_pt_encoder.sh
```
Our training pipelines take inspiration from [BEATs](https://github.com/microsoft/unilm/tree/master/beats), where we pretraining a representation model by tokens predictions instead of masked patch reconstruction. The model contains encoder-decoder and tokenizer-estimator architecture, where the two components are alternately trained, while freezing the other.
- Iter1 Encoder training: Use `submit_pt_encoder.sh`, set `--cold_start` to make sure we use a simple linear projection as the tokenizer.
- Iter2 Tokenizer training: Use `submit_pt_token.sh`, make sure to load the previous checkpoint using the variable `prev_phase`.
- Iter2 Encoder training: Use `submit_pt_encoder.sh`, do not use `--cold_start`, load the previous checkpoint with `prev_phase`.

Note that we have two different implementations for the codebooks, VQ (from [BEATs](https://github.com/microsoft/unilm/tree/master/beats)), and RQ (from [RQVAE](https://github.com/kakaobrain/rq-vae-transformer)). Adjust `codebook_type`, `codebook_set` (number of codebooks, default to 1 for VQ), `code_num` (number of codes per codebook), and `code_dim` (dimension of each code) accordingly. For RQ, we support both uniform initialization (1.0 recommended due to normalization), and k-means. From the experiments, we observe that k-means initialization works better.

### Images

For image experiments, we conduct our research on the ImageNet-1K dataset standalone. The model on the train split (~1.28M samples) and evaluate the model on the val split (50K samples, 1000 classes).

### Video

For video experiments, we utilize the Kinetics-400 dataset. Note that due to data availability, the dataset we use have dimensions 512 (W) x 288 (H) (for VideoMAE, this is 320 x 568, then read with 320 x 256 window). The model is trained on the train split (~240K samples) and evaluated on the val (same as test) split (~20K samples, 400 classes).

## Reference
If you find this project useful in your research, please consider citing us.

```bibtex
@article{nguyen2024bridle,
  title={BRIDLE: Generalized Self-supervised Learning with Quantization},
  author={Nguyen, Hoang M. and Shukla, Satya N. and Zhang, Qiang and Yu, Hanchao and Roy, Sreya D. and Tian, Taipeng and Zhu, Lingjiong and Liu, Yuchen},
  year={2024},
}
