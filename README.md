# stable-rq-beats

## References

This project builds upon the following repositories:

- [AST (Audio Spectrogram Transformer)](https://github.com/YuanGongND/ast)
- [AudioMAE](https://github.com/AudioMAE/AudioMAE)
- [VideoMAE](https://github.com/VideoMAE/VideoMAE)

## Environment Setup

To set up the environments required for this project, please follow the instructions below:

### Audio and Images

For audio and image processing, use the `env_mae.yml` file to create the environment:

```bash
conda env create -f env_mae.yml
conda activate your_env_name
```

### Video

For video processing, use the `env_videomae.yml` file to create the environment:

```bash
conda env create -f env_videomae.yml
conda activate your_env_name
```

## Experiments

### Audio

For audio experiments, we utilize the AudioSet dataset. The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/agkphysics/AudioSet). Please note that we are missing 15% of the data, as shown in the table below:

| Dataset   | Total Data | Missing Data |
|-----------|------------|--------------|
| AudioSet  | 100%       | 15%          |

Additionally, we use the ESC-50 dataset, which can be downloaded from [GitHub](https://github.com/karolpiczak/ESC-50.git).

### Images

For image experiments, we conduct our research on the ImageNet-1K dataset.

### Video

For video experiments, we utilize the Kinetics-400 dataset.
