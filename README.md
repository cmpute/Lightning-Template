# Installation

```shell
# Create conda environment
conda create -n ltn python=3.9

# Instal Pytorch and CUDA development environment
conda activate ltn 
conda install pytorch torchvision pytorch-cuda=11.8 cuda-nvcc=11.8 cuda-libraries-dev=11.8 -c pytorch -c nvidia

# Install normal Python dependencies
pip install -r requirements.txt

# Setup this development package
pip install -e .
```

# Usage

> Note that the keywords, such as `--cfg_path`, are usually omissible in the command.

## Training

- From scratch: `python tools/main.py train --cfg_path configs/local/cifar_256.toml`
- From checkpoint: `python tools/main.py train --cfg_path configs/local/cifar_256.toml --resume output/example_50epochs.ckpt`
- Visualize: Install `tensorboard` and `tensorboard --logdir output/example/tensorboard`

## Testing

- Test on validation set: `python tools/main.py test --cfg_path configs/local/cifar_256.toml --ckpt_path output/example_50epochs.ckpt --val` 
- Test on testing set: `python tools/main.py test --cfg_path configs/local/cifar_256.toml --ckpt_path output/example_50epochs.ckpt`

## Inference

To be supported.

# Tested Environment

- Python: 3.9.16
- PyTorch: 2.0.1
- Lightning: 2.0.2

# References

The basic model is modified from [the official Auto-encoder example](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html) of the Lightning framework. 
