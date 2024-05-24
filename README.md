# NDE: Neural Directional Encoding
This repo contains the training code and demo for NDE.
### [Project Page](https://lwwu2.github.io/nde/) | [Paper](https://arxiv.org/abs/2405.14847) | [Citation](#citation)

## Setup

- python 3.8
- CUDA 11.7
- pytorch 2.0.1
- pytorch-lightning 2.0.8
- nerfacc
- tinycudann (fp32)

We compile `tinycudann` with fp32 precision for stable optimization. This is done by set `TCNN_HALF_PRECISION=0` in [this line](https://github.com/NVlabs/tiny-cuda-nn/blob/235d1fde956dc04966940f9d1bec66aa3bdb705a/include/tiny-cuda-nn/common.h#L99).

### Dataset

- [NeRF-Synthetic (Materials)](https://github.com/bmild/nerf)
- [RefNeRF-Synthetic](https://dorverbin.github.io/refnerf/)
- [NeRO-GlossyReal](https://github.com/liuyuan-pal/NeRO)
- [Teaser](https://drive.google.com/file/d/1C4mP0dyb3tmtVJIKB-u8t4Io6KiqIC_8/view?usp=sharing)

### Pre-trained models

The pre-trained weights for both synthetic and real scenes can be found in [here](https://drive.google.com/file/d/1f0wFPVwXW62JnwnXjin_LmnhfFVEqk4Q/view?usp=sharing)

## Usage

1. Edit `configs/synthetic.yaml` or `configs/real.yaml` to set up dataset path and configure a training.
2. To train a model, run:
```shell
python train.py --experiment_name=EXPERIMENT_NAME --device=GPU_DEVICE\
                --config CONFIG_FILE --max_epochs=NUM_OF_EPOCHS # 4000 by default
```
3. For view synthesis results, see `demo/demo.ipynb`

## Citation
```
@inproceedings{wu2024neural,
  author = {Liwen Wu and Sai Bi and Zexiang Xu and Fujun Luan and Kai Zhang and Iliyan Georgiev and Kalyan Sunkavalli and Ravi Ramamoorthi},
  title = {Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling},
  booktitle = {CVPR},
  year = {2024}
}
```
