# Texture Inpainting with Partial Convolutions

An implementation for the texture-inpainting with partial convolutions.

[3DBooSTeR](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_49)

[Dataset](https://cvi2.uni.lu/3dbodytexv2/)

[Pretrained model](https://dropit.uni.lu/invitations?share=b8c75827528519c60e66) for inpainting with coarse textures.

The following repositories were used for the partial convolutions:

[Unofficial implementation](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv)

[Official implementation](https://github.com/NVIDIA/partialconv)



## Requirements
- Python 3.6+
- Pytorch 0.4.1+

```
pip install -r requirements.txt
```

## Usage

### Train
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py
```

### Fine-tune
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --finetune --resume <checkpoint_name>
```
### Test
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --snapshot <snapshot_path>
```