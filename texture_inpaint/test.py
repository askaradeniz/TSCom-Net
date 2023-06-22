import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from bodytex import Bodytex
from evaluation import evaluate, test
from net import PConvUNet
#from net_original import PConvUNet
from unet import UNet
from util.io import load_ckpt
import os
import torch.nn as nn

parser = argparse.ArgumentParser()
# training options
#parser.add_argument('--root', type=str, default='./3dbodytex_dataset')
parser.add_argument('--root', type=str, default='/data/3d_cluster/SHARP2022/Challenge1_clone/Track1-3DBodyTex.v2/tex-inpaint_data')
parser.add_argument('--snapshot', type=str, default='./ckpt/1000000.pth')
parser.add_argument('--image_size', type=int, default=2048)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Bodytex(args.root, img_transform, mask_transform, 'val', coarse_mode=True)

model = PConvUNet().to(device)
#model = UNet(in_channels=3, n_classes=3, padding=True, up_mode='upsample', batch_norm=True).to(device)
print(os.path.isfile(args.snapshot))
load_ckpt(args.snapshot, [('model', model)])
#model.load_state_dict(torch.load(args.snapshot))

model.eval()
#test(model, dataset_val, device)
evaluate(model, dataset_val, device, "result.png")

# dataset_train = Bodytex(args.root, img_transform, mask_transform, 'train', coarse_mode=False)
# dataset_val = Bodytex(args.root, img_transform, mask_transform, 'val', coarse_mode=False)
# dataset_eval = Bodytex(args.root, img_transform, mask_transform, 'eval', coarse_mode=False)
# print(len(dataset_train))
# print(len(dataset_val))
# print(len(dataset_eval))