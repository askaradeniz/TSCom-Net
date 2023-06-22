import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as TF

import opt
from bodytex import Bodytex
from evaluation import evaluate, test
from loss import InpaintingLoss
from net import PConvUNet
from unet import UNet
from net import VGG16FeatureExtractor
from places2 import Places2
from util.io import load_ckpt
from util.io import save_ckpt


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/data/3d_cluster/SHARP2022/Challenge1_clone/Track1-3DBodyTex.v2/tex-inpaint_data')
parser.add_argument('--save_dir', type=str, default='/data/akaradeniz/experiments/tex-inpaint/snapshots/default_check')
parser.add_argument('--log_dir', type=str, default='./logs/default_check')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1100000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10)
parser.add_argument('--vis_interval', type=int, default=10)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=2048)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--coarse', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

if args.coarse:
    dataset_train = Bodytex(args.root, img_tf, mask_tf, 'train', True)
    dataset_val = Bodytex(args.root, img_tf, mask_tf, 'eval', True)
else:
    dataset_train = Bodytex(args.root, img_tf, mask_tf, 'train', False)
    dataset_val = Bodytex(args.root, img_tf, mask_tf, 'eval', False)

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))
print(len(dataset_train))
model = PConvUNet().to(device)
#model = UNet(in_channels=3, n_classes=3, padding=True, up_mode='upsample', batch_norm=True).to(device)

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, args.max_iter)):
    model.train()
    if args.coarse:
        image, mask, bmask, gt, img_with_mask,mask_orig = [x.to(device) for x in next(iterator_train)]
        output, _, _ = model(image, mask, bmask)
        #output = model(img_with_mask)
        loss_dict = criterion(image, mask_orig, bmask, output, gt)
    else:
        image, mask, bmask, gt, img_with_mask, _ = [x.to(device) for x in next(iterator_train)]
        output, _, _ = model(image, mask, bmask)
        #output = model(img_with_mask)
        loss_dict = criterion(image, mask, bmask, output, gt)


    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)
        # torch.save(model.state_dict(), '{:s}/ckpt/{:d}_test.pth'.format(args.save_dir, i + 1))

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        # evaluate(model, dataset_val, device,
        #          '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))
        test(model, dataset_val, device)
        break

writer.close()
