import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root,bmask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_256/**/*.jpg'.format(img_root),
                              recursive=True)
        else:
            self.paths = glob('{:s}/{:s}_256/*'.format(img_root, split))


        self.mask_paths = glob('{:s}/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)
        self.bmask_paths = glob('{:s}/*.jpg'.format(bmask_root))
        self.N_bmask = len(self.bmask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))

        bmask = Image.open(self.bmask_paths[random.randint(0, self.N_bmask - 1)])
        bmask = self.mask_transform(bmask.convert('RGB'))
        return gt_img * mask, mask,bmask, gt_img

    def __len__(self):
        return len(self.paths)
