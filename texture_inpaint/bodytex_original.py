import random
import torch
from PIL import Image
from glob import glob
import os 

class Bodytex(torch.utils.data.Dataset):
    def __init__(self, img_root, img_transform, mask_transform,
                 split='train'):
        super(Bodytex, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform


        self.paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'images')))
        self.mask_paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'masks')))
        self.N_mask = len(self.mask_paths)
        self.bmask_paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'bmasks')))
        self.N_bmask = len(self.bmask_paths)

        # for i in range(len(self.paths)):
        #     a = self.paths[i].split("/")[-1].split('_normalized.png')[0]
        #     b =self.mask_paths[i].split("/")[-1].split('-completed-partial_mask.png')[0]
        #     if (a !=b):
        #         print(a)
        #         print(b)
        #         print(i)
       

    def __getitem__(self, index):

        #print(self.paths[index],self.mask_paths[index],self.bmask_paths[index])
        
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[index])
        mask = self.mask_transform(mask.convert('RGB'))

        bmask = Image.open(self.bmask_paths[index])
        bmask = self.mask_transform(bmask.convert('RGB'))
        return gt_img * mask, mask,bmask, gt_img

    def __len__(self):
        return len(self.paths)
