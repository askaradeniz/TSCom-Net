import random
import torch
from PIL import Image
from glob import glob
import os 
import numpy as np
from mask_generator import MaskGenerator
import torchvision.transforms.functional as TF
import random
import cv2
from PIL import ImageFilter

class Bodytex(torch.utils.data.Dataset):
    def __init__(self, img_root, img_transform, mask_transform,
                 split='train', coarse_mode=False):
        super(Bodytex, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.coarse_mode = coarse_mode
        self.split = split


        self.paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'images')))
        self.mask_paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'masks')))
        if coarse_mode == True:
            if "eval" in split:
                self.paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'coarse_images')))
            self.mask_paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'coarse_masks')))
            self.mask_paths_orig = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'masks')))

        self.N_mask = len(self.mask_paths)
        self.bmask_paths = sorted(glob('{:s}/{:s}/{:s}/*.png'.format(img_root, split,'bmasks')))
        self.N_bmask = len(self.bmask_paths)

        p = np.random.permutation(len(self.paths))
        self.paths = np.array(self.paths)[p]
        self.mask_paths = np.array(self.mask_paths)[p]
        if coarse_mode:
            self.mask_paths_orig = np.array(self.mask_paths_orig)[p]
        self.bmask_paths = np.array(self.bmask_paths)[p]

    def erode_texture(tex_im, kernel_size=25):
        tex_im_original = tex_im[:,:,:]
        tex_im_gray = cv2.cvtColor(tex_im_original, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        _, mask = cv2.threshold(tex_im_gray, 0, 255,cv2.THRESH_BINARY)
        mask = cv2.erode(mask, kernel)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = mask/255.
        result = tex_im * mask
        result = result.astype(np.uint8)
        #result = tex_im

        return result

    def __getitem__(self, index):

        # print(self.paths[index],self.mask_paths[index],self.bmask_paths[index])
        # print(index)
        
        gt_img = Image.open(self.paths[index]).convert("RGB")
        #gt_img = gt_img.resize(size=(512,512))
        gt_img_arr = np.array(gt_img)
        gt_img = self.img_transform(gt_img)
        

        mask = Image.open(self.mask_paths[index]).convert("RGB")
        #mask = mask.resize(size=(512,512))
        mask_arr = np.array(mask)/255.
        mask = self.mask_transform(mask)
        if self.coarse_mode:
            mask_orig = Image.open(self.mask_paths_orig[index]).convert("RGB")
            #mask_orig = mask_orig.resize(size=(512,512))
            mask_orig = self.mask_transform(mask_orig)

        img_arr = gt_img_arr*mask_arr
        img = Image.fromarray(img_arr.astype(np.uint8))
        img_with_mask = img_arr + (1-mask_arr)*127.
        img_with_mask = Image.fromarray(img_with_mask.astype(np.uint8))
        img = self.img_transform(img)
        img_with_mask = self.img_transform(img_with_mask)

        bmask = Image.open(self.bmask_paths[index])
        bmask = self.mask_transform(bmask.convert('RGB'))

        #print(img.size(), mask.size(), bmask.size(), gt_img.size(), img_with_mask.size(), mask_orig.size())

        if self.coarse_mode:
            return img, mask, bmask, gt_img, img_with_mask, mask_orig
        else:
            return img, mask, bmask, gt_img, img_with_mask, mask


    def __len__(self):
        return len(self.paths)