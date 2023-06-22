import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize
import os
import shutil
import numpy as np
import glob
import cv2
import random

# def evaluate(model, dataset, device, filename):
#     image, mask,bmask, gt = zip(*[dataset[i] for i in range(8)])
#     image = torch.stack(image)
#     mask = torch.stack(mask)
#     bmask = torch.stack(bmask)
#     gt = torch.stack(gt)
#     with torch.no_grad():
#         output, output_mask, output_bmask= model(image.to(device), mask.to(device),bmask.to(device))
#     output = output.to(torch.device('cpu'))
#     output_mask = output_mask.to(torch.device('cpu'))
#     output_bmask = output_bmask.to(torch.device('cpu'))
#     output_comp = mask * image + (1 - mask) * output

#     grid = make_grid(
#         torch.cat((unnormalize(image), mask,bmask, unnormalize(output),
#                    unnormalize(output_comp), unnormalize(gt)), dim=0))
#     save_image(unnormalize(output_comp), "result_single.png")
#     save_image(grid, filename)

def evaluate(model, dataset, device, filename):
    r = random.randint(0, len(dataset))
    # image, mask, bmask, gt, img_with_mask,mask_orig = zip(*[dataset[i] for i in range(8)])
    image, mask, bmask, gt, img_with_mask, mask_orig = zip(*[dataset[r]])
    
    image = torch.stack(image)
    mask = torch.stack(mask)
    bmask = torch.stack(bmask)
    mask_orig = torch.stack(mask_orig)
    gt = torch.stack(gt)
    img_with_mask = torch.stack(img_with_mask)
    with torch.no_grad():
        #output = model(img_with_mask.to(device))
        output, output_mask, output_bmask = model(image.to(device), mask.to(device),bmask.to(device))

    output = output.to(torch.device('cpu'))
    output_comp = mask_orig * image + (1 - mask_orig) * output

    # grid = make_grid(
    #     torch.cat((unnormalize(image), (output_mask.to(torch.device('cpu'))),(output_bmask.to(torch.device('cpu'))), unnormalize(output),
    #                unnormalize(output_comp), unnormalize(gt)), dim=0))
    grid = make_grid(
    torch.cat((unnormalize(image), mask, bmask, unnormalize(output),
                unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(unnormalize(output_comp), "result_single.png")
    save_image(grid, filename)



def dilate_texture(tex_im, kernel_size=3):
    tex_im_original = tex_im.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(tex_im, kernel)
    _, mask = cv2.threshold(tex_im_original,0,255,cv2.THRESH_BINARY)
    mask = cv2.erode(mask, kernel)
    mask = mask/255.
    result = dilated * (1-mask) + tex_im_original * mask
    return result

def test(model, dataset, device):
    base_dir = "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_54_base_2k/eval"
    out_dir = "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_54_inpaint_70000_nocoarse_test_coarse/eval"
    # base_dir = "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_64_2021_base/eval"
    # out_dir = "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_64_2021_pconv/eval"
    # base_dir = "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_janaldo_base/eval"
    # out_dir = "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_janaldo/eval"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(dataset)):
        path = dataset.paths[i]
        print(path)
        mesh_id = path.split("_normalized")[0].split(os.sep)[-1]
        fn = path.split(os.sep)[-1]
        print(mesh_id)
        mesh_folder = os.path.join(base_dir, mesh_id)
        new_mesh_folder = os.path.join(out_dir, mesh_id)
        shutil.copytree(mesh_folder, new_mesh_folder)
        out_path = os.path.join(out_dir, mesh_id, fn)
        print(out_path)
        (image, mask, bmask, gt, img_with_mask, mask_orig) = dataset[i]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        bmask = bmask.unsqueeze(0)
        gt = gt.unsqueeze(0)
        img_with_mask = img_with_mask.unsqueeze(0)
        mask_orig = mask_orig.unsqueeze(0)
        with torch.no_grad():
            #output = model(img_with_mask.to(device))
            output, output_mask, output_bmask = model(image.to(device), mask.to(device),bmask.to(device))

        output = output.to(torch.device('cpu'))
        output_comp = mask_orig * image + (1 - mask_orig) * output
        save_image(torch.clip(unnormalize(output_comp), 0, 1), out_path)
        
        # output = output.to(torch.device('cpu'))
        # output = unnormalize(output).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[:,:,::-1]
        # image = unnormalize(image).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[:,:,::-1]
        # mask = mask_orig.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
        # print(output.dtype, output.shape, output.max())
        # out = output
        # out = mask * image + (1 - mask) * out
        # out = (out*255).astype(np.uint8)
        # out = dilate_texture(out)
        # out = out.astype(np.uint8)
        # out = cv2.bilateralFilter(out,9,75,75)
        # out = mask * image + (1 - mask) * out/255.
        # out = (out*255.).astype(np.uint8)
        # out = dilate_texture(out)
        # cv2.imwrite(out_path, out)


        # img = unnormalize(image).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[:,:,::-1]
        # mask = mask_orig.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
        # mask = ((1-mask)*255.).astype(np.uint8)
        # img = (img*255.).astype(np.uint8)
        # output = cv2.inpaint(img,mask[:,:,0],35,cv2.INPAINT_TELEA)
        # cv2.imwrite(out_path, output)

        ## Dilate final texture.
        out = cv2.imread(out_path)
        out = dilate_texture(out, kernel_size=3)
        cv2.imwrite(out_path, out)