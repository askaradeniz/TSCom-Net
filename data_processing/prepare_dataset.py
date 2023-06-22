"""description: Prepares the SHARP dataset with the following structure:
    -----Challenge1
    ---------Track1-3DBodyTex.v2
    ----------------train
    ----------------------170523-003-m-hnl1-bfc9-low-res-result (mesh id)
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized.obj (complete mesh)
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized.mtl
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized.png
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized-partial-01.obj (partial mesh 1)
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized-partial-01.mtl
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized-partial-01.png
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized-partial-02.obj (partial mesh 2)
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized-partial-02.mtl
    ----------------------------- 170523-003-m-hnl1-bfc9-low-res-result_normalized-partial-02.png
    ----------------------------- ...

    Install https://gitlab.uni.lu/skali/sharp-2022
    Use the cli of sharp to convert npz to obj.
    Bring all the obj, mtl and png files for both complete and partial meshes together.
"""

from functools import partial
import glob
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import random
import trimesh
import copy

def prepare_sharp2022():
    base_dir = "/data/3d_cluster/SHARP2022/Challenge1_clone/Track1-3DBodyTex.v2/"
    gt_dir = os.path.join(base_dir, "train") ## Directory for the complete 3D files.
    partial_dir = os.path.join(base_dir, "partial_data") ## Directory for the partial 3D files.
    final_dir = os.path.join(base_dir, "track1_dataset/train") ## Final directory for the combined complete + partial 3D files.

    os.makedirs(final_dir, exist_ok=True)

    gt_files = glob.glob(gt_dir + "/*/*.npz")
    partial_files = glob.glob(partial_dir + "/*/*")

    print(len(gt_files), len(partial_files))

    ## Convert complete meshes from npz to obj.
    ## Create a folder for each mesh in the final directory.
    ## Copy complete and partial files with the same mesh id to this folder.
    for i in range(len(gt_files)):
        print("Preparing the dataset %d/%d" % ((i+1), len(gt_files)))
        gt_path = gt_files[i]
        mesh_id = gt_path.split(os.sep)[-2]
        mesh_folder = os.path.join(final_dir, mesh_id)
        os.makedirs(mesh_folder, exist_ok=True) ## Create a folder with the mesh id.
        
        gt_fn = os.path.basename(gt_path)
        gt_path_new = os.path.join(mesh_folder, gt_fn.replace("npz","obj"))
        print("Converting %s.." % (gt_fn))
        os.system("python -m sharp_challenge1 convert %s %s" % (gt_path, gt_path_new))
        print("Copying partial files for the %s" % mesh_id)
        for partial_path in partial_files:
            partial_mesh_id = partial_path.split(os.sep)[-2]
            if partial_mesh_id == mesh_id:
                shutil.copy(partial_path, mesh_folder) ## Copy the partial file.
    print("Done.")

def prepare_sharp2021_eval():
    base_dir = "/data/3d_cluster/SHARP2021/Challenge1/Track1-3DBodyTex.v2/eval-partial/"
    partial_dir = os.path.join(base_dir, "eval-all") ## Directory for the partial 3D files.
    final_dir = os.path.join(base_dir, "eval-all-data") ## Final directory for the combined complete + partial 3D files.

    os.makedirs(final_dir, exist_ok=True)

    partial_files = glob.glob(partial_dir + "/*/*")

    print(len(partial_files))

    ## Convert complete meshes from npz to obj.
    ## Create a folder for each mesh in the final directory.
    ## Copy complete and partial files with the same mesh id to this folder.
    for i in range(len(partial_files)):
        partial_path = partial_files[i]
        print("Preparing the dataset %d/%d" % ((i+1), len(partial_files)))
        mesh_id = os.path.basename(os.path.dirname(partial_path))
        mesh_folder = os.path.join(final_dir, mesh_id)
        os.makedirs(mesh_folder, exist_ok=True) ## Create a folder with the mesh id.
        
        fn = os.path.basename(partial_path)
        partial_path_new = os.path.join(mesh_folder, fn.replace("npz","obj"))
        print("Converting %s.." % (fn))
        print(partial_path, partial_path_new)
        os.system("python -m sharp_challenge1 convert %s %s" % (partial_path, partial_path_new))
    print("Done.")


def mask_to_coarse_mask(mask, mesh, texture_res=512):
    uvs = mesh.visual.uv
    texture_res = mask.shape[0]
    coords = np.clip((uvs*(texture_res-1)).astype(np.int32), 0, texture_res-1)
    coarse_mask = mask.copy()
    for i in range(len(coords)):
        u, v = coords[i][0], coords[i][1]
        v = texture_res-1-v

        coarse_mask[v,u] = 255

    return coarse_mask

def make_partial_mask(texture, background):
    """Create the mask of the missing foreground regions of the texture atlas.

    Args:
        texture: (h, w, 3) Texture image with RGB values in [0, 1].
        background: (h, w) Boolean mask of the background of the texture atlas.
            I.e. foreground is True, background id False.

    Returns:
        mask: (h, w) Boolean mask of the missing regions of the foreground of
            the texture atlas. False if missing, True otherwise.
        overlay: (h, w, 3) Missing regions overlaid over the texture image.
    """
    mask = np.ones(texture.shape[:2], dtype=bool)
    overlay = copy.deepcopy(texture)

    threshold_notcolored = 0.01
    notcolored = np.sqrt((texture ** 2).sum(axis=-1)) <= threshold_notcolored
    missing = notcolored & background

    mask[missing] = False
    overlay[missing] = 1

    return mask, overlay

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


## Dataset for training texture inpainting.
def prepare_texture_inp_data():
    base_dir = "/data/3d_cluster/SHARP2022/Challenge1_clone/Track1-3DBodyTex.v2"
    data_dir = os.path.join(base_dir, "track1_data/train")
    train_dir = os.path.join(base_dir, "tex-inpaint_data/train")
    val_dir = os.path.join(base_dir, "tex-inpaint_data/val")
    
    masks_train_dir = os.path.join(train_dir, "masks")
    bmasks_train_dir = os.path.join(train_dir, "bmasks")
    images_train_dir = os.path.join(train_dir, "images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(masks_train_dir, exist_ok=True)
    os.makedirs(bmasks_train_dir, exist_ok=True) 
    os.makedirs(images_train_dir, exist_ok=True)
    masks_val_dir = os.path.join(val_dir, "masks")
    bmasks_val_dir = os.path.join(val_dir, "bmasks")
    images_val_dir = os.path.join(val_dir, "images")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(masks_val_dir, exist_ok=True)
    os.makedirs(bmasks_val_dir, exist_ok=True) 
    os.makedirs(images_val_dir, exist_ok=True)


    gt_paths = glob.glob(data_dir + "/*/*_normalized.png")
    paths = glob.glob(data_dir + "/*/*-partial*.png")
    random.shuffle(gt_paths)
    train_gt_paths = gt_paths[int(len(gt_paths)*0.1):]
    val_gt_paths = gt_paths[:int(len(gt_paths)*0.1)]
    #print(train_gt_paths[:5])

    
    for gt_path in tqdm(gt_paths):
        gt_fn = gt_path.split(os.sep)[-1]
        mesh_dir = os.path.dirname(gt_path)
        mesh_id = mesh_dir.split(os.sep)[-1]
        partial_paths = glob.glob(mesh_dir+"/*-partial*.png")
        for path in partial_paths:
            fn = path.split(os.sep)[-1]
            mask_out_path = os.path.join(masks_train_dir, fn)
            bmask_out_path = os.path.join(bmasks_train_dir, fn)
            im_out_path = os.path.join(images_train_dir, fn)

            if gt_path in val_gt_paths:
                mask_out_path = os.path.join(masks_val_dir, fn)
                coarse_mask_out_path = os.path.join(masks_val_dir, fn)
                bmask_out_path = os.path.join(bmasks_val_dir, fn)
                im_out_path = os.path.join(images_val_dir, fn)

            # if os.path.isfile(im_out_path):
            #     continue
            
            gt_tex = cv2.imread(gt_path)
            partial_tex = cv2.imread(path)

            gt_tex = erode_texture(gt_tex)

            gt_tex_gray = cv2.cvtColor(gt_tex, cv2.COLOR_BGR2GRAY)
            partial_tex_gray = cv2.cvtColor(partial_tex, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(partial_tex_gray, 5, 255,cv2.THRESH_BINARY_INV)
            _, bmask = cv2.threshold(gt_tex_gray, 5, 255,cv2.THRESH_BINARY)

            mask = cv2.bitwise_and(mask, bmask)
            mask = 255-mask

            mask_closed = mask

            cv2.imwrite(mask_out_path, mask_closed)
            cv2.imwrite(bmask_out_path, bmask)
            cv2.imwrite(im_out_path, gt_tex)



def create_coarse_mask(gt_mesh, missing_mask):
    new_missing_mask = missing_mask.copy()

    uvs = gt_mesh.visual.uv
    texture_res = missing_mask.shape[0]
    im_coords = np.clip((uvs*(texture_res-1)).astype(np.int32), 0, texture_res-1)

    for i in range(len(im_coords)):
        u, v = im_coords[i][0], im_coords[i][1]
        v = texture_res-1-v

        new_missing_mask[v,u] = 255

    return new_missing_mask

## This was required for showing samples from val on the paper.
def prepare_coarse_images():
    base_dir = "/data/3d_cluster/SHARP2022/Challenge1_clone/Track1-3DBodyTex.v2"
    data_dir = os.path.join(base_dir, "track1_data/train")
    final_dir = os.path.join(base_dir, "tex-inpaint_data_512")
    train_dir = os.path.join(base_dir, "tex-inpaint_data_512/train")
    val_dir = os.path.join(base_dir, "tex-inpaint_data_512/val")
    
    coarse_images_val_dir = os.path.join(val_dir, "coarse_images")
    os.makedirs(coarse_images_val_dir, exist_ok=True)

    gt_paths = glob.glob(data_dir + "/*/*_normalized.obj")
    for gt_path in tqdm(gt_paths):
        gt_mesh = trimesh.load(gt_path)
        mesh_id = os.path.dirname(gt_path).split(os.sep)[-1]

        partial_mask_paths = glob.glob(final_dir + "/val/coarse_masks/%s*.png" % mesh_id)
        for mask_path in partial_mask_paths:
            coarse_mask = cv2.imread(mask_path)
            img = cv2.imread(gt_path.replace(".obj",".png"))
            img = erode_texture(img)
            img = cv2.resize(img, (512,512))
            coarse_img = img*(coarse_mask/255.)
            coarse_img = coarse_img.astype(np.uint8)
            new_img_path = mask_path.replace("coarse_masks", "coarse_images")
            cv2.imwrite(new_img_path, coarse_img)



def prepare_coarse_masks():
    base_dir = "/data/3d_cluster/SHARP2022/Challenge1_clone/Track1-3DBodyTex.v2"
    data_dir = os.path.join(base_dir, "track1_data/train")
    final_dir = os.path.join(base_dir, "tex-inpaint_data")
    train_dir = os.path.join(base_dir, "tex-inpaint_data/train")
    val_dir = os.path.join(base_dir, "tex-inpaint_data/val")
    
    coarse_masks_train_dir = os.path.join(train_dir, "coarse_masks")
    coarse_masks_val_dir = os.path.join(val_dir, "coarse_masks")
    os.makedirs(coarse_masks_train_dir, exist_ok=True)
    os.makedirs(coarse_masks_val_dir, exist_ok=True)

    gt_paths = glob.glob(data_dir + "/*/*_normalized.obj")
    for gt_path in tqdm(gt_paths):
        gt_mesh = trimesh.load(gt_path)
        mesh_id = os.path.dirname(gt_path).split(os.sep)[-1]

        partial_mask_paths = glob.glob(final_dir + "/*/masks/%s*.png" % mesh_id)
        for mask_path in partial_mask_paths:
            mask = cv2.imread(mask_path, 0)
            new_mask = create_coarse_mask(gt_mesh, mask)
            new_mask_path = mask_path.replace("masks", "coarse_masks")
            cv2.imwrite(new_mask_path, new_mask)

if __name__ == "__main__":
    prepare_coarse_images()