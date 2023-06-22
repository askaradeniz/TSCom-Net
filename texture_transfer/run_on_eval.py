from email.mime import base
import glob
import os
from tqdm import tqdm
import subprocess
import shutil
import cv2
import trimesh
import numpy as np


def create_new_texture(recon_mesh_colored, recon_mesh_transfer, missing_mask, ratio=0.25):
    original_texture = np.array(recon_mesh_transfer.visual.material.image)
    new_texture = original_texture.copy()
    new_missing_mask = missing_mask.copy()

    vertex_colors = recon_mesh_colored.visual.vertex_colors[:,:3]
    n_vertices = int(recon_mesh_transfer.vertices.shape[0]*ratio)
    random_indices = np.random.randint(0, recon_mesh_transfer.vertices.shape[0], n_vertices)
    vertices = recon_mesh_transfer.vertices.copy()[random_indices]
    print(vertices.shape, recon_mesh_transfer.vertices.shape)
    v_indices = recon_mesh_colored.kdtree.query(vertices)[1]


    #uvs = recon_mesh_transfer.visual.uv[:recon_mesh_colored.vertices.shape[0]]
    uvs = recon_mesh_transfer.visual.uv[random_indices]
    texture_res = new_texture.shape[0]-1
    im_coords = np.clip(np.rint(uvs*(texture_res)).astype(np.int), 0, texture_res)
    #im_coords = np.clip((uvs*texture_res).astype(np.int), 0, texture_res)

    print(vertex_colors.shape, im_coords.shape)

    print(new_texture.shape, im_coords.shape)

    for i in range(len(im_coords)):        
        u, v = im_coords[i][0], im_coords[i][1]
        v = texture_res-v
        #print(u, v)

        v_idx = v_indices[i]
        color = vertex_colors[v_idx]

        mask_val = missing_mask[v, u]
        if mask_val == 0:
            new_texture[v, u] = color
        new_missing_mask[v,u] = 255
        new_texture[v,u] = color

        # mask_val = missing_mask[v, u]
        # if mask_val == 0:
        #     r = np.random.randint(0, 1, 1)
        #     if r == 0:
        #         new_texture[v, u] = color
        #         new_missing_mask[v,u] = 255

    return new_texture, new_missing_mask


def generate_masks(out_path, path):
    """
    out_path: path to the mesh with partial texture transfer.
    path: path to to the partial mesh.
    """

    success = False

    mask_path = out_path.replace("_transfer.obj", "_transfer-partial_mask.png")
    bmask_path = out_path.replace("_transfer.obj", "_transfer-background_mask.png")
    im_path = out_path.replace("_transfer.obj", "_transfer_0.png")

    if os.path.isfile(os.path.join(masks_dir, mask_path.split(os.sep)[-1])):
        print("Mask exist, return.")
        return

    mask_gen_cmd = "python make_masks.py %s" % (out_path)
    os.system(mask_gen_cmd)

    if os.path.isfile(mask_path) and os.path.isfile(bmask_path) and os.path.isfile(im_path):
        shutil.move(mask_path, os.path.join(masks_dir, mask_path.split(os.sep)[-1]))
        shutil.move(bmask_path, os.path.join(bmasks_dir, bmask_path.split(os.sep)[-1]))
        shutil.copy(im_path, os.path.join(images_dir, im_path.split(os.sep)[-1]))
        success = True

    else:
        print("Failed mask generation: %s" % path.split(os.sep)[-1])
        failed_meshes.append(path)

    for overlay_png in glob.glob(os.path.dirname(path)+"/*overlay*.png"):
        os.remove(overlay_png)

    return success

def generate_textures(path):
    print("Generating coarse tex and coarse masks.")
    rec_mesh_colored_path = path.replace(".obj", "_reconstruction_colored.obj")
    rec_mesh_trnsf_path = path.replace(".obj", "_transfer.obj")
    missing_mask_path = os.path.join(masks_dir, path.replace(".obj", "_transfer-partial_mask.png").split(os.sep)[-1]) 
    tex_path = os.path.join(masks_dir, path.replace(".obj", "_transfer_0.png").split(os.sep)[-1]) 

    recon_mesh_colored = trimesh.load(rec_mesh_colored_path)
    recon_mesh_transfer = trimesh.load(rec_mesh_trnsf_path)
    print(missing_mask_path)
    missing_mask = cv2.imread(missing_mask_path, 0)

    new_texture, new_missing_mask = create_new_texture(recon_mesh_colored, recon_mesh_transfer, missing_mask)

    cv2.imwrite(os.path.join(coarse_masks_dir, path.replace(".obj", "_transfer-partial_mask.png").split(os.sep)[-1]), new_missing_mask)
    cv2.imwrite(os.path.join(coarse_images_dir, path.replace(".obj", "_transfer_0.png").split(os.sep)[-1]), cv2.cvtColor(new_texture, cv2.COLOR_RGB2BGR))

def run_tranfer():
    for path in tqdm(paths):
        if ("reconstruction" in path) or ("transfer" in path): ## loop through partials only.
            print("cont.")
            continue

        fn = path.split(os.sep)[-1]
        mesh_dir = os.path.dirname(path)
        fn_recon = fn.replace(".obj", "_reconstruction.obj")
        fn_out = fn.replace(".obj", "_transfer.obj")
        recon_path = os.path.join(mesh_dir, fn_recon)
        out_path = os.path.join(mesh_dir, fn_out)

        mesh = trimesh.load(path)
        #mesh = mesh.process(True)
        #mesh.export(path)

        if(len(glob.glob(os.path.dirname(path)+"/*transfer.obj")) > 0):
            #print("Transfer exists, continue.")
            success = generate_masks(out_path, path)
            if success == True:
                generate_textures(path)
            continue

        for old_png in glob.glob(os.path.dirname(path)+"/*transfer*.png"):
            os.remove(old_png)

        if fn in failed_files:
            failed_meshes.append(path)
            continue

        cmd = "./run_transfer.sh %s %s %s %s" % (path, recon_path, path, out_path)
        print(cmd)
        os.system(cmd)

        success = generate_masks(out_path, path)
        if success == True:
            generate_textures(path)
        
    print("Files that mask gen. failed:")
    print(failed_meshes)
    print(len(failed_meshes))
    

if __name__ == "__main__":
    base_dir = "./data/evaluation_52"

    paths = glob.glob(base_dir + "/eval/*/*.obj")
    trnsf_paths = glob.glob(base_dir + "/eval/*/*transfer.obj")
    recon_paths = glob.glob(base_dir + "/eval/*/*reconstruction.obj")
    recon_colored_paths = glob.glob(base_dir + "/eval/*/*reconstruction_colored.obj")
    paths = list((set(paths) - set(trnsf_paths)) - set(recon_paths) - set(recon_colored_paths))[:15]

    ## T2
    failed_files = ["170523-007-f-eiet-89f1-low-res-result_normalized-partial-04.obj", "170424-015-f-3pdl-ed3e-low-res-result_normalized-partial-04.obj",
    "170509-009-m-ygel-22f0-low-res-result_normalized-partial-01.obj", "170410-014-a-qrz8-a221-low-res-result_normalized-partial-04.obj",
    "180807-022-fitness-a-jmcr-963c-low-res-result_normalized-partial-04.obj", "180809-009-casual-a-xqad-610c-low-res-result_normalized-partial-04.obj",
    "171005-005-fitness-u-dydd-041f-low-res-result_normalized-partial-01.obj"]

    ## T1
    # failed_files = ["180418-012-fitness-u-dd38-f2c5-low-res-result_normalized-partial.obj","180423-009-fitness-scape013-fl3c-cb2c-low-res-result_normalized-partial.obj", 
    # "180825-001-fitness-a-rux7-e50c-low-res-result_normalized-partial.obj","170412-010-m-dtpi-cf9a-low-res-result/170412-010-m-dtpi-cf9a-low-res-result_normalized-partial.obj",
    # "170424-020-f-w186-fd74-low-res-result_normalized-partial.obj", "171214-003-fitness-run-r6bn-5f87-low-res-result_normalized-partial.obj"]

    failed_meshes = []
    masks_dir = os.path.join(base_dir, "inpaint_data/masks")
    coarse_masks_dir = os.path.join(base_dir, "inpaint_data/coarse_masks")
    bmasks_dir = os.path.join(base_dir, "inpaint_data/bmasks")
    images_dir = os.path.join(base_dir, "inpaint_data/images")
    coarse_images_dir = os.path.join(base_dir, "inpaint_data/coarse_images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(coarse_masks_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(bmasks_dir, exist_ok=True) 
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(coarse_images_dir, exist_ok=True)
    run_tranfer()

