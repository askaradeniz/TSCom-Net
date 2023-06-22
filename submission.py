import os
import shutil
import glob
from cv2 import TermCriteria_MAX_ITER
from tqdm import tqdm


def move_missing_meshes():
    for folder in remaining_folders:
        files = glob.glob(folder+"/*transfer.obj")
        rec_file = glob.glob(folder+"/*reconstruction_colored.obj")[0]
        if(len(files) == 0):
            mesh_id = folder.split(os.sep)[-1]
            new_mesh_folder = os.path.join(results_dir, mesh_id)
            if os.path.isdir(new_mesh_folder):
                shutil.rmtree(new_mesh_folder)
            print(folder, new_mesh_folder)
            shutil.copytree(folder, new_mesh_folder)

def prepare_submission(texture_transfer=False):
    for path in tqdm(paths):
        fn = os.path.basename(path)
        if texture_transfer == True:
            fn = fn.replace("reconstruction_colored", "transfer")
            tnsf_path = os.path.join(os.path.dirname(path), fn)
            # if os.path.isfile(tnsf_path) == False:
            #     continue

        scan_dir = os.path.basename(os.path.dirname(path))
        new_fn = scan_dir+"-completed.obj"
        new_scan_dir = os.path.join(final_dir, scan_dir)
        os.makedirs(new_scan_dir, exist_ok=True)
        new_path = os.path.join(new_scan_dir, new_fn)
        
        
        if texture_transfer == True:
            tnsf_png_path = tnsf_path.replace(".obj", "_0.png")
            tnsf_mtl_path = tnsf_path.replace(".obj", ".mtl")
            dirname = os.path.dirname(new_path)

            if os.path.isfile(tnsf_path) and os.path.isfile(tnsf_png_path) and os.path.isfile(tnsf_mtl_path):
                shutil.copy(tnsf_path, new_path)
                shutil.copy(tnsf_png_path, dirname)
                shutil.copy(tnsf_mtl_path, dirname)
            else:
                shutil.copy(path, new_path)

        else:
            shutil.copy(path, new_path)

    


if __name__ == "__main__":
    results_dir = "/data/akaradeniz/experiments/jin_bs6_50000_res128/evaluation_52/eval"
    final_dir = "/data/akaradeniz/experiments/jin_bs6_50000_res128/evaluation_52/submission"
    os.makedirs(final_dir, exist_ok=True)

    #move_missing_meshes()
    paths = glob.glob(results_dir+"/*/*_reconstruction_colored.obj")
    prepare_submission(texture_transfer=False)