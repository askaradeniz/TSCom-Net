import os
import shutil
import glob
from tqdm import tqdm

results_dir =  "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_54_inpaint_33500/eval/"
final_dir =  "/data/akaradeniz/experiments/ifnet-plusv2/evaluation_54_inpaint_33500/submission"
os.makedirs(final_dir, exist_ok=True)

paths = glob.glob(results_dir+"/*/*_reconstruction.obj")

def prepare_submission():
    failed_objs = 0
    for path in tqdm(paths):
        fn = os.path.basename(path)
        fn = fn.replace("reconstruction", "transfer")
        tnsf_path = os.path.join(os.path.dirname(path), fn)
        if os.path.isfile(tnsf_path) == False:
            print("Failed transfer %d: %s" % (failed_objs, fn))
            failed_objs += 1
            continue
        scan_dir = os.path.basename(os.path.dirname(path))
        new_fn = scan_dir+"-completed.obj"
        new_scan_dir = os.path.join(final_dir, scan_dir)
        os.makedirs(new_scan_dir, exist_ok=True)
        new_path = os.path.join(new_scan_dir, new_fn)
        
        
        shutil.copy(tnsf_path, new_path)
        tnsf_png_path = tnsf_path.replace(".obj", "_0.png")
        tnsf_mtl_path = tnsf_path.replace(".obj", ".mtl")
        dirname = os.path.dirname(new_path)
        shutil.copy(tnsf_png_path, dirname)
        shutil.copy(tnsf_mtl_path, dirname)


if __name__ == "__main__":
    prepare_submission()