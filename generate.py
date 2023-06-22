import models.local_model as model
import models.dataloader as dataloader
import numpy as np
import argparse
from models.generation import Generator
import config.config_loader as cfg_loader
import os
import trimesh
import shutil
import torch
from data_processing import utils
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generation Model'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    net = model.get_models()[cfg['model']]()

    mode = cfg['generation']['mode']
    dataloader = dataloader.VoxelizedDataset(mode, cfg, generation = True, num_workers=0).get_loader()

    gen = Generator(net, cfg)
    gen.checkpoint = gen.load_checkpoint(cfg['generation']['checkpoint'])
    print(gen.model)

    out_path = '{}/{}/evaluation_{}/{}'.format(cfg['experiment_prefix'], cfg['folder_name'], gen.checkpoint, mode)
    print(out_path)
    os.makedirs(out_path, exist_ok=True)



    for data in tqdm(dataloader):
        try:
            inputs = data['inputs']
            inputs_shape = data['inputs_shape']
            path = data['path'][0]
        except:
            print('none')
            continue

        path = os.path.normpath(path)
        challange = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        basename = path.split(os.sep)[-1]
        filename_partial = os.path.splitext(path.split(os.sep)[-1])[0]
        file_out_folder = out_path + '/{}/'.format(gt_file_name)
        fn = os.path.basename(path)[:-4]


        os.makedirs(file_out_folder, exist_ok=True)

        ## Shape prediction.
        file_out_path = file_out_folder + '{}_reconstruction.obj'.format(fn)
        out_mesh = gen.generate_mesh(inputs, inputs_shape, add_colors=False)
        out_mesh.export(file_out_path)
        
        ## Colored prediction.
        file_out_path = file_out_folder + '{}_reconstruction_colored.obj'.format(fn)
        out_mesh = gen.generate_mesh(inputs, inputs_shape, add_colors=True)
        out_mesh.export(file_out_path)

        
        ## Copy the partial and complete meshes.
        if mode != "eval":
            path_surface_png = os.path.join(cfg['data_path'], split, gt_file_name, gt_file_name + '_normalized.png')
            path_surface_mtl = os.path.join(cfg['data_path'], split, gt_file_name, gt_file_name + '_normalized.mtl')
            path_surface = os.path.join(cfg['data_path'], split, gt_file_name, gt_file_name + '_normalized.obj')
            path_partial = os.path.join(cfg['data_path'], split, gt_file_name, filename_partial + '.obj')
            path_partial_png = os.path.join(cfg['data_path'], split, gt_file_name, filename_partial + '.png')
            path_partial_mtl = os.path.join(cfg['data_path'], split, gt_file_name, filename_partial + '.mtl')
            shutil.copy(path_surface, file_out_folder)
            shutil.copy(path_surface_png, file_out_folder)
            shutil.copy(path_surface_mtl, file_out_folder)
            shutil.copy(path_partial, file_out_folder)
            shutil.copy(path_partial_png, file_out_folder)
            shutil.copy(path_partial_mtl, file_out_folder)
        else:
            path_partial = os.path.join(cfg['data_path'], split, gt_file_name, filename_partial + '.obj')
            path_partial_png = os.path.join(cfg['data_path'], split, gt_file_name, filename_partial + '.png')
            path_partial_mtl = os.path.join(cfg['data_path'], split, gt_file_name, filename_partial + '.mtl')
            shutil.copy(path_partial, file_out_folder)
            shutil.copy(path_partial_png, file_out_folder)
            shutil.copy(path_partial_mtl, file_out_folder)
      
