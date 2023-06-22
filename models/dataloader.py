from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch


class VoxelizedDataset(Dataset):


    def __init__(self, mode, cfg, generation = False, num_workers = 12, shuffle=True):

        self.path = cfg['data_path']
        self.cfg = cfg
        self.mode = mode
        self.data = np.load(cfg['split_file'])[mode]#[:20]
        self.res = cfg['input_resolution']
        self.bbox_str = cfg['data_bounding_box_str']
        self.bbox = cfg['data_bounding_box']
        self.num_gt_rgb_samples = cfg['preprocessing']['boundary_sampling']['sample_number']

        self.sample_points_per_object = cfg['training']['sample_points_per_object']
        if generation:
            self.batch_size = 1
        else:
            self.batch_size = cfg['training']['batch_size']
        self.num_workers = num_workers
        if cfg['input_type'] == 'voxels':
            self.voxelized_pointcloud = False
        else:
            self.voxelized_pointcloud = True
            self.pointcloud_samples = cfg['input_points_number']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
    
        path = os.path.normpath(path)
        mesh_dir = path.split(os.sep)[-2]
        gt_file_name = "{}_normalized.obj".format(mesh_dir)
        full_file_name = os.path.splitext(path.split(os.sep)[-1])[0]

        split_dir = "train" ## train or val
        if self.mode == "eval":
            split_dir = "eval"

        
        voxel_path = os.path.join(self.path, split_dir, mesh_dir, '{}_voxelized_colored_point_cloud.npz'.format(full_file_name))

        R = np.load(voxel_path)['R']
        G = np.load(voxel_path)['G']
        B = np.load(voxel_path)['B']
        S = np.load(voxel_path)['S']

        R = np.reshape(R, (self.res,)*3)
        G = np.reshape(G, (self.res,)*3)
        B = np.reshape(B, (self.res,)*3)
        S = np.reshape(S, (self.res,)*3)
        input = np.array([R,G,B])
        input_shape = np.array([S])

        if self.mode == 'eval':
            return { 'inputs': np.array(input, dtype=np.float32), 'inputs_shape': np.array(input_shape, dtype=np.float32),  'path' : path}

        rgbs_samples_path = os.path.join(self.path, split_dir, mesh_dir, '{}_rgbs_samples.npz'.format(gt_file_name[:-4]))

        rgbs_samples_npz = np.load(rgbs_samples_path)
        grid_coords_surface = rgbs_samples_npz['grid_coords_surface']
        rgb_values = rgbs_samples_npz['rgb_values']
        grid_coords = rgbs_samples_npz['grid_coords']
        occ_values = rgbs_samples_npz['occupancies']
        #normals = rgbs_samples_npz['normals']
        subsample_indices = np.random.randint(0, len(occ_values), self.sample_points_per_object)
        grid_coords = grid_coords[subsample_indices]
        rgb_values = rgb_values[subsample_indices]
        grid_coords_surface = grid_coords_surface[subsample_indices]
        occ_values = occ_values[subsample_indices]
        #normals = normals[subsample_indices]

        complete_mesh_path = os.path.join(self.path, split_dir, mesh_dir, gt_file_name)

        return {'grid_coords':np.array(grid_coords, dtype=np.float32), 'occ': np.array(occ_values, dtype=np.float32),  'grid_coords_surface': np.array(grid_coords_surface, dtype=np.float32), 'rgb': np.array(rgb_values, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'inputs_shape': np.array(input_shape, dtype=np.float32),  'path' : path, 'gt_mesh_path': complete_mesh_path}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)