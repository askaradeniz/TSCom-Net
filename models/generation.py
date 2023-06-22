import data_processing.utils as utils
import trimesh
import torch
import os
from glob import glob
import numpy as np
import mcubes
from torch.nn import functional as F

class Generator(object):
    def __init__(self, model, cfg, device = torch.device("cuda")):
        self.model = model.to(device)
        self.model.eval()

        self.experiment_prefix = cfg['experiment_prefix']
        self.checkpoint_path = '{}/{}/checkpoints/'.format(cfg['experiment_prefix'], cfg['folder_name'])
        self.exp_folder_name = cfg['folder_name']
        self.threshold = cfg['generation']['retrieval_threshold']

        self.device = device
        self.resolution = cfg['generation']['retrieval_resolution']
        self.batch_points = cfg['generation']['batch_points']

        self.bbox = cfg['data_bounding_box']
        self.min = self.bbox[::2]
        self.max = self.bbox[1::2]

        grid_points = utils.create_grid_points_from_xyz_bounds(*cfg['data_bounding_box'], self.resolution)
        self.grid_points = grid_points
        grid_coords = utils.to_grid_sample_coords(grid_points, self.bbox)
        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)


    def add_colors(self, pred_mesh, inputs, inputs_shape, n_subdivide=0):
        if(len(pred_mesh.vertices) == 0):
            return pred_mesh
        for i in range(n_subdivide):
            pred_mesh = pred_mesh.subdivide()

        vert_coords = utils.to_grid_sample_coords(pred_mesh.vertices, self.bbox)
        vert_coords = torch.tensor(vert_coords).unsqueeze(0)
        p = vert_coords.to(self.device).float()
        # p.shape is [1, n_verts, 3] 693016 -> > 21gb gram

        i = inputs.to(self.device).float()
        i_shape = inputs_shape.to(self.device).float()
        full_pred = []

        p_batches = torch.split(p, 200000, dim=1)
        
        for p_batch in p_batches:
            with torch.no_grad():
                pred_rgb = self.model.forward_color(p_batch, i, i_shape)
            full_pred.append(pred_rgb.squeeze(0).detach().cpu())

        pred_rgb = torch.cat(full_pred, dim=0).numpy()
        pred_rgb.astype(np.int)[0]
        pred_rgb = np.clip(pred_rgb, 0, 255)

        pred_mesh.visual.vertex_colors = pred_rgb
        return pred_mesh

    def mesh_from_logits(self, logits, res):
        logits = np.reshape(logits, (res,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (res - 1)
        vertices = np.multiply(vertices, step)

        vertices += self.min

        return trimesh.Trimesh(vertices, triangles)

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)

        vertices += self.min

        return trimesh.Trimesh(vertices, triangles)

    def generate_mesh(self, inputs, inputs_shape, add_colors=True, n_subdivide=0):
        #print("Generating mesh..")
        i = inputs.to(self.device).float()
        i_shape = inputs_shape.to(self.device).float()
        full_pred = []

        p_batches = self.grid_points_split

        for p_batch in p_batches:
            with torch.no_grad():
                pred_occ = self.model.forward_shape(p_batch, i_shape)
                #print(pred_occ.shape)
                pred_occ = pred_occ.squeeze(0)
            full_pred.append(pred_occ.squeeze(0).detach().cpu())

        logits = torch.cat(full_pred, dim=0).numpy()
        mesh = self.mesh_from_logits(logits) ## shape is generated.
        if add_colors == True:
            mesh = self.add_colors(pred_mesh=mesh, inputs=inputs, inputs_shape=inputs_shape, n_subdivide=n_subdivide)
        return mesh

    def load_checkpoint(self, checkpoint):
        if checkpoint == -1:
            val_min_npy = '{}/{}/val_min.npy'.format(self.experiment_prefix,
                self.exp_folder_name)
            checkpoint = int(np.load(val_min_npy)[0])
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        torch_checkpoint = torch.load(path)
        self.model.load_state_dict(torch_checkpoint['model_state_dict'])
        return checkpoint