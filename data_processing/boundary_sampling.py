import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import config.config_loader as cfg_loader
import utils
import traceback
import tqdm
import implicit_waterproofing as iw



def sample_boundary(gt_mesh_path):
    try:
        path = os.path.normpath(gt_mesh_path)
        challange = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]

        out_file = cfg['data_path'] + '/{}/{}_boundary_samples.npz'.format(gt_file_name,full_file_name)

        # if os.path.exists(out_file):
        #     print('File exists. Done.')
        #     return
        
        gt_mesh = utils.as_mesh(trimesh.load(gt_mesh_path))
        sample_points = gt_mesh.sample(num_points)

        boundary_points_close = sample_points[:num_points//2] + sigma[0] * np.random.randn(num_points//2, 3)
        boundary_points_far = sample_points[num_points//2:] + sigma[1] * np.random.randn(num_points//2, 3)
        boundary_points = np.concatenate([boundary_points_close, boundary_points_far], axis=0)
        np.random.shuffle(boundary_points)
        grid_coords = utils.to_grid_sample_coords(boundary_points, bbox)
        occupancies = iw.implicit_waterproofing(gt_mesh, boundary_points)[0]

        np.savez(out_file, points = boundary_points, grid_coords = grid_coords, occupancies = occupancies)

    except Exception as err:
        print('Error with {}: {}'.format(out_file, traceback.format_exc()))


def sample_colors_and_occuppancies(gt_mesh_path):
    try:
        path = os.path.normpath(gt_mesh_path)
        challange = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]

        out_file = cfg['data_path'] + '/{}/{}/{}_rgbs_samples.npz' \
            .format(split, gt_file_name, full_file_name, num_points, cfg['data_bounding_box_str'])



        # if os.path.exists(out_file):
        #     os.remove(out_file)
        #     print('File exists. Done.')
        #     return
        
        gt_mesh = utils.as_mesh(trimesh.load(gt_mesh_path))
        sample_points, face_idxs = gt_mesh.sample(num_points, return_index = True)

        triangles = gt_mesh.triangles[face_idxs]
        face_vertices = gt_mesh.faces[face_idxs]
        faces_uvs = gt_mesh.visual.uv[face_vertices]

        q = triangles[:, 0]
        u = triangles[:, 1]
        v = triangles[:, 2]

        uvs = []

        for i, p in enumerate(sample_points):
            barycentric_weights = utils.barycentric_coordinates(p, q[i], u[i], v[i])
            uv = np.average(faces_uvs[i], 0, barycentric_weights)
            uvs.append(uv)

        texture = gt_mesh.visual.material.image

        colors = trimesh.visual.color.uv_to_color(np.array(uvs), texture)

        boundary_points_close = sample_points[:num_points//2] + sigma[0] * np.random.randn(num_points//2, 3)
        boundary_points_far = sample_points[num_points//2:] + sigma[1] * np.random.randn(num_points//2, 3)
        boundary_points = np.concatenate([boundary_points_close, boundary_points_far], axis=0)
        p = np.random.permutation(len(sample_points))
        boundary_points = boundary_points[p]
        sample_points = sample_points[p]
        colors = colors[p]
        occupancies = iw.implicit_waterproofing(gt_mesh, boundary_points)[0]


        grid_coords = utils.to_grid_sample_coords(boundary_points, bbox)
        grid_coords_surface = utils.to_grid_sample_coords(sample_points, bbox)

        #normals = gt_mesh.face_normals[face_idxs]

        #print(grid_coords_surface.shape, normals.shape, occupancies.shape)

        # np.savez(out_file, grid_coords = grid_coords, grid_coords_surface=grid_coords_surface, occupancies=occupancies, rgb_values = colors[:,:3], normals=normals)
        np.savez(out_file, grid_coords = grid_coords, grid_coords_surface=grid_coords_surface, occupancies=occupancies, rgb_values = colors[:,:3])

    except Exception as err:
        print('Error with {}: {}'.format(out_file, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling. Samples surface points on the GT objects, and saves their coordinates along with the occupancies at their location.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    
    print(args.config)
    cfg = cfg_loader.load(args.config)

    num_points = cfg['preprocessing']['boundary_sampling']['sample_number']
    bbox = cfg['data_bounding_box']

    sigma = cfg['preprocessing']['boundary_sampling']['sigma']
    
    print('Fining all gt object paths for point and boundary sampling.')
    paths = glob(cfg['data_path'] + cfg['preprocessing']['boundary_sampling']['input_files_regex'])
    print('Start sampling.')    
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(sample_colors_and_occuppancies, paths), total=len(paths)):
        pass
    p.close()
    p.join()
