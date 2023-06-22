import trimesh
import numpy as np
import sys
from libmesh.inside_mesh import check_mesh_contains
import math
import os
import glob
from multiprocessing import Pool


def to_rotation_matrix(euler_angles):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(euler_angles[0]), -math.sin(euler_angles[0])],
                    [0, math.sin(euler_angles[0]), math.cos(euler_angles[0])]
                    ])

    R_y = np.array([[math.cos(euler_angles[1]), 0, math.sin(euler_angles[1])],
                    [0, 1, 0],
                    [-math.sin(euler_angles[1]), 0, math.cos(euler_angles[1])]
                    ])

    R_z = np.array([[math.cos(euler_angles[2]), -math.sin(euler_angles[2]), 0],
                    [math.sin(euler_angles[2]), math.cos(euler_angles[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def implicit_waterproofing(mesh_source, query_points):
    occ_list, holes_list = check_mesh_contains(mesh_source, query_points)

    for euler_angles in np.array([[0, np.pi / 2, 0], [np.pi / 2, 0, 0], [0, 0, np.pi / 2]]):

        if not holes_list.any():
            break
        print('iteration start')

        mesh = mesh_source.copy()
        r = to_rotation_matrix(euler_angles)
        r = np.pad(r, [(0, 1), (0, 1)], 'constant', constant_values=0)
        mesh.apply_transform(r)
        points = np.dot(r[:3, :3], query_points[holes_list].T).T
        occ_list_rot, holes_list_rot = check_mesh_contains(mesh, points)

        occ_list[holes_list] = occ_list_rot
        holes_list_updated = np.full(len(query_points), False)
        holes_list_updated[holes_list] = holes_list_rot
        holes_list = holes_list_updated

    return occ_list, holes_list



# # Converting to occupancy grid
def to_occ(mesh, res):
    occ, holes = implicit_waterproofing(mesh, create_grid_points(mesh, res))
    occ = np.reshape(occ,(res,res,res))
    return occ