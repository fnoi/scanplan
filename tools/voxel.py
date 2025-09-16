import os
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from tools import parallel


def voxel_purge(voxel_grid):
    # find unique faces / purge redundant
    raw_arrays_create = parallel.init_arrays_create(voxel_grid_dict=voxel_grid)

    mp_threads = os.cpu_count()

    with mp.Pool(
            processes=mp_threads,
            initializer=parallel.face_filter_init,
            initargs=(
                    raw_arrays_create['unique face coords'],
                    raw_arrays_create['unique face coords shape']
            )
    ) as pool:
        resi = list(
            tqdm(
                pool.imap(
                    parallel.face_filter_worker,
                    range(raw_arrays_create['unique face coords shape'][0])
                ),
                desc='face filter new and amazing',
                total=raw_arrays_create['unique face coords shape'][0]
            )
        )

    # combo_input = [(face_coords, voxel_grid["unique vertices"]) for face_coords in voxel_grid["unique face coords"]]
    #
    # input_0 = [_coords for _coords in voxel_grid['unique face coords']]
    # input_1 = voxel_grid['unique face coords']
    #
    # with mp.Pool(os.cpu_count()) as pool:
    #     all_faces_ind_unique = list(tqdm(pool.imap(
    #         face_filter_verts_multi, combo_input
    #     ), desc='new verts voxels multi', total=voxel_grid["unique face coords"].shape[0]))

    all_faces_ind_unique = resi

    return all_faces_ind_unique


def voxel_setup(voxel_dict, config):
    scene_vox = voxel_dict["scene"].get_voxels()
    scene_ids = np.asarray([_.grid_index for _ in scene_vox])
    test_mi = np.asarray([voxel_dict["scene"].get_voxel_bounding_points(_) for _ in scene_ids])
    test_m = test_mi.reshape(-1, test_mi.shape[-1])
    voxel_dict["center coords scene"] = np.vstack(
        [voxel_dict["scene"].get_voxel_center_coordinate(scene_ids[i])
         for i in range(len(scene_ids))])

    delta = config["scene resolution"] / 2
    all_edges = []
    all_vertices = []
    all_faces = []
    x_E = []
    y_E = []
    z_E = []
    x_E_memory = []
    y_E_memory = []
    z_E_memory = []
    c = 0

    _edge_memory = []

    count_ok = 0
    count_fail = 0

    # calculate corner point
    for center_coord in tqdm(voxel_dict["center coords scene"], desc='center coord loop, edge filter'):
        vertex_0 = center_coord + np.asarray([delta, delta, delta])
        vertex_1 = center_coord + np.asarray([-delta, delta, delta])
        vertex_2 = center_coord + np.asarray([delta, -delta, delta])
        vertex_3 = center_coord + np.asarray([-delta, -delta, delta])
        vertex_4 = center_coord + np.asarray([delta, delta, -delta])
        vertex_5 = center_coord + np.asarray([-delta, delta, -delta])
        vertex_6 = center_coord + np.asarray([delta, -delta, -delta])
        vertex_7 = center_coord + np.asarray([-delta, -delta, -delta])
        vertex = [
            vertex_0, vertex_1, vertex_2, vertex_3,
            vertex_4, vertex_5, vertex_6, vertex_7
        ]

        voxel_cube_faces = np.asarray([
            [0, 1, 2],
            [4, 5, 6],
            [1, 3, 2],
            [5, 7, 6],
            [0, 2, 4],
            [1, 3, 5],
            [2, 6, 4],
            [3, 7, 5],
            [0, 1, 4],
            [2, 3, 6],
            [1, 5, 4],
            [3, 7, 6]
        ])

        cs = np.empty_like(voxel_cube_faces)
        cs[:] = c
        voxel_cube_faces = voxel_cube_faces + cs

        all_vertices.extend(vertex)
        if c == 0:
            all_faces = voxel_cube_faces
        else:
            all_faces = np.append(all_faces, voxel_cube_faces, axis=0)
        # all_faces.append(faces)
        c += 8

        edges = [
            [list(vertex[0]), list(vertex[1])],
            [list(vertex[0]), list(vertex[2])],
            [list(vertex[1]), list(vertex[3])],
            [list(vertex[2]), list(vertex[3])],
            [list(vertex[0]), list(vertex[4])],
            [list(vertex[1]), list(vertex[5])],
            [list(vertex[2]), list(vertex[6])],
            [list(vertex[3]), list(vertex[7])],
            [list(vertex[4]), list(vertex[5])],
            [list(vertex[4]), list(vertex[6])],
            [list(vertex[5]), list(vertex[7])],
            [list(vertex[6]), list(vertex[7])]
        ]

        for edge in edges:
            _edge_memory.append(np.asarray(edge[0] + edge[1]))

    all_faces = np.stack(tuple(all_faces), axis=0)

    # identify unique vertices
    np_mem = np.asarray(_edge_memory)
    np_mem_true = np.unique(np_mem, axis=0)
    for np_edge in tqdm(np_mem_true, desc='build edges'):
        x_edge = np.asarray([np_edge[0], np_edge[3], None])
        y_edge = np.asarray([np_edge[1], np_edge[4], None])
        z_edge = np.asarray([np_edge[2], np_edge[5], None])

        x_E.extend(x_edge.tolist())
        y_E.extend(y_edge.tolist())
        z_E.extend(z_edge.tolist())

    voxel_dict["edges"] = [x_E, y_E, z_E]

    all_vertices = np.asarray(all_vertices)

    all_faces_coords = all_vertices[all_faces]
    voxel_dict["unique face coords"], count_dup_faces = np.unique(all_faces_coords, axis=0, return_counts=True)
    voxel_dict["unique vertices"] = np.unique(all_vertices, axis=0)  # legacy all_vertices_unique

    return voxel_dict
