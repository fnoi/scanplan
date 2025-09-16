import json
import pathlib
import time

import open3d as o3d
import numpy as np

from tqdm import tqdm

from tools.geometry import triangle_midpoints_area


def load_model(config):
    mesh = o3d.io.read_triangle_mesh(config["model file"])
    mesh_data = dict.fromkeys([
        'size',
        'ids',
        'area total',
        'vertices',
        'triangles',
        'midpoint',
        'normal',
        'area',
        'hit by'
    ])

    mesh_data['vertices'] = np.asarray(mesh.vertices)
    mesh_data['triangles'] = (np.asarray(mesh.triangles))
    mesh_data['size'] = len(mesh_data['triangles'])
    mesh_data['ids'] = [i for i in range(mesh_data['size'])]
    mesh.compute_vertex_normals()
    mesh_data['normals'] = (np.asarray(mesh.triangle_normals))
    mesh_data['contained by'] = [config['model file'] for i in range(mesh_data['size'])]

    # add all additionals models
    # bug here because size and face count is not equal
    if len(config["additional Models"]) != 0:
        for iter in config['additional Models']:
            mesh = o3d.io.read_triangle_mesh(iter)

            ##### depreciated hopfully
            mesh_data['contained by'] += [iter for i in
                                          range(mesh_data['size'], mesh_data['size'] + len(np.asarray(mesh.triangles)))]

            # add here the offset of vertices because the start at 0
            tmp_triangles = np.asarray(mesh.triangles) + len(mesh_data['vertices'])

            mesh_data['vertices'] = np.concatenate((mesh_data['vertices'], np.asarray(mesh.vertices)))
            mesh_data['triangles'] = np.concatenate((mesh_data['triangles'], tmp_triangles))
            mesh_data['size'] = len(mesh_data['triangles'])
            mesh_data['ids'] = [i for i in range(mesh_data['size'])]
            mesh.compute_vertex_normals()
            mesh_data['normals'] = np.concatenate((mesh_data['normals'], np.asarray(mesh.triangle_normals)))

    # compute & add midpoints and areas
    mesh_data = triangle_midpoints_area(mesh=mesh_data, config=config)

    mesh = o3d.geometry.TriangleMesh()
    face_exp = mesh_data['triangles']
    vertex_exp = np.asarray(mesh_data['vertices'])

    mesh.triangles = o3d.utility.Vector3iVector(face_exp[:][:])
    mesh.vertices = o3d.utility.Vector3dVector(vertex_exp[:][:])

    assert mesh_data["area total"] > 0, f"mesh area > 0 expected"

    # o3d.io.write_triangle_mesh("test_seperation.obj", mesh)

    return mesh_data


def model2scene(model, config):
    # transfer mesh to ray-casting scene
    scene = o3d.t.geometry.RaycastingScene()

    list_id = []
    device = o3d.core.Device("CPU:0")
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32

    for i in tqdm(range(model['size']), desc='transforming model to o3d scene', total=model['size']):
        point1 = model['triangles'][i][0]
        point2 = model['triangles'][i][1]
        point3 = model['triangles'][i][2]
        # create a new mesh with only one triangle
        mesh_new = o3d.t.geometry.TriangleMesh(device)
        mesh_new.vertex["positions"] = o3d.core.Tensor(
            [np.transpose(
                np.float32([model['vertices'][point1][0], model['vertices'][point1][1], model['vertices'][point1][2]])),
                np.transpose(np.float32(
                    [model['vertices'][point2][0], model['vertices'][point2][1], model['vertices'][point2][2]])),
                np.transpose(np.float32(
                    [model['vertices'][point3][0], model['vertices'][point3][1], model['vertices'][point3][2]])),
            ], dtype_f, device)
        mesh_new.triangle["indices"] = o3d.core.Tensor(
            [[0, 1, 2]], dtype_i, device)

        triangle_id = scene.add_triangles(mesh_new)
        list_id.append(triangle_id)

    return list_id, scene


def overlap_wrap(resulting_stuff, candidates):
    overlap_areamat = np.zeros([len(candidates), len(candidates)])
    overlap_relmat = np.zeros([len(candidates), len(candidates)])

    for thing in tqdm(resulting_stuff, desc='wrapping overlap to matrices'):
        overlap_areamat[tuple(thing[0])] = thing[1]
        overlap_relmat[tuple(thing[0])] = thing[2]

    overlap_relmat = overlap_relmat + np.transpose(overlap_relmat)

    return overlap_areamat, overlap_relmat


def table_base(model, candidates, config):
    table = np.empty(shape=(len(candidates), model['size']))
    table.fill(0.0)

    i = 0
    for candi in tqdm(candidates, desc='visibility table (single thread)'):
        if config['greedy area']:
            table[i][candi['qualified hits']] = np.array(model['area'])[candi['qualified hits']]
        else:
            table[i][candi['qualified hits']] = 1.0
        i += 1

    return table


def table_base_v2(model, candidates, config):
    # table = np.empty(shape=(len(candidates), model['size']))
    # table.fill(0.0)
    table = np.full(shape=(len(candidates), model['size']), fill_value=False)
    for i, candi in tqdm(enumerate(candidates), desc='visibility mask table, single thread'):
        table[i][candi['qualified hits']] = True

    # i = 0
    # for candi in tqdm(candidates, desc='visibility table (single thread)'):
    #     if config['greedy area']:
    #         table[i][candi['qualified hits']] = np.array(model['area'])[candi['qualified hits']]
    #     else:
    #         table[i][candi['qualified hits']] = 1.0
    #     i += 1

    return table