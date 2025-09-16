import itertools
import os
import shutil
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import pathlib
import open3d as o3d
import time
import copy
import json
import sys
import matplotlib.pyplot as plt
import networkx as nx

from tools.geometry import triangle_midpoints_area


# maybe update points to lists to check more rays at once
def directconnection(point1, point2, scene):
    connectin = False

    input = copy.deepcopy(point1)
    input.append(point2[0] - point1[0])
    input.append(point2[1] - point1[1])
    input.append(point2[2] - point1[2])

    ray = o3d.core.Tensor([input], dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(ray)

    if ans['t_hit'] >= 1:
        connectin = True
    return connectin


# check if distance between location and scene
def distancesmaller(scene, location, distance):
    location_prep = np.array(location[:], dtype=np.float32)
    if scene.compute_distance([location_prep]) < distance:
        return True
    else:
        return False


def graph2candidates(config, graph):
    candidates = []
    for _ in range(len(graph.nodes)):  # - 1):
        location = [
            graph.nodes[_]['pos'][0],
            graph.nodes[_]['pos'][1],
            graph.nodes[_]['pos'][2],
        ]

        # add here check for distance

        candidate_data = dict.fromkeys([
            'coordinates',
            'area covered',
            'max coverage area',
            'max coverage hits',
            'face ratio local',
            'face ratio global',
            'area ratio local',
            'area ratio global',
            'rays',
            'hit ids',
            'qualified hits'
        ])
        candidate_data['coordinates'] = location
        candidate_data['hit ids'] = []
        candidates.append(candidate_data)

    return candidates


def load_model(config, current):
    print(f" >>> loading mesh model {config['model path']}", flush=True)

    mesh = o3d.io.read_triangle_mesh(config['model path'])
    print(f" >>> loading mesh done")
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
    mesh_data['contained by'] = [config['model path'] for i in range(mesh_data['size'])]

    # add all additionals models
    # bug here because size and face count is not equal
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
    mesh_data, current = triangle_midpoints_area(mesh=mesh_data, config=config, current=current)

    # get size of iteration
    # size = triangles.shape[0]

    # Convert mesh to format that is supported
    # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    #  return size, triangles, vertices, list_id, normals
    #     #maybe voxel_size in config?
    #     voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
    #                                                           voxel_size=0.5)

    #     o3d.visualization.draw_geometries([voxel_grid])
    #     queries = np.asarray(mesh_data['vertices'])
    #     output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))

    #     for i in range(len(mesh_data['triangles'])):
    #         if output[mesh_data['triangles'][i][0]] or output[mesh_data['triangles'][i][1]] or output[mesh_data['triangles'][i][2]]:
    #             mesh_data['contained by'][i]=iter

    #     #test export of additionals
    #     list_triangle=[]
    #     for i in range(len(mesh_data['triangles'])):
    #         if mesh_data['contained by'][i]!=config['model path']:
    #             list_triangle.append(mesh_data['triangles'][i])
    #     mesh = o3d.geometry.TriangleMesh()
    #     face_exp = np.asarray(list_triangle)
    #     vertex_exp = np.asarray(mesh_data['vertices'])

    #     mesh.triangles = o3d.utility.Vector3iVector(face_exp[:][:])
    #     mesh.vertices = o3d.utility.Vector3dVector(vertex_exp[:][:])
    #     o3d.io.write_triangle_mesh("test_adds.obj", mesh)

    # test export of additionals
    # list_triangle=[]
    # for i in range(len(mesh_data['triangles'])):
    #    list_triangle.append(mesh_data['triangles'][i])
    mesh = o3d.geometry.TriangleMesh()
    # face_exp = np.asarray(list_triangle)
    face_exp = mesh_data['triangles']
    vertex_exp = np.asarray(mesh_data['vertices'])

    mesh.triangles = o3d.utility.Vector3iVector(face_exp[:][:])
    mesh.vertices = o3d.utility.Vector3dVector(vertex_exp[:][:])
    o3d.io.write_triangle_mesh("test_seperation.obj", mesh)

    return mesh_data, current


def model2scene(model, config, current):
    # Create a Scene for Raycasting
    scene = o3d.t.geometry.RaycastingScene()

    list_id = []
    device = o3d.core.Device("CPU:0")
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32

    # import the mesh into an format we can use
    for i in tqdm(range(model['size']), desc=' >>> model2scene', total=model['size']):
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

    return list_id, scene, current


def export_sp_meshes(model, result, scanpoints, destination):
    # new stuff for export this is surely ugly and need refinement
    # p = Path('.')
    face_already_exported = set()
    mesh = mesh_export_helper(model)

    for i, sp in enumerate(result['strategy']['scanpoint ids']):
        temp_list = sorted(scanpoints[sp]['qualified hits'])
        temp_set = set(temp_list)
        # make a diff from the lists
        face2export = np.asarray(list(temp_set - face_already_exported))
        # store the exported faces
        temp_set = set(face2export)
        face_already_exported = face_already_exported.union(temp_set);

        mesh_export = copy.deepcopy(mesh)
        export_faces = np.asarray(mesh_export.triangles)[face2export[:]]
        mesh_export.triangles = o3d.utility.Vector3iVector(export_faces[:][:])

        dataname = str(i) + '_' + str(sp) + '_refined'
        path2exp = destination + dataname + '.obj'
        o3d.io.write_triangle_mesh(path2exp, mesh_export)


# buggy maybe an wrong assignment
def heatmap_hit_export(config, model, candidates, scanpoints,
                       destination):  # listfound, facecount, points, triangls, area, switch):
    """function to export the inital model with colors according to the count of hits on the respectiv face"""
    # path_meshfile = config['model path']
    path_meshfile = 'arch/test.obj'
    count = np.zeros(model['size'])

    # get all faceid that are hit by strat
    all_hits = []
    for scanpoint in scanpoints:
        all_hits.extend(candidates[scanpoint]['qualified hits'])

    id_count = np.zeros((len(model['ids']), 2))
    id_count[:, 0] = model['ids']
    id_count = id_count.astype(int)

    unique, counts = np.unique(all_hits, return_counts=True)
    id_count[unique, 1] = counts

    hit_count = np.asarray(id_count[:, 1]) / float(np.max(np.asarray(id_count[:, 1])))
    # iterate over faces and sum up hits for points
    vertex_hits = np.zeros((len(model['vertices']), 1))
    for i, hits in enumerate(hit_count):
        vertex_ids = model['triangles'][i]
        vertex_hits[vertex_ids] += hits
        count[model['triangles'][i]] += 1
    vertex_hits_rel = vertex_hits

    # transform relative hits to rgb per vertex
    # value normalise by hits
    vertex_rgbs = np.zeros((len(model['vertices']), 3))
    for i, hits in enumerate(vertex_hits_rel):
        vertex_rgbs[i] = plt.cm.viridis(hits / count[i])[0][:3]
    # export
    mesh_export = mesh_export_helper(model=model)
    mesh_export.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_rgbs))
    path_heatmapmesh = destination + 'heatmap_debug.obj'
    o3d.io.write_triangle_mesh(path_heatmapmesh, mesh_export)

    return


def mesh_export_debug(model, filename):
    mesh = o3d.geometry.TriangleMesh()
    face_exp = np.asarray(model['triangles'])
    vertex_exp = np.asarray(model['vertices'])

    mesh.triangles = o3d.utility.Vector3iVector(face_exp[:][:])
    mesh.vertices = o3d.utility.Vector3dVector(vertex_exp[:][:])
    o3d.io.write_triangle_mesh(filename, mesh)


def mesh_export_helper(model):
    mesh = o3d.geometry.TriangleMesh()
    face_exp = np.asarray(model['triangles'])
    vertex_exp = np.asarray(model['vertices'])

    mesh.triangles = o3d.utility.Vector3iVector(face_exp[:][:])
    mesh.vertices = o3d.utility.Vector3dVector(vertex_exp[:][:])
    return (mesh)


def ids_to_mesh(meshfile, idlist, filename):
    mesh = o3d.io.read_triangle_mesh(meshfile)
    mesh_exp = []
    mesh_exp.append(copy.deepcopy(mesh))
    face_exp = np.asarray(mesh_exp.triangles)[idlist[:]]
    mesh_exp.triangles = o3d.utility.Vector3iVector(face_exp[:][:])
    o3d.io.write_triangle_mesh(filename, mesh_exp)

    a = 0


# export functions for reviewing
def export_sp_meshes_raw(model, result, scanpoints, destination):
    # exports all faces hit by all scanpoints
    mesh = mesh_export_helper(model)
    for i, sp in enumerate(result['strategy']['scanpoint ids']):
        face2exp = sorted(scanpoints[sp]['qualified hits'])
        mesh_exp = copy.deepcopy(mesh)
        exp_face = np.asarray(mesh_exp.triangles)[face2exp[:]]
        mesh_exp.triangles = o3d.utility.Vector3iVector(exp_face[:][:])
        dataname = str(i) + '_' + str(sp) + '_raw'
        path2exp = destination + dataname + '.obj'
        o3d.io.write_triangle_mesh(path2exp, mesh_exp)
        a = 0


def load_config(configfile):
    try:
        with open(configfile, 'r') as cf:
            config = json.load(cf)
        del cf
    except FileNotFoundError:
        print('config file not found')
        sys.exit()

    return config


def config_init(configfile, configpath):
    try:
        with open(configfile, 'r') as cf:
            config = json.load(cf)
            # is this still valide?
            if config['coverage goal'] == 'multi':
                config['coverage goal'] = [float(round(x, 3)) for x in list(np.linspace(0.8, 1.0, 20))]
            if config['count goal'] == 'multi':
                config['count goal'] = [x + 1 for x in range(50)]
        del cf
    except FileNotFoundError:
        print('config file not found')
        sys.exit()

    configdicts = []
    configdir = str(pathlib.Path().resolve()) + '/config_pool/'
    for f in os.listdir(configdir):
        try:
            os.remove(os.path.join(configdir, f))
        except:
            shutil.rmtree(os.path.join(configdir, f))
    multi = False
    for param in config.items():
        if type(param[1]) == list and not param[0] == 'z axis':
            multi = True
    config['z axis'] = tuple(config['z axis'])
    config['configpath'] = configpath

    # here new entrys in json
    config['additional Models'] = tuple(config['additional Models'])
    # convert to lists into dict
    try:
        config['minimum local point density add'] = dict(
            zip(config['minimum local point density class'], config['minimum local point density value']))
    except:
        print("Local density doesn´t have same number of values as classes")
        exit(1);
    # remove list that are now dict
    config.pop('minimum local point density class', None)
    config.pop('minimum local point density value', None)

    # create all config combinations
    # TODO: remove nonsense-combinations
    if multi:
        params = [p for p in config.keys()]
        values = []
        for param in config.items():
            # exclude non-level 1 parameters
            # if param[0] not in ['count goal', 'coverage goal', 'weight of the weight']:
            if param[0] in ['count goal', 'coverage goal']:
                param_list = [None]
            else:
                if type(param[1]) == list:
                    param_list = param[1]
                else:
                    param_list = [param[1]]
            values.append(param_list)

        # create all combi-tuples: level 1
        l1_configs = list(itertools.product(*values))
        del values
        # add level 2 params, calc combi tuples
        l2_configs = []
        l1_configs = [list(l) for l in l1_configs]
        for i, l1_config in enumerate(l1_configs):
            values = []
            # TODO: lift manual ID restriction
            if l1_config[12] == 'count goal':
                l1_config[15] = (config['count goal'])
                l1_configs[i] = l1_config
            if l1_config[12] == 'coverage goal':
                l1_config[16] = (config['coverage goal'])
                l1_configs[i] = l1_config
            for param in l1_config:
                if type(param) == list:
                    param_list = param
                else:
                    param_list = [param]
                values.append(param_list)
            l2_configs.extend(list(itertools.product(*values)))
            del values

        # build config dictionaries from lists with init values
        for i, configtup in enumerate(l2_configs):
            configdict = dict.fromkeys(params)
            for k, v in zip(configdict, configtup):
                configdict[k] = v
            configdict['z axis'] = list(configdict['z axis'])
            configdict['id'] = i
            if configdict['greedy type'] == 'count goal':
                configdict['coverage goal'] = None
            elif configdict['greedy type'] == 'coverage goal':
                configdict['count goal'] = None
            if not configdict['greedy weighted']:
                configdict['weight of the weight'] = None
            configdict['path'] = configpath + f"/config_{i}/"
            configdicts.append(configdict)

        # store all dicts for reference
        with open(f'config_pool/all_configs.json', 'w') as config_collection:
            json.dump(configdicts, config_collection, indent=2)

        for combi in configdicts:
            id = combi['id']
            dir = combi['path']
            os.mkdir(dir)
            with open(f"{dir}/config_{id}.json", 'w') as configfdump:
                json.dump(combi, configfdump)
    else:
        with open(f'config_pool/config.json', 'w') as configfdump:
            json.dump(config, configfdump)

    return


def multi_tool_overlap(combo, candidate_1, candidate_2, area):
    inter_ids = np.intersect1d(candidate_1, candidate_2)
    # TODO: overlap quality evaluation with aggregated normal directions
    union_ids = np.union1d(candidate_1, candidate_2)
    inter_area = np.sum(area[inter_ids])
    union_area = np.sum(area[union_ids])
    inter_perc = inter_area / union_area
    return combo, inter_area, inter_perc


def get_kth_neigbors(graph, node, k):
    nhall = nx.single_source_shortest_path_length(graph, node, cutoff=k)
    keys = list(nhall.keys())
    values = list(nhall.values())
    neighborhood = []
    for i in range(len(keys)):
        if (values[i] == k):
            neighborhood.append(keys[i])

    return (neighborhood)


def listdir_hasfile(path):
    res = False
    for f in os.listdir(path):
        if not f.startswith('.'):
            res = True
    return res
