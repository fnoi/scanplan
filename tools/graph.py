import numpy as np
import pandas as pd
import networkx as nx
import open3d as o3d

from tools.helpers import directconnection,distancesmaller
from tqdm import tqdm


def candidate_graph(config, candidates):
    coords_free = candidates["valid"]
    coords_free = pd.DataFrame(coords_free, columns=['x', 'y', 'z'])
    coords_free.sort_values(by=['x'])
    coords_free.sort_values(by=['y'])
    coords_free.sort_values(by=['z'])
    
    
    #get scene for cornercuting check
    mesh_scene = o3d.io.read_triangle_mesh(config['model file'])
    mesh_scene = o3d.t.geometry.TriangleMesh.from_legacy(mesh_scene)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_scene)
    
    toremove=[]
    
    for ind in tqdm(coords_free.index, desc='remove point that are to close to scenes'):
        x = coords_free.x[ind]
        y = coords_free.y[ind]
        z = coords_free.z[ind]
        #get rid of point that are to close to scene
        if distancesmaller(scene,[x,y,z],config['min distance']):
            #TODO: remove from list to check against
            toremove.append(ind)
    #remove points to close to scene
    coords_free.drop(toremove)
    
    C = nx.Graph()

    nodes = []
    for ind in tqdm(coords_free.index, desc='building graph'):

        C.add_node(ind)
        x = coords_free.x[ind]
        y = coords_free.y[ind]
        z = coords_free.z[ind]
        x_min = round(x - config["grid resolution"] * config["inlier factor"], config["inlier precision"])
        x_max = round(x + config["grid resolution"] * config["inlier factor"], config["inlier precision"])
        y_min = round(y - config["grid resolution"] * config["inlier factor"], config["inlier precision"])
        y_max = round(y + config["grid resolution"] * config["inlier factor"], config["inlier precision"])
        z_min = round(z - config["height delta"] * config["inlier factor"], config["inlier precision"])
        z_max = round(z + config["height delta"] * config["inlier factor"], config["inlier precision"])

        inlier_ind = coords_free.index[(
                ((coords_free.x <= x_max) & (coords_free.x >= x_min)) &
                ((coords_free.y <= y_max) & (coords_free.y >= y_min)) &
                ((coords_free.z <= z_max) & (coords_free.z >= z_min))
        )]
        inlier_ind = inlier_ind.tolist()

        
        
        inlier_ind.remove(ind)
        # print(len(inlier_ind))

        if len(inlier_ind) > 0:
            for id in inlier_ind:
                dist = np.sqrt(
                    np.sum(
                        (np.asarray([x, y, z]) - np.asarray(
                            [coords_free.x[id], coords_free.y[id], coords_free.z[id]])) ** 2
                    )
                )
                #add check if a diagonal cuts a corner
                if directconnection([x, y, z], [coords_free.x[id], coords_free.y[id], coords_free.z[id]], scene):
                    C.add_edge(ind, id, weight=dist)

        else:
            print(f'problem, isolated node {ind}')

    coord_attr = {_: (coords_free.x[_], coords_free.y[_], coords_free.z[_]) for _, data in C.nodes(data=True)}
    nx.set_node_attributes(C, coord_attr, 'pos')

    # -------------------------------------
    comps = sorted([c for c in nx.connected_components(C)], key=len)
    comps_og = len(comps)

    who = 0
    cx = 0

    coords_free_np = np.asarray(coords_free)

    while len(comps) > 1:
        who += 1
        print(f'fixing {who}/original {comps_og}, now {len(comps)}')

        smol = np.asarray(list(comps[0]))
        rest = []
        for _ in range(1, len(comps)):
            rest.extend(list(comps[_]))
        rest = np.asarray(rest)

        rep_smol = np.repeat(smol, rest.shape[0])
        rep_rest = np.tile(rest, smol.shape[0])

        coords_smol = coords_free_np[rep_smol]
        coords_rest = coords_free_np[rep_rest]

        dists = np.sqrt(np.sum(np.power(np.abs(coords_smol - coords_rest), 2), axis=1))

        mindist = np.min(dists)
        ind = np.argwhere(dists == mindist)
        if float(mindist) <= 2 * config["grid resolution"]:
            for __ in ind:
                C.add_edge(int(rep_smol[__]), int(rep_rest[__]), weight=float(mindist))
                cx += 1
        else:
            C.add_edge(int(rep_smol[ind[0]]), int(rep_rest[ind[0]]), weight=float(mindist))
            cx += 1

        comps = sorted([c for c in nx.connected_components(C)], key=len)

    candidates["graph"] = C

    print(f'moment of truth, graph is complete: {nx.is_connected(C)}, fixed {cx}')

    return candidates
