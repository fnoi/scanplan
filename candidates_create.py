import copy
import math
import numpy as np
import open3d as o3d
import multiprocessing as mp
import os
import pathlib
import pickle
import sys

from tqdm import tqdm

from tools.voxel import voxel_setup, voxel_purge
from tools.graph import candidate_graph
from tools.utils_nongeom import gather_config
from tools import timer
from tools import parallel

from tools import parallel_rev

sys.path.append(f'{pathlib.Path().resolve()}/draw')
from draw import plot


def candidates_create():
    print(f'\n[start - candidates_create ]')
    t = timer.Timer()
    t.start()

    config = gather_config(config_list=['candidates_create.json'])

    config["height delta"] = None
    if len(config["height options"]) == 1:
        config["height delta"] = config["height options"][0]
    else:
        config["height delta"] = max(
            [round(config["height options"][i + 1] - config["height options"][i], 2)
             for i in range(len(config["height options"]) - 1)]
        )
    assert config["height delta"] is not None, f"height delta missing"
    config["project path"] = pathlib.Path().resolve()

    t.report_lap(achieved='config load complete')

    # read objs
    mesh_scene = o3d.io.read_triangle_mesh(config['model file'])
    assert mesh_scene.get_surface_area() > 0, f"mesh area > 0 expected"
    t.report_lap(achieved=f'mesh loaded: {config["model file"]}')

    mesh_planes = o3d.io.read_triangle_mesh(config["plane file"])
    assert mesh_planes.get_surface_area() > 0, f"mesh area > 0 expected"
    t.report_lap(achieved=f'mesh loaded: {config["plane file"]}')

    # create occupancy / voxel grid with point sampling workaround. logic: 9pts per area as per resolution^2
    voxel_grid = {"cloud": mesh_scene.sample_points_uniformly(
        number_of_points=10 * round(mesh_scene.get_surface_area() * 9 / math.pow(config["scene resolution"], 2))
    )}
    t.report_lap(achieved='intermediate point cloud sampled')

    voxel_grid["scene"] = o3d.geometry.VoxelGrid.create_from_point_cloud(
        input=voxel_grid["cloud"],
        voxel_size=config["scene resolution"]
    )
    t.report_lap('scene grid created')

    t.report_lap("sandbox in")

    voxel_grid = voxel_setup(voxel_dict=voxel_grid, config=config)

    ra_c = parallel_rev.init_arrays_create(voxel_grid_dict=voxel_grid)

    mp_threads = os.cpu_count()
    parallel.config = config

    with mp.Pool(
            processes=mp_threads,
            initializer=parallel_rev.face_filter_init,
            initargs=(ra_c['unique face coords'],
                      ra_c['unique face coords shape'],
                      ra_c['unique vertices'],
                      ra_c['unique vertices shape'])
    ) as pool:
        resi = list(
            tqdm(
                pool.imap(
                    parallel_rev.face_filter_worker,
                    range(ra_c['unique vertices shape'][0])
                ),
                desc='face filter new and amazing',
                total=ra_c['unique vertices shape'][0]
            )
        )

    voxel_grid['unique face ids'] = np.array(resi)

    t.report_lap("sandbox out")




    # voxel_grid["unique face ids"] = voxel_purge(voxel_grid=voxel_grid)

    # re-build voxel scene for visualization


    # t.report_lap("sandbox out")

    # sample points on candidate planes. logic: 9pts per area as per resolution^2
    mesh_planes.compute_vertex_normals()
    candidates = {"initial": mesh_planes.sample_points_uniformly(
        number_of_points=10 * round(mesh_planes.get_surface_area() * 9 / math.pow(config["grid resolution"], 2))
    )}
    candidates["sampled"] = np.asarray(
        candidates["initial"].voxel_down_sample(voxel_size=config["grid resolution"]).points
    )

    # copy along elevation options
    __ = copy.deepcopy(candidates["sampled"])
    __[:, 2] = __[:, 2] + config["height options"][0]
    candidates["elevated"] = __
    if len(config["height options"]) > 1:
        for _ in config["height options"][1:]:
            __ = copy.deepcopy(candidates["sampled"])
            __[:, 2] = __[:, 2] + _
            candidates["elevated"] = np.concatenate(
                [candidates["elevated"], __]
            )
    t.report_lap('candidate grids created')

    _ = []
    for n, raw_point in enumerate(candidates["elevated"]):
        o3_coords = o3d.utility.Vector3dVector([raw_point])
        included = voxel_grid["scene"].check_if_included(o3_coords)
        if included[0]:
            _.append(n)
    candidates["valid"] = copy.deepcopy(candidates["elevated"])
    candidates["valid"] = np.delete(candidates["valid"], _, axis=0)
    del _, o3_coords, included
    t.report_lap('inclusion test & purge complete')

    # build fully connected candidate graph using nearest neighbor points and clusters
    candidates = candidate_graph(config=config, candidates=candidates)

    os.makedirs(f'{config["project path"]}{config["work dir"]}', exist_ok=True)
    deletees = ["cloud", "scene", "unique face coords", "center coords scene"]
    for _ in deletees:
        voxel_grid.pop(_)
    pickle.dump(voxel_grid,
                open(f"{config['project path']}{config['work dir']}"
                     f"{'0_occupancy_grid_scene.pkl'}", "wb")
                )
    pickle.dump(candidates["graph"],
                open(f"{config['project path']}{config['work dir']}"
                     f"{'0_candidate_graph.pkl'}", "wb")
                )

    t.report('completed candidates_create')
    print(f'[  end - candidates_create ]')

    # plot.plotter(meshfiles=["/sphere/sphere.obj", "/sphere/plane.obj"]) #TODO: robust relative path fixes


if __name__ == "__main__":
    candidates_create()
