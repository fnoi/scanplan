import time
import copy
import numpy as np
import open3d as o3d
from tqdm import tqdm


# this prep is the new one but leads to an error
# def ray_prep(scanpoint, midpoints):
#     directions = midpoints - scanpoint

#     # raytensor = np.c_[
#     #     np.tile(
#     #         scanpoint,
#     #         (len(directions), 1)),
#     #     directions
#     # ]
#     raytensor_new = (scanpoint, directions)

#     return raytensor_new

# just an backup function can be deletet if new one works
def ray_prep(iter, model, candidates, config, current):
    midpoints = np.asarray(model['midpoint'])
    scanpoint = np.asarray(candidates[iter]['coordinates'])
    directions = midpoints - scanpoint
    directions_full = copy.deepcopy(directions)

    tmp = np.tile(scanpoint, (len(directions), 1))
    raytensor = np.c_[tmp, directions]

    return raytensor, directions_full, current


def prep_cast_multi(jj, candidates, size, midpoints, log_time, restrict_scan, z_axis=[0.0, 0.0, 1.0]):
    current = time.perf_counter()
    midpoints = np.asarray(midpoints)
    scanpoint = np.asarray(candidates[jj])
    directions = midpoints - scanpoint
    directions_full = copy.deepcopy(directions)
    z_axis = np.array(z_axis)

    tmp = np.tile(scanpoint, (len(directions), 1))
    raytensor = np.c_[tmp, directions]

    return raytensor, directions_full


def ray_cast(scene, raytensor, config):
    """open3d ray cast"""
    current = time.perf_counter()
    temps = np.c_[
        np.tile(
            raytensor[0],
            (len(raytensor[1]), 1)),
        raytensor[1]
    ]
    ray = o3d.core.Tensor(np.asarray(temps), dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(ray, nthreads=0)
    # ans = ans['geometry_ids'].numpy()
    # ans = np.unique(ans)
    return ans


def eval_cast(ans, config):
    """evaluation of ray cast raw result: list of (unique) hits"""
    current = time.perf_counter()
    list_found = np.unique(ans['geometry_ids'].numpy())
    # 4294967295 indicates 'no hit' exception, excluded
    if list_found.any:
        if list_found[-1] == 4294967295:
            temp_list = list_found.tolist()
            temp_list.pop()
            list_found = np.asarray(temp_list)
            # list_found = np.unique(temp_list)
    return list_found


def cast_plus_eval(scene, raytensor, config):
    temps = np.c_[
        np.tile(
            raytensor[0],
            (len(raytensor[1]), 1)),
        raytensor[1]
    ]
    ray = o3d.core.Tensor(np.asarray(temps), dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(ray, nthreads=0)

    list_found = np.unique(ans['geometry_ids'].numpy())
    if len(list(list_found)) != 0:
        if list_found[-1] == 4294967295:
            temp_list = list_found.tolist()
            temp_list.pop()
            list_found = np.asarray(temp_list)

    return list_found


def final_eval(model, candidates, config):
    """final evaluation after all candidates have been evaluated individually"""
    # print(f'start final eval')
    start_fe = time.time()
    hit_ids_global = []
    for candi in candidates:
        hit_ids_global.extend(candi['qualified hits'])
    hit_ids_global = np.unique(hit_ids_global)
    hit_area_global = np.sum(np.array(model['area'])[hit_ids_global])
    for candi in tqdm(candidates, desc='storing evaluation results'):
        candi['max coverage hits'] = len(hit_ids_global)
        candi['max coverage area'] = hit_area_global
        candi['face ratio local'] = len(candi['qualified hits']) / candi['max coverage hits']
        candi['area ratio local'] = candi['area covered'] / candi['max coverage area']
        # not sure if this is the right place
        # get area of all hits
        max_area_covered = np.sum(np.array(model['area'])[candi['hit ids'][0]])

        candi['disqualified area in %'] = 1 - (candi['area covered'] / max_area_covered)

    # print(f'end fe {time.time() - start_fe}')
    return candidates
