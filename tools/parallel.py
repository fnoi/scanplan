import multiprocessing as mp
import numpy as np
import itertools

#
# from numba import jit

var_dict = {}


def init_arrays(model, candidates):  # TODO: set up neutral for all, fill where needed once workers are individual
    midpoints_shape = model['midpoint'].shape
    midpoints_ra = mp.RawArray('d', midpoints_shape[0] * midpoints_shape[1])
    midpoints_ra_np = np.frombuffer(midpoints_ra, dtype=np.float64).reshape(midpoints_shape)
    np.copyto(midpoints_ra_np, model['midpoint'])

    candidates_shape = (len(candidates), 3)
    candidates_ra = mp.RawArray('d', candidates_shape[0] * candidates_shape[1])
    candidates_ra_np = np.frombuffer(candidates_ra, dtype=np.float64).reshape(candidates_shape)
    np.copyto(candidates_ra_np, np.asarray([_['coordinates'] for _ in candidates]))

    candidate_hits_shape = (len(candidates), model['size'])
    candidate_hits_ra = mp.RawArray('i', candidate_hits_shape[0] * candidate_hits_shape[1])
    candidate_hits_ra_np = np.frombuffer(candidate_hits_ra, dtype=np.int32).reshape(candidate_hits_shape)
    candidate_hits_ra_np[:] = -1

    visibility_mask_shape = (len(candidates), model['size'])
    visibility_mask_ra = mp.RawArray('b', visibility_mask_shape[0] * visibility_mask_shape[1])
    visibility_mask_ra_np = np.frombuffer(visibility_mask_ra, dtype=bool).reshape(visibility_mask_shape)
    visibility_mask_ra_np[:] = False

    face_normals_shape = model['normals'].shape
    face_normals_ra = mp.RawArray('d', face_normals_shape[0] * face_normals_shape[1])
    face_normals_ra_np = np.frombuffer(face_normals_ra, dtype=np.float64).reshape(face_normals_shape)
    np.copyto(face_normals_ra_np, model['normals'])

    face_areas_shape = (model['size'], 1)
    face_areas_ra = mp.RawArray('d', face_areas_shape[0] * face_areas_shape[1])
    face_areas_ra_np = np.frombuffer(face_areas_ra, dtype=np.float64).reshape(face_areas_shape)
    np.copyto(face_areas_ra_np, model['area'].reshape(face_areas_shape))

    indies = [i for i in range(len(candidates))]
    combos = list(itertools.combinations(indies, 2))

    overlap_combos_shape = (len(combos), 2)
    overlap_combos_ra = mp.RawArray('i', overlap_combos_shape[0] * overlap_combos_shape[1])
    overlap_combos_ra_np = np.frombuffer(overlap_combos_ra, dtype=np.int32).reshape(overlap_combos_shape)
    np.copyto(overlap_combos_ra_np, np.asarray(combos))

    return {
        "midpoints": midpoints_ra,
        "midpoints shape": midpoints_shape,
        "candidates": candidates_ra,
        "candidates shape": candidates_shape,
        "candidate hits": candidate_hits_ra,
        "candidate hits ra np": candidate_hits_ra_np,
        "candidate hits shape": candidate_hits_shape,
        "face normals": face_normals_ra,
        "face normals shape": face_normals_shape,
        "face areas": face_areas_ra,
        "face areas ra np": face_areas_ra_np,
        "face areas shape": face_areas_shape,
        "overlap combos": overlap_combos_ra,
        "overlap combos shape": overlap_combos_shape,
        "visibility mask": visibility_mask_ra,
        "visibility mask ra np": visibility_mask_ra_np,
        "visibility mask shape": visibility_mask_shape
    }


def rayprep_init(ra_midpoints, ra_midpoints_shape, ra_candidates, ra_candidates_shape):
    var_dict['midpoints'] = ra_midpoints
    var_dict['midpoints_shape'] = ra_midpoints_shape
    var_dict['candidates'] = ra_candidates
    var_dict['candidates_shape'] = ra_candidates_shape


def rayprep_worker(candidate):
    midpoints_ra_np = np.frombuffer(var_dict['midpoints']).reshape(var_dict['midpoints_shape'])
    candidates_ra_np = np.frombuffer(var_dict['candidates']).reshape(var_dict['candidates_shape'])
    candidate_coords = candidates_ra_np[candidate]
    candidate_stack = np.tile(candidate_coords, (var_dict['midpoints_shape'][0], 1))
    directions = midpoints_ra_np - candidate_stack

    return (candidate_coords, directions)


def eval_init(ra_midpoints, ra_midpoints_shape, ra_candidates, ra_candidates_shape,
              ra_candidate_hits, ra_candidate_hits_shape, ra_face_normals, ra_face_normals_shape, config):
    var_dict['midpoints'] = ra_midpoints
    var_dict['midpoints_shape'] = ra_midpoints_shape
    var_dict['candidates'] = ra_candidates
    var_dict['candidates_shape'] = ra_candidates_shape
    var_dict['candidate_hits'] = ra_candidate_hits
    var_dict['candidate_hits_shape'] = ra_candidate_hits_shape
    var_dict['face_normals'] = ra_face_normals
    var_dict['face_normals_shape'] = ra_face_normals_shape
    var_dict['config'] = config


def eval_worker(candidate):
    config = var_dict['config']
    midpoints_ra_np = np.frombuffer(var_dict['midpoints']).reshape(var_dict['midpoints_shape'])
    candidates_ra_np = np.frombuffer(var_dict['candidates']).reshape(var_dict['candidates_shape'])
    candidate_hits_ra_np = np.frombuffer(var_dict['candidate_hits'], dtype=np.int32).reshape(
        var_dict['candidate_hits_shape'])
    candidate_hits_ra_np = candidate_hits_ra_np[candidate]
    candidate_hits = np.delete(candidate_hits_ra_np, np.where(candidate_hits_ra_np == -1))
    face_normals_ra_np = np.frombuffer(var_dict['face_normals']).reshape(var_dict['face_normals_shape'])

    candidate_coords = candidates_ra_np[candidate]

    disqualified = []

    midpoint_hit = midpoints_ra_np[candidate_hits]
    hit_vectors = midpoint_hit - candidate_coords
    dist = np.sqrt(
        np.sum(
            hit_vectors * hit_vectors,
            axis=1
        )
    )
    if config['maximum distance'] != 0:
        _ = list(np.where(dist > config['maximum distance'])[0])
        if len(_) != 0:
            disqualified.extend(
                candidate_hits[_]
            )

    if config['minimum distance'] != 0:
        _ = list(np.where(dist < config['minimum distance'])[0])
        if len(_) != 0:
            disqualified.extend(
                candidate_hits[_]
            )

    # max scanner angle check
    directions = midpoint_hit - candidate_coords
    dotpro = np.sum(
        config['z axis'] * directions[:],
        axis=1
    )
    absolute = np.sqrt(
        np.sum(
            directions * directions,
            axis=1
        )
    )
    angle = np.arccos(dotpro / absolute)
    angle = np.rad2deg(angle)
    for i in range(len(angle)):
        if not (config['minimum output angle'] < angle[i] < config['maximum output angle']) and not (
                (360 - config['maximum output angle']) < angle[i] < (360 - config['minimum output angle'])):
            disqualified.append(candidate_hits[i])

    normals_hit = face_normals_ra_np[candidate_hits]

    dotpro = np.sum(
        hit_vectors * normals_hit,
        axis=1
    )

    absolute = np.sqrt(
        np.sum(
            hit_vectors * hit_vectors,
            axis=1
        )
        *
        np.sum(
            normals_hit * normals_hit,
            axis=1
        )
    )

    # % 90 solves acute angle issue, itera-loop obsolete
    inci_angle = np.absolute(np.rad2deg(np.arccos(dotpro / absolute)[:])[:] % 90)

    # calc local density here
    spacm = config['scanner point spacing at 10m'] / 10 / 1e3  # 1e3 to bring res to meter

    local_density = np.multiply(
        np.divide(
            np.ones_like(inci_angle),
            np.multiply(
                np.ones_like(inci_angle) * spacm, dist
            )
        )
        ,
        np.divide(
            np.ones_like(inci_angle),
            np.multiply(
                np.ones_like(inci_angle) * spacm,
                np.divide(
                    dist, np.sin(inci_angle * np.pi / 180.)
                )
            )
        )
    )

    # disqualify angle below requirement
    if config['minimum incidence angle'] != 0:
        _ = list(np.where(inci_angle < config['minimum incidence angle'])[0])
        if len(_) != 0:
            disqualified.extend(
                candidate_hits[_]
            )

    if config['minimum local point density'] != 0:
        _ = list(np.where(local_density < config['minimum local point density'])[0])
        if len(_) != 0:
            disqualified.extend(
                candidate_hits[_]
            )

    disqualified = np.unique(disqualified)

    qualified = list(set(candidate_hits) - set(disqualified))

    return qualified


def overlap_init_rev(ra_visibility_mask, ra_visibility_mask_shape, ra_face_areas, ra_face_areas_shape,
                     ra_overlap_combos, ra_overlap_combos_shape):
    var_dict["visibility_mask"] = ra_visibility_mask
    var_dict["visibility_mask_shape"] = ra_visibility_mask_shape
    var_dict["face_areas"] = ra_face_areas
    var_dict["face_areas_shape"] = ra_face_areas_shape
    var_dict["overlap_combos"] = ra_overlap_combos
    var_dict["overlap_combos_shape"] = ra_overlap_combos_shape


def overlap_worker_rev(combo):
    visibility_mask_np = np.frombuffer(
        var_dict['visibility_mask'], dtype=bool).reshape(var_dict['visibility_mask_shape'])
    face_areas_ra_np = np.frombuffer(
        var_dict['face_areas']).reshape(var_dict['face_areas_shape'])
    overlap_combo_ra_np = np.frombuffer(
        var_dict['overlap_combos'], dtype=np.int32).reshape(var_dict['overlap_combos_shape'])

    current_combo = overlap_combo_ra_np[combo]

    candidate_hits_a = visibility_mask_np[current_combo[0]]
    candidate_hits_b = visibility_mask_np[current_combo[1]]
    inter_ids = np.logical_and(
        candidate_hits_a,
        candidate_hits_b
    )
    if inter_ids.any():
        union_ids = np.logical_or(
            candidate_hits_a,
            candidate_hits_b
        )
        inter_area = np.sum(face_areas_ra_np[inter_ids])
        union_area = np.sum(face_areas_ra_np[union_ids])
        inter_perc = inter_area / union_area

    else:
        inter_area = 0.0
        inter_perc = 0.0

    return current_combo, inter_area, inter_perc


def overlap_init(ra_candidate_hits, ra_candidate_hits_shape, ra_face_areas, ra_face_areas_shape,
                 ra_overlap_combos, ra_overlap_combos_shape):
    var_dict["candidate_hits"] = ra_candidate_hits
    var_dict["candidate_hits_shape"] = ra_candidate_hits_shape
    var_dict["face_areas"] = ra_face_areas
    var_dict["face_areas_shape"] = ra_face_areas_shape
    var_dict["overlap_combos"] = ra_overlap_combos
    var_dict["overlap_combos_shape"] = ra_overlap_combos_shape


def overlap_worker(combo):
    candidate_hits_ra_np = np.frombuffer(
        var_dict['candidate_hits'], dtype=np.int32).reshape(var_dict['candidate_hits_shape'])
    # candidate_hits_ra_np = np.delete(candidate_hits_ra_np, np.where(candidate_hits_ra_np == -1))
    face_areas_ra_np = np.frombuffer(
        var_dict['face_areas']).reshape(var_dict['face_areas_shape'])
    overlap_combo_ra_np = np.frombuffer(
        var_dict['overlap_combos'], dtype=np.int32).reshape(var_dict['overlap_combos_shape'])

    current_combo = overlap_combo_ra_np[combo]

    candidate_hits_a = candidate_hits_ra_np[current_combo[0]]
    candidate_hits_a = np.delete(candidate_hits_a, np.where(candidate_hits_a == -1))
    candidate_hits_b = candidate_hits_ra_np[current_combo[1]]
    candidate_hits_b = np.delete(candidate_hits_b, np.where(candidate_hits_b == -1))

    inter_ids = np.intersect1d(
        candidate_hits_a,
        candidate_hits_b,
        assume_unique=True
    )
    if inter_ids.any():
        union_ids = np.union1d(
            candidate_hits_a,
            candidate_hits_b
        )
        inter_area = np.sum(face_areas_ra_np[inter_ids])
        union_area = np.sum(face_areas_ra_np[union_ids])
        inter_perc = inter_area / union_area

    else:
        inter_area = 0.0
        inter_perc = 0.0

    return current_combo, inter_area, inter_perc


