import multiprocessing as mp
import numpy as np

var_dict_create = {}


def init_arrays_create(voxel_grid_dict):
    voxel_grid_face_coords_shape = voxel_grid_dict['unique face coords'].shape
    voxel_grid_face_coords_ra = mp.RawArray(
        'd',
        voxel_grid_face_coords_shape[0] * voxel_grid_face_coords_shape[1] * voxel_grid_face_coords_shape[2]
    )
    voxel_grid_face_coords_ra_np = np.frombuffer(
        voxel_grid_face_coords_ra, dtype=np.float64
    ).reshape(voxel_grid_face_coords_shape)
    np.copyto(voxel_grid_face_coords_ra_np, voxel_grid_dict['unique face coords'])

    voxel_grid_vertices_shape = voxel_grid_dict['unique vertices'].shape
    voxel_grid_vertices_ra = mp.RawArray(
        'd',
        voxel_grid_vertices_shape[0] * voxel_grid_vertices_shape[1]
    )
    voxel_grid_vertices_ra_np = np.frombuffer(
        voxel_grid_vertices_ra, dtype=np.float64
    ).reshape(voxel_grid_vertices_shape)
    np.copyto(voxel_grid_vertices_ra_np, voxel_grid_dict['unique vertices'])

    return {
        'unique face coords': voxel_grid_face_coords_ra,
        'unique face coords shape': voxel_grid_face_coords_shape,
        'unique face coords ra np': voxel_grid_face_coords_ra_np,
        'unique vertices': voxel_grid_vertices_ra,
        'unique vertices shape': voxel_grid_vertices_shape,
        'unique vertices ra np': voxel_grid_vertices_ra_np
    }


def face_filter_init(ra_voxel_grid_face_coords, ra_voxel_grid_face_coords_shape,
                     ra_voxel_grid_vertices, ra_voxel_grid_vertices_shape):
    var_dict_create['unique face coords'] = ra_voxel_grid_face_coords
    var_dict_create['unique face coords shape'] = ra_voxel_grid_face_coords_shape
    var_dict_create['unique vertices'] = ra_voxel_grid_vertices
    var_dict_create['unique vertices shape'] = ra_voxel_grid_vertices_shape


def face_filter_worker_rev(i):
    vox_coords_np = np.frombuffer(
        var_dict_create['unique face coords']).reshape(
        var_dict_create['unique face coords shape']
    )
    vox_verts_np = np.frombuffer(
        var_dict_create['unique vertices']).reshape(
        var_dict_create['unique vertices shape']
    )

    res = []
    for vox_coord in vox_coords_np:
        res.append(
            int(
                np.where(
                    np.all(
                        vox_coord == vox_verts_np, axis=1
                    )
                )
                [0]
            )
        )

    return res


def face_filter_worker(i):
    # print(i)
    face = np.frombuffer(
        var_dict_create['unique face coords']).reshape(
        var_dict_create['unique face coords shape']
    )[i]
    vert = np.frombuffer(
        var_dict_create['unique vertices']).reshape(
        var_dict_create['unique vertices shape']
    )
    # investigated face
    # face = ra_c['unique face coords ra np'][0]
    # vert = ra_c['unique vertices ra np']

    # print(vert)
    res = []
    for face_coord in face:
        res.append(
            np.where(
                np.all(
                    face_coord == vert, axis=1
                )
            )
            [0]
        )

    return res
