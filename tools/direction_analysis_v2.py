import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from structure.sphere import Sphere, sph2cart, cart2sph

if __name__ == "__main__":
    # samplesize = 9000
    # normals = np.random.randint(0, 90, (samplesize, 2))

    # mesh_model = o3d.io.read_triangle_mesh('C:/Users/ga25mal/PycharmProjects/scanplan/data/bunny.obj')
    mesh_model = o3d.io.read_triangle_mesh('C:/Users/ga25mal/PycharmProjects/scanplan/data/00_testroom/testroom5_1.obj')
    # mesh_model = o3d.io.read_triangle_mesh('C:/Users/ga25mal/PycharmProjects/scanplan/data/sphere/sphere.obj')
    mesh_model.compute_vertex_normals()
    xyz = np.asarray(mesh_model.triangle_normals)  # TODO: integrate with legacy, handover normals as np.array
    xyz = xyz / np.linalg.norm(xyz, axis=1)[:, None]

    rpt = cart2sph(xyz)
    print(rpt)
    xyz = sph2cart(rpt)
    print(xyz)

    print(f'normals range sph_theta ({np.min(rpt[:, 1])},{np.max(rpt[:, 1])})'
          f'\nnormals range sph_phi   ({np.min(rpt[:, 2])},{np.max(rpt[:, 2])})')

    n_seg = 10

    sphere = Sphere(no_seg=n_seg)
    winner = sphere.analyze(normals_xyz=xyz, normals_rpt=rpt)

    print(winner)

    if min(winner) < 0.3:
        print('WARNING: unbalanced distribution of normals')

    a = 0
