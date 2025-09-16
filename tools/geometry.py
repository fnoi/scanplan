import time
import numpy as np
import os
#from tools import helpers


def midpoints_core(triangles, vertices):
    triangles = mesh['triangles']
    vertices = mesh['vertices']

    area_total = 0
    area_tris = []
    midpoints = []



    for m in range(mesh['size']):
        # get midpoint of triangle
        point1 = triangles[m][0]
        point2 = triangles[m][1]
        point3 = triangles[m][2]
        vertex1 = np.asarray([vertices[point1][0], vertices[point1][1], vertices[point1][2]])
        vertex2 = np.asarray([vertices[point2][0], vertices[point2][1], vertices[point2][2]])
        vertex3 = np.asarray([vertices[point3][0], vertices[point3][1], vertices[point3][2]])

        midpoint = vertex1[:] / 3 + vertex2[:] / 3 + vertex3[:] / 3
        length1 = vertex2 - vertex1
        length2 = vertex3 - vertex1
        normal_area = np.absolute((np.cross(length1, length2)))
        area_small = 0.5 * (np.dot(normal_area, normal_area)) ** 0.5
        area_total += area_small
        area_tris.append(area_small)

        midpoints.append(midpoint)

def triangle_midpoints_area_multi(mesh, config, current):
    combo_combos_2 = [(combo,
                       qualified_hitlist[combo[0]],
                       qualified_hitlist[combo[1]],
                       area
                       ) for combo in combos]

    start_multi = time.time()
    threads = os.cpu_count()
    print(f'- multi is on {threads} cores')

    with mp.Pool(threads) as pool:
        midpoints = pool.starmap(multi_tool_overlap, combo_combos_2)


def triangle_midpoints_area(mesh, config):
    # (size, triangles, vertices, log_time)
    """triangle face midpoint and area calculation"""
    triangles = mesh['triangles']
    vertices = mesh['vertices']

    points1 = triangles[:, 0]
    points2 = triangles[:, 1]
    points3 = triangles[:, 2]
    vertices1 = np.asarray([
        vertices[points1, 0],
        vertices[points1, 1],
        vertices[points1, 2]
    ]).transpose()
    vertices2 = np.asarray([
        vertices[points2, 0],
        vertices[points2, 1],
        vertices[points2, 2]
    ]).transpose()
    vertices3 = np.asarray([
        vertices[points3, 0],
        vertices[points3, 1],
        vertices[points3, 2]
    ]).transpose()

    midpoints_2 = vertices1 / 3 + vertices2 / 3 + vertices3 / 3

    AB = vertices3 - vertices1
    AC = vertices2 - vertices1
    A = 0.5 * np.linalg.norm(np.cross(AB, AC, axis=1), axis=1)

    mesh['area total'] = np.sum(A) # area_total
    mesh['area'] = A # area_tris
    mesh['midpoint'] = midpoints_2

    return mesh


def calc_local_points():
    a=0
    # ray_normplane =
    # tri_on_plane =
    # sin_alpha =
    # local_spacing =
    # local_density =
    # proj_area =
    # real_area =
    # real_points =
    # local_spacing
