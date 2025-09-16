import numpy as np
import math as m
import itertools


# convert cartesian to sphere coordinates
def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    alpha = m.atan2(z, m.sqrt(XsqPlusYsq)) * 180 / np.pi  # theta
    beta = m.atan2(y, x) * 180 / np.pi  # phi
    return r, alpha, beta


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew[-3:]

# TODO food for thought: sphere class, initiate once (segments), use classmethod and idlist for eval
# TODO needs to be VERY FAST, overlap (count) alone is already bottleneck computation time-wise


# this evaluats the quality of the overlap
def overlap_quality(model, ids):
    normals = model['normals'][ids]

    seg = 10

    sperecoords = []
    # this can be speed up by numpy
    for iter in normals:
        sperecoords.append(cart2sph(iter[0], iter[1], iter[2]))

    # init sphere list
    # this also can be speed up
    discret_sphere = []
    for i in range(seg):
        discret_sphere.append([])
        for j in range(seg * 2):
            discret_sphere[i].append([])

    # realy ugly but i don´t know a better approach for now
    # check all normals on which face they are pointing
    for l in range(len(sperecoords)):
        iter = sperecoords[l]
        for i, j, in zip(range(-90, 90, int(180 / seg)), range(seg)):
            for m, n in zip(range(-180, 180, int(180 / seg)), range(seg * 2)):
                if i <= iter[1] < i + 30:
                    if m <= iter[2] < m + 30:
                        discret_sphere[j][n].append(l)

    # search for mdd face
    mosthit = [0, 0, 0]
    for i in range(len(discret_sphere)):
        for j in range(len(discret_sphere[0])):
            if len(discret_sphere[i][j]) > mosthit[0]:
                mosthit[0] = len(discret_sphere[i][j])
                mosthit[1] = i
                mosthit[2] = j

    mdd = np.zeros(3)
    # getting main direction
    for iter in discret_sphere[mosthit[1]][mosthit[2]]:
        mdd += normals[iter]

    mdd = mdd / len(discret_sphere[mosthit[1]][mosthit[2]])
    mdd = mdd / np.linalg.norm(mdd)

    # setup list for sdd
    sdd_list = []
    for i in normals:
        # to get projection vector - (vector*mdd)*mdd
        tmp = i - (np.dot(i, mdd)) * mdd
        sdd_list.append(tmp)

    mosthit = 0
    for i in range(len(sdd_list)):
        if np.linalg.norm(sdd_list[i]) > mosthit:
            mosthit = np.linalg.norm(sdd_list[i])
            sdd = normals[i]

    sdd = sdd / np.linalg.norm(sdd)

    tdd = np.cross(mdd, sdd)

    return mdd, sdd, tdd
