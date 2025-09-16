import numpy as np
import matplotlib.pyplot as plt


class Sphere:
    """
    0  theta  azimuth  H
    1  phi    polar    V  (from z-axis down)
    """

    def __init__(self, no_seg):
        # init params
        self.steps = no_seg
        self.offset = (1 / no_seg) / 2

        # angles / radial distance always 1
        self.theta = np.linspace(start=0, stop=1, num=self.steps)
        self.phi = np.linspace(start=-1, stop=1, num=self.steps)
        self.theta_lims = self.theta - self.offset
        self.theta_lims = np.append(self.theta_lims, np.asarray([1 + self.offset]), axis=0)
        self.phi_lims = self.phi - self.offset
        self.phi_lims = np.append(self.phi_lims, np.asarray([1 + self.offset]), axis=0)

        temp = np.meshgrid(self.theta, self.phi)
        self.normals_spherical = np.stack((np.ones_like(temp[0]), temp[0], temp[1]), axis=0)

        self.normals_spherical_flat = self.normals_spherical.flatten(order='C').reshape(3, -1).T
        self.normals_cartesian_flat = sph2cart(self.normals_spherical_flat)

        self.normals_cartesian = self.normals_cartesian_flat.T.reshape(self.normals_spherical.shape, order='C')

    def analyze(self, normals_xyz, normals_rpt):
        histogram_in = np.histogram2d(
            x=normals_rpt[:, 1],
            y=normals_rpt[:, 2],
            bins=[self.theta_lims, self.phi_lims]
        )[0]
        mdd_ind = np.unravel_index(
            np.argmax(
                histogram_in, axis=None
            ),
            histogram_in.shape
        )
        max_hit = histogram_in[mdd_ind]
        histogram_in = histogram_in / max_hit
        mdd_rpt = np.asarray([1, self.theta[mdd_ind[0]], self.phi[mdd_ind[1]]])
        mdd_xyz = sph2cart(mdd_rpt)[0]
        mdd = mdd_xyz
        dir_rel = [np.max(histogram_in), 0, 0]

        weighted_grid_normals = np.multiply(
            self.normals_cartesian,
            histogram_in
        )

        weighted_grid_normals_flat = weighted_grid_normals.flatten(order='C').reshape(3, -1).T
        mdd_proj_new = np.asarray([normal - (np.dot(normal, mdd) * mdd) for normal in weighted_grid_normals_flat])
        # mdd_proj_new[np.where(mdd_proj_new[:, 1] == -0)] = 0
        mdd_proj_new_sph = cart2sph(mdd_proj_new)
        mdd_proj_new_sph[np.isnan(mdd_proj_new_sph)] = 0

        sdd_ind = np.argmax(np.linalg.norm(mdd_proj_new_sph, axis=1))
        sdd_rpt = mdd_proj_new_sph[sdd_ind]
        sdd_xyz = mdd_proj_new[sdd_ind]
        dir_rel[1] = round(np.linalg.norm(sdd_xyz), 2)

        tdd = np.cross(mdd, sdd_xyz)
        dir_rel[2] = round(np.linalg.norm(tdd), 2)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        z = [0, mdd[2], 0, sdd_xyz[2], 0, tdd[2]]
        x = [0, mdd[0], 0, sdd_xyz[0], 0, tdd[0]]
        y = [0, mdd[1], 0, sdd_xyz[1], 0, tdd[1]]

        ax.plot3D(x, y, z)
        plt.show()

        return dir_rel


def sph2cart(rpt):
    if rpt.ndim == 1:
        rpt = rpt.reshape(1, -1)
    xyz = np.empty(rpt.shape)
    rpt[:, 1] *= np.pi
    rpt[:, 2] *= np.pi
    xyz[:, 0] = rpt[:, 0] * np.sin(rpt[:, 1]) * np.cos(rpt[:, 2])
    xyz[:, 1] = rpt[:, 0] * np.sin(rpt[:, 1]) * np.sin(rpt[:, 2])
    xyz[:, 2] = rpt[:, 0] * np.cos(rpt[:, 1])

    return xyz


def cart2sph(xyz):
    if xyz.ndim == 1:
        xyz = xyz.reshape(1, -1)
    rpt = np.empty(xyz.shape)
    rpt[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    rpt[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0]) / np.pi
    rpt[:, 1] = np.arccos(xyz[:, 2] / rpt[:, 0]) / np.pi

    return rpt
