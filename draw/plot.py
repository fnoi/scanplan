import pathlib
import sys

import control
import triangle
import voxel
import graph
import sys
import os


def plotter(meshes: list):
    # mesh needs manual input
    dicts = []
    for mesh in meshes:
        _dict = triangle.file2dict(filename=mesh)
        _dict = triangle.dict2trace(objdict=_dict)
        dicts.append(_dict)

    # voxel grid, candidate graph and strategy load from convention
    # define filepath for diverging input in dict2trace (voxel) and file2dict (graph)
    dicts.append(voxel.dict2trace(color_edges='grey', color_faces='green', opac=0.1))
    _dict = graph.file2dict()
    dicts.append(graph.dict2trace(graphdict=_dict,
                                  nodes_all=True,
                                  edges_neighbor=True,
                                  nodes_strat=False,
                                  edges_travel=False
                                  ))

    control.exec_plot(objlist=dicts, show=True, save=True, savefile='yooo.png')


if __name__ == "__main__":
    plotter(meshes=["/sphere/sphere.obj", "/sphere/plane.obj"])
    #plotter(meshes=[
    #    "/01_factory_building/factory_building_tri_showcase.obj",
    #    "/01_factory_building/factory_building_floors_planes_2.obj"
    #])