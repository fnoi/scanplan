import os
import pathlib
import pickle
import plotly.graph_objects as go
import numpy as np


def dict2trace(
        color_faces='red',
        color_edges='blue',
        opac=0.3,
        filepath="/handover/0_occupancy_grid_scene.pkl"
) -> dict:

    with open(f"{os.path.dirname(pathlib.Path().resolve())}{filepath}", 'rb') as jar:
        occu_grid = pickle.load(jar)

    occu_grid['x'] = occu_grid['unique vertices'][:, 0]
    occu_grid['y'] = occu_grid['unique vertices'][:, 1]
    occu_grid['z'] = occu_grid['unique vertices'][:, 2]

    occu_grid['trace_edges'] = go.Scatter3d(x=occu_grid['edges'][0], y=occu_grid['edges'][1], z=occu_grid['edges'][2],
                                            mode='lines', line=dict(color=color_edges, width=3))
    occu_grid['trace_faces'] = go.Mesh3d(x=occu_grid['unique vertices'][:, 0],
                                         y=occu_grid['unique vertices'][:, 1],
                                         z=occu_grid['unique vertices'][:, 2],
                                         color=color_faces, alphahull=3, flatshading=True, opacity=opac,
                                         i=np.asarray(occu_grid['unique face ids'])[:, 0],
                                         j=np.asarray(occu_grid['unique face ids'])[:, 1],
                                         k=np.asarray(occu_grid['unique face ids'])[:, 2],
                                         hoverinfo='none')

    occu_grid['data'] = [occu_grid['trace_edges'], occu_grid['trace_faces']]

    return occu_grid
