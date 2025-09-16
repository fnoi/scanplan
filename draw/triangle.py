import os
import pathlib
import meshio
import numpy as np
import plotly.graph_objects as go


def file2dict(filename: str) -> dict:
    mesh_model = f"{os.path.dirname(pathlib.Path().resolve())}/data{filename}"
    mesh = meshio.read(mesh_model)
    plot_mesh = {
        "x": mesh.points[:, 0],
        "y": mesh.points[:, 1],
        "z": mesh.points[:, 2],
        "i": mesh.cells_dict['triangle'][:, 0],
        "j": mesh.cells_dict['triangle'][:, 1],
        "k": mesh.cells_dict['triangle'][:, 2],
    }
    null = np.stack((mesh.cells_dict['triangle'][:, 0], mesh.cells_dict['triangle'][:, 1]))
    ones = np.stack((mesh.cells_dict['triangle'][:, 1], mesh.cells_dict['triangle'][:, 2]))
    twos = np.stack((mesh.cells_dict['triangle'][:, 2], mesh.cells_dict['triangle'][:, 0]))
    intermed = np.hstack((null, ones, twos)).transpose()
    plot_mesh['edges'] = [tuple(row) for row in intermed]
    non = np.full((intermed.shape[0], 1), None)
    plot_mesh['x_e'] = np.hstack((plot_mesh['x'][intermed], non)).flatten().tolist()
    plot_mesh['y_e'] = np.hstack((plot_mesh['y'][intermed], non)).flatten().tolist()
    plot_mesh['z_e'] = np.hstack((plot_mesh['z'][intermed], non)).flatten().tolist()

    return plot_mesh


def dict2trace(objdict, color_face='white', color_edge='black') -> dict:
    objdict['trace_faces'] = go.Mesh3d(
        x=objdict['x'],
        y=objdict['y'],
        z=objdict['z'],
        color=color_face,
        alphahull=3,
        flatshading=True,
        i=objdict['i'],
        j=objdict['j'],
        k=objdict['k']
    )
    objdict['trace_edges'] = go.Scatter3d(
        x=objdict['x_e'],
        y=objdict['y_e'],
        z=objdict['z_e'],
        mode='lines',
        line=dict(
            color=color_edge,
            width=2
        )
    )
    objdict['data'] = [objdict['trace_faces'], objdict['trace_edges']]

    return objdict


# def trace2plot(trace) -> None:



if __name__ == "__main__":
    dict_1 = file2dict("/sphere/sphere.obj")
    a=0