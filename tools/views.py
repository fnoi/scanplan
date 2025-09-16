import pathlib
import os
import meshio
import time
import numpy as np
import pickle
import plotly
import plotly.graph_objects as go
import plotly.express as px

if __name__ == "__main__":



# load something
# create dicts
# create traces
# select- define - show





def plotly_mesh_dict(mesh_dict_list, vgrid_dict_list, show=False, save=False, filename=None, color_face='grey',
                     color_edge='white'):
    data = []
    for mesh_dict in mesh_dict_list:
        mesh_dict['trace_faces'] = go.Mesh3d(x=mesh_dict['x'], y=mesh_dict['y'], z=mesh_dict['z'], color=color_face,
                                             alphahull=3, flatshading=True,
                                             i=mesh_dict['i'], j=mesh_dict['j'], k=mesh_dict['k'])
        mesh_dict['trace_edges'] = go.Scatter3d(x=mesh_dict['x_e'], y=mesh_dict['y_e'], z=mesh_dict['z_e'],
                                                mode='lines',
                                                line=dict(color=color_edge, width=2))
        data.append(mesh_dict['trace_edges'])
        data.append(mesh_dict['trace_faces'])

    for vgrid_dict in vgrid_dict_list:
        data.append(vgrid_dict['trace_faces'])
        data.append(vgrid_dict['trace_edges'])

    data.append(trace_nodes_all)
    data.append(trace_edges_path)
    data.append(trace_nodes_choice)

    fig = go.Figure(data=data, layout=layout)
    fig.update_scenes(camera_projection_type='orthographic')

    fig.update_layout(
        # scene_aspectmode='data',
        # scene_aspectratio=dict(x=aspectratio_x, y=aspectratio_y, z=aspectratio_z),
        # margin=dict(t=5),
        scene_camera=camera
    )

    if show:
        fig.show()
    if save:
        if not filename:
            raise ('no filename provided')
        else:
            plotly.io.write_image(fig, filename, format='png')


def plotly_vgrid(show=False, color_faces='red', color_edges='blue', opac=0.3) -> dict:
    with open(f"{os.path.dirname(pathlib.Path().resolve())}/handover/0_occupancy_grid_scene.pkl", 'rb') as jar:
        occu_grid = pickle.load(jar)
    occu_grid.keys()

    vgrid_dict = {}
    vgrid_dict['trace_edges'] = go.Scatter3d(x=occu_grid['edges'][0], y=occu_grid['edges'][1], z=occu_grid['edges'][2],
                                             mode='lines', line=dict(color=color_edges, width=3))
    vgrid_dict['trace_faces'] = go.Mesh3d(x=occu_grid['unique vertices'][:, 0],
                                          y=occu_grid['unique vertices'][:, 1],
                                          z=occu_grid['unique vertices'][:, 2],
                                          color=color_faces, alphahull=3, flatshading=True, opacity=opac,
                                          i=np.asarray(occu_grid['unique face ids'])[:, 0],
                                          j=np.asarray(occu_grid['unique face ids'])[:, 1],
                                          k=np.asarray(occu_grid['unique face ids'])[:, 2],
                                          hoverinfo='none')

    data = [vgrid_dict['trace_faces'], vgrid_dict['trace_edges']]

    fig = go.Figure(data=data)#, layout=layout)
    fig.update_scenes(camera_projection_type='orthographic')

    # fig.update_layout(
    #     scene_aspectmode='data',
    #     scene_aspectratio=dict(x=aspectratio_x, y=aspectratio_y, z=aspectratio_z),
    #     margin=dict(t=5),
    #     scene_camera=camera
    # )

    if show:
        fig.show()

    return vgrid_dict


vgrid_dict = plotly_vgrid(show=False)


########################


## layout
axis = dict(showbackground=False,
           showline=False,
           zeroline=False,
           showgrid=False,
           showticklabels=False,
           title='')
layout = go.Layout(#title='dummy title idk',
                  width=1000,
                  height=1000,
                  showlegend=False,
                  scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),
                  margin=dict(t=100),
                  hovermode='closest')

#stolen from below
#range_i min, max, size, center
# range_x = [min(plot_mesh['x']),
#            max(plot_mesh['x']),
#            max(plot_mesh['x']) - min(plot_mesh['x']),
#            min(plot_mesh['x']) + 0.5 * (max(plot_mesh['x']) - min(plot_mesh['x']))]
# range_y = [min(plot_mesh['y']),
#            max(plot_mesh['y']),
#            max(plot_mesh['y']) - min(plot_mesh['y']),
#            min(plot_mesh['y']) + 0.5 * (max(plot_mesh['y']) - min(plot_mesh['y']))]
# range_z = [min(plot_mesh['z']), max(plot_mesh['z']),
#            max(plot_mesh['z']) - min(plot_mesh['z']),
#            min(plot_mesh['z']) + 0.5 * (max(plot_mesh['z']) - min(plot_mesh['z']))]
#
# axis_size = 1 * max([range_x[2], range_y[2], range_z[2]])
# axis_range_x = [range_x[3] - axis_size / 2, range_x[3] + axis_size / 2]
# axis_range_y = [range_y[3] - axis_size / 2, range_y[3] + axis_size / 2]
# axis_range_z = [range_z[3] - axis_size / 2, range_z[3] + axis_size / 2]
#
# aspectratio_x = range_x[2] / axis_size
# aspectratio_y = range_y[2] / axis_size
# aspectratio_z = range_z[2] / axis_size

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-.6, y=-1.25, z=0.65)
)

with open(f"{os.path.dirname(pathlib.Path().resolve())}/handover/0_candidate_graph.pkl", 'rb') as jar:
    candidate_graph = pickle.load(jar)

with open(f"{os.path.dirname(pathlib.Path().resolve())}/handover/2_greedy_strategy.pkl", 'rb') as jar:
    strategy = pickle.load(jar)



trace_edges_neighbors = go.Scatter3d(x=x_edges,
                                     y=y_edges,
                                     z=z_edges,
                                     mode='lines',
                                     line=dict(color='black', width=1),
                                     hoverinfo='none')

trace_nodes_all = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               text=n_nodes,
                               hoverinfo='text',
                               mode='markers',
                               marker=dict(symbol='circle', size=2, line=dict(color='black', width=0.25))
                               )

trace_nodes_choice = go.Scatter3d(x=x_nodes_choice,
                                  y=y_nodes_choice,
                                  z=z_nodes_choice,
                                  text=n_nodes_choice,
                                  mode='markers',
                                  marker=dict(symbol='circle', size=7.5, color=px.colors.qualitative.Plotly[1],
                                              line=dict(color='black', width=0.5)),
                                  hoverinfo='text'
                                  )

# trace_nodes_connectivity = go.Scatter3d(x=x_nodes_connect,
#                                        y=y_nodes_connect,
#                                        z=z_nodes_connect,
#                                         mode='markers',
#                                         marker=dict(symbol='circle', size=7.5, line=dict(color='blue', width=0.5))
#                                        )

trace_edges_connectivity = go.Scatter3d(x=x_edges_C,
                                        y=y_edges_C,
                                        z=z_edges_C,
                                        text=w_edges_C,
                                        mode='lines',
                                        line=dict(color=px.colors.qualitative.Plotly[2], width=5),
                                        hoverinfo='text'
                                        )

trace_edges_path = go.Scatter3d(x=x_path,
                                y=y_path,
                                z=z_path,
                                mode='lines',
                                line=dict(color=px.colors.qualitative.Plotly[0], width=2.5),
                                # marker=dict(symbol='cross', size=20),
                                hoverinfo='none')

mesh_dict = objmesh2mesh_dict(filename="/sphere/sphere.obj")
mesh_dict_2 = objmesh2mesh_dict(filename="/sphere/plane.obj")
plotly_mesh_dict(mesh_dict_list=[mesh_dict, mesh_dict_2], vgrid_dict_list=[vgrid_dict],
                 show=True, save=False, filename='yo_remote.png')