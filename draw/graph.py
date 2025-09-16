import os
import pickle
import pathlib
import plotly.graph_objects as go
import plotly.express as px

import numpy as np


def file2dict(file_graph="/handover/0_candidate_graph.pkl", file_strat="/handover/2_greedy_strategy.pkl") -> dict:
    with open(f"{os.path.dirname(pathlib.Path().resolve())}{file_graph}", 'rb') as jar:
        c_graph = pickle.load(jar)

    with open(f"{os.path.dirname(pathlib.Path().resolve())}{file_strat}", 'rb') as jar:
        strategy = pickle.load(jar)

    graph_dict = {'x': np.asarray([c_graph.nodes[_]['pos'][0] for _ in c_graph.nodes]),
                  'y': np.asarray([c_graph.nodes[_]['pos'][1] for _ in c_graph.nodes]),
                  'z': np.asarray([c_graph.nodes[_]['pos'][2] for _ in c_graph.nodes]),
                  'x_choice': np.asarray([c_graph.nodes[_]['pos'][0] for _ in strategy['route']]),
                  'y_choice': np.asarray([c_graph.nodes[_]['pos'][1] for _ in strategy['route']]),
                  'z_choice': np.asarray([c_graph.nodes[_]['pos'][2] for _ in strategy['route']]),
                  'n_nodes_all': list(c_graph.nodes), 'n_nodes_choice': strategy['route']}

    x_nodes_connect = [c_graph.nodes[_]['pos'][0] for _ in c_graph.nodes]
    y_nodes_connect = [c_graph.nodes[_]['pos'][1] for _ in c_graph.nodes]
    z_nodes_connect = [c_graph.nodes[_]['pos'][2] for _ in c_graph.nodes]

    edge_list_N = c_graph.edges()
    x_edges = []
    y_edges = []
    z_edges = []
    for edge in edge_list_N:
        x_coords = [c_graph.nodes[edge[0]]['pos'][0], c_graph.nodes[edge[1]]['pos'][0], None]
        y_coords = [c_graph.nodes[edge[0]]['pos'][1], c_graph.nodes[edge[1]]['pos'][1], None]
        z_coords = [c_graph.nodes[edge[0]]['pos'][2], c_graph.nodes[edge[1]]['pos'][2], None]

        x_edges += x_coords
        y_edges += y_coords
        z_edges += z_coords

    edge_list_C = c_graph.edges()
    x_edges_C = []
    y_edges_C = []
    z_edges_C = []
    w_edges_C = []

    for edge in edge_list_C:
        x_coords_C = [c_graph.nodes[edge[0]]['pos'][0], c_graph.nodes[edge[1]]['pos'][0], None]
        y_coords_C = [c_graph.nodes[edge[0]]['pos'][1], c_graph.nodes[edge[1]]['pos'][1], None]
        z_coords_C = [c_graph.nodes[edge[0]]['pos'][2], c_graph.nodes[edge[1]]['pos'][2], None]
        w_edges_C.append(c_graph[edge[0]][edge[1]]['weight'])

        x_edges_C += x_coords_C
        y_edges_C += y_coords_C
        z_edges_C += z_coords_C

    graph_dict['edges_neighbors_x'] = x_edges_C
    graph_dict['edges_neighbors_y'] = y_edges_C
    graph_dict['edges_neighbors_z'] = z_edges_C

    x_path = []
    y_path = []
    z_path = []

    for ind in range(len(strategy['full_path']) - 1):
        waypoint = c_graph.nodes[strategy['full_path'][ind]]['pos']
        nextpoint = c_graph.nodes[strategy['full_path'][ind + 1]]['pos']
        x_edges_path = [waypoint[0], nextpoint[0], None]
        y_edges_path = [waypoint[1], nextpoint[1], None]
        z_edges_path = [waypoint[2], nextpoint[2], None]

        x_path += x_edges_path
        y_path += y_edges_path
        z_path += z_edges_path

    graph_dict['edges_path_x'] = x_path
    graph_dict['edges_path_y'] = y_path
    graph_dict['edges_path_z'] = z_path

    return graph_dict


def dict2trace(graphdict,
               nodes_all=True,
               nodes_strat=True,
               edges_neighbor=False,
               edges_travel=True,
               edges_connectivity=False):
    graphdict['data'] = []

    if nodes_strat:
        graphdict['trace_nodes_strategy'] = go.Scatter3d(
            x=graphdict['x_choice'],
            y=graphdict['y_choice'],
            z=graphdict['z_choice'],
            text=graphdict['n_nodes_choice'],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=7.5, color=px.colors.qualitative.Plotly[1],
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            hoverinfo='text'
        )
        graphdict['data'].append(graphdict['trace_nodes_strategy'])

    if nodes_all:
        graphdict['trace_nodes_all'] = go.Scatter3d(
            x=graphdict['x'],
            y=graphdict['y'],
            z=graphdict['z'],
            text=graphdict['n_nodes_all'],
            hoverinfo='text',
            mode='markers',
            marker=dict(
                symbol='circle',
                size=2,
                line=dict(
                    color='black',
                    width=0.25)
            )
        )
        graphdict['data'].append(graphdict['trace_nodes_all'])

    if edges_neighbor:
        graphdict['trace_edges_neighbors'] = go.Scatter3d(
            x=graphdict['edges_neighbors_x'],
            y=graphdict['edges_neighbors_y'],
            z=graphdict['edges_neighbors_z'],
            mode='lines',
            line=dict(
                color='black', width=1
            ),
            hoverinfo='none')
        graphdict['data'].append(graphdict['trace_edges_neighbors'])

    if edges_travel:
        graphdict['trace_edges_path'] = go.Scatter3d(
            x=graphdict['edges_path_x'],
            y=graphdict['edges_path_y'],
            z=graphdict['edges_path_z'],
            mode='lines',
            line=dict(
                color=px.colors.qualitative.Plotly[0],
                width=3
            ),
            marker=dict(symbol='cross', size=20),
            hoverinfo='none'
        )
        graphdict['data'].append(graphdict['trace_edges_path'])

    # TODO: add "travelable"



    # trace_nodes_connectivity = go.Scatter3d(x=x_nodes_connect,
    #                                        y=y_nodes_connect,
    #                                        z=z_nodes_connect,
    #                                         mode='markers',
    #                                         marker=dict(symbol='circle', size=7.5, line=dict(color='blue', width=0.5))
    #                                        )

    # trace_edges_connectivity = go.Scatter3d(x=x_edges_C,
    #                                         y=y_edges_C,
    #                                         z=z_edges_C,
    #                                         text=w_edges_C,
    #                                         mode='lines',
    #                                         line=dict(color=px.colors.qualitative.Plotly[2], width=5),
    #                                         hoverinfo='text'
    #                                         )

    return graphdict
