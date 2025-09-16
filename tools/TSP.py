import numpy as np
import networkx as nx
import itertools
import time
import copy
import open3d as o3d
# from itertools import pairwise


from structure.class_point import Point
from tools.helpers import directconnection


def point_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

def point_dist_np(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# this function is used to get a graph of all candidats to get the weights for the TSP
def build_location_graph(candidates):
    time_build = time.time()
    print('start building graph')
    distx = 0
    disty = 0
    distz = 0

    # get equidistance
    samplepoint = candidates[0]['coordinates']
    for i in range(1, len(candidates)):
        if (not np.isclose(samplepoint[0], candidates[i]['coordinates'][0]) and distx == 0):
            distx = np.abs(samplepoint[0] - candidates[i]['coordinates'][0])
        if (not np.isclose(samplepoint[1], candidates[i]['coordinates'][1]) and disty == 0):
            disty = np.abs(samplepoint[1] - candidates[i]['coordinates'][1])
        if (not np.isclose(samplepoint[2], candidates[i]['coordinates'][2]) and distz == 0):
            distz = np.abs(samplepoint[2] - candidates[i]['coordinates'][2])
        if (not np.isclose(distx, 0)) and (not np.isclose(disty, 0)) and (not np.isclose(distz, 0)):
            break

    points = []
    edges = []
    # create points for graph
    for iter in candidates:
        points.append(Point(iter['coordinates']))

    # neighborhood recognition
    # this can probaply be speed up by numpy where
    for i in range(len(candidates)):

        for j in range(i + 1, len(candidates)):
            # check if candidest are near to each other in 1 cooardinate direction
            if np.isclose(np.abs(candidates[i]['coordinates'][0] - candidates[j]['coordinates'][0]), distx) \
                    and np.isclose(candidates[i]['coordinates'][1], candidates[j]['coordinates'][1]) \
                    and np.isclose(candidates[i]['coordinates'][2], candidates[j]['coordinates'][2]):
                edges.append([points[i], points[j], distx])
                continue
            if np.isclose(candidates[i]['coordinates'][0], candidates[j]['coordinates'][0]) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][1] - candidates[j]['coordinates'][1]), disty) \
                    and np.isclose(candidates[i]['coordinates'][2], candidates[j]['coordinates'][2]):
                edges.append([points[i], points[j], disty])
                continue
            if np.isclose(candidates[i]['coordinates'][0], candidates[j]['coordinates'][0]) \
                    and np.isclose(candidates[i]['coordinates'][1], candidates[j]['coordinates'][1]) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][2] - candidates[j]['coordinates'][2]), distz):
                edges.append([points[i], points[j], distz])
                continue
            # not sure if needed
            # chech for diagonals in 2D
            if np.isclose(np.abs(candidates[i]['coordinates'][0] - candidates[j]['coordinates'][0]), distx) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][1] - candidates[j]['coordinates'][1]), disty) \
                    and np.isclose(candidates[i]['coordinates'][2], candidates[j]['coordinates'][2]):
                edges.append([points[i], points[j], np.sqrt(distx ** 2 + disty ** 2)])
                continue
            # break from loop if candidates are only 2D
            if (distz == 0):
                continue
            if np.isclose(candidates[i]['coordinates'][0], candidates[j]['coordinates'][0]) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][1] - candidates[j]['coordinates'][1]), disty) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][2] - candidates[j]['coordinates'][2]), distz):
                edges.append([points[i], points[j], np.sqrt(distz ** 2 + disty ** 2)])
                continue
            if np.isclose(np.abs(candidates[i]['coordinates'][0] - candidates[j]['coordinates'][0]), distx) \
                    and np.isclose(candidates[i]['coordinates'][1], candidates[j]['coordinates'][1]) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][2] - candidates[j]['coordinates'][2]), distz):
                edges.append([points[i], points[j], np.sqrt(distx ** 2 + distz ** 2)])
                continue
            # check for diagonal in 3d
            if np.isclose(np.abs(candidates[i]['coordinates'][0] - candidates[j]['coordinates'][0]), distx) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][1] - candidates[j]['coordinates'][1]), disty) \
                    and np.isclose(np.abs(candidates[i]['coordinates'][2] - candidates[j]['coordinates'][2]), distz):
                edges.append([points[i], points[j], np.sqrt(distx ** 2 + disty ** 2 + distz ** 2)])
                continue

    graph = nx.Graph()
    graph.add_nodes_from(points)
    # this may can be enter in the loop above
    # documentation saied its ok without np.unique
    for iter in edges:
        graph.add_edge(iter[0], iter[1], weight=iter[2])

    print(f'graph built in {time.time() - time_build}')

    return graph, edges, points


# def get_travaling_ways(candidates, strategy):
#     graph_candidates, edges, points = build_location_graph(candidates=candidates)
#
#     time_tsp = time.time()
#     print('start tsp calc on graph')
#     travaling_ways = list(itertools.combinations(strategy['strategy']['scanpoint ids'], 2))
#     paths = []

def get_travaling_ways(candidates, strategy, neighborhood_graph, config):
    graph_candidates = neighborhood_graph[0]
    edges = np.asarray(neighborhood_graph.edges)
    points = np.asarray(neighborhood_graph.nodes)

    #create scene
    mesh = o3d.io.read_triangle_mesh(config['model file'])
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh)

    # graph_candidates = neighborhood_graph.nodes
    # edges = neighborhood_graph.edges
    # points = neighborhood_graph.nodes
    # points = [point for point in points]

    travaling_ways = list(itertools.combinations(strategy['scanpoint ids'], 2))

    if config['debug skip candidate'] != 1:
        tmp_scanpoints = [x * config['debug skip candidate'] for x in strategy['scanpoint ids']]
        travaling_ways = list(itertools.combinations(tmp_scanpoints, 2))

    paths = []
    for iter in travaling_ways:
        #if direct connection possible -> take this route
        #ToDo: Need to be testet
        if directconnection(points[iter[0]],points[iter[1]], scene=scene):
            paths.append([iter[0],iter[1]])
            continue

        paths.append(nx.dijkstra_path(graph_candidates, source=points[iter[0]], target=points[iter[1]]))
    dist = []
    for iter in paths:
        for i in range(len(iter) - 1):
            if (i == 0):
                dist.append(point_dist(iter[i], iter[i + 1]))
            dist[-1] += point_dist(iter[i], iter[i + 1])

    return travaling_ways, dist


def get_route_new_graph(strategy_ptids, neighborhood_graph, scene):

    points=[]
    #not possible to iterate throught nodes with for becasue nodes get typcasted into ints
    for i in range(len(neighborhood_graph.nodes)):
        points.append(neighborhood_graph.nodes[i]['pos'])

    #first iteration of TSP to get the Route
    wal = nx.approximation.traveling_salesman_problem(G=neighborhood_graph, weight='weight',
                                                      nodes=strategy_ptids, cycle=True)
    #add all TSP Points to cut corners if possible
    travaling_ways = list(itertools.combinations(wal, 2))

    for iter in travaling_ways:
        #if direct connection possible -> this route is also possible
        #this is maybe a problem for ditchs
        #ToDo: Need to be testet
        if directconnection(list(points[iter[0]]),list(points[iter[1]]), scene=scene):
            neighborhood_graph.add_edge(iter[0],iter[1],weight=point_dist_np(points[iter[0]],points[iter[1]]))
    
    #get improved TSP Route
    wal = nx.approximation.traveling_salesman_problem(G=neighborhood_graph, weight='weight',
                                                      nodes=strategy_ptids, cycle=True)
    # connections = list(itertools.permutations(strategy['scanpoint ids'], r=2))
    # paths = []
    # dists = []
    # for _ in connections:
    #     path = nx.dijkstra_path(
    #         G=neighborhood_graph,
    #         source=_[0],
    #         target=_[1],
    #         weight='weight'
    #     )
    #     dist = nx.path_weight(
    #         G=neighborhood_graph,
    #         path=path,
    #         weight='weight'
    #     )
    #     paths.append(path)
    #     dists.append(dist)
    #
    # meta_graph = nx.Graph()
    # for _ in strategy_ptids:
    #     meta_graph.add_node(_)
    # for connection, path, dist in zip(connections, paths, dists):
    #     meta_graph.add_edge(connection[0], connection[1], weight=dist)
    #
    # route = nx.approximation.traveling_salesman_problem(meta_graph)

    #### extension explainability

    # pts = []
    # for no, id in enumerate(route):
    #     if no == 0:
    #         pt1 = id
    #     else:
    #         pts.append((pt1, id))
    #         pt1 = id
    # full_path = []
    # for pt in pts:
    #     print(pt)
    #     if pt in connections:
    #         index = connections.index(pt)
    #         full_path.extend(paths[index])
    #     elif reversed(pt) in connections:
    #         index = connections.index(reversed(pt))
    #         full_path.extend(reversed(paths[index]))

    # path_coordinates = []
    # for pt in full_path:
    #     path_coordinates.append(tuple(candidates[pt]['coordinates']))

    return strategy_ptids, wal #, path_coordinates




def get_travaling_route(candidates, strategy, neighborhood_graph, config):
    # maybe concatinate this into an dict or class

    # this is where we insert w/ nu graph logic



    travaling_ways, dist = get_travaling_ways(candidates, strategy, neighborhood_graph, config)

    graph = nx.Graph()
    if config['debug skip candidate'] != 1:
        tmp_scanpoints = [x * config['debug skip candidate'] for x in strategy['scanpoint ids']]
        for iter in tmp_scanpoints:
            graph.add_node(iter)
    else:
        for iter in strategy['scanpoint ids']:
            graph.add_node(iter)

    for i in range(len(travaling_ways)):
        graph.add_edge(travaling_ways[i][0], travaling_ways[i][1], weight=dist[i])

    route = nx.approximation.traveling_salesman_problem(graph)

    if config['debug skip candidate'] != 1:
        route = [int(x / config['debug skip candidate']) for x in route]


    #get better route

    graph_with_tsp=copy.deepcop(neighborhood_graph)
    for i in range(len(route)):
        graph_with_tsp.add_edge(route[i],)

    return route

