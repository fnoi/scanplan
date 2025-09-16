import matplotlib.pyplot as plt
import time
from tools.helpers import uhrenvergleich
import numpy as np
import networkx as nx
from labellines import labelLines
from tools.evaluation import best_neighbors


def plot_per_step(config, candidates, greedy_step, greedy_choice, greedy_score):
    """function to create plots in 2D/3D for candidate value per greedy step, show and save optional"""
    current = uhrenvergleich(time.perf_counter(), 'plot per step init', config['time log option'])
    # if 2D, 3D, store (same level options!)
    if config['show heatmaps'] or config['save heatmaps'] or config['3d']:
        coords = np.asarray([candidate['coordinates'] for candidate in candidates])
        coords_choice = coords[greedy_choice]
        # candidates_local = copy.deepcopy(candidates)
        # v = np.asarray([candidate['area ratio local'] for candidate in candidates_local])
        v = greedy_score

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle(f"{config['greedy type']} search, iteration {greedy_step}")
        scat = ax.scatter(coords[:, 0],
                          coords[:, 1],
                          s=10, c=v, alpha=0.75)

        ax.scatter(coords[:, 0],
                   coords[:, 1],
                   s=10, c=v, alpha=0.75)

        ax.set_aspect('equal')
        if (config['greedy weighted']):
            plt.colorbar(scat, label="weighted viewpoint value = covered area per candidate", orientation="vertical")
        else:
            plt.colorbar(scat, label="viewpoint value = covered area per candidate", orientation="vertical")

        ax.plot(coords_choice[0],
                coords_choice[1],
                '+', alpha=1)
        ax.text(coords_choice[0],
                coords_choice[1],
                str(greedy_step))

        a = 0
        if config['save heatmaps']:
            fig.savefig(config['path'] + f"HM_2D_{greedy_step}.png")
        if config['show heatmaps']:
            plt.show()
        else:
            plt.close()

        if config['3d']:
            fig = plt.figure()
            fig.suptitle(f"{config['greedy type']} search, iteration {greedy_step}")
            ax = fig.add_subplot(projection='3d')
            ax.set_zlim(np.min(coords[:, 2]) - 1, np.max(coords[:, 2]) + 1)
            ax.scatter(coords[:, 0],
                       coords[:, 1],
                       coords[:, 2],
                       s=10, c=v, alpha=0.5, zorder=1)
            ax.scatter(coords_choice[0],
                       coords_choice[1],
                       coords_choice[2],
                       s=15, marker='+', alpha=1, zorder=2)

            ax.text(coords_choice[0],
                    coords_choice[1],
                    coords_choice[2],
                    str(greedy_step), zorder=2)

            if config['save heatmaps']:
                fig.savefig(config['path'] + f"HM_3D_{greedy_step}.png")
            if config['show heatmaps']:
                plt.show()
            else:
                plt.close()

            # # this needs to be adjusted
            # # update area and hits
            # # remove faces that are alrady hit
            # already_hit = candidates_local[scanpoints[i]]['qualified hits']
            # for j in range(len(candidates_local)):
            #     candidates_local[j]['qualified hits'] = list(
            #         set(candidates_local[j]['qualified hits']) - set(already_hit))
            #     # update area values in dicts
            #     # this is copyed from raycast full_evalb because an import is not possible circular dependency
            # hit_ids_global = []
            # for candi in candidates_local:
            #     hit_ids_global.extend(candi['qualified hits'])
            # hit_ids_global = np.unique(hit_ids_global)
            # hit_area_global = np.sum(np.array(model['area'])[hit_ids_global])
            # # update coverd area
            # # i dont know how to the fancy syntaxing like in scanplan
            # # TODO: follow up, what is going on here?
            # for m in range(len(candidates_local)):
            #     candidates_local[m]['area covered'] = np.sum(
            #         np.array(model['area'])[candidates_local[m]['qualified hits']])
            # for candi in candidates_local:
            #     candi['max coverage hits'] = len(hit_ids_global)
            #     candi['max coverage area'] = hit_area_global
            #     candi['face ratio local'] = len(candi['qualified hits']) / candi['max coverage hits']
            #     candi['area ratio local'] = candi['area covered'] / candi['max coverage area']

    current = uhrenvergleich(time.perf_counter(), 'plot per step done', config['time log option'])


def plot_TSP(config, candidates, scanpoints, destination, route, graph, points):
    """function to create plot in 2D/3D for chosen strategy, no candidate values, show and save optional"""
    coords = np.asarray([candidate['coordinates'] for candidate in candidates])
    coords_choice = np.asarray([coords[choice] for choice in route])

    fig = plt.figure()
    fig.suptitle(f"{config['greedy type']} search, full strategy")
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=10, alpha=0.5)
    ax.scatter(coords_choice[:, 0],
               coords_choice[:, 1],
               s=15, marker='x', color='k', alpha=1)
    ax.set_aspect('equal')
    # for i, coord in enumerate(coords_choice):
    #     plt.text(coord[0], coord[1], i + 1)

    for i in range(len(route) - 1):
        plt.text(coords_choice[i][0], coords_choice[i][1], i + 1, ha='left', va='top')
        plt.text(coords_choice[i][0], coords_choice[i][1], route[i], ha='right', va='bottom')
        connection = nx.dijkstra_path(graph, source=points[route[i] * config['debug skip candidate']],
                                      target=points[route[i + 1] * config['debug skip candidate']])
        for j in range(len(connection) - 1):
            x_values = [connection[j].x, connection[j + 1].x]
            y_values = [connection[j].y, connection[j + 1].y]
            plt.plot(x_values, y_values, color='g', linestyle="--")
    plt.show()
    fig.savefig(destination + "TSP.png")
    plt.close()


def plot_registration_connectivity(config, candidates, destination, route, overlap_table):
    """function to create plot in 2D/3D for chosen strategy, no candidate values, show and save optional"""
    coords = np.asarray([candidate['coordinates'] for candidate in candidates])
    coords_choice = np.asarray([coords[choice] for choice in route])
    basis = best_neighbors(route=route[:-1], overlap_table=overlap_table, candidates=candidates)[0]

    fig = plt.figure()
    fig.suptitle(f"{config['greedy type']} search, full strategy")
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=10, alpha=0.5)
    ax.scatter(coords_choice[:, 0],
               coords_choice[:, 1],
               s=15, marker='x', color='k', alpha=1)
    ax.set_aspect('equal')
    for i, coord in enumerate(coords_choice):
        plt.text(coord[0], coord[1], i + 1)
#TSP line
    # for i in range(len(route) - 1):
    #     plt.text(coords_choice[i][0], coords_choice[i][1], i + 1)
    #     connection = nx.dijkstra_path(graph, source=points[route[i] * config['debug skip candidate']],
    #                                   target=points[route[i + 1] * config['debug skip candidate']])
    #     for j in range(len(connection) - 1):
    #         x_values = [connection[j].x, connection[j + 1].x]
    #         y_values = [connection[j].y, connection[j + 1].y]
    #         plt.plot(x_values, y_values, color='g', linestyle="--")
#minimum weight graph

    for line in basis:
        x = [candidates[line['ids'][0]]['coordinates'][0], candidates[line['ids'][1]]['coordinates'][0]]
        y = [candidates[line['ids'][0]]['coordinates'][1], candidates[line['ids'][1]]['coordinates'][1]]
        # x1 = [line['start'][0], line['end'][0]]
        # y1 = [line['start'][1], line['end'][1]]
        label = round(line['relative overlap'], 2)
        plt.plot(x, y, label=label, color='b')
    labelLines(plt.gca().get_lines(), zorder=2.5)

    plt.show()
    fig.savefig(destination + "connectedness.png")
    plt.close()


def plot_once(config, candidates, scanpoints, destination):
    """function to create plot in 2D/3D for chosen strategy, no candidate values, show and save optional"""
    coords = np.asarray([candidate['coordinates'] for candidate in candidates])
    coords_choice = np.asarray([coords[choice] for choice in scanpoints])

    fig = plt.figure()
    fig.suptitle(f"{config['greedy type']} search, full strategy")
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=10, alpha=0.5)
    ax.scatter(coords_choice[:, 0],
               coords_choice[:, 1],
               s=15, marker='x', color='k', alpha=1)
    ax.set_aspect('equal')
    for i, coord in enumerate(coords_choice):
        plt.text(coord[0], coord[1], i + 1)
    plt.show()
    fig.savefig(destination + "strategy_2D.png")
    plt.close()

    if (config['3d']):
        fig = plt.figure()
        fig.suptitle(f"{config['greedy type']} search, full strategy")
        ax = fig.add_subplot(projection='3d')
        ax.set_zlim(np.min(coords[:, 2]) - 1, np.max(coords[:, 2]) + 1)
        ax.scatter(coords[:, 0],
                   coords[:, 1],
                   coords[:, 2],
                   s=10, alpha=0.5)
        ax.scatter(coords_choice[:, 0],
                   coords_choice[:, 1],
                   coords_choice[:, 2],
                   s=15, marker='x', color='k', alpha=1)
        for i, coord in enumerate(coords_choice):
            ax.text(coord[0], coord[1], coord[2], str(i + 1))
        plt.show()
        fig.savefig(destination + "strategy_3D.png")
        plt.close()


# INDEPENDENT PLOTTING

def plot_strategy_TSP(config, candidates, scanpoints, destination, route, graph, points):#\
        #candidates, route, :
    """function to create plot in 2D/3D for chosen strategy, no candidate values, show and save optional"""
    coords = np.asarray([candidate['coordinates'] for candidate in candidates])
    coords_choice = np.asarray([coords[choice] for choice in route])

    fig = plt.figure()
    #fig.suptitle(f"{config['greedy type']} search, full strategy")
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=10, alpha=0.5)
    ax.scatter(coords_choice[:, 0],
               coords_choice[:, 1],
               s=15, marker='x', color='k', alpha=1)
    ax.set_aspect('equal')
    # for i, coord in enumerate(coords_choice):
    #     plt.text(coord[0], coord[1], i + 1)

    for i in range(len(route) - 1):
        plt.text(coords_choice[i][0], coords_choice[i][1], i + 1, ha='left', va='top')
        plt.text(coords_choice[i][0], coords_choice[i][1], route[i], ha='right', va='bottom')
        connection = nx.dijkstra_path(graph, source=points[route[i] * config['debug skip candidate']],
                                      target=points[route[i + 1] * config['debug skip candidate']])
        for j in range(len(connection) - 1):
            x_values = [connection[j].x, connection[j + 1].x]
            y_values = [connection[j].y, connection[j + 1].y]
            plt.plot(x_values, y_values, color='g', linestyle="--")
    plt.show()
    fig.savefig(destination + "TSP.png")
    plt.close()

def plot_strategy_connectivity(config, candidates, destination, route, overlap_table):
    """function to create plot in 2D/3D for chosen strategy, no candidate values, show and save optional"""
    coords = np.asarray([candidate['coordinates'] for candidate in candidates])
    coords_choice = np.asarray([coords[choice] for choice in route])
    basis = best_neighbors(route=route[:-1], overlap_table=overlap_table, candidates=candidates)[0]

    fig = plt.figure()
    fig.suptitle(f"{config['greedy type']} search, full strategy")
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=10, alpha=0.5)
    ax.scatter(coords_choice[:, 0],
               coords_choice[:, 1],
               s=15, marker='x', color='k', alpha=1)
    ax.set_aspect('equal')
    for i, coord in enumerate(coords_choice):
        plt.text(coord[0], coord[1], i + 1)
#TSP line
    # for i in range(len(route) - 1):
    #     plt.text(coords_choice[i][0], coords_choice[i][1], i + 1)
    #     connection = nx.dijkstra_path(graph, source=points[route[i] * config['debug skip candidate']],
    #                                   target=points[route[i + 1] * config['debug skip candidate']])
    #     for j in range(len(connection) - 1):
    #         x_values = [connection[j].x, connection[j + 1].x]
    #         y_values = [connection[j].y, connection[j + 1].y]
    #         plt.plot(x_values, y_values, color='g', linestyle="--")
#minimum weight graph

    for line in basis:
        x1 = [line['start'][0], line['end'][0]]
        y1 = [line['start'][1], line['end'][1]]
        label = round(line['relative overlap'], 2)
        plt.plot(x1, y1, label=label, color='b')
    labelLines(plt.gca().get_lines(), zorder=2.5)

    plt.show()
    fig.savefig(destination + "connectedness.png")
    plt.close()


def plot_much(config, candidates, scanpoints, destination, route, graph, points):

    coords = np.asarray([np.asarray(node['coord']) for _ in graph])
    coords_choice = np.asarray([coords[choice] for choice in route])

    fig = plt.figure()
    fig.suptitle(f"{config['greedy type']} search, full strategy")
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=10, alpha=0.5)
    ax.scatter(coords_choice[:, 0],
               coords_choice[:, 1],
               s=15, marker='x', color='k', alpha=1)
    ax.set_aspect('equal')
    # for i, coord in enumerate(coords_choice):
    #     plt.text(coord[0], coord[1], i + 1)

    for i in range(len(route) - 1):
        plt.text(coords_choice[i][0], coords_choice[i][1], i + 1, ha='left', va='top')
        plt.text(coords_choice[i][0], coords_choice[i][1], route[i], ha='right', va='bottom')
        connection = nx.dijkstra_path(graph, source=points[route[i] * config['debug skip candidate']],
                                      target=points[route[i + 1] * config['debug skip candidate']])
        for j in range(len(connection) - 1):
            x_values = [connection[j].x, connection[j + 1].x]
            y_values = [connection[j].y, connection[j + 1].y]
            plt.plot(x_values, y_values, color='g', linestyle="--")
    plt.show()
    fig.savefig(destination + "TSP.png")
    plt.close()

def plot_much_2D():
    coords = np.asarray([np.asarray(neighborhood_graph.nodes[_]['coord']) for _ in neighborhood_graph])
    coords_choice = np.asarray([np.asarray(neighborhood_graph.nodes[_]['coord']) for _ in route])

    fig = plt.figure()
    fig.suptitle(f"the new and improved plot from scratch 2D")
    ax = fig.add_subplot(111)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=5, alpha=0.5
    )
    ax.scatter(
        coords_choice[:, 0],
        coords_choice[:, 1],
        s=15, marker='+', color='k', alpha=1
    )
    for i, coord in enumerate(coords_choice):
        plt.text(coord[0], coord[1], i + 1)
    ax.set_aspect('equal')

    plt.show()

def plot_much_3D():
    coords = np.asarray([np.asarray(neighborhood_graph.nodes[_]['coord']) for _ in neighborhood_graph])
    coords_choice = np.asarray([np.asarray(neighborhood_graph.nodes[_]['coord']) for _ in route])

    fig = plt.figure()
    fig.suptitle(f"the new and improved plot from scratch 3D")
    ax = fig.add_subplot(projection='3d')
    master_min = np.min(coords)
    master_max = np.max(coords)
    ax.set_zlim(master_min, master_max)
    ax.set_ylim(master_min, master_max)
    ax.set_xlim(master_min, master_max)
    # ax.set_zlim(np.min(coords[:, 2]) - 1, np.max(coords[:, 2]) + 1)
    # ax.set_aspect('equal')
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=5, alpha=0.5, zorder=1
    )
    ax.scatter(
        coords_choice[:, 0],
        coords_choice[:, 1],
        coords_choice[:, 2],
        s=15, marker='+', color='k', alpha=1, zorder=2
    )
    for i, coord in enumerate(coords_choice):
        ax.text(coord[0], coord[1], coord[2], str(i + 1))

    plt.show()