import itertools
from typing import Tuple, Any, Union

import networkx as nx
import numpy as np

# TODO: seperate fitness from greedy and EA properly
# maybe put evaluation of fitness values in an seperat function
# -> Bug when optimize for count with greedy  
# Workaround copyed fitness for greedy
def one_fitness(individual, overlap_table, candidates, model, config, visibility_table) -> float:
    """calculation of strategy fitness"""
    fit_count = len(individual)
    fit_overlap = None

    hit_ids = []
    for vp in individual:
        hit_ids.extend(candidates[vp]['qualified hits'])
    hit_ids = np.unique(np.asarray(hit_ids))

    if hit_ids.size == 0:
        fit_coverage = 0
    else:
        fit_coverage = np.sum(np.asarray(model['area'])[hit_ids]) / candidates[0]['max coverage area']

    if fit_count > 1:
        m_tree, overlap_critical = best_neighbors_light(
            individual=individual,
            overlap_table=overlap_table,
            candidates=candidates
        )

        if overlap_critical < config['minimum overlap']:
            fit_overlap = -config['overlap penalty']
        else:
            fit_overlap = 0
            
    
    if fit_coverage is None or fit_overlap is None or fit_count is None:
        raise Exception("No individuals in list -> probably caused by length penalty")

    if config['experiment type'] == ['count']:  # TODO: add experiment type// split
        if fit_coverage <= config['coverage goal']:
            fit_coverage = -config['coverage penalty']
        fit_sum = - fit_coverage - fit_overlap + fit_count  # Maybe fit_coverage set 0 if coverage goal is reached
    elif config['experiment type'] == ['coverage']:
        fit_sum = - fit_coverage - fit_overlap   # no penalty for length in the EA the lenght varation is deactivated
    # not sure if needed anymore 
    # to empiric to find length penalty
    elif config['experiment type'] == ['mixed']:
        fit_sum = - fit_coverage - fit_overlap + config['length penalty'] * fit_count  # TODO: case study dynamic vs. static length penalty

    return fit_sum

def one_fitness_greedy(individual, overlap_table, candidates, model, config, visibility_table) -> float:
    """calculation of strategy fitness"""
    fit_count = len(individual)
    fit_overlap = None

    hit_ids = []
    for vp in individual:
        hit_ids.extend(candidates[vp]['qualified hits'])
    hit_ids = np.unique(np.asarray(hit_ids))

    if hit_ids.size == 0:
        fit_coverage = 0
    else:
        fit_coverage = np.sum(np.asarray(model['area'])[hit_ids]) / candidates[0]['max coverage area']

    if fit_count > 1:
        m_tree, overlap_critical = best_neighbors_light(
            individual=individual,
            overlap_table=overlap_table,
            candidates=candidates
        )

        if overlap_critical < config['minimum overlap']:
            fit_overlap = -config['overlap penalty']
        else:
            fit_overlap = 0

        fit_sum = - fit_coverage - fit_overlap

    return fit_sum



def best_neighbors_light(individual, overlap_table, candidates) -> Tuple[nx.Graph, float]:
    g = nx.Graph()
    for vp in individual:
        g.add_node(vp)

    all_edges = itertools.combinations(individual, 2)
    for edge in all_edges:
        weight = overlap_table[edge[0], edge[1]]  # TODO food for thought: only add non-critical edges to g
        g.add_edge(*edge, weight=weight)

    m = nx.maximum_spanning_tree(g, weight="weight")

    overlaps = []
    for n1, n2, data in m.edges.data():
        overlaps.append(data['weight'])

    # TODO: check here if overlaps is empty what return
    overlap_critical = 100
    if overlaps:
        overlap_critical = min(overlaps)

    return m, overlap_critical


def best_neighbors(route, overlap_table, candidate_graph):  # , candidates):
    # neigh_choice = []
    # for _ in route:
    #     choice_candis = copy.deepcopy(route)
    #     choice_candis.pop(route.index(_))
    #     bestie = choice_candis[np.argmax(overlap_table[_, choice_candis])]
    #     li = [_, bestie]
    #     li.sort()
    #     neigh_choice.append(tuple(li))
    #
    # neigh_choice = list(dict.fromkeys(neigh_choice))
    # neigh_choice_plot = []
    # for _ in neigh_choice:
    #     weigh = overlap_table[_]
    #     start = candidates[_[0]]['coordinates']
    #     end = candidates[_[1]]['coordinates']
    #     neigh_choice_plot.append({
    #         'ids': _,
    #         'start': start,
    #         'end': end,
    #         'relative overlap': weigh
    #     })

    g = nx.Graph()
    for _ in route:
        # here check again candidate_graph is to small? one node missing?
        if _ == 144:
            # dirty fix yet again
            _ = 143
        g.add_node(_)
        pos_attr = {_: tuple(candidate_graph.nodes[_]['pos'])}
        nx.set_node_attributes(G=g, values=pos_attr, name='pos')

    all_edges = itertools.combinations(route, 2)

    for _ in all_edges:
        try:
            g.add_edge(*_, weight=overlap_table[_[0], _[1]])
        except:
            a = 0
    #
    # nx.draw(G)
    # plt.show()

    t = nx.maximum_spanning_tree(g, weight='weight')
    # nx.draw(T)
    # plt.show()

    neigh_choice_plot = []
    overlaps = []
    for _ in t.edges():
        rel_overlap = overlap_table[(_[0], _[1])]
        overlaps.append(rel_overlap)
        neigh_choice_plot.append({
            'ids': _,
            # 'start': candidates[_[0]]['coordinates'],
            # 'end': candidates[_[1]]['coordinates'],
            'relative overlap': rel_overlap
        })
    if route[0] == route[1]:
        critical_overlap = 0
    else:
        critical_overlap = min(overlaps)

    return neigh_choice_plot, critical_overlap, t
