import time

import numpy as np
import random
import pickle
from tqdm import tqdm
from deap import algorithms, base, creator, tools
import multiprocessing as mp
import os
import copy
import argparse


# fitness function
def fitness(individual, overlap_rel, qualified_hitlists_source, arealist, max_cov):
    """(lightweight) calculation of strategy fitness based on genome only"""

    fit_count = len(individual)
    qualified_hitlists = []
    for vp in individual:
        #print(vp)
        hitlist = qualified_hitlists_source[vp]
        qualified_hitlists.append(hitlist)

    coverage = np.sum(np.asarray(arealist)[np.unique(np.concatenate(qualified_hitlists))])

    fit_coverage = coverage / max_cov

    # individual = [1112, 1113, 1114, 1115]
    best_overlapping_neighbor = []
    for loc, pt in enumerate(individual):
        #neighbors = individual.copy().pop(loc[individual])
        neighbors = copy.deepcopy(individual)
        neighbors.pop(neighbors.index(pt))
        #neighbors_remain = neighbors.pop(neighbors.index(pt))
        one_mans_overlaps = np.max(overlap_rel[pt, neighbors])
        best_overlapping_neighbor.append(one_mans_overlaps)
    fit_overlap = min(best_overlapping_neighbor)

    return fit_count, fit_coverage, fit_overlap


def one_fit(individual, overlap_rel, qualified_hitlists_source, arealist, max_cov):
    (count, coverage, overlap) = fitness(
        individual=individual,
        overlap_rel=overlap_rel,
        qualified_hitlists_source=qualified_hitlists_source,
        arealist=arealist,
        max_cov=max_cov
    )

    if overlap >= 0.4:
        overlap_fit = 0
    elif overlap < 0.4:
        overlap_fit = - 100
    coverage_fit = coverage
    fit = - coverage_fit - overlap_fit

    return (fit, )



if __name__ == "__main__":

    # load context for model
    with open('overlap_relative.pickle', 'rb') as char:
        overlap_relative = pickle.load(char)
    with open('candidate_eval.pickle', 'rb') as char:
        (vp_candidates, model) = pickle.load(char)
    hitlists = [_['qualified hits'] for _ in vp_candidates]
    max_cov = vp_candidates[0]['max coverage area']
    range_candidates = len(vp_candidates)-1
    arealist = model['area']

    creator.create("FitnessStrat", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessStrat)

    pop_size = 100
    hof_size = 20
    n_gen = 100

    # indi_sizes =
    parser = argparse.ArgumentParser()
    # parser.add_argument('--size', type=int, required=True)
    parser.add_argument('-l', '--list', nargs='+', type=int, required=True)
    args = parser.parse_args()
    indi_sizes = args.list

    #indi_sizes = [_ for _ in range(2, 11)]

    for indi_size in tqdm(indi_sizes):
        toolbox = base.Toolbox()
        toolbox.register("attr_ID", random.randint, 0, range_candidates) # random int, min max for sampling // OK
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_ID, indi_size)  #random.randint(1, 20))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, pop_size)
        toolbox.register("evaluate", one_fit,
                         overlap_rel=overlap_relative,
                         qualified_hitlists_source=hitlists,
                         arealist=arealist,
                         max_cov=max_cov)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=range_candidates, indpb=0.25)
        toolbox.register("select", tools.selTournament, tournsize=3)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population()
        hof = tools.HallOfFame(hof_size)

        threads = os.cpu_count()
        with mp.Pool(threads) as pool:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.6, ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
        winner = tools.selBest(pop, k=1)[0]
        winner = winner[:]
        print(f' :: {winner}')

        with open(f'jar/fixed_count/logbook_EA_{indi_size}.pkl', 'wb') as jar:
            pickle.dump(logbook, jar)
        with open(f'jar/fixed_count/hof_EA_{indi_size}.pkl', 'wb') as jar:
            pickle.dump(hof, jar)
        with open(f'jar/fixed_count/set_EA_{indi_size}.pkl', 'wb') as jar:
            pickle.dump(winner, jar)
