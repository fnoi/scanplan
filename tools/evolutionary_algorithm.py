import time

import numpy as np
import random
import pickle
from tqdm import tqdm
from deap import algorithms, base, creator, tools
import scipy.stats as ss
import multiprocessing as mp
import os
import copy
import argparse
from tools.helpers import get_kth_neigbors
from tools.evaluation import best_neighbors, one_fitness
from tools.TSP import get_route_new_graph


# fitness function
# is this still needed?
def fitness(individual, overlap_rel, qualified_hitlists_source, arealist, max_cov, candidate_graph):
    """(lightweight) calculation of strategy fitness based on genome only"""

    fit_count = len(individual)
    qualified_hitlists = []
    for vp in individual:
        # print(vp)
        try:
            hitlist = qualified_hitlists_source[vp]
        except IndexError:
            print(vp)
        qualified_hitlists.extend(hitlist)

    if len(qualified_hitlists) != 0:
        coverage = np.sum(np.asarray(arealist)[np.unique(np.asarray(qualified_hitlists))])
        fit_coverage = coverage / max_cov

        if fit_count > 1:
            neigh_choice, fit_overlap, some_graph = best_neighbors(
                route=individual,
                overlap_table=overlap_rel,
                candidate_graph=candidate_graph
            )
        # else:

        a = 0

    else:
        fit_coverage = 0
        fit_overlap = 0

    # individual = [1112, 1113, 1114, 1115]
    # best_overlapping_neighbor = []
    # for loc, pt in enumerate(individual):
    #     #neighbors = individual.copy().pop(loc[individual])
    #     neighbors = copy.deepcopy(individual)
    #     neighbors.pop(neighbors.index(pt))
    #     #neighbors_remain = neighbors.pop(neighbors.index(pt))
    #     one_mans_overlaps = np.max(overlap_rel[pt, neighbors])
    #     best_overlapping_neighbor.append(one_mans_overlaps)
    # fit_overlap = min(best_overlapping_neighbor)
    ### fit_overlap = best_neighbors(
    ###    route=individual,
    ###    overlap_table=overlap_rel,
    ###    candidate_graph=candidate_graph
    ###)[1]

    return fit_count, fit_coverage, fit_overlap


def varleng(population, toolbox, pb, range_candidates,overlap_table,config):
    
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring)):
        #check if length gets modified
        if random.random() < pb:
            # symmetric chance to remove or add entries
            # maybe change this probability and put into config
            if random.random() < 0.5:
                toremove=random.randrange(0,len(offspring[i]))
                del offspring[i][toremove]
                del offspring[i].fitness.values
                continue
            else:
                toadd=random.randrange(0,range_candidates)
                #adds only candidates that are not in the individual
                while(True):
                    if toadd in offspring[i]:
                        toadd=random.randrange(0,range_candidates)
                        continue
                    else:
                        # only if random point satisfy the operlap condition
                        for j in range(len(offspring[i])):
                            if overlap_table[offspring[i][j]][toadd] > config['minimum overlap']:
                                toadd=random.randrange(0,range_candidates)
                                continue
                        break
                offspring[i].append(toadd)
                del offspring[i].fitness.values


    return offspring





def one_fit(individual, overlap_rel, qualified_hitlists_source, arealist, max_cov, candidate_graph):
    (count, coverage, overlap) = fitness(
        individual=individual,
        overlap_rel=overlap_rel,
        qualified_hitlists_source=qualified_hitlists_source,
        arealist=arealist,
        max_cov=max_cov,
        candidate_graph=candidate_graph
    )

    thresh_overlap = 0.25

    if overlap >= 0.25:
        overlap_fit = 0
    elif overlap < 0.25:
        overlap_fit = - 100
    coverage_fit = coverage
    fit = - coverage_fit - overlap_fit

    return (fit,)


# def initIndividual(icls, content):
#     return icls(content)

# def initPopulation(pcls, ind_init, guesses):
#     return pcls(ind_init(c) for c in guesses)


def smart_muatat(individual, graph, k, _max):
    toolbox = base.Toolbox()
    mutant = toolbox.clone(individual)
    select = 0

    # create an Gauss distribution on an intigers
    x = np.arange(-k - 1, k)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=1, p=prob)
    # only get possitive ints
    if (nums < 0):
        nums = nums * -1
    # shift it by 1 so the 0th entry can´t get selected
    nums = nums + 1

    leng = random.randint(0, len(mutant) - 1)
    # if mutant[leng]==_max:
    #     mutant[leng]=_max-1
    # here we get the max in in
    possiblemutate = get_kth_neigbors(graph, mutant[leng], nums)
    # check if there is a mating possible
    if possiblemutate:
        select = possiblemutate[random.randint(0, len(possiblemutate) - 1)]
        # fix to work should not have to much influence size both are near
        # if select==_max:
        #     select=_max-1
        mutant[leng] = select

    del mutant.fitness.values
    return mutant,


# TODO: modifiy to change to children if wanted
def smart_mating(ind1, ind2, overlap):
    # clone 2 ind for modification
    toolbox = base.Toolbox()
    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
    toselect = {}
    # get all viewpoints of ind 1 that has an 0.75 overlap with one of the
    # viewpoint in the secend ind
    for i in range(len(child1)):
        tmp = []
        mating_canidates = []
        tmp = overlap[child1[i]][:]
        for j in range(len(tmp)):
            if tmp[j] > 0.75:  # may needs an adjustment to another value
                for m in range(len(ind2)):
                    if ind2[m] == j:
                        mating_canidates.append(j)

        toselect[i] = mating_canidates
    indextochose = []
    if len(toselect) != 0:
        for i in range(len(toselect)):
            if toselect[i]:
                indextochose.append(i)
        if len(indextochose) != 0:
            # select random which point to replace
            tochange_row = random.randint(0, len(indextochose) - 1)
            # select point which should be replaceing the old
            tochange_point = random.randint(0, len(toselect[indextochose[tochange_row]]) - 1)
            child1[i] = toselect[indextochose[tochange_row]][tochange_point]

    del child1.fitness.values
    del child2.fitness.values
    return child1, child2,


def evolutionary_algorithm_search(config, overlap_relative, model, strategy, candidates, candidate_graph):
    # load context for model
    # this needs to be adjusted

    hitlists = [_['qualified hits'] for _ in candidates]
    max_cov = candidates[0]['max coverage area']
    range_candidates = len(candidates) - 1
    arealist = model['area']

    creator.create("FitnessStrat", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessStrat)


    # Parameter maybe this in config?
    pop_size = config["population size"]
    hof_size = 20
    n_gen = config["generations"]
    smartmate = config["smart mate"]
    smartmutat = config["smart mutat"]
    kmax = config["max neighborhood"]
    matingprob = config["mating probability"]
    mutationprob = config["mutating probability"]
    lenpb = config["length probability"]
    
    # TODO: maybe the fitness function can be used for this
    # modify length penalty -> gets an average coverage value for points -> remove low performing points
    coverage_per_point=[]
    for j in range(len(strategy)):
        fit_count = len(strategy[j])
        fit_overlap = None
    
        hit_ids = []
        for vp in strategy[j]:
            hit_ids.extend(candidates[vp]['qualified hits'])
        hit_ids = np.unique(np.asarray(hit_ids))
    
        if hit_ids.size == 0:
            fit_coverage = 0
        else:
            fit_coverage = np.sum(np.asarray(model['area'])[hit_ids]) / candidates[0]['max coverage area']
        
        coverage_per_point.append(fit_coverage/len(strategy[j]))
    
    average_coverage=np.average(coverage_per_point)
    
    config['length penalty']*=average_coverage
    
    # probably this is better to make the tournamentsize dependent on pop size
    tournamentsize = int(0.1 * pop_size)
    
    
    if pop_size < len(strategy):
        raise Exception("population size needs to be bigger then the varient size of greedys")


    indi_sizes = [len(strategy[0])]
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--size', type=int, required=True)
    # parser.add_argument('-l', '--list', nargs='+', type=int, required=False)
    # args = parser.parse_args()
    # indi_sizes = args.list
    # indi_sizes = [5] #args.list

    # indi_sizes = [_ for _ in range(2, 11)]
    # print('[  end - candidates_select ]')

    # do we need this loop?
    for indi_size in tqdm(indi_sizes):
        toolbox = base.Toolbox()
        toolbox.register("attr_ID", random.randint, 0, range_candidates)  # random int, min max for sampling // OK
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_ID,
                         indi_size)  # random.randint(1, 20))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, pop_size)

        # this should creat the first guesses but does not work
        # toolbox.register("individual_guess", initIndividual, creator.Individual)
        # toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, list(greedyguess))
        # population = toolbox.population_guess()

        toolbox.register("evaluate", one_fitness,
                         overlap_table=overlap_relative,
                         candidates=candidates,
                         model=model,
                         config=config,
                         visibility_table=candidate_graph)
        if smartmate:
            toolbox.register("mate", smart_mating, overlap=overlap_relative)
        else:
            toolbox.register("mate", tools.cxOnePoint)

        if smartmutat:
            # maybe change her the random k ?
            toolbox.register("mutate", smart_muatat, graph=candidate_graph, k=kmax, _max=range_candidates)
        else:
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=range_candidates, indpb=0.25)

        toolbox.register("select", tools.selTournament, tournsize=tournamentsize)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population()
        population = pop

        # not clean but works maybe 
        #saving the greedy guesses into the indivudals of the population
        for j in range(len(strategy)):
            for i in range(len(strategy[j])):
                if len(pop[j]) <= i:
                    continue
                pop[j][i] = strategy[j][i]

        hof = tools.HallOfFame(hof_size)

        threads = os.cpu_count()
        with mp.Pool(threads) as pool:
            # this is algorithms.eaSimple with modification
            
            #maybe add this 
            # pool = multiprocessing.Pool()
            # toolbox.register("map", pool.map)
            #for multiprocessing
            
            cxpb=matingprob 
            mutpb=mutationprob
            ngen=n_gen
            halloffame=hof
            verbose=False
            
            
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit, )
        
            if halloffame is not None:
                halloffame.update(population)
        
            record = stats.compile(population) if stats else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
        
            # Begin the generational process
            for gen in tqdm(range(1, ngen + 1),desc="survival of the fittest"):
                # Select the next generation individuals
                offspring = toolbox.select(population, len(population))
        
                # Vary the pool of individuals
                offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
                
                # only modify the length if opmtimize for coverage
                if config['experiment type'] != ['coverage']:
                    offspring = varleng(offspring, toolbox, pb=lenpb,range_candidates=range_candidates,
                                        overlap_table=overlap_relative,config=config)

                
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = (fit, )
        
                # Update the hall of fame with the generated individuals
                if halloffame is not None:
                    halloffame.update(offspring)
        
                # Replace the current population by the offspring
                population[:] = offspring
                
                # maybe adding best of halloffame to population?
                # addding = tools.selBest(hof, k=1)[0]
                
                
                # Append the current generation statistics to the logbook
                record = stats.compile(population) if stats else {}
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                if verbose:
                    print(logbook.stream)



        # probalby this is better -> best=tools.selBest(hof,k=1)[0]
        winner = tools.selBest(hof, k=1)[0]
        winner = winner[:]

        qualified_hitlists = []
        for vp in winner:
            # print(vp)
            try:
                hitlist = hitlists[vp]
            except IndexError:
                print(vp)
            qualified_hitlists.extend(hitlist)

        if len(qualified_hitlists) != 0:
            coverage = np.sum(np.asarray(arealist)[np.unique(np.asarray(qualified_hitlists))])

        return winner, coverage
