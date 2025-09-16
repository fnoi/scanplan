import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt 
import random
import pickle
from tqdm import tqdm
from deap import algorithms, base, creator, tools
import multiprocessing as mp
import os

from tools.helpers import get_kth_neigbors
from tools.evaluation import best_neighbors
from tools.TSP import get_route_new_graph


# fitness function
def fitness(individual, overlap_rel, qualified_hitlists_source, arealist, max_cov, candidate_graph):
    """(lightweight) calculation of strategy fitness based on genome only"""

    fit_count = len(individual)
    qualified_hitlists = []
    for vp in individual:
        #print(vp)
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
    overlap_fit = None
    if overlap >= thresh_overlap:
        overlap_fit = 0
    elif overlap < thresh_overlap:
        overlap_fit = - 100
    coverage_fit = coverage
    fit = - coverage_fit - overlap_fit

    return (fit, )


# def initIndividual(icls, content):
#     return icls(content)

# def initPopulation(pcls, ind_init, guesses):
#     return pcls(ind_init(c) for c in guesses)


def smart_muatat(individual, graph,k):
    mutant=toolbox.clone(individual)
    select=0
    
    #create an Gauss distribution on an intigers
    x = np.arange(-k-1, k)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size = 1, p = prob)
    #only get possitive ints
    if(nums<0):
        nums=nums*-1
    #shift it by 1 so the 0th entry can´t get selected
    nums = nums+1

    leng=random.randint(0, len(mutant)-1)
    #here we get the 388 in
    possiblemutate=get_kth_neigbors(graph, mutant[leng], nums)
    select=possiblemutate[random.randint(0, len(possiblemutate)-1)]
    #fix to work should not have to much influence size both are near
    if select==388:
        select=387
    mutant[leng]=select

    del mutant.fitness.values
    return mutant,

#TODO: modifiy to change to children if wanted
def smart_mating(ind1,ind2, overlap):
    #clone 2 ind for modification
    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
    toselect={}
    #get all viewpoints of ind 1 that has an 0.75 overlap with one of the 
    #viewpoint in the secend ind
    for i in range(len(child1)):
        tmp=[]
        mating_canidates=[]
        tmp=overlap[child1[i]][:]
        for j in range(len(tmp)):
            if tmp[j]>0.75: #may needs an adjustment to another value
                for m in range(len(ind2)):
                    if ind2[m]==j:
                        mating_canidates.append(j)
                        
        toselect[i]=mating_canidates
    indextochose=[]
    if len(toselect)!=0:
        for i in range(len(toselect)):
            if toselect[i]:
                indextochose.append(i)
        if len(indextochose)!=0:
            #select random which point to replace
            tochange_row=random.randint(0, len(indextochose)-1)
            #select point which should be replaceing the old
            tochange_point=random.randint(0, len(toselect[indextochose[tochange_row]])-1)
            child1[i]=toselect[indextochose[tochange_row]][tochange_point]

    
    del child1.fitness.values
    del child2.fitness.values
    return child1,child2,



if __name__ == "__main__":

    # load context for model
    with open('overlap_relative.pickle', 'rb') as jar:
        overlap_relative = pickle.load(jar)
    with open('candidate_eval.pickle', 'rb') as jar:
        (vp_candidates, model) = pickle.load(jar)
    with open('00_voxel_candidate_graph.pkl', 'rb') as jar:
        candidate_graph = pickle.load(jar)
    hitlists = [_['qualified hits'] for _ in vp_candidates]
    max_cov = vp_candidates[0]['max coverage area']
    range_candidates = len(vp_candidates)-1
    arealist = model['area']
    
    
    greedyguess=[262, 22, 237, 58, 108, 208, 1, 113, 313, 251, 184, 18, 88]

    creator.create("FitnessStrat", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessStrat)

    #Parameter 
    pop_size = 100
    hof_size = 20
    n_gen = 100
    smartmate=True
    smartmutat=True
    kmax=20
    matingprob=0.6
    mutationprob=0.7
    #probably this is better to make the tournamentsize dependent on pop size
    tournamentsize=int(0.1*pop_size)

    
    indi_sizes = [9]
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--size', type=int, required=True)
    # parser.add_argument('-l', '--list', nargs='+', type=int, required=False)
    # args = parser.parse_args()
    # indi_sizes = args.list
    # indi_sizes = [5] #args.list

    #indi_sizes = [_ for _ in range(2, 11)]

    for indi_size in tqdm(indi_sizes):
        toolbox = base.Toolbox()
        toolbox.register("attr_ID", random.randint, 0, range_candidates) # random int, min max for sampling // OK
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_ID, indi_size)  #random.randint(1, 20))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, pop_size)
        
        
        #this should creat the first guesses but does not work
        # toolbox.register("individual_guess", initIndividual, creator.Individual)
        # toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, list(greedyguess))
        # population = toolbox.population_guess()
        
        
        toolbox.register("evaluate", one_fit,
                         overlap_rel=overlap_relative,
                         qualified_hitlists_source=hitlists,
                         arealist=arealist,
                         max_cov=max_cov,
                         candidate_graph=candidate_graph)
        if smartmate:
            toolbox.register("mate", smart_mating,overlap=overlap_relative)
        else:
            toolbox.register("mate", tools.cxOnePoint)
        
        if smartmutat:
            #maybe change her the random k ? 
            toolbox.register("mutate", smart_muatat, graph=candidate_graph, k=kmax)
        else:
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=range_candidates, indpb=0.25)
        
        toolbox.register("select", tools.selTournament, tournsize=tournamentsize)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population()
        
        #not clean but works maybe
        for i in range(indi_size):
            pop[0][i]=greedyguess[i]
        
        
        hof = tools.HallOfFame(hof_size)

        threads = os.cpu_count()
        with mp.Pool(threads) as pool:
            pop, logbook = algorithms.eaSimple(
                pop, toolbox,
                cxpb=matingprob, mutpb=mutationprob, ngen=n_gen,
                stats=stats, halloffame=hof, verbose=True
            )
            
        #probalby this is better -> best=tools.selBest(hof,k=1)[0]
        winner = tools.selBest(pop, k=1)[0]
        winner = winner[:]
        print(f' :: {winner}')
##################################################
        route, full_path = get_route_new_graph(
            strategy_ptids=winner,
            neighborhood_graph=candidate_graph
        )
        
        
        #plot
        if smartmate:
            strmate="SmartMating"
        else:
            strmate="RandomMating"
        
        if smartmutat:
            strmut="SmartMutation"
        else:
            strmut="RandomMutation"
        
            
        plt.title("Plot Pop:"+str(pop_size)+" Gen:"+str(n_gen)+" Neigborhood: "+ str(kmax)+ "\nMate: "
                  +strmate+ " Mateprob: "+str(matingprob) +"\nMutation: "+strmut +" Mutationprob: "
                  +str(mutationprob)+ " Selection: "+ "Tournament " +str(tournamentsize),fontsize=10) 
        plt.xlabel("Iterations") 
        plt.ylabel("Fitness") 
        x=np.arange(0,n_gen+1)
        ymin=[d['min'] for d in logbook]
        plt.plot(x,ymin,"k") 
        plt.show() 
        plt.savefig("image_dump\\Plot Pop"+str(pop_size)+" Gen"+str(n_gen)+" Neigborhood "+ str(kmax)+ "Mate "
                  +strmate+ " Mateprob "+str(matingprob) +"Mutation "+strmut +" Mutationprob "
                  +str(mutationprob)+ " Selection "+ "Tournament Size " +str(tournamentsize)+".png")
        

        #Plot end


        with open('00_route_path.pkl', 'wb') as jar:
            pickle.dump([route, full_path], jar)

        # calculate best neighbor things
        ncfp_0, co, C = best_neighbors(
            route=winner,
            overlap_table=overlap_relative,
            candidate_graph=candidate_graph
        )
        with open('00_another_pkl.pkl', 'wb') as jar:
            pickle.dump([ncfp_0, co, C], jar)


        # with open(f'jar/fixed_count/logbook_EA_{indi_size}.pkl', 'wb') as jar:
        #     pickle.dump(logbook, jar)
        # with open(f'jar/fixed_count/hof_EA_{indi_size}.pkl', 'wb') as jar:
        #     pickle.dump([list(_) for _ in hof.items], jar)
        # with open(f'jar/fixed_count/set_EA_{indi_size}.pkl', 'wb') as jar:
        #     pickle.dump(winner, jar)

        a = 0
