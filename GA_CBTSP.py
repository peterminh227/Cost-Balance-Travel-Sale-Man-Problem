# -*- coding: utf-8 -*-
"""
@author: Peter Minh
@problem: Cost Balanced Travelling Saleman Problem
"""

import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
# useful class
class City:
    def __init__(self, index):
        self.index = index # wanna write sth but dont know what to write
    def __repr__(self):
        return str(self.index)
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self, table_cost):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += table_cost[fromCity.index, toCity.index]
            self.distance = pathDistance
            self.distance= abs(self.distance)
        return self.distance
    def routeFitness(self, table_cost):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance(table_cost))
        return self.fitness 
# supportive function;    
def createRoute(cityList, pop_GLS, row_ind):
    #route = random.sample(cityList, len(cityList))
    route = []
    for i in range(0, len(cityList)):
        route.append(City(index = int(pop_GLS[row_ind,i])))
    return route
def initialPopulation(popSize, cityList, pop_GLS):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList, pop_GLS, i))
    return population
def rankRoutes(population, table_cost):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness(table_cost)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)    
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
   
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    #            
   
    return selectionResults
def matingPool(population, selectionResults):
   
    matingpool = []
    
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool
def breed(parent1, parent2, table_cost, bigM, count_neighbors_GLS):
    child = [100]*len(parent1)
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    child[startGene:endGene] = parent1[startGene:endGene]
    childP1 = parent1[startGene:endGene]
    childP2 = [item for item in parent2 if item not in childP1]
    childP2 = random.sample(childP2, len(childP2))
    if (len(childP2) + len(childP1)) > len(parent1):
        temp_breed = [item for item in parent1 if item not in childP1]
        childP2 = random.sample(temp_breed, len(temp_breed))
    
    child = childP1 + childP2
    # ensure child will be a feasible move ... --> perform a local search here?
    nodeCount = len(parent1)
    solution = (nodeCount + 1) * np.zeros(shape = nodeCount, dtype = np.int)
    for i in range(0, len(parent1)):
        solution[i] = int(child[i].index)
    solution = np.append(solution, solution[0]) 
    obj_inf = cal_inf(solution, table_cost, bigM)
    #print('count inf = ' + str(obj_inf))
    if (obj_inf > 0):
        table_cost_inf = copy.deepcopy(table_cost)
        table_cost_inf[np.where(table_cost_inf != bigM)] = np.sum(\
                       count_neighbors_GLS[np.asarray\
                               (np.where(table_cost_inf !=bigM))], axis = 0)        
        #
        solution, obj_inf, pop_GLS = GLS_infeasible(table_cost_inf, solution, obj_inf, bigM, alpha = 1, \
                                           local_search_optimal = obj_inf, iterations = 1000, popSize = 1)
        # pop_GLS to child
        child = []
        for i in range(0, len(parent1)):
            child.append(City(index = int(pop_GLS[0,i])))
        
    #
    return child
def breedPopulation(matingpool, eliteSize, table_cost, bigM, count_neighbors_GLS):
    children = []
    length = len(matingpool) - eliteSize
  
    pool = random.sample(matingpool, len(matingpool))
   
    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1], table_cost, bigM, count_neighbors_GLS)
        children.append(child)
    return children
def mutate(individual, mutationRate, table_cost):
    #print(1/Fitness(individual).routeFitness())
    #print(individual)
    individual_swap = copy.deepcopy(individual)
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            city1 = individual_swap[swapped]
            city2 = individual_swap[swapWith]
            individual_swap[swapped] = city2
            individual_swap[swapWith] = city1
            if (1/Fitness(individual).routeFitness(table_cost) > \
                1/Fitness(individual_swap).routeFitness(table_cost)):
                individual = copy.deepcopy(individual_swap)
            #individual =  copy.deepcopy(individual_swap)

    return individual
def mutatePopulation(population, mutationRate, table_cost):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate, table_cost)
        mutatedPop.append(mutatedInd)
    return mutatedPop
def nextGeneration(currentGen, eliteSize, mutationRate, table_cost,bigM,count_neighbors_GLS):
    popRanked = rankRoutes(currentGen, table_cost)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, table_cost, bigM, count_neighbors_GLS)
    nextGeneration = mutatePopulation(children, mutationRate, table_cost)
    #nextGeneration = breedPopulation(matingpool, eliteSize)
    return nextGeneration

def geneticAlgorithmPlot(population, popSize, eliteSize, \
                         mutationRate, generations, table_cost, pop_GLS, bigM, count_neighbors_GLS):
    pop = initialPopulation(popSize, population, pop_GLS)
    print("Initial distance: " + str(1 / rankRoutes(pop, table_cost)[0][1]))
    progress = []
    progress.append(1 / rankRoutes(pop, table_cost)[0][1])
    best_obj = float('inf')
    #best_route = []
    for i in range(0, generations):
        print(i)
        pop = nextGeneration(pop, eliteSize, mutationRate, table_cost, bigM, count_neighbors_GLS)
        temp_obj = 1/ rankRoutes(pop, table_cost)[0][1]
        temp_route = rankRoutes(pop,table_cost)[0][0]
        if temp_obj < best_obj:
            best_obj = copy.deepcopy(temp_obj)
            best_route = copy.deepcopy(pop[temp_route])
        progress.append(1 / rankRoutes(pop, table_cost)[0][1])
    print("Final distance: " + str(1 / rankRoutes(pop, table_cost)[0][1]))
    print("Best object:" + str(best_obj))
    print("Best route:" + str(best_route))
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    return best_obj, best_route
# local search area
def GLS_infeasible(table_cost, cycle, obj, bigM, alpha, local_search_optimal,\
                        iterations, popSize):
    pop_GLS = np.zeros(shape = (popSize,len(cycle)))
    pop_count = 0
    count = 0
    limit = alpha * (local_search_optimal / len(cycle))
    penalty = np.zeros((table_cost.shape)) 
    # increase penalty for the infeasible moves
    penalty[np.where(table_cost == bigM)] = 0.5
    bits = np.ones(table_cost.shape[0])
    solution = copy.deepcopy(cycle)
    best_solution = []
    best_obj = float('inf')
    while (count  < iterations):
        #alpha = alpha * count
        solution, obj, bits = FLS_infeasible(table_cost, solution, obj, \
                                                penalty, limit, bits,bigM)
        utilities = utility_inf(table_cost, solution, penalty, limit)
        penalty, bits = update_penalty_inf(penalty, solution, utilities, bits) 
        #print("obj = " + str(obj))
        if (obj == 0):
            #print(solution)
            pop_GLS[pop_count][:] = solution.astype(int)
            pop_count += 1
            print("pop_count = " + str(pop_count))
            # reset
            limit = alpha * (local_search_optimal / len(cycle))
            penalty = np.zeros((table_cost.shape)) 
            # increase penalty for the infeasible moves
            penalty[np.where(table_cost == bigM)] = 0.5
            bits = np.ones(table_cost.shape[0])
            #
            # simply doing swap 
            #i, j  = random.sample(range(0, len(solution)-1), 2)
            #if (i > j):
            #    i, j = j, i
            #solution[i:j+1] = list(reversed(solution[i:j+1]))    
            #solution[-1] = solution[0]
            temp = solution[:-1]
            np.random.shuffle(temp)
            temp = np.append(temp,temp[0])
            solution = copy.deepcopy(temp)
            best_solution = []
            best_obj = float('inf')
            #cal obj
            obj = cal_inf(solution, table_cost, bigM)
            #
            #print(solution)
            #stoppp
            # 
            if (pop_count == popSize):
                break
        else:
            if (obj < best_obj):
                best_solution = copy.deepcopy(solution)
                best_obj = copy.deepcopy(obj)
                # update limit
                limit = alpha * best_obj/(len(cycle) - 1)
    return best_solution, best_obj, pop_GLS
## FLS_infeasible
def FLS_infeasible(table_cost, solution, obj, penalty, limit, bits,bigM):
    # for the first time
    ag_cost = obj 
    n = len(bits)
    array = np.arange(n)
    indexes = copy.deepcopy(array)
    np.random.shuffle(array)
    #while (1 in bits): # --> original FLS implemented this line
    for i in array: # trigger to due with the large search of the final problem
        #iter_inner = 0
        #iter_outer += 1
        
        if bits[i] == 1: 
            # turn off the bit
            bits[i] = 0
            # i presents the city has bit = 1
            t1 = (np.where(solution[:-1] == i))[0].item()
            t_next = t1 + 1
            t_prev = t1 - 1
            mask = np.ones(n, dtype = bool)
            if (t1 < n-1):
                mask[np.array([t_prev,t1, t_next])] = False
            else:
                mask[np.array([t_prev,t1, 0])] = False
            feasible_swap_points = indexes[mask]
            # randomizing    
            np.random.shuffle(feasible_swap_points)
            for _, t3 in np.ndenumerate(feasible_swap_points):
                if (t3 < t_next):
                    exchange_1 = t3 + 1
                    exchange_2 = t1
                    
                else:
                    exchange_1 = t_next
                    exchange_2 = t3
                if (exchange_1 ==0):
                        #
                        old_edge = int(table_cost[solution[exchange_2], solution[exchange_2+1]] == bigM) +\
                                int(table_cost[solution[n-1], solution[1]] == bigM)
                        new_edge = int(table_cost[solution[n-1],solution[exchange_2]] == bigM) +\
                        int(table_cost[solution[exchange_1],solution[exchange_2 + 1]] == bigM)
                        
                        old_penalty = limit * (penalty[solution[n-1],solution[exchange_1]] +\
                                               penalty[solution[exchange_2],solution[exchange_2+1]])
                        
                        new_penalty = limit * (penalty[solution[n-1],solution[exchange_2]] +\
                                               penalty[solution[exchange_1],solution[exchange_2+1]])
                else:
                        
                            
                        old_edge = int(table_cost[solution[exchange_1-1],solution[exchange_1]]== bigM) +\
                            int(table_cost[solution[exchange_2], solution[exchange_2+1]] == bigM)
                        new_edge = int(table_cost[solution[exchange_1-1],solution[exchange_2]]==bigM) +\
                        int(table_cost[solution[exchange_1],solution[exchange_2 + 1]] == bigM)
                        old_penalty = limit * (penalty[solution[exchange_1-1],solution[exchange_1]] +\
                                               penalty[solution[exchange_2],solution[exchange_2+1]])
                        new_penalty = limit * (penalty[solution[exchange_1-1],solution[exchange_2]] +\
                                               penalty[solution[exchange_1],solution[exchange_2+1]])

                delta_obj = new_edge - old_edge
                delta_augumented =  delta_obj + (new_penalty - old_penalty)            

                if (delta_augumented < 0):
                    solution [exchange_1:exchange_2+1] = \
                    list(reversed(solution[exchange_1:exchange_2+1]))
                    solution[-1] = solution[0]
                    ag_cost += delta_augumented
                    bits[solution[t1]] = 1
                    bits[solution[t_next]] = 1
                    bits[solution[t3]] = 1
                    bits[solution[t3+1]] = 1
                    obj += delta_obj
                    break # go out of for-loop
       
    return solution, obj, bits
# important functions utilities and penalty   
def utility_inf(table_cost, cycle, penalty, limit):
    n = len(cycle) - 1
    utilities = np.zeros(n)
    for i in range(n):
        utilities[i] = (table_cost[cycle[i], cycle[i+1]]) /(1 + \
                 (penalty[cycle[i],cycle[i+1]]))  
    return utilities    
def update_penalty_inf(penalty, cycle, utilities, bits):
    indexes = np.where(utilities == max(utilities))
    indexes = np.asarray(indexes)
    penalty[cycle[indexes], cycle[indexes +1]] +=1
    penalty[cycle[indexes+1], cycle[indexes]] +=1
    bits[cycle[indexes]] = 1
    bits[cycle[indexes+1]] = 1
    return penalty, bits
def cal_inf(solution, table_cost, bigM):
    temp = table_cost[solution[:-1], solution[1:]]
    check_compare = np.asarray(np.where(temp == bigM))
    size_compare = check_compare.size
    return size_compare # infeasible size
def collect_moves(solution, table_cost, bigM):
    row_cost = table_cost[solution[-1],:]
    feasible_moves = np.asarray(np.where(row_cost != bigM)) # store city th
    # mask the moves
    mask_moves = np.isin(feasible_moves, solution[:-1])
    feasible_moves = feasible_moves[np.logical_not(mask_moves)]
    return feasible_moves


    
#    
if __name__ == '__main__':
    #if len(sys.argv) > 1:
    #    file_location = sys.argv[1].strip()
    #    with open(file_location, 'r') as input_data_file:
    #        input_data = input_data_file.read()
    #    print(solve_it(input_data))
    #else:
    #    print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
    # modify below code    
    cityList = []
    file_location = sys.argv[1].strip()
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    
    lines = input_data.split('\n')
    #nodeCount = int(lines[0])
    nodeCount = int(float(lines[0].split()[0])) 
    roadCount = int(float(lines[0].split()[1]))
    table_cost = np.zeros((nodeCount, nodeCount))
    # inf * 0 = nan --> be careful
    table_cost[np.isnan(table_cost)] = 0
    for i in range(1, roadCount+1):
        line = lines[i]
        parts = line.split()
        #cityList.append(City(index = i))
        table_cost[int(parts[0]), int(parts[1])] = parts[2]
        table_cost[int(parts[1]), int(parts[0])] = parts[2]
    for i in range(0, nodeCount):
        cityList.append(City(index = i))
    bigM = 1000 * nodeCount* np.max(np.abs(table_cost))
    table_cost[np.where(table_cost ==0)] = bigM        
    print("Starting - Local Search ...")
    # LOCAL SEARCH -  create feasible solution
    solution = (nodeCount + 1) * np.zeros(shape = nodeCount, dtype = np.int)
    

     # neighbor sizes
    neighbors = (nodeCount + 1) * np.ones((nodeCount, nodeCount), dtype = np.int)
    for i in range(nodeCount):
        row_cost = table_cost[i, :]
        feasible_moves = np.asarray(np.where(row_cost != bigM))
        neighbors[i, feasible_moves] = feasible_moves
    #
    count_neighbors = np.count_nonzero(neighbors < nodeCount + 1, axis = 1)   
    count_neighbors_GLS = copy.deepcopy(count_neighbors)
    #neighbors_GLS = copy.deepcopy(neighbors)
    sort_indices = np.argsort(count_neighbors) 
    # Gready Algorithm to create feasible solution ~ cost = min(infeasible solutions)
    solution[0] = sort_indices[0] # city with the smallest neighbors
    # neighbors of the first city
    temp = neighbors[solution[0],:]
    neighbors_firstCity =  temp[(temp < nodeCount + 1)]
    sort_neighbors_firstCity = np.argsort(count_neighbors[neighbors_firstCity])
    #print(count_neighbors[neighbors_firstCity])
    solution[1] = neighbors_firstCity[sort_neighbors_firstCity[0]]
    solution[-1] = neighbors_firstCity[sort_neighbors_firstCity[-1]]
    
    # update neighbors table
    neighbors[:,solution[0]] = nodeCount + 1 # reduce 
    neighbors[:,solution[1]] = nodeCount + 1 # reduce
    neighbors[solution[0], :] = -1
    neighbors[solution[-1],:] = -1
    
    for i in range(2, nodeCount - 1):
        
        count_neighbors = np.count_nonzero(neighbors < nodeCount + 1, axis = 1) 
        #print(count_neighbors)
        temp = neighbors[solution[i-1],:]
        neighbors_icity = temp[temp< nodeCount +1]
        temp_1 = count_neighbors[neighbors_icity]
        sort_neighbors_iCity = np.argsort(temp_1)
        flag_inf = 0
        if (neighbors_icity.size != 0):
            be_taken_city = neighbors_icity[sort_neighbors_iCity[0]]
            if (be_taken_city == solution[-1]):
                # 
                total_point = np.arange(nodeCount)
                
                visited = solution[0:i]
                visited = np.append(visited,solution[-1])
                mask_visit = np.asarray(np.isin(total_point, visited))
                solution[i:nodeCount - 1] = total_point[mask_visit == False]
                print('appear infeasible solution')
                flag_inf = 1
                break
            else:
                solution[i] = neighbors_icity[sort_neighbors_iCity[0]]
        else:
            flag_inf = 1
            total_point = np.arange(nodeCount)
            visited = solution[0:i]
            visited = np.append(visited,solution[-1])
            mask_visit = np.asarray(np.isin(total_point, visited))
            solution[i:nodeCount - 1] = total_point[mask_visit == False]
            print('[Info] Infeasible solution found')
            break
        neighbors[:,solution[i]] = nodeCount + 1
        neighbors[solution[0:i],:] = -1
        neighbors[solution[-1],:] = -1
    solution = np.append(solution, solution[0])    
    size_compare = cal_inf(solution, table_cost, bigM)
    # guided local search - FLS for infeasible value - hope it will work
    obj_inf = size_compare
    table_cost_inf = copy.deepcopy(table_cost)
    table_cost_inf[np.where(table_cost_inf != bigM)] = np.sum(\
                   count_neighbors_GLS[np.asarray\
                           (np.where(table_cost_inf !=bigM))], axis = 0)
         
    #if (flag_inf ==1):
    solution, obj_inf, pop_GLS = GLS_infeasible(table_cost_inf, solution, obj_inf, bigM, alpha = 1, \
                                       local_search_optimal = obj_inf, iterations = 1000, popSize = 10)
    print('Finishing - Local Search --> create feasible solution')  
    
    #print(solution)
    #print(cal_inf(solution, table_cost, bigM))
    geneticAlgorithmPlot(population=cityList, \
                         popSize= 10, eliteSize= 5, mutationRate = 0.01,\
                         generations= 50, table_cost = table_cost, \
                         pop_GLS = pop_GLS, bigM = bigM, count_neighbors_GLS = count_neighbors_GLS )