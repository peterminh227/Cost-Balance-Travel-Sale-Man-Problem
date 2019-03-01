#!/usr/bin/python
# -*- coding: utf-8 -*-

#import math
import sys
from collections import namedtuple
#from itertools import combinations
import copy
#import random
import time
#import timeit
import numpy as np
from os import listdir
from os.path import isfile, join
Road = namedtuple("Road", ['x', 'y','c']) 


def solve_it(input_data, set_tim):
    start_time = time.time()
    # parse the input
    lines = input_data.split('\n')
    nodeCount = int(float(lines[0].split()[0])) 
    roadCount = int(float(lines[0].split()[1]))
    roads = []
    table_cost = np.zeros((nodeCount, nodeCount))
    # inf * 0 = nan --> be careful
    table_cost[np.isnan(table_cost)] = 0
    for i in range(1, roadCount+1):
        line = lines[i]
        parts = line.split()
        roads.append(Road(int(parts[0]), int(parts[1]), float(parts[2]))) 
        table_cost[int(parts[0]), int(parts[1])] = parts[2]
        table_cost[int(parts[1]), int(parts[0])] = parts[2]
    bigM = 1000 * nodeCount* np.max(np.abs(table_cost))    

    table_cost[np.where(table_cost ==0)] = bigM
    
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
         
    if (flag_inf ==1):
        solution, obj_inf = GLS_infeasible(table_cost_inf, solution, obj_inf, bigM, alpha = 1, \
                                       local_search_optimal = obj_inf, iterations = 1000)

    time_used = time.time() - start_time - 10 # offset 10s for all of solving
    #if (size_compare ==0):
    print('[Info] Start with the feasible solution')
    
    obj, sign = calc_cost_travel(solution, table_cost)
    solution, obj = guided_local_search(table_cost, solution, obj, sign, bigM, set_time, time_used, alpha = 250, \
                            local_search_optimal = obj, iterations = 20000)
    
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution[:-1]))

    return output_data
def GLS_infeasible(table_cost, cycle, obj, bigM, alpha, local_search_optimal,\
                        iterations):
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
                    
        if (obj < best_obj):
            best_solution = copy.deepcopy(solution)
            best_obj = copy.deepcopy(obj)
            # update limit
            limit = alpha * best_obj/(len(cycle) - 1)
            if (best_obj ==0):
                break
    return best_solution, best_obj
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
def calc_cost_travel(solution, table_cost):

    cost_real = np.sum(table_cost[solution[:-1],solution[1:]])
    sign = np.sign(cost_real)
    cost_travel = np.abs(cost_real)
    return  cost_travel, sign
# Skeleton of Guided - Local Search (GLS) algorithm  
def guided_local_search(table_cost, cycle, obj, sign, bigM, set_time,time_used, alpha, local_search_optimal,\
                        iterations):
    
    limit = alpha * (local_search_optimal / len(cycle))
    penalty = np.zeros((table_cost.shape)) 
    # increase penalty for the infeasible moves
    penalty[np.where(table_cost == bigM)] = bigM
    bits = np.ones(table_cost.shape[0])
    solution = copy.deepcopy(cycle)
    best_solution = []
    best_obj = float('inf')
    start = time.time()
    elapsed = 0
    time_left = set_time * 60 - time_used # times already used ~ T.T
    while (elapsed  < time_left):
        solution, obj, bits, sign = fast_local_search(table_cost, solution, obj, sign, \
                                                penalty, limit, bits,bigM)
        utilities = utility(table_cost, solution, penalty, limit)
        penalty, bits = update_penalty(penalty, solution, utilities, bits) 
        if (obj < best_obj):
            best_solution = copy.deepcopy(solution)
            best_obj = copy.deepcopy(obj)
            # update limit
            limit = alpha * best_obj/(len(cycle) - 1)
        elapsed = time.time() - start
    return best_solution, best_obj
# function local search called augumented cost    
# 2
def augumented_cost(table_cost, cycle, penalty, limit):
    
    mask = np.zeros((table_cost.shape[0], table_cost.shape[0]))
    mask[[cycle[:-1]],[cycle[1:]]] = 1
    temp =  np.multiply(mask, table_cost)
    temp[np.isnan(temp)] = 0
    augumented = np.abs(np.sum(temp)) + np.sum(np.multiply(mask, penalty))
    return augumented     
   
# Fast local search (FLS), sub-neighbor = cities in the existing cycle  
def fast_local_search(table_cost, solution, obj, sign, penalty, limit, bits,bigM):
    # for the first time
    ag_cost = augumented_cost(table_cost, solution, penalty , limit)
    n = len(bits)
    array = np.arange(n)
    indexes = copy.deepcopy(array)
    np.random.shuffle(array)
    iter_outer= 0
    while (1 in bits): # --> original FLS implemented this line
        for i in array: # trigger to due with the large search of the final problem
            iter_outer += 1
            
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
                    #candidate = copy.deepcopy(solution)
                    if (t3 < t_next):
                        exchange_1 = t3 + 1
                        exchange_2 = t1
                        
                    else:
                        exchange_1 = t_next
                        exchange_2 = t3
                    # delta
                    # check the valid swap first
                    if (exchange_1 ==0):
                        if ((table_cost[solution[n-1],solution[exchange_2]] == bigM) | \
                        (table_cost[solution[exchange_1],solution[exchange_2 + 1]] == bigM)):
                            continue
                        else:
                            old_edge = table_cost[solution[exchange_2], solution[exchange_2+1]] +\
                                    table_cost[solution[exchange_1], solution[n-1]]
                            old_penalty = limit * (penalty[solution[n-1],solution[exchange_1]] +\
                                                   penalty[solution[exchange_2],solution[exchange_2+1]])
                            new_edge = table_cost[solution[n-1],solution[exchange_2]] +\
                            table_cost[solution[exchange_1],solution[exchange_2 + 1]]
                            new_penalty = limit * (penalty[solution[n-1],solution[exchange_2]] +\
                                                   penalty[solution[exchange_1],solution[exchange_2+1]])
                    else:
                        if ((table_cost[solution[exchange_1-1],solution[exchange_2]]== bigM) |\
        (table_cost[solution[exchange_1],solution[exchange_2 + 1]] == bigM)):
                            continue
                        else:
                            old_edge = table_cost[solution[exchange_1-1],solution[exchange_1]] +\
                                table_cost[solution[exchange_2], solution[exchange_2+1]]
                            old_penalty = limit * (penalty[solution[exchange_1-1],solution[exchange_1]] +\
                                                   penalty[solution[exchange_2],solution[exchange_2+1]])
                            new_edge = table_cost[solution[exchange_1-1],solution[exchange_2]] +\
                            table_cost[solution[exchange_1],solution[exchange_2 + 1]] 
                            new_penalty = limit * (penalty[solution[exchange_1-1],solution[exchange_2]] +\
                                                   penalty[solution[exchange_1],solution[exchange_2+1]])
                    
                    if (sign == -1):
                        new_obj = - obj - old_edge + new_edge
                    else:
                        new_obj = obj - old_edge + new_edge
                        
                    update_sign = np.sign(new_obj)        
                    delta_obj = (abs(new_obj) - obj)
                    delta_augumented =  delta_obj + (new_penalty - old_penalty)            
                    if (delta_augumented < 0):
                        solution [exchange_1:exchange_2+1] = \
                        list(reversed(solution[exchange_1:exchange_2+1]))
                        solution[-1] = solution[0]
                        #solution = copy.deepcopy(new_route)
                        ag_cost += delta_augumented
                        sign =copy.deepcopy(update_sign) 
                        # turn on bit
                        bits[solution[t1]] = 1
                        bits[solution[t_next]] = 1
                        bits[solution[t3]] = 1
                        bits[solution[t3+1]] = 1
                        obj += delta_obj
                        break # go out of for-loop
    return solution, obj, bits, sign    
# important functions utilities and penalty 
def calc_delta(table_cost, solution, exchange_1, exchange_2, limit, penalty, \
               old_obj, sign):
    n = len(solution) -1
    
    if (exchange_1 == 0 ):
        # do something
        old_edge = table_cost[solution[n-1], solution[exchange_1]] +\
        table_cost[solution[exchange_2], solution[exchange_2+1]] 
        old_penalty = limit * (penalty[solution[n-1],solution[exchange_1]] +\
                 penalty[solution[exchange_2],solution[exchange_2+1]])
        new_edge = table_cost[solution[n-1],solution[exchange_2]] +\
        table_cost[solution[exchange_1],solution[exchange_2 + 1]]
        new_penalty = limit * (penalty[solution[n-1],solution[exchange_2]] +\
                 penalty[solution[exchange_1],solution[exchange_2+1]])
        
    else:
        old_edge = table_cost[solution[exchange_1-1],solution[exchange_1]] +\
        table_cost[solution[exchange_2], solution[exchange_2+1]]
        old_penalty = limit * (penalty[solution[exchange_1-1],solution[exchange_1]] +\
                 penalty[solution[exchange_2],solution[exchange_2+1]])
        new_edge = table_cost[solution[exchange_1-1],solution[exchange_2]] +\
        table_cost[solution[exchange_1],solution[exchange_2 + 1]] 
        new_penalty = limit * (penalty[solution[exchange_1-1],solution[exchange_2]] +\
                 penalty[solution[exchange_1],solution[exchange_2+1]])
    
    if (sign == -1):
        new_obj = - old_obj - old_edge + new_edge
    else:
        new_obj = old_obj - old_edge + new_edge
    update_sign = np.sign(new_obj)    
    delta_obj = (abs(new_obj) - old_obj)
    delta_aug =  delta_obj + (new_penalty - old_penalty)
    return delta_aug, delta_obj, update_sign
def utility(table_cost, cycle, penalty, limit):
    n = len(cycle) - 1
    utilities = np.zeros(n)
    for i in range(n):
        utilities[i] = (table_cost[cycle[i], cycle[i+1]]) /(1 + \
                 (penalty[cycle[i],cycle[i+1]]))  
    return utilities
#Function: Update Penalty
def update_penalty(penalty, cycle, utilities, bits):
    # speed-up the code here
    indexes = np.where((utilities == max(utilities)) |\
                       (utilities == - max(-utilities)))    
    indexes = np.asarray(indexes)
    penalty[cycle[indexes], cycle[indexes +1]] +=1
    penalty[cycle[indexes+1], cycle[indexes]] +=1
    bits[cycle[indexes]] = 1
    bits[cycle[indexes+1]] = 1
    return penalty, bits
    
if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) > 1:
        
        path_dir = sys.argv[1].strip()
        files = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
        files =files[:-1]   # remove 'desktop.ini'
        # 27 files 
        # 0 ... 200: 35 minutes per instance [0 - 13]
        # 250 ... 3000: [14 - 27] 15 minutes per instance
        set_time = 14.5
        for i in range(27):
            file_path = path_dir + files[i]
            for k in range(10):
                    print('Start solving problem:' + files[i] + ' [' + str(k) + ']')
                    with open(file_path, 'r') as input_data_file:
                        input_data = input_data_file.read()
                    f = open(path_dir + 'result/' + files[i], "a+")
                    f.write(solve_it(input_data,set_time) + '\n')
                    f.close()
        #print('Start solving problem:' + files[0] )    
        #print(solve_it(input_data))
        #print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

