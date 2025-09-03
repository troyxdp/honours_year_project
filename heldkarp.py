import itertools
import time

import numpy as np
import matplotlib.pyplot as plt

# TIME COMPLEXITY: O(2^{n}n^{2})
# g(S, k) is the cost of starting at 0, going through all the vertices in S, and ending at one of the vertices in S, k (which counts as going through k)

# dists = np.array([
#     [0, 1, 2, 3],
#     [1, 0, 1, 2],
#     [2, 1, 0, 3],
#     [3, 2, 3, 0]
# ])
# dists = np.array([
#     [0, 2, 9, 10, 7, 3],
#     [2, 0, 6, 4, 3, 8],
#     [9, 6, 0, 5, 4, 7],
#     [10, 4, 5, 0, 6, 2],
#     [7, 3, 4, 6, 0, 5],
#     [3, 8, 7, 2, 5, 0]
# ])

def get_subsets(set, length):
    set_list = list(set)
    return itertools.combinations(set_list, length)

# Made with assistance from https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
# Made with assistance from https://stackoverflow.com/questions/69902373/can-you-help-explain-this-held-karp-tsp-pseudocode
def held_karp(dists):
    cities = [_ for _ in range(len(dists))]
    g = {} # keeps track of the G values for each "subpath" 
    parent = {} # keeps track of second-last city visited in each "subpath" 
    e = cities[0] # number of initial city

    # Initialize g with 1 step subpaths
    for k in range(1, len(dists)):
        g[((k,), k)] = dists[0][k]
        parent[((k,), k)] = 0

    # Add subpaths of increasing length s - this is the "dynamic programming loop"
    for s in range(2, len(dists)):
        # Get the subsets of cities excluding the starting city of length s
        subsets = get_subsets(cities[1:], s)
        for S in subsets:
            # Go through each end destination k
            for k in S:
                # Find the path of minimum cost (or rather set of vertices in the path) that ends at k
                S_minus_k = [i for i in S if i != k]
                S_minus_k = tuple(S_minus_k)
                min_cost = np.inf
                arg_min_m = -1
                for m in S:
                    if m == k:
                        continue
                    g_val = g[(S_minus_k, m)]
                    cost_val = dists[m][k]
                    cost = g_val + cost_val
                    if cost < min_cost:
                        min_cost = cost
                        arg_min_m = m
                g[(tuple(S), k)] = min_cost
                parent[(tuple(S), k)] = arg_min_m

    # Find optimal cost
    optimal_cost = np.inf
    arg_min_k = -1
    for k in cities[1:]:
        cost = g[(tuple(cities[1:]), k)] + dists[k][e]
        if cost < optimal_cost:
            optimal_cost = cost
            arg_min_k = k

    # Find optimal path
    # Made with assistance from https://www.youtube.com/watch?v=-JjA4BLQyqE
    path = []
    subset = tuple(cities[1:])
    k = arg_min_k
    while len(subset) != 0:
        path.append(k)
        m = parent[(subset, k)]
        subset = tuple([city for city in subset if city != k])
        k = m
    path.append(0)
    path = list(reversed(path))

    return path, optimal_cost

    
if __name__ == '__main__':
    min_size = int(input("Input the min size to test the algorithm with: "))
    max_size = int(input("Input the max size to test the algorithm with (going above 18 will take really long): "))
    times = []
    for size in range(min_size, max_size+1):
        print()
        dists = np.random.randint(1, 10, (size, size))
        for i in range(len(dists)): # iterate over rows
            dists[i][i] = 0
            for j in range(i + 1): # iterate over column of values below current row
                dists[j][i] = dists[i][j]
        print(dists)

        start_time = time.time()
        path, optimal_cost = held_karp(dists)
        end_time = time.time() - start_time
        times.append(end_time)

        print("Size of graph:", size)
        print("Optimal path:", path)
        print("Optimal path length:", optimal_cost)
        print("Time elapsed:", end_time, " seconds")
        
    sizes = list(range(min_size, max_size+1))
    plt.plot(sizes, times)
    plt.xlabel("Number of vertices in graph")
    plt.ylabel("Time elapsed")
    plt.title("Speed of Held-Karp algorithm")
    plt.show()