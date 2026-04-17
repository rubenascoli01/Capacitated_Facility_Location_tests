import numpy as np
import networkx as nx
"""
Finds a separating hyperplane for the subproblem in MFN
"""
def Dijkstra_oracle(var, extras, debug=False):
    # print("Called Dijkstra-based oracle...")
    # var has form (ell, z)

    graph = extras[0]
    num_D = extras[1]
    num_F = extras[2]
    demand = extras[3]
    S = extras[4]

    edge_list = list(graph.edges())

    n = len(var)
    m = n - num_D

    # check 1: box constraints

    # print("--Checking box constraints...")

    for i in range(n):
        if var[i] > 1:
            return False, (np.eye(n)[i], 1.0)
        elif var[i] < 0:
            return False, (-1 * np.eye(n)[i], 0.0)
    
    # check 2: shortest j^s - j^t path must be less than z[j]

    # print("--Computing shortest paths...")

    idx = 0
    for u, v in graph.edges():
        graph[u][v]['weight'] = var[idx]
        idx += 1

    for j in range(num_D):
        # print(f"--Client {j}")
        length, path = nx.single_source_dijkstra(
            G=graph,
            source= 2 * num_F + j,
            target= 2 * num_F + num_D + j,
            cutoff= 2, # if no path exists, returns length 2 (constraint feasible)
            weight='weight'
        )

        path_edges = list(zip(path, path[1:]))

        if length < var[m + j]:
            constraint_indices = [edge_list.index(e) for e in path_edges]
            arr = np.zeros(n)
            arr[constraint_indices] = -1.0
            arr[m + j] = 1.0

            return False, (arr, 0.0)

    return True, (var, )