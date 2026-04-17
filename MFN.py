import numpy as np
import networkx as nx
import gurobipy as gp
from ellipsoid_lp import ellipsoid_oracle
from Dijkstra_oracle import Dijkstra_oracle

def MFN_feas_test(g, x, y, prob, S, d):
    # print(f"Running multicommodity flow network feasibility test...")
    arcs = [] # A in multicommodity flow network
    facility_nodes = [i for i in range(2 * prob.num_F)] # two nodes per facility
    client_nodes = [j for j in range(2 * prob.num_F, 2 * prob.num_F + 2 * prob.num_D)] # two nodes per client (j^s, j^t)

    violated_const = 0.0

    # arcs are added with 4 different capacity types; we keep track of the indices in the arc capacity list
    arcs1 = []
    arcs2 = []
    arcs3 = []
    arcs4 = []

    idx = 0

    for i in range(prob.num_F): # first set of arcs to add
        arcs.append((i, i + prob.num_F, {'capacity': y[i] * (prob.capacities[i] - g[i, :].sum()), 'name': f"{i},0"}))
        arcs1.append(idx)
        idx += 1

    for i in range(prob.num_F): # second set of arcs to add
        for j in range(prob.num_D):
            # add (j^s, i) with capacity x_ij
            arcs.append((2 * prob.num_F + j, i, {'capacity': x[i][j], 'name': f"{i},{j}"}))
            arcs2.append(idx)
            idx += 1

            # add (i, j^s) with capacity g_ij
            arcs.append((i, 2 * prob.num_F + j, {'capacity': g[i][j], 'name': f"{i},{j}"}))
            arcs3.append(idx)
            idx += 1

            # add (i', j^t) with capacity y[i]d[j]
            arcs.append((prob.num_F + i, 2 * prob.num_F + prob.num_D + j, {'capacity': y[i] * d[j], 'name': f"{i},{j}"}))
            arcs4.append(idx)
            idx += 1

    # Construct the MFN graph
    MFN_graph = nx.DiGraph()
    MFN_graph.add_edges_from(arcs)

    m = MFN_graph.number_of_edges()

    obj = np.array([MFN_graph[u][v]['capacity'] for u, v in MFN_graph.edges()] + [-1.0 * d_j for d_j in d])

    result_dict = ellipsoid_oracle(
        n = m + prob.num_D,
        c = obj,
        oracle = Dijkstra_oracle,
        max_iter = 1_000,
        tol = 1e-3,
        # x0 = x_0,
        extras = (MFN_graph, prob.num_D, prob.num_F, d, S),
        printer=True,
        stop_early=True # if we get any feasible point with a negative objective, we can save time by returning the violated constraint
    )

    arr = result_dict["x"]
    ell = arr[:m]
    z = arr[m:]

    val = result_dict["obj"]
    if val < 0:
        # We use Lemma 4 to recover a violated constraint on x, y
        a = np.zeros(prob.num_F) # infeasibility on y
        b = np.zeros((prob.num_F, prob.num_D)) # infeasibility on x

        for j in range(prob.num_D):
            violated_const += z[j]
            for i in range(prob.num_F):
                violated_const -= z[j] * g[i][j]

        for idx in arcs1:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(","))) # j is vestigial
            a[i] -= ell[idx] * (prob.capacities[i] - g[i, :].sum())  

        for idx in arcs2:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(",")))
            b[i][j] = -1 * ell[idx]

        for idx in arcs3:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(",")))
            violated_const -= ell[idx] * g[i][j]

        for idx in arcs4:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(",")))
            a[i] -= ell[idx] * d[j]

        violated_constr = np.concatenate((a, b.flatten()))

        # print("Infeasible :(")
        return False, (violated_constr, -1 * violated_const)
    else:
        # print("Feasible!")
        return True, (np.array([0]), 0)