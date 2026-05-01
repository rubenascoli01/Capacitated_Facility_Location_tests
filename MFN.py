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

    obj = np.array([-1 * MFN_graph[u][v]['capacity'] for u, v in MFN_graph.edges()] + [d_j for d_j in d])

    run=0
    mod = gp.Model() 
    ell_z = mod.addMVar(shape=m + prob.num_D, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="ell_z")
    mod.setObjective(obj @ ell_z, gp.GRB.MAXIMIZE)
    mod.setParam('OutputFlag', 0)
    while True: # use column gen now instead of ellipsoid
        # print(f"Internal iteration {run}")
        mod.optimize()

        opt_ell_z = ell_z.X
        objective = mod.ObjVal

        feas, (arr, h) = Dijkstra_oracle(
            var = opt_ell_z,
            extras = (MFN_graph, prob.num_D, prob.num_F, d, S)
        )
        
        run += 1

        if feas == True:
            del mod
            break
        else:
            mod.addConstr(arr @ ell_z <= h)
            continue

    # input()

    # result_dict = ellipsoid_oracle(
    #     n = m + prob.num_D,
    #     c = obj,
    #     oracle = Dijkstra_oracle,
    #     max_iter = 20_000,
    #     tol = 1e-3,
    #     # x0 = x_0,
    #     extras = (MFN_graph, prob.num_D, prob.num_F, d, S),
    #     printer=False,
    #     stop_early=True # if we get any feasible point with a negative objective, we can save time by returning the violated constraint
    # )

    # print(f"Finished on {run} iterations")

    arr = opt_ell_z
    ell = arr[:m]
    z = arr[m:]

    val = objective
    if val > 0:
        print_thing = 0.0
        print_thing2 = 0.0
        # We use Lemma 4 to recover a violated constraint on x, y
        a = np.zeros(prob.num_F) # infeasibility on y
        b = np.zeros((prob.num_F, prob.num_D)) # infeasibility on x

        for j in range(prob.num_D):
            violated_const += z[j] * (1 - g[:, j].sum())
            # for i in range(prob.num_F):
            #     violated_const -= z[j] * g[i][j]

        for idx in arcs1:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(","))) # j is vestigial
            new_idx = list(MFN_graph.edges()).index((i, i + prob.num_F))
            a[i] -= ell[new_idx] * (prob.capacities[i] - g[i, :].sum())

        for idx in arcs2:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(",")))
            new_idx = list(MFN_graph.edges()).index((2 * prob.num_F + j, i))
            b[i][j] -= ell[new_idx]

        for idx in arcs3:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(",")))
            new_idx = list(MFN_graph.edges()).index((i, 2 * prob.num_F + j))
            violated_const -= ell[new_idx] * g[i][j]

        for idx in arcs4:
            (i, j) = tuple(map(int, arcs[idx][2]["name"].split(",")))
            new_idx = list(MFN_graph.edges()).index((prob.num_F + i, 2 * prob.num_F + prob.num_D + j))
            a[i] -= ell[new_idx] * d[j]

        # print(f"val={val}")
        # print(f"objective={a @ y + (b * x).sum() + violated_const}")
        # input()

        violated_constr = np.concatenate((a, b.flatten()))

        # print("Infeasible :(")
        return False, (violated_constr, -1 * violated_const)
    else:
        # print("Feasible!")
        return True, (np.array([0]), 0)