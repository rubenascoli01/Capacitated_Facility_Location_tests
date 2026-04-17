import numpy as np
import gurobipy as gp
import networkx as nx
from FacilityProb import FacilityLocationProb
from MFN import MFN_feas_test
"""
Boxed Algorithm substep from page 286

if MFN(g,x,y) infeasible: returns violated constraint
else: returns x_hat, y_hat
"""

def CFL_oracle(var, prob, debug=False): # var is y + flattened x in one
    # print("Called boxed algorithm...")

    num_F = prob.num_F
    num_D = prob.num_D
    U = prob.capacities

    y = var[:num_F]
    x = var[num_F:].reshape((num_F, num_D))

    y_prime = np.zeros(num_F)

    # Step 1
    for i in range(num_F):
        if y[i] >= 0.25: y_prime[i] = 1
        else: y_prime[i] = y[i]

    I = np.where(y_prime == 1)[0]
    S = np.where(y_prime != 1)[0]

    if debug: 
        print(f"I, S = {I}, {S}")
        input()

    # Step 2 - 3; combine into one by solving the b-matching using Gurobi
    m = gp.Model()
    m.Params.LogToConsole = 0
    m.Params.OutputFlag = 0

    z = m.addMVar((num_F, num_D), vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")

    for i in range(num_F):
        m.addConstr(z[i, :].sum() <= U[i], name=f"B_facilities{i}")

    for j in range(num_D):
        m.addConstr(z[:, j].sum() <= 1.0, name=f"B_clients{j}")

    obj_vec = 2 * x
    obj_vec[S, :] = 0

    # print("Solving b-matching subproblem...")
    m.setObjective((obj_vec * z).sum(), gp.GRB.MAXIMIZE)
    m.optimize()

    """
    -------NOTICE-------

    Since Gurobi uses simplex in the case of continuous variables, 
    and the polyhedron for bipartite b-matching is integral, we may
    assume that z is an integral solution :) This implies g is integral,
    which we need for the MFN subproblem.
    """

    z_vals = np.array(z.X)
    if debug: 
        print(f"LP solution: {z.X}")
        input()

    # Step 4
    rows1, cols1 = np.where(z_vals < 2 * x)
    rows2, cols2 = np.where(z_vals > 0.0)

    for i in range(len(rows1)): rows1[i] = rows1[i] + num_D
    for i in range(len(rows2)): rows2[i] = rows2[i] + num_D

    coords1 = list(zip(rows1, cols1))
    coords2 = list(zip(rows2, cols2))

    if debug:
        print(f"rows1 = {rows1}, cols1 = {cols1}")
        print(f"rows2 = {rows2}, cols2 = {cols2}")

    edge_list = coords1 + coords2

    if debug: 
        # print(f"edges = {edge_list}") 
        input()

    H = nx.Graph()
    H.add_nodes_from([j for j in range(num_D)] + [num_D + i for i in range(num_F)])
    H.add_edges_from(edge_list)

    if debug: 
        # print(f"H = {H.nodes()}")
        input()

    # Step 5
    col_sums = np.sum(z_vals, axis=0)
    unsaturated = np.where(abs(col_sums - 1.0) > 1e-3)[0]

    if debug:
        # print(f"column sums = {col_sums}")
        print(f"unsaturated clients = {unsaturated}")
        input()

    I_H = []
    D_H = []

    for j in range(num_D):
        for k in unsaturated:
            if nx.has_path(H, j, k):
                D_H.append(j)

    for i in I:
        for k in unsaturated:
            if nx.has_path(H, num_D + i, k):
                I_H.append(i)

    if debug:
        print(f"I_H = {I_H}")
        print(f"D_H = {D_H}")

    # Step 6
    g = np.zeros_like(x)

    for i in range(num_F):
        for j in range(num_D):
            if i in I_H:
                g[i][j] = z_vals[i][j]
            elif (i in I) and (j in D_H):
                g[i][j] = 0
            elif (i in I):
                g[i][j] = z_vals[i][j]
            else: # i in S
                g[i][j] = 0

    if debug:
        print(f"g = {g}")
        input()

    # Step 7
    d = [1 - g[:, j].sum() for j in range(prob.num_D)] # demands
    feas, (arr1, arr2) = MFN_feas_test(g, x, y, prob, S, d)

    if not feas:
        return feas, (arr1, arr2) # if infeasible, return the violated constraint
    else:
        return True, (None, None)
        # y_hat = np.ones_like(y)
        # x_hat = np.copy(g)

        # h = arr1

        # h_S = h[S, :]
        # h_sum_S = h_S.sum(axis=0)

        # for i in S:
        #     y_hat[i] = 2 * y[i]
        #     for j in range(num_D):
        #         x_hat[i][j] = h[i][j] / h_sum_S[j] * d[j]


        # return True, (y_hat, x_hat)
    
# DEBUG TESTING
if __name__ == "__main__":
    pass