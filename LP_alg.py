import gurobipy as gp
import numpy as np
import csv
from FacilityProb import FacilityLocationProb
from ellipsoid_lp import ellipsoid_oracle
from boxed_alg import CFL_oracle
"""
Implementation of ellipsoid algorithm using boxed separation oracle.
"""

def LP_approx(prob, printer=False, writer=False, verbose=False):
    # print("Running LP semi-integral approximation algorithm...")
    prob.flatten()
    # obj = np.concatenate((prob.opening_costs, prob.flat_connection_costs))

    num_F = prob.num_F
    num_D = prob.num_D

    # 1. Make model
    m = gp.Model()

    # 2. Add Variables
    x_shape = prob.connection_costs.shape
    x = m.addMVar(x_shape, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

    y_shape = prob.num_F
    y = m.addMVar(shape=(y_shape,), vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")

    # 3. Add objective
    expr1 = prob.opening_costs @ y
    expr2 = (prob.connection_costs * x).sum()

    m.setObjective(expr1 + expr2, gp.GRB.MINIMIZE)

    # 4. Add Constraints
    # 4.1 x[i][j] <= y[j]
    for i in range(prob.num_F):
        for j in range(prob.num_D):
            m.addConstr(x[i][j] - y[i] <= 0.0, name=f"Coverage({i},{j})")

    # 4.2 sum of x[i][j] over all i is >= demand
    for j in range(prob.num_D):
        m.addConstr(x[:, j].sum() - prob.demands[j] >= 0.0, name=f"Demand{j}")

    # 4.3 cannot exceed facility capacity.
    for i in range(prob.num_F):
        m.addConstr(x[i, :] @ prob.demands - prob.capacities[i] * y[i] <= 0.0, name=f"Capacity{i}")

    # 5 Solve
    m.setParam('OutputFlag', 0)
    m.optimize()
    if m.status == gp.GRB.INFEASIBLE: 
        print(f"Infeasible capacities: {prob.capacities}")
        return -1, -1

    x_test = np.concatenate([y.X, x.X.flatten()])
    objective_1 = m.ObjVal
    objective_2 = objective_1


    feas = False
    while not feas:
        feas, (g, h) = CFL_oracle(x_test, prob)
        if feas: break
        
        g_unflat_x = g[num_F:].reshape(x_shape)
        g_unflat_y = g[:num_F]
        m.addConstr((g_unflat_x * x).sum() + g_unflat_y @ y <= h)
        print(f"--------- MFN-LP cutting ----------")
        # print(f"h={h} versus {(g_unflat_x * x.X).sum() + g_unflat_y @ y.X}")
        # print(g)
        m.optimize()
        x_test = np.concatenate([y.X, x.X.flatten()])
        # input()
        objective_2 = m.ObjVal

    # result_dict = ellipsoid_oracle(
    #     n = prob.num_F * (1 + prob.num_D),
    #     c = obj,
    #     oracle = CFL_oracle,
    #     max_iter = 5_000, # need a ton of iterations to clear all the box constraints; only < 100 of these will be for the actual constraint
    #     tol = 1e-3,
    #     x0 = x0,
    #     extras = prob,
    #     R = np.sqrt(prob.num_F * prob.num_D + prob.num_F),
    #     # find_feas=True,
    #     printer=True
    # )

    # if result_dict["status"] == "infeasible":
    #     print("Error!!!") # DEBUG
    # else:
    #     print(f"relaxed objective: {result_dict["obj"]}")
    #     # print(f"relaxed assignment: {result_dict["x"]}")
    #     print(f"relaxed openings: {result_dict["x"][:prob.num_F]}")

    #     assignments = result_dict["x"][prob.num_F:].reshape((prob.num_F, prob.num_D))
    #     # print(assignments.sum(axis=0))
    #     # input()


    return (objective_1, objective_2)