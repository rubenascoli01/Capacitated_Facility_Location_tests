import gurobipy as gp
import numpy as np
import csv
from FacilityProb import FacilityLocationProb
from ellipsoid_lp import ellipsoid_oracle
from boxed_alg import CFL_oracle
"""
Implementation of ellipsoid algorithm using boxed separation oracle.
"""

def LP_approx(prob, printer=True, writer=False, verbose=False):
    print("Running LP semi-integral approximation algorithm...")
    prob.flatten()
    obj = np.concatenate((prob.opening_costs, prob.flat_connection_costs))
    # first prob.num_F are opening costs
    # y[i] = var[i]
    # x[i][j] = var[prob.num_F + prob.num_D * i + j]
    var = np.zeros(prob.num_F * (1 + prob.num_D))

    max_iter = 1_000
    i = 0


    result_dict = ellipsoid_oracle(
        n = prob.num_F * (1 + prob.num_D),
        c = obj,
        oracle = CFL_oracle,
        max_iter = 100,
        tol = 1e-3,
        x0 = var,
        extras = prob,
        find_feas=True,
        printer=True
    )

    if result_dict["status"] == "infeasible":
        print("Error!!!") # DEBUG
    else:
        print(f"relaxed objective: {result_dict["obj"]}")
        print(f"relaxed assignment: {result_dict["x"]}")

    return result_dict["obj"]