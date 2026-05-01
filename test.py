import gurobipy as gp
import numpy as np
import csv
from FacilityProb import OldFacilityLocationProb, FacilityLocationProb
from boxed_alg import CFL_oracle
from LP_alg import LP_approx
from standard import StandardLP

def test1():
    n = 4 # 4 facilities
    m = 20 # 20 customers

    dems = np.ones(m)
    opens = 300 * np.ones(n)
    caps = 25 * np.ones(n) 
    conn_costs = np.random.randint(50, size=(n, m))
    my_prob = FacilityLocationProb(
        opening_costs = opens,
        connection_costs = conn_costs,
        capacities = caps,
        demands = dems
    )

    control_val = StandardLP(my_prob, printer=False, verbose=False)

    print(f"Naive relaxation: {control_val}")
    # input()

    val = LP_approx(my_prob)

def test2(K):
    """
    This shows unbounded integrality gap with K as a parameter
    """
    n = 2 # 2 facilities
    m = K + 1 # K+1 customers

    dems = np.ones(m)
    opens = np.array([1, K])
    caps = K * np.ones(m)
    conn_costs = (1/100000) * np.random.rand(n, m) # tiny
    my_prob = OldFacilityLocationProb(
        opening_costs = opens,
        connection_costs = conn_costs,
        capacities = caps,
        demands = dems
    )

    val = StandardLP(my_prob, printer=True, verbose=True)

test2(100)