import gurobipy as gp
import numpy as np
import csv
from FacilityProb import FacilityLocationProb
from boxed_alg import CFL_oracle
from LP_alg import LP_approx
from standard import StandardLP, StandardIP
from localsearch import LocalSearch
from brute_force import Brute_Force

"""
Generates random instances & runs full test suite
"""

outputfile = "results.csv"

if __name__ == "__main__":
    for (n, m) in [
        # (5, 10), 
        # (5, 25), 
        # (5, 50),
        # (10, 25), 
        # (10, 50),
        # (25, 50)
        (10, 100),
        (25, 100),
        (50, 100)
    ]:
        for i in range(100): # 100 tests
            # n = 5 # number of facilities
            # m = 25 # number of customers
            gridsize = 100 #everything placed in a grid of size gridsize x gridsize

            dems = np.ones(m)
            caps = np.random.randint(10,int(m/2),size=n)
            opens = np.array([np.random.randint(4*caps[i],8*caps[i]) for i in range(n)])
            faclocs = np.random.randint(0, gridsize, size=(n, 2))
            clilocs = np.random.randint(0, gridsize, size=(m, 2))
            my_prob = FacilityLocationProb(
                opening_costs = opens,
                facility_locations = faclocs,
                client_locations = clilocs,
                capacities = caps,
                demands = dems
            )

            # run tests
            print("--------------> Running Brute Force...")
            OPT_val = StandardIP(my_prob, printer=True, verbose=False)
            print("--------------> Running LP relaxation...")
            CONTROL_val = StandardLP(my_prob, printer=False, verbose=False)
            print("--------------> Running Local Search...")
            LS_val = LocalSearch(my_prob)
            print("--------------> Running MFN-LP...")
            # LP_APPROX_val = LP_approx(my_prob)
            LP_APPROX_val = -1.0

            # print(CONTROL_val, LS_val, LP_APPROX_val)

            arr = [n, m, CONTROL_val, LP_APPROX_val, OPT_val, LS_val]

            with open(outputfile, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(arr)