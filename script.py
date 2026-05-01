import gurobipy as gp
import numpy as np
import csv
import sys
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
    if len(sys.argv) > 1:
        write = (sys.argv[1] == "True")
    else:
        write = False
        
    print_gaps = False

    for (n, m) in [
        (5, 10), 
        (5, 25), 
        (5, 50),
        (10, 25), 
        (10, 50),
        (25, 50),
        # (25, 100),
        # (50, 100)
    ]:
        for i in range(100): # 100 tests
            # n = number of facilities
            # m = number of customers
            gridsize = 100 # everything placed in a grid of size gridsize x gridsize

            dems = np.ones(m)
            
            caps = np.zeros(n)
            while caps.sum() < m:
                caps = np.random.choice(list(range(1, m - 1)),size=n) # results_1X
                # caps = np.random.choice([1, m - 1],size=n) # results_2X

            # opens = np.random.choice([1000, 2000, 4000, 8000, 16000], size=n) # results_X1
            opens = np.random.choice([100, 200, 300, 400, 500], size=n) # results_X2
            faclocs = np.random.randint(0, gridsize, size=(n, 2))
            clilocs = np.random.randint(0, gridsize, size=(m, 2))
            my_prob = FacilityLocationProb(
                opening_costs = opens,
                facility_locations = faclocs,
                client_locations = clilocs,
                capacities = caps,
                demands = dems
            )

            # print(my_prob.connection_costs)
            # input()

            # run tests
            # print("--------------> Running Brute Force...")
            OPT_val = StandardIP(my_prob, printer=False, verbose=False)
            # print("--------------> Running LP relaxation...")
            # x0, CONTROL_val = StandardLP(my_prob, printer=False, verbose=False)
            # print("--------------> Running Local Search...")
            LS_val = LocalSearch(my_prob, printer=False)


            # print("--------------> Running MFN-LP...")
            CONTROL_val, LP_APPROX_val = LP_approx(my_prob)
            # LP_APPROX_val = -1

            if OPT_val > CONTROL_val and print_gaps == 1:
                print(f"Integrality gap at {(n, m)} rd. {i + 1} of {OPT_val/CONTROL_val:.4f}")
            if CONTROL_val < LP_APPROX_val:
                print(f"MFN-LP Improvement at {(n, m)} rd. {i + 1} with ratio {LP_APPROX_val/CONTROL_val:.4f}")
            if OPT_val < LS_val:
                # print(my_prob.opening_costs)
                # print(my_prob.capacities)
                print(f"Suboptimal Local Search Performance at {(n, m)} rd. {i + 1} with ratio {LS_val/OPT_val:.4f}")
                # input()

            # LP_APPROX_val = -1.0

            arr = [n, m, CONTROL_val, LP_APPROX_val, OPT_val, LS_val, CONTROL_val/OPT_val, LS_val/OPT_val, LP_APPROX_val/OPT_val]

            if write:
                with open(outputfile, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(arr)