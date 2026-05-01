import numpy as np
import csv
from FacilityProb2 import FacilityLocationProb2
from localsearch import LocalSearch

n = 30 # number of facilities
m = 60 # number of customers
gridsize = 80 #everything placed in a grid of size gridsize x gridsize

dems = np.ones(m)
caps = np.random.randint(4,25,size=n)
opens = np.array([np.random.randint(4*caps[i],8*caps[i]) for i in range(n)])
print(caps)
print(opens)
faclocs = np.random.randint(0, gridsize, size=(n, 2))
clilocs = np.random.randint(0, gridsize, size=(m, 2))
my_prob = FacilityLocationProb2(
    opening_costs = opens,
    facility_locations = faclocs,
    client_locations = clilocs,
    capacities = caps,
    demands = dems
)

#print(my_prob.client_locations)
#print(my_prob.facility_locations)
#print(my_prob.connection_costs)

print(LocalSearch(my_prob))