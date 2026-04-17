import numpy as np
import csv
from FacilityProb2 import FacilityLocationProb2
import networkx as nx
from itertools import combinations

def Brute_Force(prob, printer=True, writer=False, verbose=False):
	n = prob.num_F
	best_val = float('inf')
	facs = list(range(0, n))
	for r in range(n+1):
		for c in combinations(facs, r):
			S = np.array(c)
			valS = determine_val(prob, S)
			if valS < best_val:
				best_val = valS
	return best_val

def determine_val(prob, facilities_subset, return_flow_dict=False): #given the list of facilities we choose to open, compute objective value
	#If flow_dict is True, then instead of returning the objective value, we return the flow dictionary.
	facilities_subset = [int(i) for i in facilities_subset]
	distances = prob.connection_costs#[facilities_subset, :]
	capacities = prob.capacities#[facilities_subset]
	demands = prob.demands
	m = prob.num_D
	if np.sum(demands) > np.sum(capacities):
		raise ValueError("Infeasible: total demand exceeds total capacity")
	G = nx.DiGraph()

	# Add nodes with demand attribute
	# Negative = supply, Positive = demand
	G.add_node("s", demand=-np.sum(demands))
	G.add_node("t", demand=np.sum(demands))

	# Facilities
	for i in facilities_subset:
		G.add_node(f"F{i}", demand=0)
		G.add_edge("s", f"F{i}", capacity=capacities[i], weight=0)

	# Clients
	for j in range(m):
		G.add_node(f"C{j}", demand=0)
		G.add_edge(f"C{j}", "t", capacity=demands[j], weight=0)

	# Facility → Client edges
	for i in facilities_subset:
		for j in range(m):
			G.add_edge(f"F{i}", f"C{j}", capacity=demands[j], weight=distances[i, j])

	# Solve min cost flow
	try:
		flow_dict = nx.min_cost_flow(G)
	except nx.NetworkXUnfeasible:
		#raise ValueError("Infeasible flow (capacity constraints violated)")
		return float('inf')
	if return_flow_dict:
		return flow_dict
	# Compute total connection cost
	cost1 = 0
	for i in facilities_subset:
		for j in range(m):
			flow = flow_dict[f"F{i}"][f"C{j}"]
			cost1 += flow * distances[i, j]
	#print("connection cost = " +str(cost1))
	#Then, add facility opening cost
	cost2 = 0
	for i in facilities_subset:
		cost2 += prob.opening_costs[i]
	#print("opening cost = "+str(cost2))
	return cost1 + cost2