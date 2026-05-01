import numpy as np
import csv
from FacilityProb import FacilityLocationProb
import networkx as nx

epsilon = 0.01 # Algorithm of Pal et al. guarantees a (9+epsilon)-approximation

def LocalSearch(prob, printer=True, writer=False, verbose=False):
	S = np.array([j for j in range(prob.num_F)]) #take all facilities for now :)
	val = determine_val(prob, S)
	step = 0
	if printer: print("Starting S: "+str(S)+"; starting val: "+str(val))
	while True: #check to see whether any of the local search operations can lower the objective value
		step += 1
		best_op = ""
		flow_dict = determine_val(prob, S, return_flow_dict=True)
		bestNewS = S
		bestNewVal = val
         # Part 1: add
		for j in range(prob.num_F):
			if j not in S: # Try adding j to S
				newS = np.append(S, j)
				valNewS = determine_val(prob, newS)
				if valNewS <= bestNewVal: # If better than other sets considered so far, take note!
					bestNewS = newS
					bestNewVal = valNewS
					best_op = "add("+str(j)+")"

		# Part 2: open
		for j in range(prob.num_F):
			# We add j to S and choose a set of facilities in S to close
			#if j not in S: 
			T = open_subroutine_choose_facilities_to_close(prob, S, flow_dict, j)
			newS=S
			if j not in newS:
				newS = np.append(newS, j) # Add j
			mask = ~np.isin(newS, T)
			newS = newS[mask] # Remove T
			valNewS = determine_val(prob, newS)
			if valNewS <= bestNewVal: # If better than other sets considered so far, take note!
				bestNewS = newS
				bestNewVal = valNewS
				best_op = "open("+str(j)+", "+str([int(t) for t in T])+")"

		# Part 3: close
		for j in S:
			# We remove j from S and choose a set of facilities not in S to open
			T = close_subroutine_choose_facilities_to_open(prob, S, flow_dict, j)
			newS = S[S != j] # Remove j
			if len(T)>0:
				newS = np.union1d(newS, T) # Add T
			valNewS = determine_val(prob, newS)
			if valNewS <= bestNewVal: # If better than other sets considered so far, take note!
				bestNewS = newS
				bestNewVal = valNewS
				best_op = "close("+str(j)+", "+str([int(t) for t in T])+")"

		# If one of the operations yielded at least some small improvement, take that and continue to the next iteration of the while loop.
		if bestNewVal <= val*(1-epsilon/(prob.num_F)):
			S = bestNewS
			val = bestNewVal
			if printer: print("Step = "+str(step)+"; chosen operation: "+best_op+"; new S = "+str(S)+"; new cost = "+str(val))
		else: # If none of the operations yields a small improvement, we're done.
			if printer: print("Step = "+str(step)+"; no operation yielded enough improvement.")
			break
	return val

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

def open_subroutine_choose_facilities_to_close(prob, S, flow_dict, s):
	items = []

	# Extract profit and weight for each t in S
	for t in S:
		if t == s:
			continue
		total_flow = 0
		reroute_cost_increase = 0

		for j_node, flow in flow_dict[f"F{t}"].items():
			if j_node.startswith("C"):
				j = int(j_node[1:])
				total_flow += flow
				reroute_cost_increase += flow * (prob.connection_costs[s, j] - prob.connection_costs[t, j])

		profit = prob.opening_costs[t] - reroute_cost_increase
		weight = total_flow

		# Only keep items with positive profit
		if profit > 0 and weight > 0:
			items.append((t, weight, profit))

	current_load = 0
	if f"F{s}" in flow_dict:
		for j_node, flow in flow_dict[f"F{s}"].items():
			if j_node.startswith("C"):
				current_load += flow

	cap = int(prob.capacities[s] - current_load)

	if cap <= 0:
		return []

	# Knapsack DP
	dp = [0] * (cap + 1)
	keep = [[False]*len(items) for _ in range(cap + 1)]

	for i, (t, w, v) in enumerate(items):
		w=int(w)
		for c in range(cap, w - 1, -1):
			if dp[c - w] + v > dp[c]:
				dp[c] = dp[c - w] + v
				keep[c] = keep[c - w].copy()
				keep[c][i] = True

	# Recover solution
	best_c = max(range(cap + 1), key=lambda c: dp[c])
	chosen = keep[best_c]

	T = [items[i][0] for i in range(len(items)) if chosen[i]]
	return T

def close_subroutine_choose_facilities_to_open(prob, S, flow_dict, s):
	n = prob.num_F
	current_load = np.zeros(n)
	for t in S:
		for j_node, flow in flow_dict[f"F{t}"].items():
			if j_node.startswith("C"):
				current_load[t] += flow
	current_load = [int(load) for load in current_load]
	# total demand currently served by s
	d_s = current_load[s]

	if d_s == 0:
		return []  # nothing to reroute

	residual_capacity = prob.capacities - current_load
	residual_capacity = [int(rescap) for rescap in residual_capacity]

	best_cost = float("inf")
	best_T = None

	# ---- Step 2: try each f as t* ----
	for f in range(n):
		if f == s:
			continue

		cap_f = residual_capacity[f]
		dist_sf = prob.facility_distances[s, f]

		# Need to cover this much demand with T'
		demand_needed = max(0, d_s - cap_f)

		items = []

		for t in range(n):
			if t == s or t == f:
				continue

			cap_t = residual_capacity[t]
			if cap_t <= 0:
				continue

			open_cost = 0 if t in S else prob.opening_costs[t]
			cost_t = open_cost + cap_t * prob.facility_distances[s, t]

			# transformed cost for DP
			value = cost_t - dist_sf * cap_t

			items.append((t, cap_t, value))

		# ---- Step 3: covering knapsack DP ----
		# dp[x] = min cost to achieve capacity x
		max_cap = sum(cap for _, cap, _ in items)
		max_cap = int(min(max_cap, demand_needed))

		dp = [float("inf")] * (max_cap + 1)
		choice = [None] * (max_cap + 1)

		dp[0] = 0
		choice[0] = []

		for t, cap_t, val in items:
			for c in range(max_cap, -1, -1):
				if dp[c] < float("inf"):
					new_c = min(max_cap, c + cap_t)
					new_cost = dp[c] + val
					if new_cost < dp[new_c]:
						dp[new_c] = new_cost
						choice[new_c] = choice[c] + [t]

		# find best feasible solution
		best_sub_cost = float("inf")
		best_subset = []

		for c in range(demand_needed, max_cap + 1):
			if dp[c] < best_sub_cost:
				best_sub_cost = dp[c]
				best_subset = choice[c]

		if best_sub_cost == float("inf"):
  			  continue  # infeasible for this f

		# ---- Step 4: compute full cost ----
		open_cost_f = 0 if f in S else prob.opening_costs[f]

		total_cost = (
			-prob.opening_costs[s]
			+ open_cost_f
			+ dist_sf * d_s
			+ best_sub_cost
		)

		# include f in T
		T_candidate = best_subset + [f]

		if total_cost < best_cost:
			best_cost = total_cost
			best_T = T_candidate

	return best_T if best_T is not None else []