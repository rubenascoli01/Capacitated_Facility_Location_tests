import gurobipy as gp
import numpy as np
import csv
from FacilityProb import FacilityLocationProb

def StandardLP(prob, printer=True, writer=False, verbose=False):
    """
    Takes in a "FacilityLocationProb" object and outputs the standard LP relaxation solution.
    """

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
    if not printer: m.setParam('OutputFlag', 0)
    m.optimize()

    # 6. Print
    if printer:
        if m.status == gp.GRB.OPTIMAL:
            if verbose:
                print("x =", x.X)
                print("y =", y.X)
                print("obj =", m.ObjVal)
                # for c in m.getConstrs():
                #     print(c.ConstrName, "slack:", c.Slack)
            return m.ObjVal
        elif m.status == gp.GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded")
        elif m.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible")
        elif m.status == gp.GRB.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Optimization stopped with status:", m.status)

    return np.concatenate([y.X, x.X.flatten()]), m.ObjVal

def StandardIP(prob, printer=True, writer=False, verbose=False):
    """
    Takes in a "FacilityLocationProb" object and outputs the standard ILP solution.
    """

    # 1. Make model
    m = gp.Model()

    # 2. Add Variables
    x_shape = prob.connection_costs.shape
    x = m.addMVar(x_shape, vtype=gp.GRB.BINARY, name="x")

    y_shape = prob.num_F
    y = m.addMVar(shape=(y_shape,), vtype=gp.GRB.BINARY, name="y")

    # 3. Add objective
    expr1 = prob.opening_costs @ y
    expr2 = (prob.connection_costs * x).sum()

    m.setObjective(expr1 + expr2, gp.GRB.MINIMIZE)

    # 4. Add Constraints
    # 4.1 x[i][j] <= y[j]
    # for i in range(prob.num_F):
    #     for j in range(prob.num_D):
    #         m.addConstr(x[i][j] - y[i] <= 0.0001, name=f"Coverage({i},{j})")

    # 4.2 sum of x[i][j] over all i is >= demand
    for j in range(prob.num_D):
        m.addConstr(x[:, j].sum() - prob.demands[j] >= -0.00001, name=f"Demand{j}")

    # 4.3 cannot exceed facility capacity.
    for i in range(prob.num_F):
        m.addConstr(x[i, :] @ prob.demands - prob.capacities[i] * y[i] <= 0.0001, name=f"Capacity{i}")

    # 5 Solve
    if not printer: m.setParam('OutputFlag', 0)
    m.optimize()

    # print("y =", y.X)

    # 6. Print
    if printer:
        if m.status == gp.GRB.OPTIMAL:
            if verbose:
                print("x =", x.X)
                print("y =", y.X)
                print("obj =", m.ObjVal)
                # for c in m.getConstrs():
                #     print(c.ConstrName, "slack:", c.Slack)
            return m.ObjVal
        elif m.status == gp.GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded")
        elif m.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible")
        elif m.status == gp.GRB.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Optimization stopped with status:", m.status)

    if m.status == gp.GRB.INFEASIBLE: 
        print(f"Infeasible capacities: {prob.capacities}")
        input()
        return -1

    return m.ObjVal

def test1():
    n = 40 # 40 facilities
    m = 190 # 190 customers

    dems = np.ones(m)
    opens = 300 * np.ones(n)
    caps = 25 * np.ones(m)
    conn_costs = np.random.randint(50, size=(n, m))
    my_prob = FacilityLocationProb(
        opening_costs = opens,
        connection_costs = conn_costs,
        capacities = caps,
        demands = dems
    )

    val = StandardLP(my_prob, printer=True)
    print(val)

def test2(K):
    """
    This shows unbounded integrality gap with K as a parameter
    """
    n = 2 # 2 facilities
    m = K + 1 # K+1 customers

    dems = np.ones(m)
    opens = np.array([1, K])
    caps = K * np.ones(m)
    conn_costs = (1/K) * np.random.rand(n, m) # tiny
    my_prob = FacilityLocationProb(
        opening_costs = opens,
        connection_costs = conn_costs,
        capacities = caps,
        demands = dems
    )

    val = StandardLP(my_prob, printer=True, verbose=True)
    # print(val)

    # integral solution is 101 + small connection costs
if __name__ == "__main__":  
    test2(10000)