"""
Ellipsoid Algorithm with Separation Oracle
==========================================
Solves an LP of the form:
    minimize    c @ x
    subject to  x in P  (a polytope with potentially exponentially many constraints)

instead of a constraint matrix, the feasible region is described by a
**separation oracle** — a callable that, given a point x, either:
  - certifies x is feasible (returns None), or
  - returns a violated constraint as a vector g and scalar h such that
        g @ x > h   but   g @ z <= h  for all z in P
    i.e. the hyperplane {z : g @ z = h} separates x from P.

This is the standard approach for combinatorial LPs (e.g. matching polytope,
spanning tree polytope, submodular polyhedra) where the constraint matrix has
exponentially many rows but violated constraints can be found in polynomial time.

Oracle contract
---------------
    oracle(x: np.ndarray) -> None | tuple[np.ndarray, float]

    - Return None            if x is feasible (no constraint violated).
    - Return (g, h)          if x is infeasible:
                                 g : (n,) separating hyperplane normal
                                 h : float RHS  (g @ x > h, g @ z <= h for z in P)

Optimality
----------
Optimality is handled by augmenting the oracle with an objective cut:
once a feasible point x* is found with value v* = c @ x*, we add the
half-space  {z : c @ z <= v*}  as an additional constraint so the ellipsoid
is forced to search for improving solutions.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
from FacilityProb import FacilityLocationProb


# Type alias for the separation oracle
Oracle = Callable[[np.ndarray], Optional[Tuple[np.ndarray, float]]]


def ellipsoid_oracle(
    n: int,
    c: np.ndarray,
    oracle: Oracle,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-2,
    R: float = 12,# final polytope fits in a hypercube in all cases, max norm sqrt(m * n)
    x0: np.ndarray = None,
    extras = None,
    stop_early=False,
    find_feas=False,
    printer=False
) -> dict:
    """
    Minimize c @ x over a convex set P described by a separation oracle.

    Parameters
    ----------
    n        : dimension of the decision variable x
    c        : (n,) objective coefficient vector (minimization)
    oracle   : separation oracle  x -> None | (g, h)
                 None    => x is feasible
                 (g, h)  => g @ x > h, and g @ z <= h for all z in P
                            (g is the separating hyperplane normal, h is the RHS)
    max_iter : maximum number of ellipsoid iterations
    tol      : numerical tolerance for feasibility / convergence
    R        : initial ball radius; must satisfy ||x*|| <= R for some optimal x*
    x0       : initial center (defaults to zero vector)

    Returns
    -------
    dict with keys:
        status  : 'optimal' | 'infeasible' | 'max_iter_reached'
        x       : best feasible solution found (or None)
        obj     : objective value c @ x  (or None)
        iters   : number of iterations performed
    """
    # print("Running ellipsoid algorithm...")
    c = np.asarray(c, dtype=float)
    assert c.shape == (n,), f"c must be shape ({n},), got {c.shape}"

    # ------------------------------------------------------------------ #
    # Initialise ellipsoid E(x, P) = { z : (z-x)^T P^{-1} (z-x) <= 1 } #
    # as a ball of radius R centred at x0.                                #
    # ------------------------------------------------------------------ #
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    P = (R ** 2) * np.eye(n)
    # print(f"R={R}")
    # print(f"P={P}")
    # input()

    best_x: Optional[np.ndarray] = None
    best_obj: float = np.inf

    shrink = (n ** 2 / (n ** 2 - 1)) if n > 1 else 2.0   # precompute

    def ellipsoid_update(
        x: np.ndarray, P: np.ndarray, g: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Half-ellipsoid update in the direction of subgradient g.

        The new ellipsoid is the minimum-volume ellipsoid containing
            { z in E(x,P) : g @ (z - x) <= 0 }

        Shor / Nemirovski-Yudin update rules:
            g~ = P g / sqrt(g^T P g)
            x' = x - 1/(n+1) * g~
            P' = n²/(n²-1) * ( P - 2/(n+1) * g~ g~^T )
        """
        # print("updating objective...")
        # print(f"g={g}")
        Pg = P @ g
        gPg = float(g @ Pg)
        # print(f"gPg = {gPg}")
        # input()
        if gPg <= 0.0:
            # print("Error: degenerate case")
            # exit()
            return x, np.zeros_like(P)          # degenerate / zero subgradient — done
        # print("nondegenerate: calculating g_tilde")
        g_tilde = Pg / np.sqrt(gPg)
        # print("nondegenerate: calculated g_tilde")
        x_new = x - g_tilde / (n + 1)
        P_new = shrink * (P - (2.0 / (n + 1)) * np.outer(g_tilde, g_tilde))
        # print("objective updated...")
        return x_new, P_new

    status = "max_iter_reached"

    sep_count = 0
    for k in range(max_iter):
        # if printer: print(f"Iteration {k + 1}")

        # ---- Phase I: check feasibility via oracle ----------------------
        feasible, result = oracle(x, extras, debug=False)

        # print(f"Feasible? {feasible}")

        if not feasible:
            if printer: print(f"Iteration: {k + 1} --- Separating {sep_count}...")
            sep_count += 1
            # x is infeasible; oracle returns a separating hyperplane (g, h)
            # with g @ x > h.  Use g as the cutting direction.
            g, h = result
            g = np.asarray(g, dtype=float)
            assert g.shape == (n,), \
                f"oracle must return g of shape ({n},), got {g.shape}"
            x, P = ellipsoid_update(x, P, g)
            if np.max(P) == 0:
                break

            if sep_count == 3000: # just give up if separating 3000 iterations in a row
                break


        else:
            sep_count = 0
            # ---- Phase II: x is feasible — try to improve objective -----
            obj_val = float(c @ x)
            if printer: 
                print(f"found feasible point at iteration {k + 1} with objective {obj_val} versus {best_obj}")
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = x.copy()
            elif obj_val > 2 * best_obj and best_obj > 0.0 and k > 20000:
                break

            if stop_early and obj_val < 0.0:
                break

            if find_feas:
                break

            # Cut with the objective: push toward { z : c @ z < best_obj }
            # The separating hyperplane is c (pointing in the direction of
            # increasing objective), cutting off the current center.
            g = c
            x, P = ellipsoid_update(x, P, g)
            if np.max(P) == 0:
                break
            # if printer: print(f"Next objective: {c @ x}")


        # ---- Convergence check ------------------------------------------
        # Volume of the ellipsoid shrinks by factor exp(-1/(2n)) per step,
        # so after O(n² log(R/ε)) steps the ellipsoid is tiny.
        # We detect this via the smallest eigenvalue of P.
        eigmax = np.linalg.eigvalsh(P).max()
        if eigmax < tol:
            status = "optimal" if best_x is not None else "infeasible"
            break

    else:
        # Exhausted iterations
        if best_x is not None:
            status = "optimal"

    return {
        "status": status,
        "x": best_x,
        "obj": best_obj if best_x is not None else None,
        "iters": k + 1,
    }


# ======================================================================= #
#  Helper: build an oracle from an explicit (A, b) matrix pair            #
#  (useful for testing / bridging with the matrix-based version)          #
# ======================================================================= #

def matrix_oracle(A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> Oracle:
    """
    Construct a separation oracle for the polytope { x : A @ x <= b }.

    Returns the most-violated constraint, or None if x is feasible.
    This lets you use ellipsoid_oracle as a drop-in replacement for
    ellipsoid_lp when you happen to have A and b explicitly.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    def oracle(x: np.ndarray):
        slacks = A @ x - b          # positive => violated
        idx = int(np.argmax(slacks))
        if slacks[idx] > tol:
            return A[idx], b[idx]   # (g, h): g @ x > h
        return None                 # feasible

    return oracle


# ======================================================================= #
#  Demo / tests                                                            #
# ======================================================================= #

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    from scipy.optimize import linprog

    # ------------------------------------------------------------------ #
    # Example 1: Same simple 2-D LP as before, via matrix oracle          #
    #   minimize  -x1 - x2                                                 #
    #   s.t.      x1 + x2 <= 4,  x1 <= 3,  x2 <= 3,  x1,x2 >= 0         #
    #   Optimal: obj = -4                                                  #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Example 1: 2-D LP via matrix_oracle wrapper")
    A1 = np.array([[ 1,1],[ 1,0],[ 0,1],[-1,0],[ 0,-1]], dtype=float)
    b1 = np.array([4, 3, 3, 0, 0], dtype=float)
    c1 = np.array([-1., -1.])
    res1 = ellipsoid_oracle(2, c1, matrix_oracle(A1, b1), R=1e4)
    print(f"  Status : {res1['status']}")
    print(f"  x*     : {res1['x']}")
    print(f"  obj*   : {res1['obj']:.6f}  (expected -4.0)")
    print(f"  iters  : {res1['iters']}")

    # ------------------------------------------------------------------ #
    # Example 2: Minimum-weight spanning tree polytope (combinatorial!)   #
    #                                                                      #
    # The spanning-tree polytope on a complete graph K_n has              #
    # exponentially many constraints (one per subset S of vertices):      #
    #   x(E(S)) <= |S| - 1  for every S ⊆ V, |S| >= 2                    #
    #   x(E)    = n - 1     (total edges = n-1 for a spanning tree)       #
    #   0 <= x_e <= 1                                                      #
    #                                                                      #
    # Separation oracle (Cunningham 1984 / Padberg-Wolsey):               #
    #   The most violated subset constraint can be found in poly time via  #
    #   max-flow / min-cut. Here we use a simple O(n³) sweep over cuts     #
    #   for pedagogical clarity on small instances.                        #
    #                                                                      #
    # We model x as a vector indexed by edges of K_n (upper triangle).    #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Example 2: Minimum-weight spanning tree polytope on K_5")

    import itertools

    def spanning_tree_lp(n_nodes: int, weights: np.ndarray):
        """
        Solve the minimum spanning tree LP on K_{n_nodes} via ellipsoid.

        weights : (n_nodes*(n_nodes-1)//2,) edge weights in upper-triangle order
        Returns the LP solution (should match MST weight for integer weights).
        """
        edges = list(itertools.combinations(range(n_nodes), 2))
        m = len(edges)  # number of edges = n*(n-1)/2
        edge_idx = {e: i for i, e in enumerate(edges)}

        def edges_in(S):
            """Indices of edges with both endpoints in S."""
            S = set(S)
            return [edge_idx[(u, v)] for u, v in edges if u in S and v in S]

        def spanning_tree_oracle(x: np.ndarray):
            # --- Box constraints 0 <= x_e <= 1 ---
            for i in range(m):
                if x[i] < -1e-8:
                    g = np.zeros(m); g[i] = -1.0
                    return g, 0.0          # -x_e <= 0  =>  x_e >= 0
                if x[i] > 1 + 1e-8:
                    g = np.zeros(m); g[i] = 1.0
                    return g, 1.0          # x_e <= 1

            # --- Total edge constraint: sum(x) = n-1 (as two inequalities) ---
            total = x.sum()
            if total > (n_nodes - 1) + 1e-8:
                return np.ones(m), float(n_nodes - 1)
            if total < (n_nodes - 1) - 1e-8:
                return -np.ones(m), -float(n_nodes - 1)

            # --- Subset constraints x(E(S)) <= |S| - 1 ---
            # Sweep all subsets of size 2..n-1 (feasible for small n_nodes).
            # For large graphs replace with a min-cut computation.
            worst_g, worst_h, worst_viol = None, None, 0.0
            for size in range(2, n_nodes):
                for S in itertools.combinations(range(n_nodes), size):
                    idx = edges_in(S)
                    val = sum(x[i] for i in idx)
                    rhs = len(S) - 1
                    viol = val - rhs
                    if viol > worst_viol:
                        worst_viol = viol
                        worst_g = np.zeros(m)
                        for i in idx:
                            worst_g[i] = 1.0
                        worst_h = float(rhs)
            if worst_viol > 1e-8:
                return worst_g, worst_h
            return None   # feasible

        c = np.asarray(weights, dtype=float)
        return ellipsoid_oracle(m, c, spanning_tree_oracle, R=float(n_nodes), max_iter=50_000)

    # K_5 with known MST
    # Edges in order: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
    w = np.array([1, 4, 3, 6, 2, 5, 7, 8, 9, 10], dtype=float)
    # MST by inspection (Kruskal): (0,1)=1, (1,2)=2, (0,3)=3, (0,2) or others...
    # Kruskal picks edges in weight order: 1,2,3,4 => edges (0,1),(1,2),(0,2),(0,3) — total=10? 
    # Let's just compare LP relaxation to scipy.
    res2 = spanning_tree_lp(5, w)
    print(f"  Status : {res2['status']}")
    print(f"  x*     : {res2['x']}")
    print(f"  obj*   : {res2['obj']:.6f}")
    print(f"  iters  : {res2['iters']}")

    # Cross-check with scipy (enumerate all subset constraints explicitly)
    edges5 = list(itertools.combinations(range(5), 2))
    m5 = len(edges5)
    edge_idx5 = {e: i for i, e in enumerate(edges5)}
    rows_A, rows_b = [], []
    # subset constraints
    for size in range(2, 5):
        for S in itertools.combinations(range(5), size):
            S_set = set(S)
            idx = [edge_idx5[(u,v)] for u,v in edges5 if u in S_set and v in S_set]
            if idx:
                row = np.zeros(m5); row[idx] = 1.0
                rows_A.append(row); rows_b.append(float(len(S)-1))
    # box
    for i in range(m5):
        row = np.zeros(m5); row[i] = 1.0;  rows_A.append(row);  rows_b.append(1.0)
        row = np.zeros(m5); row[i] = -1.0; rows_A.append(row); rows_b.append(0.0)
    # total = n-1 as two inequalities
    rows_A.append(np.ones(m5));  rows_b.append(4.0)
    rows_A.append(-np.ones(m5)); rows_b.append(-4.0)

    ref = linprog(w, A_ub=np.array(rows_A), b_ub=np.array(rows_b), method="highs")
    print(f"  scipy ref obj: {ref.fun:.6f}  (status: {ref.message})")

    # ------------------------------------------------------------------ #
    # Example 3: Custom hand-written oracle                               #
    # L-inf ball: { x : -1 <= x_i <= 1 for all i }                       #
    # minimize sum(x) => optimal x* = (-1,-1,...,-1), obj = -n            #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Example 3: L-inf ball oracle, minimize sum(x), n=6")

    def linf_oracle(x: np.ndarray):
        """Separation oracle for the L-inf unit ball [-1,1]^n."""
        for i, xi in enumerate(x):
            if xi > 1 + 1e-8:
                g = np.zeros(len(x)); g[i] = 1.0
                return g, 1.0
            if xi < -1 - 1e-8:
                g = np.zeros(len(x)); g[i] = -1.0
                return g, -1.0
        return None

    n3 = 6
    c3 = np.ones(n3)   # minimize sum(x)
    res3 = ellipsoid_oracle(n3, c3, linf_oracle, R=float(n3))
    print(f"  Status : {res3['status']}")
    print(f"  x*     : {res3['x']}")
    print(f"  obj*   : {res3['obj']:.6f}  (expected {-n3}.0)")
    print(f"  iters  : {res3['iters']}")