"""
Part C: synthetic end-to-end check that the pipeline matches Theorem 3.1.
Build a tiny directed graph, compute kappa via the same scipy max-flow primitive used
in the paper, read off the predicted min-cut edge, delete it, and verify the demand
becomes unreachable (kappa -> 0). Also check that a kappa=2 demand survives one deletion.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

BIG = 1 << 24


def kappa_and_cut(edges, src, access, N):
    SINK = N
    eu = [u for u, v in edges]; ev = [v for u, v in edges]
    rows = eu + list(access); cols = ev + [SINK] * len(access)
    vals = [1] * len(eu) + [BIG] * len(access)
    M = csr_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(N + 1, N + 1))
    res = maximum_flow(M, src, SINK); k = int(res.flow_value)
    cut = []
    if k >= 1:
        F = res.flow; resid = (((M - F) > 0).astype(np.int8) + (F > 0).astype(np.int8).T).tocsr()
        order = breadth_first_order(resid, src, directed=True, return_predecessors=False)
        R = np.zeros(N + 1, bool); R[order] = True
        cut = [(u, v) for (u, v) in edges if R[u] and not R[v]]
    return k, cut


def reachable(edges, src, access, N):
    if not edges:
        return False
    rows = [u for u, v in edges]; cols = [v for u, v in edges]
    A = csr_matrix((np.ones(len(edges)), (np.array(rows), np.array(cols))), shape=(N, N))
    order = breadth_first_order(A, src, directed=True, return_predecessors=False)
    return bool(set(order) & set(access))


def main():
    ok = True
    # Case 1: chain d=0 -> 1 -> 2 -> access=3.  kappa should be 1; cut isolates d.
    E = [(0, 1), (1, 2), (2, 3)]; N = 4; acc = [3]
    k, cut = kappa_and_cut(E, 0, acc, N)
    E2 = [e for e in E if e != cut[0]]
    still = reachable(E2, 0, acc, N)
    print(f"Case 1 (chain): kappa={k} (expect 1), predicted cut={cut[0]}, "
          f"reachable after deleting cut={still} (expect False)")
    ok &= (k == 1 and not still)
    # Case 2: two edge-disjoint routes d=0 ->1->3 and 0->2->3, access=3. kappa=2; survives 1 cut.
    E = [(0, 1), (1, 3), (0, 2), (2, 3)]; N = 4; acc = [3]
    k, cut = kappa_and_cut(E, 0, acc, N)
    E2 = [e for e in E if e != (0, 1)]  # delete one edge
    still = reachable(E2, 0, acc, N)
    print(f"Case 2 (two disjoint): kappa={k} (expect 2), reachable after deleting one edge={still} (expect True)")
    ok &= (k == 2 and still)
    # Case 3: many near-optimal routes sharing a bottleneck e0=(0,1); kappa=1.
    E = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]; N = 6; acc = [5]
    k, cut = kappa_and_cut(E, 0, acc, N)
    E2 = [e for e in E if e != (0, 1)]
    still = reachable(E2, 0, acc, N)
    print(f"Case 3 (bottleneck w/ many continuations): kappa={k} (expect 1), "
          f"reachable after deleting bottleneck={still} (expect False)")
    ok &= (k == 1 and not still)
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
    return ok


if __name__ == "__main__":
    main()
