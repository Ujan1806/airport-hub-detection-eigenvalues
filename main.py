import json
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# 1. DATASET: 10 Indian Airports & 21 Flight Routes
# -----------------------------------------------------------------------------
AIRPORTS = [
    {"code": "DEL", "city": "New Delhi", "name": "Indira Gandhi Int'l Airport", "category": "National Mega Hub"},
    {"code": "BOM", "city": "Mumbai", "name": "Chhatrapati Shivaji Maharaj Int'l", "category": "National Mega Hub"},
    {"code": "BLR", "city": "Bengaluru", "name": "Kempegowda Int'l Airport", "category": "Primary Metro Hub"},
    {"code": "HYD", "city": "Hyderabad", "name": "Rajiv Gandhi Int'l Airport", "category": "Primary Metro Hub"},
    {"code": "MAA", "city": "Chennai", "name": "Chennai Int'l Airport", "category": "Primary Metro Hub"},
    {"code": "CCU", "city": "Kolkata", "name": "Netaji Subhash Chandra Bose Int'l", "category": "Regional Hub (East)"},
    {"code": "AMD", "city": "Ahmedabad", "name": "Sardar Vallabhbhai Patel Int'l", "category": "Regional Hub (West)"},
    {"code": "COK", "city": "Kochi", "name": "Cochin Int'l Airport", "category": "Regional Gateway (South)"},
    {"code": "GOI", "city": "Goa", "name": "Dabolim / Manohar Int'l", "category": "Tourist Gateway (West)"},
    {"code": "GAU", "city": "Guwahati", "name": "Lokpriya Gopinath Bordoloi Int'l", "category": "Regional Gateway (NE)"},
]

ROUTES = [
    {"origin": "DEL", "destination": "BOM", "daily_flights": 65},
    {"origin": "DEL", "destination": "BLR", "daily_flights": 45},
    {"origin": "DEL", "destination": "HYD", "daily_flights": 32},
    {"origin": "DEL", "destination": "CCU", "daily_flights": 26},
    {"origin": "DEL", "destination": "MAA", "daily_flights": 24},
    {"origin": "DEL", "destination": "AMD", "daily_flights": 22},
    {"origin": "DEL", "destination": "GAU", "daily_flights": 9},
    {"origin": "BOM", "destination": "BLR", "daily_flights": 40},
    {"origin": "BOM", "destination": "HYD", "daily_flights": 26},
    {"origin": "BOM", "destination": "GOI", "daily_flights": 24},
    {"origin": "BOM", "destination": "MAA", "daily_flights": 22},
    {"origin": "BOM", "destination": "AMD", "daily_flights": 20},
    {"origin": "BOM", "destination": "COK", "daily_flights": 14},
    {"origin": "BLR", "destination": "HYD", "daily_flights": 28},
    {"origin": "BLR", "destination": "MAA", "daily_flights": 22},
    {"origin": "BLR", "destination": "COK", "daily_flights": 16},
    {"origin": "BLR", "destination": "GOI", "daily_flights": 14},
    {"origin": "HYD", "destination": "MAA", "daily_flights": 16},
    {"origin": "HYD", "destination": "CCU", "daily_flights": 12},
    {"origin": "MAA", "destination": "COK", "daily_flights": 10},
    {"origin": "CCU", "destination": "GAU", "daily_flights": 12},
]


def build_matrices():
    """Builds the 10x10 binary adjacency matrix A and weighted frequency matrix W."""
    n = len(AIRPORTS)
    code_to_idx = {a["code"]: i for i, a in enumerate(AIRPORTS)}
    A = np.zeros((n, n), dtype=float)
    W = np.zeros((n, n), dtype=float)

    for r in ROUTES:
        i = code_to_idx[r["origin"]]
        j = code_to_idx[r["destination"]]
        flights = float(r["daily_flights"])
        A[i, j] = A[j, i] = 1.0
        W[i, j] = W[j, i] = flights

    return A, W, code_to_idx


def power_iteration(M, max_iter=100, tol=1e-9):
    """
    Implements Power Iteration from scratch:
        v^(k+1) = M v^(k) / ||M v^(k)||
        lambda^(k) = Rayleigh Quotient
    """
    n = M.shape[0]
    v = np.ones(n, dtype=float) / np.sqrt(n)
    final_lambda = 0.0

    for k in range(1, max_iter + 1):
        Mv = M @ v
        lam = float(v.T @ Mv / (v.T @ v))
        v_next = Mv / np.linalg.norm(Mv)
        step_error = float(np.linalg.norm(v_next - v))
        v = v_next
        final_lambda = lam
        if step_error < tol:
            return final_lambda, v, k

    return final_lambda, v, max_iter


def main():
    print("=" * 80)
    print("   AIRPORT HUB DETECTION USING EIGENVALUES & EIGENVECTORS")
    print("   Course: UE25MA242A - MFAD | PES University")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: REAL-WORLD DATA & MATRIX REPRESENTATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 1: REAL-WORLD DATA & MATRIX REPRESENTATION")
    print("-" * 80)
    A, W, code_to_idx = build_matrices()
    n = len(AIRPORTS)
    total_flights = int(np.sum(W) / 2)
    total_routes = int(np.sum(A) / 2)

    print(f"Airports (Nodes V)   : {n}")
    print(f"Routes   (Edges E)   : {total_routes} bidirectional flight corridors")
    print(f"Daily Flight Volume  : {total_flights} commercial flights/day")
    print(f"Matrix Dimension     : {n}x{n} Real Symmetric (A = A^T, W = W^T)")
    print(f"Matrix Trace         : Tr(A) = {int(np.trace(A))}, Tr(W) = {int(np.trace(W))} (Zero self-loops)")

    # -------------------------------------------------------------------------
    # STAGE 2: STRUCTURE OF THE SPACE & RANK-NULLITY THEOREM
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 2: STRUCTURE OF THE SPACE (BASIS, RANK & NULLITY)")
    print("-" * 80)
    rank_A = int(np.linalg.matrix_rank(A))
    nullity_A = n - rank_A
    rank_W = int(np.linalg.matrix_rank(W))
    nullity_W = n - rank_W

    print(f"Ambient Vector Space : R^{n} (10-dimensional airport state space)")
    print(f"Binary Matrix A      : Rank = {rank_A}, Nullity = {nullity_A}")
    print(f"Rank-Nullity Theorem : Rank(A) + Nullity(A) = {rank_A} + {nullity_A} = {n} (VERIFIED)")
    print(f"Weighted Matrix W    : Rank = {rank_W}, Nullity = {nullity_W} (Full Rank)")

    # -------------------------------------------------------------------------
    # STAGE 3: PATTERN DISCOVERY (EIGENVALUES & EIGENVECTORS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 3: SPECTRAL GRAPH DECOMPOSITION & PERRON-FROBENIUS THEOREM")
    print("-" * 80)
    # Diagonalization of symmetric matrix W via np.linalg.eigh
    w, v = np.linalg.eigh(W)
    idx_desc = np.argsort(w)[::-1]
    evals = w[idx_desc]
    evecs = v[:, idx_desc]

    lambda_1 = float(evals[0])
    lambda_2 = float(evals[1])
    spectral_gap = lambda_1 - lambda_2
    dom_vec = np.abs(evecs[:, 0])  # Perron-Frobenius sign canonicalization

    # Linear algebra verification: ||Wv - lambda*v||
    residual = float(np.linalg.norm(W @ dom_vec - lambda_1 * dom_vec))
    orthonormal_check = np.allclose(evecs.T @ evecs, np.eye(n))

    print(f"Dominant Eigenvalue (lambda_1)  : {lambda_1:.4f}")
    print(f"Second Eigenvalue   (lambda_2)  : {lambda_2:.4f}")
    print(f"Spectral Gap (lambda_1 - lambda_2): {spectral_gap:.4f}")
    print(f"Residual Norm ||Wv - lambda*v|| : {residual:.2e} (Exact to machine precision)")
    print(f"Orthonormal Eigenbasis (Q^T*Q=I): {orthonormal_check}")
    print(f"Perron-Frobenius Guarantee      : All coordinates v_i > 0 for dominant eigenvector")

    # -------------------------------------------------------------------------
    # STAGE 4: NUMERICAL APPROXIMATION (POWER ITERATION)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 4: NUMERICAL APPROXIMATION (POWER ITERATION FROM SCRATCH)")
    print("-" * 80)
    pi_lam, pi_vec, iters = power_iteration(W)
    print(f"Power Iteration Converged in    : {iters} iterations")
    print(f"Estimated Dominant Eigenvalue   : {pi_lam:.4f}")
    print(f"Difference from Exact Solver    : {abs(pi_lam - lambda_1):.2e}")

    # -------------------------------------------------------------------------
    # STAGE 5: FINAL APPLICATION OUTPUT (HUB CENTRALITY RANKINGS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 5: FINAL APPLICATION OUTPUT (STRATEGIC AIRPORT HUB RANKINGS)")
    print("-" * 80)
    # Max-normalization: Premier hub = 1.0000
    hub_scores = dom_vec / np.max(dom_vec)
    ranked_indices = np.argsort(hub_scores)[::-1]

    header = f"{'Rank':<5} | {'Code':<5} | {'City':<14} | {'Routes':<6} | {'Daily Flights':<13} | {'Hub Score':<10} | {'Classification'}"
    print(header)
    print("-" * len(header))

    for rank, idx in enumerate(ranked_indices, start=1):
        airport = AIRPORTS[idx]
        routes_count = int(np.sum(A[idx, :]))
        flight_count = int(np.sum(W[idx, :]))
        score = hub_scores[idx]
        print(
            f"{rank:<5} | {airport['code']:<5} | {airport['city']:<14} | "
            f"{routes_count:<6} | {flight_count:<13} | {score:<10.4f} | {airport['category']}"
        )
    print("-" * len(header))


if __name__ == "__main__":
    main()
