"""
spectral_analysis.py
====================
Core Linear Algebra Engine for Airport Hub Detection.
Computes eigenvalues, eigenvectors, dominant spectral properties,
Perron-Frobenius verified eigenvector centrality, and the power iteration method.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import networkx as nx
from src.network_model import AirportNetwork


class SpectralHubDetector:
    """Performs eigenvalue-eigenvector decomposition and centrality scoring on airport networks."""

    def __init__(self, network: AirportNetwork):
        self.network = network
        self.A = network.adjacency_matrix.copy()
        self.airport_codes = network.airport_codes
        self.n = len(self.airport_codes)

        # Spectral outputs
        self.eigenvalues: np.ndarray = np.array([])
        self.eigenvectors: np.ndarray = np.array([])
        self.dominant_eigenvalue: float = 0.0
        self.dominant_eigenvector: np.ndarray = np.array([])
        self.spectral_gap: float = 0.0

        # Normalized centrality scores
        self.max_normalized_scores: np.ndarray = np.array([])
        self.l2_normalized_scores: np.ndarray = np.array([])
        self.sum_normalized_scores: np.ndarray = np.array([])

        # Power iteration trace
        self.power_iteration_history: List[Dict] = []

        # Comprehensive metrics table
        self.results_df: pd.DataFrame = pd.DataFrame()

    def compute_eendecomposition(self) -> Tuple[float, np.ndarray]:
        """
        Calculates all eigenvalues and eigenvectors using NumPy linear algebra.
        Since A is real symmetric (A = A^T), np.linalg.eigh is utilized for optimal
        numerical stability and guaranteed real eigenvalues.
        """
        # np.linalg.eigh returns eigenvalues in ascending order
        w, v = np.linalg.eigh(self.A)

        # Sort eigenvalues and eigenvectors in descending order
        idx_desc = np.argsort(w)[::-1]
        self.eigenvalues = w[idx_desc]
        self.eigenvectors = v[:, idx_desc]

        # The dominant (largest) eigenvalue and its corresponding eigenvector
        self.dominant_eigenvalue = float(self.eigenvalues[0])
        raw_dominant_vec = self.eigenvectors[:, 0].copy()

        # Perron-Frobenius sign adjustment:
        # For a connected graph, the dominant eigenvector has strictly positive entries.
        # Numerical solvers may return the negative vector (-v); we flip signs if necessary.
        if np.sum(raw_dominant_vec) < 0:
            raw_dominant_vec = -raw_dominant_vec

        # Enforce all values positive
        raw_dominant_vec = np.abs(raw_dominant_vec)
        self.dominant_eigenvector = raw_dominant_vec

        # Spectral gap: difference between dominant and second largest eigenvalue
        # Governs the rate of convergence of random walks and power iteration
        self.spectral_gap = float(self.eigenvalues[0] - self.eigenvalues[1])

        # Normalization schemes:
        # 1. Max Normalization: highest hub score = 1.00 (user-friendly comparison)
        self.max_normalized_scores = self.dominant_eigenvector / np.max(self.dominant_eigenvector)

        # 2. L2 Normalization: Euclidean norm = 1.00
        self.l2_normalized_scores = self.dominant_eigenvector / np.linalg.norm(self.dominant_eigenvector)

        # 3. Sum Normalization: sum of scores = 1.00 (probability distribution)
        self.sum_normalized_scores = self.dominant_eigenvector / np.sum(self.dominant_eigenvector)

        return self.dominant_eigenvalue, self.dominant_eigenvector

    def run_power_iteration(self, max_iter: int = 100, tol: float = 1e-9) -> Tuple[float, np.ndarray, int]:
        """
        Implements the classical Power Iteration Algorithm from scratch.
        Demonstrates the iterative convergence towards the dominant eigenvector:
            v^(k+1) = (A * v^(k)) / ||A * v^(k)||
            lambda^(k) = (v^(k)^T * A * v^(k)) / (v^(k)^T * v^(k))  [Rayleigh Quotient]
        """
        # Initialize with a uniform positive vector (guaranteed non-orthogonal to v1)
        v = np.ones(self.n, dtype=float) / np.sqrt(self.n)
        self.power_iteration_history = []

        converged = False
        final_iter = 0

        for k in range(1, max_iter + 1):
            # Matrix-vector multiplication
            Av = self.A @ v

            # Rayleigh Quotient for eigenvalue estimate
            lam = float(v.T @ Av / (v.T @ v))

            # Normalize next iterate
            norm_Av = np.linalg.norm(Av)
            v_next = Av / norm_Av

            # Compute step error
            step_error = float(np.linalg.norm(v_next - v))

            self.power_iteration_history.append({
                "iteration": k,
                "eigenvalue_estimate": lam,
                "step_error": step_error
            })

            v = v_next
            final_iter = k

            if step_error < tol:
                converged = True
                break

        # Max-normalize the converged vector
        v_final = v / np.max(v)
        return lam, v_final, final_iter

    def verify_eigen_equation(self) -> Dict[str, float]:
        """
        Verifies the fundamental linear algebra equation Av = lambda * v.
        Computes the residual norm ||A*v - lambda*v|| to prove mathematical exactness.
        """
        Av = self.A @ self.dominant_eigenvector
        lam_v = self.dominant_eigenvalue * self.dominant_eigenvector
        residual = np.linalg.norm(Av - lam_v)
        relative_residual = residual / (self.dominant_eigenvalue * np.linalg.norm(self.dominant_eigenvector))

        return {
            "dominant_eigenvalue": round(self.dominant_eigenvalue, 6),
            "residual_norm": float(residual),
            "relative_residual": float(relative_residual),
            "is_valid": residual < 1e-6
        }

    def build_comparison_dataframe(self) -> pd.DataFrame:
        """
        Constructs a comprehensive comparison DataFrame contrasting:
        - Eigenvector Centrality (Our Linear Algebra calculation)
        - NetworkX Eigenvector Centrality (Reference validation)
        - Degree Centrality & Raw Connection Count (Naive counting)
        - Betweenness Centrality (Shortest path bridges)
        - Closeness Centrality (Average travel distance)
        """
        if len(self.dominant_eigenvector) == 0:
            self.compute_eendecomposition()

        # Compute graph centralities via NetworkX
        deg_centrality = nx.degree_centrality(self.network.graph)
        nx_eigen_centrality = nx.eigenvector_centrality(self.network.graph, max_iter=1000)
        betweenness = nx.betweenness_centrality(self.network.graph)
        closeness = nx.closeness_centrality(self.network.graph)

        data = []
        for i, code in enumerate(self.airport_codes):
            meta = self.network.code_to_meta[code]
            connections = int(np.sum(self.A[i, :]))

            data.append({
                "code": code,
                "city": meta["city"],
                "airport_name": meta["name"],
                "category": meta["category"],
                "connections": connections,
                "eigenvector_score": round(float(self.max_normalized_scores[i]), 4),
                "l2_score": round(float(self.l2_normalized_scores[i]), 4),
                "sum_share_pct": round(float(self.sum_normalized_scores[i] * 100), 2),
                "nx_eigen_val": round(float(nx_eigen_centrality[code]), 4),
                "degree_centrality": round(float(deg_centrality[code]), 4),
                "betweenness": round(float(betweenness[code]), 4),
                "closeness": round(float(closeness[code]), 4)
            })

        df = pd.DataFrame(data)

        # Compute rank orders
        df["eigen_rank"] = df["eigenvector_score"].rank(ascending=False, method="min").astype(int)
        df["degree_rank"] = df["connections"].rank(ascending=False, method="min").astype(int)
        df["rank_shift"] = df["degree_rank"] - df["eigen_rank"]  # Positive means boosted by quality connections

        # Sort by Eigenvector Centrality descending
        df = df.sort_values(by="eigenvector_score", ascending=False).reset_index(drop=True)
        self.results_df = df
        return df

    def save_results(self, filepath: str = None) -> Path:
        """Saves calculated centrality and ranking results to CSV."""
        if filepath is None:
            filepath = self.network.data_dir / "centrality_results.csv"
        self.results_df.to_csv(filepath, index=False)
        return Path(filepath)


if __name__ == "__main__":
    net = AirportNetwork()
    detector = SpectralHubDetector(net)
    lam, v = detector.compute_eendecomposition()
    verify = detector.verify_eigen_equation()
    print("=" * 60)
    print(f"Dominant Eigenvalue (lambda_max): {lam:.6f}")
    print(f"Spectral Gap (lambda_1 - lambda_2): {detector.spectral_gap:.6f}")
    print(f"Residual Check ||Av - lambda*v||: {verify['residual_norm']:.2e} (Valid: {verify['is_valid']})")
    print("=" * 60)

    # Run power iteration
    lam_pi, v_pi, iters = detector.run_power_iteration()
    print(f"Power Iteration Converged in {iters} iterations to lambda = {lam_pi:.6f}")

    df = detector.build_comparison_dataframe()
    print("\n--- Airport Hub Centrality Ranking ---")
    print(df[["eigen_rank", "code", "city", "connections", "eigenvector_score", "rank_shift"]])
