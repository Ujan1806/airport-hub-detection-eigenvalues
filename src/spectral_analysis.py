"""
spectral_analysis.py
====================
Core Linear Algebra Engine for Airport Hub Detection.
Computes:
1. Spectral Decomposition for both Unweighted (A) and Weighted (W) matrices.
2. Perron-Frobenius Verified Dominant Eigenvectors (Centrality Rankings).
3. Spectral Theorem & Orthonormal Eigenbasis (Q^T Q = I, A = Q Lambda Q^T).
4. Power Iteration Algorithm with Rayleigh Quotient.
5. In-depth Hub Comparison (Topological Prestige vs. Flight Volume Capacity).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import networkx as nx
from src.network_model import AirportNetwork


class SpectralHubDetector:
    """Performs dual spectral analysis (Unweighted Topology vs Weighted Flight Volume)."""

    def __init__(self, network: AirportNetwork):
        self.network = network
        self.A = network.adjacency_matrix.copy()
        self.W = network.weighted_matrix.copy()
        self.airport_codes = network.airport_codes
        self.n = len(self.airport_codes)

        # Spectral outputs container for unweighted (A) and weighted (W)
        self.unweighted: Dict = {}
        self.weighted: Dict = {}

        # Master results DataFrame
        self.results_df: pd.DataFrame = pd.DataFrame()

        # Run dual spectral analysis on initialization
        self.compute_all_spectral_analyses()

    # Convenience properties for backward compatibility
    @property
    def eigenvalues(self) -> np.ndarray:
        return self.unweighted.get("eigenvalues", np.array([]))

    @property
    def dominant_eigenvalue(self) -> float:
        return self.unweighted.get("dominant_eigenvalue", 0.0)

    @property
    def dominant_eigenvector(self) -> np.ndarray:
        return self.unweighted.get("dominant_eigenvector", np.array([]))

    @property
    def spectral_gap(self) -> float:
        return self.unweighted.get("spectral_gap", 0.0)

    @property
    def max_normalized_scores(self) -> np.ndarray:
        return self.unweighted.get("max_normalized_scores", np.array([]))

    @property
    def l2_normalized_scores(self) -> np.ndarray:
        return self.unweighted.get("l2_normalized_scores", np.array([]))

    @property
    def sum_normalized_scores(self) -> np.ndarray:
        return self.unweighted.get("sum_normalized_scores", np.array([]))

    @property
    def power_iteration_history(self) -> List[Dict]:
        return self.unweighted.get("power_iteration_history", [])

    def _decompose_symmetric_matrix(self, M: np.ndarray, label: str) -> Dict:
        """
        Calculates all eigenvalues and eigenvectors using np.linalg.eigh.
        Guarantees strictly real eigenvalues and an orthonormal eigenbasis Q.
        """
        w, v = np.linalg.eigh(M)

        # Sort descending by eigenvalue magnitude
        idx_desc = np.argsort(w)[::-1]
        eigenvalues = w[idx_desc]
        eigenvectors = v[:, idx_desc]  # Columns are orthonormal eigenvectors

        dominant_val = float(eigenvalues[0])
        dominant_vec = eigenvectors[:, 0].copy()

        # Perron-Frobenius sign adjustment:
        # For non-negative connected graphs, dominant eigenvector has strictly positive entries.
        if np.sum(dominant_vec) < 0:
            dominant_vec = -dominant_vec
        dominant_vec = np.abs(dominant_vec)

        # Spectral gap
        spectral_gap = float(eigenvalues[0] - eigenvalues[1])

        # Normalization schemes
        max_norm = dominant_vec / np.max(dominant_vec)
        l2_norm = dominant_vec / np.linalg.norm(dominant_vec)
        sum_norm = dominant_vec / np.sum(dominant_vec)

        # Eigen-equation residual: ||M*v - lambda*v||_2
        residual = float(np.linalg.norm(M @ dominant_vec - dominant_val * dominant_vec))

        # Spectral Theorem verification: Q^T * Q = I and M = Q * Lambda * Q^T
        Q = eigenvectors
        I_check = np.allclose(Q.T @ Q, np.eye(self.n), atol=1e-12)
        diag_recon = Q @ np.diag(eigenvalues) @ Q.T
        recon_check = np.allclose(M, diag_recon, atol=1e-12)

        return {
            "label": label,
            "matrix": M,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "dominant_eigenvalue": dominant_val,
            "dominant_eigenvector": dominant_vec,
            "spectral_gap": spectral_gap,
            "max_normalized_scores": max_norm,
            "l2_normalized_scores": l2_norm,
            "sum_normalized_scores": sum_norm,
            "residual_norm": residual,
            "is_eigen_exact": residual < 1e-9,
            "is_eigenbasis_orthonormal": I_check,
            "is_spectral_decomposition_exact": recon_check,
            "power_iteration_history": []
        }

    def compute_all_spectral_analyses(self):
        """Executes full spectral decomposition on both Unweighted (A) and Weighted (W) models."""
        self.unweighted = self._decompose_symmetric_matrix(self.A, "Unweighted Topology (A)")
        self.weighted = self._decompose_symmetric_matrix(self.W, "Weighted Flight Frequency (W)")

        # Run power iterations
        self.run_power_iteration(matrix_type="unweighted")
        self.run_power_iteration(matrix_type="weighted")

        # Build master dataframe
        self.build_comparison_dataframe()

    def compute_eendecomposition(self) -> Tuple[float, np.ndarray]:
        """Backward-compatible method returning dominant eigenvalue and vector for A."""
        return self.dominant_eigenvalue, self.dominant_eigenvector

    def run_power_iteration(
        self, matrix_type: str = "unweighted", max_iter: int = 100, tol: float = 1e-9
    ) -> Tuple[float, np.ndarray, int]:
        """
        Implements the classical Power Iteration Algorithm from scratch:
            v^(k+1) = (M * v^(k)) / ||M * v^(k)||
            lambda^(k) = Rayleigh Quotient
        """
        M = self.A if matrix_type == "unweighted" else self.W
        store = self.unweighted if matrix_type == "unweighted" else self.weighted

        # Uniform positive unit vector (guaranteed non-orthogonal to principal eigenvector)
        v = np.ones(self.n, dtype=float) / np.sqrt(self.n)
        history = []
        final_iter = 0

        for k in range(1, max_iter + 1):
            Mv = M @ v
            lam = float(v.T @ Mv / (v.T @ v))
            norm_Mv = np.linalg.norm(Mv)
            v_next = Mv / norm_Mv
            step_error = float(np.linalg.norm(v_next - v))

            history.append({
                "iteration": k,
                "eigenvalue_estimate": lam,
                "step_error": step_error
            })

            v = v_next
            final_iter = k

            if step_error < tol:
                break

        v_final = v / np.max(v)
        store["power_iteration_history"] = history
        return lam, v_final, final_iter

    def verify_eigen_equation(self, matrix_type: str = "unweighted") -> Dict:
        """Verifies linear algebraic accuracy of Av = lambda*v."""
        store = self.unweighted if matrix_type == "unweighted" else self.weighted
        return {
            "dominant_eigenvalue": round(store["dominant_eigenvalue"], 6),
            "residual_norm": store["residual_norm"],
            "is_valid": store["is_eigen_exact"],
            "is_eigenbasis_orthonormal": store["is_eigenbasis_orthonormal"],
            "is_spectral_decomposition_exact": store["is_spectral_decomposition_exact"]
        }

    def build_comparison_dataframe(self) -> pd.DataFrame:
        """
        Constructs an enriched DataFrame contrasting:
        - Unweighted Eigenvector Centrality (Route Topology)
        - Weighted Eigenvector Centrality (Daily Flight Volume)
        - Degree (Routes count) & Weighted Degree (Daily Flight Count)
        - NetworkX Validations
        - Betweenness & Closeness Centrality
        """
        # Graph metrics via NetworkX
        deg_centrality = nx.degree_centrality(self.network.graph)
        betweenness = nx.betweenness_centrality(self.network.graph)
        closeness = nx.closeness_centrality(self.network.graph)

        # Weighted degree (strength)
        weighted_degrees = {
            node: sum(data.get("daily_flights", 1) for _, _, data in self.network.graph.edges(node, data=True))
            for node in self.network.airport_codes
        }

        data = []
        u_max = self.unweighted["max_normalized_scores"]
        u_l2 = self.unweighted["l2_normalized_scores"]
        w_max = self.weighted["max_normalized_scores"]
        w_l2 = self.weighted["l2_normalized_scores"]

        for i, code in enumerate(self.airport_codes):
            meta = self.network.code_to_meta[code]
            connections = int(np.sum(self.A[i, :]))
            flights = int(weighted_degrees[code])

            data.append({
                "code": code,
                "city": meta["city"],
                "airport_name": meta["name"],
                "category": meta["category"],
                "routes": connections,
                "daily_flights": flights,
                "unweighted_eigen_score": round(float(u_max[i]), 4),
                "unweighted_l2_score": round(float(u_l2[i]), 4),
                "weighted_eigen_score": round(float(w_max[i]), 4),
                "weighted_l2_score": round(float(w_l2[i]), 4),
                "degree_centrality": round(float(deg_centrality[code]), 4),
                "betweenness": round(float(betweenness[code]), 4),
                "closeness": round(float(closeness[code]), 4)
            })

        df = pd.DataFrame(data)

        # Compute rank orders
        df["unweighted_rank"] = df["unweighted_eigen_score"].rank(ascending=False, method="min").astype(int)
        df["weighted_rank"] = df["weighted_eigen_score"].rank(ascending=False, method="min").astype(int)
        df["degree_rank"] = df["routes"].rank(ascending=False, method="min").astype(int)
        df["flight_rank"] = df["daily_flights"].rank(ascending=False, method="min").astype(int)

        # Rank shift: How accounting for flight volume shifts topological rank
        df["volume_boost"] = df["unweighted_rank"] - df["weighted_rank"]

        # Sort primarily by weighted hub importance, then unweighted
        df = df.sort_values(by=["weighted_eigen_score", "unweighted_eigen_score"], ascending=False).reset_index(drop=True)
        self.results_df = df
        return df

    def save_results(self, filepath: Optional[Path] = None) -> Path:
        """Saves enriched results to CSV."""
        if filepath is None:
            filepath = self.network.data_dir / "centrality_results.csv"
        self.results_df.to_csv(filepath, index=False)
        return filepath


if __name__ == "__main__":
    net = AirportNetwork()
    detector = SpectralHubDetector(net)
    print("Spectral Analysis Completed:")
    print(f"Unweighted lambda_max: {detector.dominant_eigenvalue:.4f} (Gap: {detector.spectral_gap:.4f})")
    print(f"Weighted lambda_max  : {detector.weighted['dominant_eigenvalue']:.4f} (Gap: {detector.weighted['spectral_gap']:.4f})")
    print("\nMaster Ranking Table:")
    print(detector.results_df[["code", "city", "routes", "daily_flights", "unweighted_eigen_score", "weighted_eigen_score", "weighted_rank"]])
