"""
network_model.py
================
Constructs the Airport Flight Network graph and generates linear algebraic
matrix representations:
1. Binary Adjacency Matrix A (Topological route connectivity)
2. Weighted Frequency Matrix W (Daily flight volume and capacity)
3. Space Structure Analysis: Rank, Nullity, Basis vectors, and Subspaces
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import null_space, qr


class AirportNetwork:
    """Represents an airport flight network as graph and matrix models (Unweighted & Weighted)."""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.airports: List[Dict] = []
        self.routes: List[Dict] = []
        self.airport_codes: List[str] = []
        self.code_to_meta: Dict[str, Dict] = {}
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_code: Dict[int, str] = {}
        self.graph: nx.Graph = nx.Graph()
        self.adjacency_matrix: np.ndarray = np.array([])
        self.weighted_matrix: np.ndarray = np.array([])
        self.adjacency_df: pd.DataFrame = pd.DataFrame()
        self.weighted_df: pd.DataFrame = pd.DataFrame()

        self._load_data()
        self._build_graph()
        self._build_matrices()

    def _load_data(self):
        """Loads airport metadata and flight route connections from data directory."""
        airports_file = self.data_dir / "airports.json"
        routes_file = self.data_dir / "routes.json"

        if not airports_file.exists() or not routes_file.exists():
            raise FileNotFoundError(f"Airport or route data missing in {self.data_dir}")

        with open(airports_file, "r", encoding="utf-8") as f:
            self.airports = json.load(f)

        with open(routes_file, "r", encoding="utf-8") as f:
            self.routes = json.load(f)

        self.airport_codes = [a["code"] for a in self.airports]
        self.code_to_meta = {a["code"]: a for a in self.airports}
        self.code_to_idx = {code: i for i, code in enumerate(self.airport_codes)}
        self.idx_to_code = {i: code for i, code in enumerate(self.airport_codes)}

    def _build_graph(self):
        """Constructs a NetworkX undirected graph with node and edge attributes."""
        self.graph = nx.Graph()

        # Add airport nodes with geographical and metadata attributes
        for a in self.airports:
            self.graph.add_node(
                a["code"],
                name=a["name"],
                city=a["city"],
                state=a["state"],
                lat=a["lat"],
                lon=a["lon"],
                category=a["category"],
            )

        # Add flight route edges with daily flights and distance
        for r in self.routes:
            u, v = r["origin"], r["destination"]
            daily_flights = r.get("daily_flights", 1)
            dist = r.get("distance_km", 0)
            if u in self.code_to_meta and v in self.code_to_meta:
                self.graph.add_edge(
                    u, v,
                    daily_flights=daily_flights,
                    weight=daily_flights,
                    distance_km=dist,
                    description=r.get("description", "")
                )

    def _build_matrices(self):
        """
        Builds both the binary Adjacency Matrix A and the Weighted Frequency Matrix W.
        - A[i, j] = 1 if route exists, else 0
        - W[i, j] = daily_flights if route exists, else 0
        """
        n = len(self.airport_codes)
        A = np.zeros((n, n), dtype=float)
        W = np.zeros((n, n), dtype=float)

        for u, v, data in self.graph.edges(data=True):
            i = self.code_to_idx[u]
            j = self.code_to_idx[v]
            flights = float(data.get("daily_flights", 1.0))

            # Symmetric bidirectional route modeling
            A[i, j] = 1.0
            A[j, i] = 1.0

            W[i, j] = flights
            W[j, i] = flights

        self.adjacency_matrix = A
        self.weighted_matrix = W

        self.adjacency_df = pd.DataFrame(
            A.astype(int),
            index=self.airport_codes,
            columns=self.airport_codes
        )
        self.weighted_df = pd.DataFrame(
            W.astype(int),
            index=self.airport_codes,
            columns=self.airport_codes
        )

    def save_matrices(self) -> Tuple[Path, Path]:
        """Saves both adjacency and weighted matrices as CSV files."""
        unweighted_path = self.data_dir / "adjacency_matrix.csv"
        weighted_path = self.data_dir / "weighted_adjacency_matrix.csv"

        self.adjacency_df.to_csv(unweighted_path)
        self.weighted_df.to_csv(weighted_path)

        return unweighted_path, weighted_path

    def analyze_vector_space_structure(self) -> Dict:
        """
        Analyzes the structure of the space (Vector Space -> Subspaces -> Basis -> Rank & Nullity):
        - Column Space Dimension = Rank(M)
        - Null Space Dimension = Nullity(M)
        - Rank-Nullity Theorem: Rank(M) + Nullity(M) = n
        - Extracts basis vectors for Column Space and Null Space
        """
        n = len(self.airport_codes)

        # Unweighted Matrix A
        rank_A = int(np.linalg.matrix_rank(self.adjacency_matrix))
        nullity_A = n - rank_A
        null_basis_A = null_space(self.adjacency_matrix)

        # Weighted Matrix W
        rank_W = int(np.linalg.matrix_rank(self.weighted_matrix))
        nullity_W = n - rank_W
        null_basis_W = null_space(self.weighted_matrix)

        # Basis of Column Space via pivoted QR decomposition
        Q_A, R_A, P_A = qr(self.adjacency_matrix, mode="full", pivoting=True)
        col_basis_indices_A = [self.airport_codes[i] for i in P_A[:rank_A]]

        return {
            "dimension_n": n,
            "unweighted": {
                "rank": rank_A,
                "nullity": nullity_A,
                "rank_nullity_verified": (rank_A + nullity_A == n),
                "col_space_basis_airports": col_basis_indices_A,
                "has_non_trivial_nullspace": nullity_A > 0,
                "null_vector": null_basis_A[:, 0].tolist() if nullity_A > 0 else []
            },
            "weighted": {
                "rank": rank_W,
                "nullity": nullity_W,
                "rank_nullity_verified": (rank_W + nullity_W == n),
                "has_non_trivial_nullspace": nullity_W > 0
            }
        }

    def get_network_summary(self) -> Dict:
        """Computes comprehensive graph-theoretic and operational network metrics."""
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        is_conn = nx.is_connected(self.graph)
        density = nx.density(self.graph)
        degrees = dict(self.graph.degree())
        avg_deg = sum(degrees.values()) / n if n > 0 else 0

        # Weighted operational metrics
        total_daily_flights = sum(nx.get_edge_attributes(self.graph, "daily_flights").values())
        busiest_edge = max(self.graph.edges(data=True), key=lambda x: x[2].get("daily_flights", 0))

        # Node total daily flights
        airport_daily_flights = {
            code: int(np.sum(self.weighted_matrix[self.code_to_idx[code], :]))
            for code in self.airport_codes
        }

        # Linear algebra matrix properties
        is_A_sym = bool(np.allclose(self.adjacency_matrix, self.adjacency_matrix.T))
        is_W_sym = bool(np.allclose(self.weighted_matrix, self.weighted_matrix.T))
        trace_A = int(np.trace(self.adjacency_matrix))
        trace_W = int(np.trace(self.weighted_matrix))

        return {
            "num_airports": n,
            "num_routes": m,
            "total_daily_flights": total_daily_flights,
            "busiest_route": f"{busiest_edge[0]} <-> {busiest_edge[1]} ({busiest_edge[2]['daily_flights']} flights/day)",
            "is_connected": is_conn,
            "graph_density": round(density, 4),
            "average_degree": round(avg_deg, 2),
            "is_matrix_symmetric": is_A_sym and is_W_sym,
            "matrix_trace": trace_A,
            "weighted_matrix_trace": trace_W,
            "degrees": degrees,
            "airport_daily_flights": airport_daily_flights
        }


if __name__ == "__main__":
    net = AirportNetwork()
    summary = net.get_network_summary()
    space = net.analyze_vector_space_structure()
    print("Airport Network Loaded:")
    print(f"Airports: {summary['num_airports']}, Routes: {summary['num_routes']}, Total Flights: {summary['total_daily_flights']}")
    print(f"Busiest Route: {summary['busiest_route']}")
    print(f"Space Analysis (Unweighted): Rank = {space['unweighted']['rank']}, Nullity = {space['unweighted']['nullity']}")
    print(f"Rank-Nullity Theorem: Rank + Nullity = {space['unweighted']['rank'] + space['unweighted']['nullity']} == {space['dimension_n']}")
