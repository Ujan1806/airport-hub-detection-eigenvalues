"""
network_model.py
================
Constructs the Airport Flight Network graph and generates the mathematical
adjacency matrix A representation for linear algebra analysis.
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


class AirportNetwork:
    """Represents an airport flight route network as a mathematical graph and adjacency matrix."""

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
        self.adjacency_df: pd.DataFrame = pd.DataFrame()

        self._load_data()
        self._build_graph()
        self._build_adjacency_matrix()

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
        """Constructs a NetworkX undirected graph from nodes and edges."""
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

        # Add flight route edges (undirected bidirectional routes)
        for r in self.routes:
            u, v = r["origin"], r["destination"]
            if u in self.code_to_meta and v in self.code_to_meta:
                self.graph.add_edge(u, v, description=r.get("description", ""))

    def _build_adjacency_matrix(self):
        """
        Converts the graph into a symmetric binary Adjacency Matrix A.
        A[i, j] = 1 if airports i and j have a direct flight, else 0.
        """
        n = len(self.airport_codes)
        A = np.zeros((n, n), dtype=float)

        for u, v in self.graph.edges():
            i = self.code_to_idx[u]
            j = self.code_to_idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0  # Symmetric: bidirectional flight connection

        self.adjacency_matrix = A
        self.adjacency_df = pd.DataFrame(
            A.astype(int),
            index=self.airport_codes,
            columns=self.airport_codes
        )

    def save_adjacency_matrix(self, filepath: Path = None) -> Path:
        """Saves the adjacency matrix as a CSV file."""
        if filepath is None:
            filepath = self.data_dir / "adjacency_matrix.csv"
        self.adjacency_df.to_csv(filepath)
        return filepath

    def get_network_summary(self) -> Dict:
        """Computes fundamental graph-theoretic and linear algebraic network properties."""
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        is_conn = nx.is_connected(self.graph)
        density = nx.density(self.graph)
        degrees = dict(self.graph.degree())
        avg_deg = sum(degrees.values()) / n if n > 0 else 0

        # Matrix properties
        is_symmetric = np.allclose(self.adjacency_matrix, self.adjacency_matrix.T)
        trace_val = np.trace(self.adjacency_matrix)

        return {
            "num_airports": n,
            "num_routes": m,
            "is_connected": is_conn,
            "graph_density": round(density, 4),
            "average_degree": round(avg_deg, 2),
            "is_matrix_symmetric": is_symmetric,
            "matrix_trace": int(trace_val),
            "degrees": degrees
        }


if __name__ == "__main__":
    net = AirportNetwork()
    summary = net.get_network_summary()
    print("Airport Network Loaded:")
    print(f"Airports: {summary['num_airports']}, Flight Routes: {summary['num_routes']}")
    print(f"Graph Connected: {summary['is_connected']}")
    print(f"Matrix Symmetric: {summary['is_symmetric']} (Trace = {summary['matrix_trace']})")
    print("\nAdjacency Matrix A:")
    print(net.adjacency_df)
