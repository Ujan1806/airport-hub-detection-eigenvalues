"""
visualizer.py
=============
Publication-Quality Visual Dashboard Generator for Airport Hub Detection.
Produces high-resolution figures:
1. Flight Network Graph (Route thickness scaled by daily flights, nodes scaled by centrality)
2. Dual Matrix Heatmaps (Binary Adjacency A vs. Weighted Frequency W)
3. Three-Way Centrality Comparison (Topology vs Flight Volume vs Degree)
4. Dual Eigenvalue Spectrum & Power Iteration Convergence
5. Multi-Panel Executive Presentation Dashboard
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.network_model import AirportNetwork
from src.spectral_analysis import SpectralHubDetector


class HubVisualizer:
    """Generates modern, presentation-grade visualizations for the linear algebra project."""

    def __init__(self, detector: SpectralHubDetector, output_dir: Path = None):
        self.detector = detector
        self.network = detector.network
        if output_dir is None:
            output_dir = PROJECT_ROOT / "figures"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style preferences for academic and executive clarity
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#CBD5E1'
        plt.rcParams['axes.linewidth'] = 0.9

    def plot_network_graph(self, save_path: Optional[Path] = None, ax: Optional[plt.Axes] = None) -> Path:
        """
        Visualizes the Indian flight transit network.
        - Edge widths are proportional to daily flight frequency (e.g. DEL-BOM is thickest).
        - Nodes are sized and colored according to Weighted Eigenvector Centrality.
        - Coordinates follow actual airport geographic coordinates.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(12, 10), facecolor="#F8FAFC")
        else:
            fig = ax.figure

        G = self.network.graph
        df = self.detector.results_df.set_index("code")

        pos = {
            code: (self.network.code_to_meta[code]["lon"], self.network.code_to_meta[code]["lat"])
            for code in self.network.airport_codes
        }

        # Centrality scores for node sizing and coloring
        w_scores = np.array([df.loc[code, "weighted_eigen_score"] for code in self.network.airport_codes])
        node_sizes = 700 + (w_scores ** 1.6) * 3200

        # Draw flight routes (edges) with width scaled by daily flight count
        edges = G.edges(data=True)
        flight_counts = [d.get("daily_flights", 1) for _, _, d in edges]
        max_flights = max(flight_counts)
        edge_widths = [1.2 + (f / max_flights) * 6.5 for f in flight_counts]

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#3B82F6",
            width=edge_widths,
            alpha=0.55,
            style="solid"
        )

        # Draw airport nodes
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=w_scores,
            cmap=plt.cm.plasma,
            edgecolors="#0F172A",
            linewidths=2.0,
            alpha=0.95
        )

        # Labels with airport code, city, and daily flights
        for code, (x, y) in pos.items():
            w_score = df.loc[code, "weighted_eigen_score"]
            city = self.network.code_to_meta[code]["city"]
            flights = df.loc[code, "daily_flights"]
            ax.text(
                x, y + 0.58,
                f"{code}\n{city}\n({flights} flt/d | {w_score:.2f})",
                fontsize=8.5,
                fontweight='bold' if w_score > 0.8 else 'normal',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", alpha=0.92, edgecolor="#94A3B8", linewidth=0.8)
            )

        cbar = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Weighted Eigenvector Centrality Score", fontsize=9.5, fontweight='bold', labelpad=8)
        cbar.ax.tick_params(labelsize=8.5)

        ax.set_title(
            "India Aviation Transit Network: Flight Frequency & Eigenvector Hubs",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0F172A"
        )
        ax.set_xlabel("Longitude (°E)", fontsize=9, color="#475569")
        ax.set_ylabel("Latitude (°N)", fontsize=9, color="#475569")
        ax.set_facecolor("#F8FAFC")
        ax.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")

        if standalone:
            plt.tight_layout()
            if save_path is None:
                save_path = self.output_dir / "01_network_graph.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return None

    def plot_adjacency_heatmap(self, save_path: Optional[Path] = None, ax: Optional[plt.Axes] = None) -> Path:
        """
        Visualizes the matrices:
        - When standalone: Creates side-by-side heatmaps of A (Binary) and W (Flight Counts).
        - When embedded in multi-panel: Displays the Weighted Matrix W with full flight counts.
        """
        standalone = ax is None
        if standalone:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor="#F8FAFC")
        else:
            fig = ax.figure
            ax2 = ax

        df_A = self.network.adjacency_df
        df_W = self.network.weighted_df

        if standalone:
            # Heatmap 1: Binary Adjacency A
            cmap_A = sns.color_palette(["#F1F5F9", "#2563EB"], as_cmap=True)
            sns.heatmap(
                df_A,
                ax=ax1,
                cmap=cmap_A,
                cbar=False,
                annot=True,
                fmt="d",
                linewidths=1.0,
                linecolor="#CBD5E1",
                square=True,
                annot_kws={"fontsize": 9, "fontweight": "bold"}
            )
            ax1.set_title("Binary Adjacency Matrix A (Topological Routes)", fontsize=11.5, fontweight="bold", pad=10)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8.5)

        # Heatmap 2: Weighted Frequency W
        cmap_W = sns.cubehelix_palette(start=2.8, rot=0.1, dark=0.15, light=0.96, reverse=False, as_cmap=True)
        sns.heatmap(
            df_W,
            ax=ax2,
            cmap="YlGnBu",
            cbar=True,
            annot=True,
            fmt="d",
            linewidths=1.0,
            linecolor="#CBD5E1",
            square=True,
            annot_kws={"fontsize": 8.5, "fontweight": "bold"},
            cbar_kws={"label": "Daily Flights between Airports"}
        )
        ax2.set_title("Weighted Adjacency Matrix W (Daily Flight Frequency)", fontsize=11.5, fontweight="bold", pad=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8.5)

        if standalone:
            plt.tight_layout()
            if save_path is None:
                save_path = self.output_dir / "02_adjacency_heatmap.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return None

    def plot_centrality_ranking(self, save_path: Optional[Path] = None, ax: Optional[plt.Axes] = None) -> Path:
        """
        Contrasts:
        - Topological Centrality (Unweighted A)
        - Operational Capacity Centrality (Weighted W)
        - Total Daily Flights
        Shows how flight volume elevates true transit hubs.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(11, 7), facecolor="#F8FAFC")
        else:
            fig = ax.figure

        df = self.detector.results_df.copy()
        df = df.sort_values(by="weighted_eigen_score", ascending=True)

        y_pos = np.arange(len(df))
        height = 0.35

        # Weighted Eigenvector score bars
        bars1 = ax.barh(
            y_pos + height / 2,
            df["weighted_eigen_score"],
            height=height,
            color="#7C3AED",  # Deep Purple
            edgecolor="#4C1D95",
            linewidth=1.0,
            label="Weighted Hub Score (Flight Volume W)",
            alpha=0.92
        )

        # Unweighted Eigenvector score bars
        bars2 = ax.barh(
            y_pos - height / 2,
            df["unweighted_eigen_score"],
            height=height,
            color="#0284C7",  # Sky Blue
            edgecolor="#0369A1",
            linewidth=1.0,
            label="Unweighted Hub Score (Topology A)",
            alpha=0.75
        )

        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row["weighted_eigen_score"] + 0.02, y_pos[i] + height / 2, f"{row['weighted_eigen_score']:.2f} ({row['daily_flights']} flt)", va="center", fontsize=8, fontweight="bold", color="#581C87")
            ax.text(row["unweighted_eigen_score"] + 0.02, y_pos[i] - height / 2, f"{row['unweighted_eigen_score']:.2f}", va="center", fontsize=8, color="#075985")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['code']} - {row['city']}" for _, row in df.iterrows()], fontsize=9, fontweight="bold")
        ax.set_xlabel("Normalized Centrality Score (Max = 1.00)", fontsize=9.5, fontweight="bold", color="#0F172A")
        ax.set_title("Hub Centrality Comparison: Topological Routes vs. Daily Flight Volume", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0, 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.35, color="#94A3B8")
        ax.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
        ax.set_facecolor("#F8FAFC")

        if standalone:
            plt.tight_layout()
            if save_path is None:
                save_path = self.output_dir / "03_centrality_ranking.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return None

    def plot_eigenvalue_spectrum(self, save_path: Optional[Path] = None, ax: Optional[plt.Axes] = None) -> Path:
        """
        Visualizes the Eigenvalue Spectrum of both matrices and Power Iteration convergence.
        """
        standalone = ax is None
        if standalone:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="#F8FAFC")
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        u_evals = self.detector.unweighted["eigenvalues"]
        w_evals = self.detector.weighted["eigenvalues"]
        x_idx = np.arange(1, len(u_evals) + 1)

        # Bar chart of eigenvalues
        ax1.bar(
            x_idx - 0.18, u_evals, width=0.35, color="#0284C7", edgecolor="#0369A1",
            label=f"Unweighted A ($\\lambda_1={u_evals[0]:.2f}$, gap={self.detector.unweighted['spectral_gap']:.2f})"
        )
        ax1.bar(
            x_idx + 0.18, w_evals / (w_evals[0] / u_evals[0]), width=0.35, color="#7C3AED", edgecolor="#4C1D95",
            label=f"Weighted W (Scaled, $\\lambda_1={w_evals[0]:.1f}$, gap={self.detector.weighted['spectral_gap']:.1f})"
        )

        ax1.axhline(0, color="#0F172A", linestyle="-", linewidth=0.8)
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels([f"$\\lambda_{{{i}}}$" for i in x_idx], fontsize=8.5)
        ax1.set_xlabel("Eigenvalue Index", fontsize=9.5, fontweight="bold")
        ax1.set_ylabel("Eigenvalue Magnitude", fontsize=9.5, fontweight="bold")
        ax1.set_title("Eigenvalue Spectrum & Spectral Gap", fontsize=11.5, fontweight="bold", pad=12)
        ax1.legend(loc="upper right", fontsize=8.5)
        ax1.grid(axis="y", linestyle="--", alpha=0.35)
        ax1.set_facecolor("#F8FAFC")

        if standalone and ax2 is not None:
            # Convergence rate plot for Power Iteration
            u_hist = [h["step_error"] for h in self.detector.unweighted["power_iteration_history"]]
            w_hist = [h["step_error"] for h in self.detector.weighted["power_iteration_history"]]

            ax2.plot(range(1, len(u_hist) + 1), u_hist, marker="o", markersize=4, color="#0284C7", label=f"Unweighted A ({len(u_hist)} iters)")
            ax2.plot(range(1, len(w_hist) + 1), w_hist, marker="s", markersize=4, color="#7C3AED", label=f"Weighted W ({len(w_hist)} iters)")
            ax2.set_yscale("log")
            ax2.set_xlabel("Iteration Step (k)", fontsize=9.5, fontweight="bold")
            ax2.set_ylabel("Step Residual $||v^{(k+1)} - v^{(k)}||_2$", fontsize=9.5, fontweight="bold")
            ax2.set_title("Power Iteration Numerical Convergence (Log Scale)", fontsize=11.5, fontweight="bold", pad=12)
            ax2.grid(True, linestyle="--", alpha=0.35)
            ax2.legend(loc="upper right", fontsize=8.5)
            ax2.set_facecolor("#F8FAFC")

        if standalone:
            plt.tight_layout()
            if save_path is None:
                save_path = self.output_dir / "04_eigenvalue_spectrum.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return None

    def plot_executive_dashboard(self, save_path: Optional[Path] = None) -> Path:
        """
        Combines all key analyses into a unified 2x2 executive presentation dashboard.
        """
        fig = plt.figure(figsize=(19, 15), facecolor="#F8FAFC")
        gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.24)

        ax_net = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_rank = fig.add_subplot(gs[1, 0])
        ax_spec = fig.add_subplot(gs[1, 1])

        fig.suptitle(
            "Airport Hub Detection Executive Dashboard — Applied Linear Algebra (MFAD)",
            fontsize=17,
            fontweight="bold",
            color="#0F172A",
            y=0.97
        )

        self.plot_network_graph(ax=ax_net)
        self.plot_adjacency_heatmap(ax=ax_heat)
        self.plot_centrality_ranking(ax=ax_rank)
        self.plot_eigenvalue_spectrum(ax=ax_spec)

        if save_path is None:
            save_path = self.output_dir / "airport_hub_executive_dashboard.png"

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path


if __name__ == "__main__":
    net = AirportNetwork()
    detector = SpectralHubDetector(net)
    viz = HubVisualizer(detector)
    f1 = viz.plot_network_graph()
    f2 = viz.plot_adjacency_heatmap()
    f3 = viz.plot_centrality_ranking()
    f4 = viz.plot_eigenvalue_spectrum()
    f_dash = viz.plot_executive_dashboard()
    print("All visualizations generated successfully.")
