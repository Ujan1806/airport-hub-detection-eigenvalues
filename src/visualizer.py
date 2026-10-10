"""
visualizer.py
=============
Visual Dashboard Generator for Airport Hub Detection.
Produces high-resolution figures:
1. Airport Flight Network Graph (Eigenvector Centrality scaled)
2. Adjacency Matrix Heatmap
3. Airport Centrality Ranking & Comparison Charts
4. Eigenvalue Spectrum & Power Iteration Convergence
5. Unified Multi-Panel Executive Dashboard
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
    """Generates publication-quality visualizations for the linear algebra airport model."""

    def __init__(self, detector: SpectralHubDetector, output_dir: Path = None):
        self.detector = detector
        self.network = detector.network
        if output_dir is None:
            output_dir = PROJECT_ROOT / "figures"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style preferences
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#CCCCCC'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_network_graph(self, save_path: Optional[Path] = None, ax: Optional[plt.Axes] = None) -> Path:
        """
        Visualizes the flight route network.
        Nodes are scaled and colored according to their Eigenvector Centrality score.
        Positions correspond to geographic coordinates (lon, lat) of the airports.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(11, 9), facecolor="#F8FAFC")
        else:
            fig = ax.figure

        G = self.network.graph
        df = self.detector.results_df.set_index("code")

        # Position using geographic coordinates (lon = x, lat = y)
        pos = {
            code: (self.network.code_to_meta[code]["lon"], self.network.code_to_meta[code]["lat"])
            for code in self.network.airport_codes
        }

        # Centrality scores for node sizing and color mapping
        scores = np.array([df.loc[code, "eigenvector_score"] for code in self.network.airport_codes])
        node_sizes = 800 + (scores ** 1.5) * 2600  # Non-linear scaling for contrast

        # Draw flight routes (edges)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#94A3B8",
            width=1.6,
            alpha=0.65,
            style="solid"
        )

        # Draw airport nodes
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=scores,
            cmap=plt.cm.plasma,
            edgecolors="#1E293B",
            linewidths=1.8,
            alpha=0.92
        )

        # Draw labels with IATA code and City name
        labels = {
            code: f"{code}\n({self.network.code_to_meta[code]['city']})"
            for code in self.network.airport_codes
        }

        # Adjust label offsets for clarity
        for code, (x, y) in pos.items():
            score = df.loc[code, "eigenvector_score"]
            city = self.network.code_to_meta[code]["city"]
            ax.text(
                x, y + 0.55,
                f"{code}\n{city}\n[{score:.2f}]",
                fontsize=8.5,
                fontweight='bold' if score > 0.8 else 'normal',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#CBD5E1", linewidth=0.7)
            )

        # Add colorbar
        cbar = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Eigenvector Centrality Score (Normalized)", fontsize=10, fontweight='bold', labelpad=8)
        cbar.ax.tick_params(labelsize=9)

        ax.set_title(
            "Airport Flight Network Graph (Eigenvector Centrality)",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0F172A"
        )
        ax.set_xlabel("Longitude (°E)", fontsize=9, color="#475569")
        ax.set_ylabel("Latitude (°N)", fontsize=9, color="#475569")
        ax.set_facecolor("#F8FAFC")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

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
        Visualizes the Adjacency Matrix A as an annotated heatmap.
        Highlights symmetric bidirectional routes (1) and non-connections (0).
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#F8FAFC")
        else:
            fig = ax.figure

        df_A = self.network.adjacency_df

        # Custom colormap: 0 = light grey-blue, 1 = deep navy
        cmap = sns.color_palette(["#F1F5F9", "#1E40AF"], as_cmap=True)

        sns.heatmap(
            df_A,
            ax=ax,
            cmap=cmap,
            cbar=False,
            annot=True,
            fmt="d",
            linewidths=1.2,
            linecolor="#CBD5E1",
            square=True,
            annot_kws={"fontsize": 10, "fontweight": "bold"}
        )

        ax.set_title(
            f"Adjacency Matrix A ({len(df_A)}×{len(df_A)} Symmetric)",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0F172A"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9, fontweight="bold")

        # Custom legend patch
        patch_0 = mpatches.Patch(color="#F1F5F9", label="0 = No Direct Flight")
        patch_1 = mpatches.Patch(color="#1E40AF", label="1 = Direct Route")
        ax.legend(handles=[patch_1, patch_0], loc="upper right", bbox_to_anchor=(1.35, 1.0), frameon=True)

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
        Displays a horizontal ranking bar chart comparing:
        Eigenvector Centrality vs Degree Centrality (Connection Count).
        Clearly visualizes structural prestige vs naive counting.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#F8FAFC")
        else:
            fig = ax.figure

        df = self.detector.results_df.copy()
        df = df.sort_values(by="eigenvector_score", ascending=True)  # ascending for horizontal bars

        y_pos = np.arange(len(df))
        height = 0.42

        # Color mapping based on eigenvector score
        colors = plt.cm.plasma(df["eigenvector_score"])

        # Eigenvector Centrality bars
        bars1 = ax.barh(
            y_pos + height / 2,
            df["eigenvector_score"],
            height=height,
            color=colors,
            edgecolor="#1E293B",
            linewidth=1.0,
            label="Eigenvector Centrality (Score)",
            alpha=0.95
        )

        # Degree Centrality (normalized to max degree for visual comparison)
        max_deg = df["connections"].max()
        deg_scaled = df["connections"] / max_deg
        bars2 = ax.barh(
            y_pos - height / 2,
            deg_scaled,
            height=height,
            color="#94A3B8",
            edgecolor="#475569",
            linewidth=1.0,
            label="Degree Centrality (Scaled Routes)",
            alpha=0.65
        )

        # Value annotations on bars
        for i, (score, deg, count) in enumerate(zip(df["eigenvector_score"], deg_scaled, df["connections"])):
            ax.text(score + 0.02, y_pos[i] + height / 2, f"{score:.2f}", va="center", fontsize=8.5, fontweight="bold")
            ax.text(deg + 0.02, y_pos[i] - height / 2, f"{count} routes", va="center", fontsize=8, color="#475569")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['code']} - {row['city']}" for _, row in df.iterrows()], fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Normalized Score (Max = 1.00)", fontsize=10, fontweight="bold", color="#0F172A")
        ax.set_title(
            "Airport Importance: Eigenvector Centrality vs Route Count",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0F172A"
        )
        ax.set_xlim(0, 1.22)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#94A3B8")
        ax.legend(loc="lower right", framealpha=0.95)
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
        Visualizes the full Eigenvalue Spectrum (lambda_1 to lambda_n).
        Highlights the dominant eigenvalue lambda_max and the spectral gap.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#F8FAFC")
        else:
            fig = ax.figure

        evals = self.detector.eigenvalues
        x_indices = np.arange(1, len(evals) + 1)

        # Highlight dominant eigenvalue with distinctive gold/ruby color
        bar_colors = ["#DC2626" if i == 0 else "#2563EB" if val > 0 else "#64748B" for i, val in enumerate(evals)]

        bars = ax.bar(x_indices, evals, color=bar_colors, edgecolor="#0F172A", linewidth=1.1, width=0.65, alpha=0.9)

        # Annotate dominant eigenvalue
        ax.annotate(
            f"Dominant $\\lambda_1 = {evals[0]:.2f}$\n(Governs Centrality)",
            xy=(1, evals[0]),
            xytext=(1.8, evals[0] - 0.2),
            arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=7),
            fontsize=9.5,
            fontweight="bold",
            color="#991B1B"
        )

        # Annotate spectral gap
        gap = self.detector.spectral_gap
        ax.text(
            0.65, 0.85,
            f"Spectral Gap ($\\lambda_1 - \\lambda_2$): {gap:.2f}\nTrace $\\sum \\lambda_i = {np.sum(evals):.1f}$",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.0)
        )

        ax.axhline(0, color="#0F172A", linestyle="-", linewidth=0.8)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"$\\lambda_{{{i}}}$" for i in x_indices], fontsize=9)
        ax.set_xlabel("Eigenvalue Index", fontsize=10, fontweight="bold", color="#0F172A")
        ax.set_ylabel("Eigenvalue Magnitude ($\\lambda$)", fontsize=10, fontweight="bold", color="#0F172A")
        ax.set_title(
            "Eigenvalue Spectrum of Airport Adjacency Matrix",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0F172A"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#94A3B8")
        ax.set_facecolor("#F8FAFC")

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
        Combines all four core visualizations into a unified 2x2 presentation dashboard.
        Ideal for reports, printouts, and project defense presentations.
        """
        fig = plt.figure(figsize=(18, 14), facecolor="#F8FAFC")
        gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

        ax_net = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_rank = fig.add_subplot(gs[1, 0])
        ax_spec = fig.add_subplot(gs[1, 1])

        # Header title
        fig.suptitle(
            "Airport Hub Detection Dashboard — Linear Algebra & Eigenvector Centrality",
            fontsize=18,
            fontweight="bold",
            color="#0F172A",
            y=0.97
        )

        # Plot individual panels
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
    detector.compute_eendecomposition()
    detector.build_comparison_dataframe()

    viz = HubVisualizer(detector)
    f1 = viz.plot_network_graph()
    f2 = viz.plot_adjacency_heatmap()
    f3 = viz.plot_centrality_ranking()
    f4 = viz.plot_eigenvalue_spectrum()
    f_dash = viz.plot_executive_dashboard()

    print("Visualizations created successfully:")
    print(f"1. Network Graph: {f1}")
    print(f"2. Heatmap: {f2}")
    print(f"3. Ranking: {f3}")
    print(f"4. Spectrum: {f4}")
    print(f"5. Executive Dashboard: {f_dash}")
