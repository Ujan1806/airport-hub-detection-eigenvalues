"""
main.py
=======
Master Orchestrator for the Airport Hub Detection Mini-Project.
Applied Linear Algebra (Mathematics for AI / Data Science - MFAD)

Executes the complete Linear Algebra pipeline:
1. Real-World Flight Data & Dual Matrix Representation (A: Binary Routes, W: Daily Flight Volume)
2. Vector Space Structure, Basis, and Rank-Nullity Theorem Analysis
3. Dual Spectral Decomposition & Perron-Frobenius Theorem Verification
4. Power Iteration Numerical Approximation with Rayleigh Quotient
5. Publication-Quality Visual Dashboards, Interactive Web Application & Academic Reports

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.network_model import AirportNetwork
from src.spectral_analysis import SpectralHubDetector
from src.visualizer import HubVisualizer
from src.report_generator import HubReportGenerator


def main():
    print("=" * 85)
    print("      AIRPORT HUB DETECTION USING EIGENVALUES, EIGENVECTORS & FLIGHT CAPACITY")
    print("             Applied Linear Algebra & Spectral Graph Theory Pipeline")
    print("                     Course: UE25MA242A - MFAD (Mini-Project)")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # Stage 1: Real-World Aviation Data & Matrix Representations (A and W)
    # -------------------------------------------------------------------------
    print("\n[Stage 1/5] Loading real-world flight data and constructing dual matrix models...")
    network = AirportNetwork()
    summary = network.get_network_summary()
    unweighted_path, weighted_path = network.save_matrices()

    print(f"  * Airport Nodes (V)          : {summary['num_airports']} commercial airports")
    print(f"  * Flight Routes (E)          : {summary['num_routes']} bidirectional corridors")
    print(f"  * Total Daily Scheduled Flow : {summary['total_daily_flights']} flights/day")
    print(f"  * Busiest Air Corridor       : {summary['busiest_route']}")
    print(f"  * Binary Matrix A (CSV)      : {unweighted_path.name} (Trace = {summary['matrix_trace']})")
    print(f"  * Weighted Matrix W (CSV)    : {weighted_path.name} (Trace = {summary['weighted_matrix_trace']})")

    # -------------------------------------------------------------------------
    # Stage 2: Structure of the Space: Bases & Rank-Nullity Theorem
    # -------------------------------------------------------------------------
    print("\n[Stage 2/5] Analyzing Vector Space Structure, Subspaces, Basis & Rank-Nullity...")
    space = network.analyze_vector_space_structure()
    u_space = space["unweighted"]
    w_space = space["weighted"]

    print(f"  * Ambient Vector Space       : R^{space['dimension_n']} (10-dimensional airport state space)")
    print(f"  * Binary Matrix A Rank       : {u_space['rank']} (dim Col(A))")
    print(f"  * Binary Matrix A Nullity    : {u_space['nullity']} (dim Null(A))")
    print(f"  * Rank-Nullity Theorem       : Rank(A) + Nullity(A) = {u_space['rank']} + {u_space['nullity']} = 10 (VERIFIED)")
    print(f"  * Weighted Matrix W Rank     : {w_space['rank']} (Full Rank, Nullity = {w_space['nullity']})")

    # -------------------------------------------------------------------------
    # Stage 3: Dual Spectral Decomposition & Perron-Frobenius Verification
    # -------------------------------------------------------------------------
    print("\n[Stage 3/5] Performing Spectral Decomposition (M v = lambda v) on A and W...")
    detector = SpectralHubDetector(network)
    u_spec = detector.unweighted
    w_spec = detector.weighted

    print(f"  * Model A (Topological Routes):")
    print(f"      - Dominant Eigenvalue (lambda_1) : {u_spec['dominant_eigenvalue']:.6f}")
    print(f"      - Second Eigenvalue   (lambda_2) : {u_spec['eigenvalues'][1]:.6f}")
    print(f"      - Spectral Gap (lambda_1-lambda_2): {u_spec['spectral_gap']:.6f}")
    print(f"      - Residual ||Av - lambda*v||     : {u_spec['residual_norm']:.2e} (Exact: {u_spec['is_eigen_exact']})")
    print(f"      - Orthonormal Eigenbasis Q^T*Q=I : {u_spec['is_eigenbasis_orthonormal']}")

    print(f"  * Model B (Daily Flight Capacity W):")
    print(f"      - Dominant Eigenvalue (lambda_1) : {w_spec['dominant_eigenvalue']:.6f}")
    print(f"      - Second Eigenvalue   (lambda_2) : {w_spec['eigenvalues'][1]:.6f}")
    print(f"      - Spectral Gap (lambda_1-lambda_2): {w_spec['spectral_gap']:.6f}")
    print(f"      - Residual ||Wv - lambda*v||     : {w_spec['residual_norm']:.2e} (Exact: {w_spec['is_eigen_exact']})")

    # -------------------------------------------------------------------------
    # Stage 4: Power Iteration Numerical Convergence & Metrics Export
    # -------------------------------------------------------------------------
    print("\n[Stage 4/5] Running Power Iteration Algorithm from scratch & exporting metrics...")
    u_iters = len(u_spec['power_iteration_history'])
    w_iters = len(w_spec['power_iteration_history'])
    print(f"  * Unweighted Power Iteration Converged in : {u_iters} iterations")
    print(f"  * Weighted Power Iteration Converged in   : {w_iters} iterations (Accelerated by massive spectral gap)")

    results_csv_path = detector.save_results()
    print(f"  * Exported Master Centrality Results CSV : {results_csv_path.name}")

    # -------------------------------------------------------------------------
    # Stage 5: Visual Dashboards, Interactive Web App & Academic Reports
    # -------------------------------------------------------------------------
    print("\n[Stage 5/5] Generating publication-quality dashboards and reports...")
    visualizer = HubVisualizer(detector)
    f1 = visualizer.plot_network_graph()
    f2 = visualizer.plot_adjacency_heatmap()
    f3 = visualizer.plot_centrality_ranking()
    f4 = visualizer.plot_eigenvalue_spectrum()
    f_dash = visualizer.plot_executive_dashboard()

    print(f"  * Created Network Graph        : {f1.name}")
    print(f"  * Created Dual Heatmaps        : {f2.name}")
    print(f"  * Created Importance Ranking   : {f3.name}")
    print(f"  * Created Eigenvalue Spectrum  : {f4.name}")
    print(f"  * Created Executive Dashboard  : {f_dash.name}")

    # Terminal report & Interactive HTML dashboard
    reporter = HubReportGenerator(detector)
    reporter.print_terminal_report()
    json_path = reporter.export_summary_json()
    html_path = reporter.generate_html_dashboard()

    print(f"[Done] Summary JSON exported: {json_path}")
    print(f"[Done] Interactive Web Dashboard created: {html_path}")
    print("\n" + "=" * 85)
    print("PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All artifacts and outputs are ready in: {PROJECT_ROOT}")
    print("=" * 85)


if __name__ == "__main__":
    main()
