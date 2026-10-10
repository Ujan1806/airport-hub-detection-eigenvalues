"""
main.py
=======
Master Orchestrator for the Airport Hub Detection Mini-Project.

Executes the end-to-end mathematical pipeline:
1. Loads Airport Network and builds Adjacency Matrix A
2. Computes Eigenvalues, Eigenvectors, and Perron-Frobenius Centrality
3. Executes Power Iteration convergence
4. Compares Eigenvector Centrality against Degree and Betweenness Centrality
5. Generates high-resolution visualization charts and multi-panel executive dashboard
6. Exports results to CSV, JSON, and creates an Interactive HTML Dashboard

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
    print("=" * 80)
    print("       AIRPORT HUB DETECTION USING EIGENVALUES AND EIGENVECTORS")
    print("                 Applied Linear Algebra Mini-Project")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Model Airport Network & Construct Adjacency Matrix A
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading airport network data and building Adjacency Matrix A...")
    network = AirportNetwork()
    summary = network.get_network_summary()
    adj_csv_path = network.save_adjacency_matrix()
    print(f"  * Nodes (Airports): {summary['num_airports']}")
    print(f"  * Edges (Routes)  : {summary['num_routes']}")
    print(f"  * Graph Density   : {summary['graph_density']}")
    print(f"  * Matrix Size     : {network.adjacency_matrix.shape[0]}x{network.adjacency_matrix.shape[1]}")
    print(f"  * Saved Matrix CSV: {adj_csv_path.name}")

    # -------------------------------------------------------------
    # Step 2: Spectral Decomposition & Eigenvector Centrality
    # -------------------------------------------------------------
    print("\n[Step 2/5] Performing Spectral Decomposition (Av = lambda * v)...")
    detector = SpectralHubDetector(network)
    lam_max, v_dom = detector.compute_eendecomposition()
    verify = detector.verify_eigen_equation()
    print(f"  * Dominant Eigenvalue (lambda_max): {lam_max:.6f}")
    print(f"  * Second Eigenvalue   (lambda_2)  : {detector.eigenvalues[1]:.6f}")
    print(f"  * Spectral Gap (lambda_1 - lambda_2): {detector.spectral_gap:.6f}")
    print(f"  * Linear Algebra Verification     : ||Av - lambda*v|| = {verify['residual_norm']:.2e} (Exact: {verify['is_valid']})")

    # -------------------------------------------------------------
    # Step 3: Power Iteration Convergence Simulation
    # -------------------------------------------------------------
    print("\n[Step 3/5] Executing Power Iteration algorithm from scratch...")
    lam_pi, v_pi, iters = detector.run_power_iteration()
    print(f"  * Iterative Method Converged in {iters} iterations to lambda = {lam_pi:.6f}")

    # -------------------------------------------------------------
    # Step 4: Metric Comparison & Result Export
    # -------------------------------------------------------------
    print("\n[Step 4/5] Computing comparative graph metrics and ranking...")
    results_df = detector.build_comparison_dataframe()
    results_csv_path = detector.save_results()
    print(f"  * Saved Results CSV: {results_csv_path.name}")

    # -------------------------------------------------------------
    # Step 5: Visual Dashboard & Report Generation
    # -------------------------------------------------------------
    print("\n[Step 5/5] Generating publication-quality visual dashboards...")
    visualizer = HubVisualizer(detector)
    f1 = visualizer.plot_network_graph()
    f2 = visualizer.plot_adjacency_heatmap()
    f3 = visualizer.plot_centrality_ranking()
    f4 = visualizer.plot_eigenvalue_spectrum()
    f_dash = visualizer.plot_executive_dashboard()

    print(f"  * Created Network Graph        : {f1.name}")
    print(f"  * Created Adjacency Heatmap    : {f2.name}")
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
    print("\n" + "=" * 80)
    print("PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All artifacts are ready in: {PROJECT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
