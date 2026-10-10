"""
report_generator.py
===================
Generates console summary reports, JSON/CSV exports, and an interactive
HTML dashboard for the Airport Hub Detection Linear Algebra project.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.network_model import AirportNetwork
from src.spectral_analysis import SpectralHubDetector


class HubReportGenerator:
    """Produces detailed text reports, JSON summaries, and an interactive HTML dashboard."""

    def __init__(self, detector: SpectralHubDetector):
        self.detector = detector
        self.network = detector.network
        self.output_dir = PROJECT_ROOT / "figures"
        self.data_dir = PROJECT_ROOT / "data"

    def print_terminal_report(self):
        """Prints a beautifully formatted academic summary to the terminal."""
        df = self.detector.results_df
        summary = self.network.get_network_summary()
        verify = self.detector.verify_eigen_equation()

        # Ensure UTF-8 output on Windows terminal if available
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ['utf-8', 'utf8']:
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass

        print("\n" + "=" * 80)
        print("   AIRPORT HUB DETECTION USING EIGENVALUES & EIGENVECTORS")
        print("   A Linear Algebra Applied Network Model")
        print("=" * 80)
        print(f"Network Scale        : {summary['num_airports']} Airports | {summary['num_routes']} Direct Flight Routes")
        print(f"Graph Topology       : Undirected, Connected = {summary['is_connected']} (Density: {summary['graph_density']})")
        print(f"Adjacency Matrix A   : {self.detector.n}x{self.detector.n} Real Symmetric (A = A^T, Trace = {summary['matrix_trace']})")
        print("-" * 80)
        print("1. SPECTRAL DECOMPOSITION RESULTS:")
        print(f"   * Dominant Eigenvalue (lambda_max) : {self.detector.dominant_eigenvalue:.6f}")
        print(f"   * Second Eigenvalue (lambda_2)     : {self.detector.eigenvalues[1]:.6f}")
        print(f"   * Spectral Gap (lambda_1 - lambda_2): {self.detector.spectral_gap:.6f}")
        print(f"   * Eigen-equation Check ||Av - lambda*v|| : {verify['residual_norm']:.2e} (Exact: {verify['is_valid']})")
        print(f"   * Perron-Frobenius Property        : All v_i > 0 guaranteed for connected graph")
        print("-" * 80)
        print("2. AIRPORT HUB CENTRALITY RANKINGS (Descending Eigenvector Importance):")
        header = f"{'Rank':<5} | {'Code':<5} | {'City':<14} | {'Routes':<6} | {'Eigen Score':<11} | {'L2 Norm':<8} | {'Degree Rank':<11} | {'Shift'}"
        print(header)
        print("-" * len(header))
        for _, row in df.iterrows():
            shift_symbol = "+ " if row['rank_shift'] > 0 else "- " if row['rank_shift'] < 0 else "  "
            shift_str = f"{shift_symbol}{abs(row['rank_shift'])}" if row['rank_shift'] != 0 else "0"
            print(f"{row['eigen_rank']:<5} | {row['code']:<5} | {row['city']:<14} | {row['connections']:<6} | {row['eigenvector_score']:<11.4f} | {row['l2_score']:<8.4f} | {row['degree_rank']:<11} | {shift_str}")
        print("-" * 80)
        print("3. MATHEMATICAL INTERPRETATION OF HIGHLIGHTS:")
        print("   * Quality of Connections vs Raw Quantity:")
        print("     - Ahmedabad (AMD) has 2 connections, yet scores 0.3950 (connected to BOM & DEL).")
        print("     - Guwahati (GAU) has 2 connections, but scores 0.2778 (connected to DEL & CCU).")
        print("       --> AMD is 42% higher because BOM & DEL are top national dominant hubs!")
        print("     - Kochi (COK) & Kolkata (CCU) both have 3 connections.")
        print("       --> COK (0.5657) beats CCU (0.4195) by 35% because COK bridges BOM, BLR, MAA.")
        print("=" * 80 + "\n")

    def export_summary_json(self, filepath: Path = None) -> Path:
        """Exports spectral properties and node scores to JSON."""
        if filepath is None:
            filepath = self.data_dir / "spectral_analysis_summary.json"

        data = {
            "project_title": "Airport Hub Detection using Eigenvalues and Eigenvectors",
            "dominant_eigenvalue": round(float(self.detector.dominant_eigenvalue), 6),
            "second_eigenvalue": round(float(self.detector.eigenvalues[1]), 6),
            "spectral_gap": round(float(self.detector.spectral_gap), 6),
            "all_eigenvalues": [round(float(x), 6) for x in self.detector.eigenvalues],
            "residual_norm": float(self.detector.verify_eigen_equation()["residual_norm"]),
            "power_iteration_converged_iters": len(self.detector.power_iteration_history),
            "rankings": self.detector.results_df.to_dict(orient="records")
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def generate_html_dashboard(self, filepath: Path = None) -> Path:
        """
        Creates an interactive, self-contained single-page HTML Dashboard.
        Can be opened directly in any modern web browser without dependencies.
        """
        if filepath is None:
            filepath = self.output_dir / "airport_hub_interactive_dashboard.html"

        df = self.detector.results_df
        airports_meta = self.network.code_to_meta
        routes = self.network.routes
        summary = self.network.get_network_summary()

        # Build table rows HTML
        table_rows = []
        for _, row in df.iterrows():
            shift_badge = (
                f'<span class="badge badge-up">▲ {row["rank_shift"]}</span>'
                if row["rank_shift"] > 0
                else f'<span class="badge badge-down">▼ {abs(row["rank_shift"])}</span>'
                if row["rank_shift"] < 0
                else '<span class="badge badge-same">— 0</span>'
            )
            pct_bar = f'<div class="progress-bar"><div class="progress-fill" style="width: {row["eigenvector_score"]*100:.1f}%;"></div></div>'
            table_rows.append(f"""
            <tr>
                <td><strong>#{row['eigen_rank']}</strong></td>
                <td><span class="iata-code">{row['code']}</span></td>
                <td><strong>{row['city']}</strong><br><small style="color:#64748B">{row['airport_name']}</small></td>
                <td><span class="category-tag">{row['category']}</span></td>
                <td><strong>{row['connections']}</strong></td>
                <td>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <strong>{row['eigenvector_score']:.4f}</strong>
                        {pct_bar}
                    </div>
                </td>
                <td>{row['l2_score']:.4f}</td>
                <td>{row['degree_rank']}</td>
                <td>{shift_badge}</td>
            </tr>
            """)
        table_html = "\n".join(table_rows)

        # Build adjacency matrix table HTML
        adj_headers = "".join([f"<th>{code}</th>" for code in self.network.airport_codes])
        adj_rows = []
        for i, r_code in enumerate(self.network.airport_codes):
            cells = []
            for j, c_code in enumerate(self.network.airport_codes):
                val = int(self.detector.A[i, j])
                cell_class = "cell-one" if val == 1 else "cell-zero"
                cells.append(f'<td class="{cell_class}">{val}</td>')
            adj_rows.append(f"<tr><th>{r_code}</th>{''.join(cells)}</tr>")
        adj_table_html = "\n".join(adj_rows)

        # Build SVG Interactive Graph
        # Map lon: [70, 94] -> x: [80, 720], lat: [8, 30] -> y: [520, 80] (inverted)
        def to_svg_x(lon):
            return 80 + (lon - 71.0) / (93.0 - 71.0) * 640

        def to_svg_y(lat):
            return 530 - (lat - 9.0) / (30.0 - 9.0) * 440

        svg_edges = []
        for r in routes:
            u, v = r["origin"], r["destination"]
            if u in airports_meta and v in airports_meta:
                x1, y1 = to_svg_x(airports_meta[u]["lon"]), to_svg_y(airports_meta[u]["lat"])
                x2, y2 = to_svg_x(airports_meta[v]["lon"]), to_svg_y(airports_meta[v]["lat"])
                svg_edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94A3B8" stroke-width="2" stroke-opacity="0.6" />')

        svg_nodes = []
        for _, row in df.iterrows():
            code = row["code"]
            meta = airports_meta[code]
            cx, cy = to_svg_x(meta["lon"]), to_svg_y(meta["lat"])
            score = row["eigenvector_score"]
            radius = 12 + score * 22
            fill_color = "#DC2626" if score >= 0.95 else "#EA580C" if score >= 0.8 else "#3B82F6" if score >= 0.5 else "#64748B"

            svg_nodes.append(f"""
            <g class="node-group" data-code="{code}" data-city="{meta['city']}" data-score="{score:.4f}" data-deg="{row['connections']}">
                <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill_color}" stroke="#FFFFFF" stroke-width="3" filter="drop-shadow(0px 3px 6px rgba(0,0,0,0.25))" />
                <text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" fill="#FFFFFF" font-size="11" font-weight="bold">{code}</text>
                <text x="{cx:.1f}" y="{cy - radius - 5:.1f}" text-anchor="middle" fill="#0F172A" font-size="10" font-weight="bold">{meta['city']} ({score:.2f})</text>
            </g>
            """)

        svg_content = f"""
        <svg viewBox="0 0 800 600" width="100%" height="520" style="background:#F8FAFC; border-radius:12px; border:1px solid #E2E8F0;">
            <!-- Routes -->
            {''.join(svg_edges)}
            <!-- Airport Hubs -->
            {''.join(svg_nodes)}
        </svg>
        """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Airport Hub Detection — Linear Algebra Mini-Project</title>
    <style>
        :root {{
            --primary: #1E40AF;
            --primary-dark: #1E3A8A;
            --accent: #DC2626;
            --bg: #0F172A;
            --card-bg: #FFFFFF;
            --text-dark: #0F172A;
            --text-muted: #64748B;
            --border: #E2E8F0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background-color: #F1F5F9; color: var(--text-dark); line-height: 1.5; padding: 24px; }}
        .header {{
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            color: white; padding: 32px; border-radius: 16px; margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        }}
        .header h1 {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
        .header p {{ font-size: 15px; opacity: 0.9; max-width: 900px; }}
        
        .metrics-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px; margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--card-bg); padding: 20px; border-radius: 12px;
            border: 1px solid var(--border); box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }}
        .metric-title {{ font-size: 13px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; }}
        .metric-value {{ font-size: 26px; font-weight: 800; color: var(--primary); }}
        .metric-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

        .card {{
            background: var(--card-bg); border-radius: 14px; padding: 24px;
            margin-bottom: 24px; border: 1px solid var(--border);
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        .card-title {{ font-size: 18px; font-weight: 700; margin-bottom: 16px; color: #1E293B; }}

        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 992px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #F8FAFC; color: #475569; font-weight: 700; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background-color: #F8FAFC; }}

        .iata-code {{ background: #EFF6FF; color: #1D4ED8; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 12px; }}
        .category-tag {{ background: #F1F5F9; color: #475569; padding: 3px 8px; border-radius: 12px; font-size: 11px; }}
        .badge {{ padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; display: inline-block; }}
        .badge-up {{ background: #DCFCE7; color: #15803D; }}
        .badge-down {{ background: #FEE2E2; color: #B91C1C; }}
        .badge-same {{ background: #F1F5F9; color: #64748B; }}

        .progress-bar {{ background: #E2E8F0; border-radius: 4px; height: 8px; width: 80px; overflow: hidden; }}
        .progress-fill {{ background: linear-gradient(90deg, #3B82F6, #DC2626); height: 100%; border-radius: 4px; }}

        .matrix-table th, .matrix-table td {{ text-align: center; padding: 8px; font-weight: bold; font-size: 12px; }}
        .cell-one {{ background: #1E40AF; color: white; }}
        .cell-zero {{ background: #F8FAFC; color: #94A3B8; }}

        .math-box {{
            background: #F8FAFC; border-left: 4px solid var(--primary);
            padding: 16px; border-radius: 0 8px 8px 0; font-size: 14px; margin-bottom: 16px;
        }}
        .formula {{ font-family: "Courier New", monospace; font-size: 15px; font-weight: 700; color: #1E3A8A; }}

        .comparison-table {{ margin-top: 16px; }}
        .node-group {{ cursor: pointer; transition: transform 0.2s; }}
        .node-group:hover circle {{ stroke: #F59E0B; stroke-width: 4; }}
    </style>
</head>
<body>

    <div class="header">
        <h1>Airport Hub Detection using Eigenvalues & Eigenvectors</h1>
        <p>A Linear Algebra mathematical network model identifying the most strategically critical airport hubs in India via Spectral Decomposition & Perron-Frobenius Eigenvector Centrality.</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Dominant Eigenvalue (λ_max)</div>
            <div class="metric-value">{self.detector.dominant_eigenvalue:.4f}</div>
            <div class="metric-sub">Spectral Radius ρ(A)</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Spectral Gap (λ_1 - λ_2)</div>
            <div class="metric-value">{self.detector.spectral_gap:.4f}</div>
            <div class="metric-sub">High gap = Rapid network convergence</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Network Scale</div>
            <div class="metric-value">{summary['num_airports']} Airports</div>
            <div class="metric-sub">{summary['num_routes']} Direct flight routes</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Matrix Symmetry & Residual</div>
            <div class="metric-value">||Av - λv|| ≈ 0</div>
            <div class="metric-sub">Residual norm: {self.detector.verify_eigen_equation()['residual_norm']:.2e}</div>
        </div>
    </div>

    <div class="two-col">
        <!-- Network Graph -->
        <div class="card">
            <div class="card-title">Flight Network Graph (Node Size ∝ Eigenvector Centrality)</div>
            {svg_content}
            <div style="font-size: 12px; color: #64748B; margin-top: 8px;">
                * Larger nodes and redder hues represent higher eigenvector centrality scores.
            </div>
        </div>

        <!-- Math Foundation -->
        <div class="card">
            <div class="card-title">Mathematical Foundation & Intuition</div>
            
            <div class="math-box">
                <div class="formula">A v = λ v &nbsp;&iff;&nbsp; v_i = (1 / λ) &sum;_(j &isin; N(i)) v_j</div>
                <p style="margin-top: 8px; color:#334155;">
                    The centrality score of airport <strong>i</strong> is proportional to the sum of importance scores of all airports <strong>j</strong> directly connected to it. Connecting to major hubs confers significantly greater structural prestige than connecting to isolated airports.
                </p>
            </div>

            <div class="math-box" style="border-left-color: #10B981;">
                <strong>Perron-Frobenius Theorem:</strong> Since the airport network is connected (irreducible) and non-negative, the dominant eigenvalue &lambda;<sub>max</sub> is guaranteed to be real and positive, with a strictly positive eigenvector <strong>v</strong> &gt; 0.
            </div>

            <h4 style="margin: 16px 0 8px 0; font-size:14px; color:#1E293B;">Key Mathematical Discovery from Network:</h4>
            <ul style="padding-left: 20px; font-size: 13px; color: #475569; line-height: 1.8;">
                <li><strong>Ahmedabad (AMD) vs Guwahati (GAU):</strong> Both airports have exactly 2 direct flight connections. Yet AMD scores <strong>0.3950</strong> while GAU scores <strong>0.2778</strong> (+42% boost) because AMD connects directly to India's top two hubs (DEL and BOM).</li>
                <li><strong>Kochi (COK) vs Kolkata (CCU):</strong> Both have 3 connections. COK achieves <strong>0.5657</strong> (+35% over CCU's 0.4195) by strategically bridging BOM, BLR, and MAA.</li>
            </ul>
        </div>
    </div>

    <!-- Ranked Table -->
    <div class="card">
        <div class="card-title">Airport Hub Strategic Ranking</div>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Code</th>
                    <th>Airport & City</th>
                    <th>Classification</th>
                    <th>Routes</th>
                    <th>Eigenvector Score</th>
                    <th>L2 Score</th>
                    <th>Degree Rank</th>
                    <th>Rank Shift</th>
                </tr>
            </thead>
            <tbody>
                {table_html}
            </tbody>
        </table>
    </div>

    <!-- Adjacency Matrix Heatmap Table -->
    <div class="card">
        <div class="card-title">Network Adjacency Matrix A (10×10 Symmetric)</div>
        <p style="font-size: 13px; color: #64748B; margin-bottom: 12px;">
            A<sub>ij</sub> = 1 if airports i and j have a bidirectional direct flight connection, 0 otherwise.
        </p>
        <div style="overflow-x: auto;">
            <table class="matrix-table">
                <thead>
                    <tr><th></th>{adj_headers}</tr>
                </thead>
                <tbody>
                    {adj_table_html}
                </tbody>
            </table>
        </div>
    </div>

    <!-- PageRank Comparison -->
    <div class="card">
        <div class="card-title">Comparison: Eigenvector Centrality vs Google PageRank</div>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Airport Hub Detection (Our Project)</th>
                    <th>Google PageRank</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Domain</strong></td>
                    <td>Airports & Physical Flight Corridors</td>
                    <td>Webpages & Hyperlinks</td>
                </tr>
                <tr>
                    <td><strong>Matrix Type</strong></td>
                    <td>Adjacency Matrix A (Symmetric, Undirected)</td>
                    <td>Column-Stochastic Transition Matrix P with Damping</td>
                </tr>
                <tr>
                    <td><strong>Formulation</strong></td>
                    <td>A v = &lambda;<sub>max</sub> v (Dominant Eigenvector)</td>
                    <td>&pi; = M &pi; (Stationary Distribution, &lambda; = 1)</td>
                </tr>
                <tr>
                    <td><strong>Directionality</strong></td>
                    <td>Bidirectional routes (A = A<sup>T</sup>)</td>
                    <td>Directed hyper-links (asymmetric)</td>
                </tr>
                <tr>
                    <td><strong>Damping Factor</strong></td>
                    <td>None required (Pure Adjacency Spectrum)</td>
                    <td>Required (d &approx; 0.85) to prevent spider traps / dead ends</td>
                </tr>
            </tbody>
        </table>
    </div>

</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_template)
        return filepath


if __name__ == "__main__":
    net = AirportNetwork()
    detector = SpectralHubDetector(net)
    detector.compute_eendecomposition()
    detector.build_comparison_dataframe()

    reporter = HubReportGenerator(detector)
    reporter.print_terminal_report()
    json_path = reporter.export_summary_json()
    html_path = reporter.generate_html_dashboard()

    print(f"Summary JSON exported: {json_path}")
    print(f"Interactive HTML Dashboard generated: {html_path}")
