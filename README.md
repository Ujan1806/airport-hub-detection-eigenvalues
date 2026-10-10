# Airport Hub Detection using Eigenvalues & Eigenvectors
### Applied Linear Algebra Mini-Project
**Location:** `D:\Ujan Dutta\maths mini project`

---

## Overview

This project models an Indian domestic aviation network across 10 major airports and uses **Eigenvector Centrality** (via Spectral Graph Theory and the **Perron-Frobenius Theorem**) to determine the most strategically vital airport hubs.

Unlike simple connection counting (Degree Centrality), Eigenvector Centrality answers:
> **"If an airport connects to other important airports, which airports are the most strategically important in the overall flight network?"**

---

## Key Mathematical Results

- **Dominant Eigenvalue ($\lambda_{\max}$):** `4.952856`
- **Second Eigenvalue ($\lambda_2$):** `1.835707`
- **Spectral Gap ($\lambda_1 - \lambda_2$):** `3.117149` (guarantees fast convergence in 27 iterations)
- **Numerical Residual $\|Av - \lambda v\|$:** `3.09e-15` (exact to machine precision)

### Strategic Hub Rankings:
1. **BOM (Mumbai):** Score `1.0000` (7 routes) — Dominant National Transit Hub
2. **DEL (New Delhi):** Score `0.9565` (7 routes) — Capital Mega Hub
3. **BLR (Bengaluru):** Score `0.9336` (6 routes) — Southern Aviation / IT Hub
4. **MAA (Chennai):** Score `0.8681` (5 routes) — Southern Coastal Hub
5. **HYD (Hyderabad):** Score `0.8435` (5 routes) — Deccan Central Hub
6. **COK (Kochi):** Score `0.5657` (3 routes) — High prestige through BOM, BLR, MAA
7. **CCU (Kolkata):** Score `0.4195` (3 routes) — Eastern Regional Hub
8. **AMD (Ahmedabad):** Score `0.3950` (2 routes) — Directly connects to top 2 hubs (DEL & BOM)
9. **GOI (Goa):** Score `0.3904` (2 routes) — Connects to BOM & BLR
10. **GAU (Guwahati):** Score `0.2778` (2 routes) — North-Eastern Gateway

> **Key Linear Algebra Insight:** Ahmedabad (AMD) and Guwahati (GAU) both have exactly 2 routes. Yet AMD scores **+42% higher** than GAU because AMD connects directly to India's top two hubs (DEL and BOM)!

---

## How to Run

### Method 1: One-Click Execution (Windows)
Double-click:
```
D:\Ujan Dutta\maths mini project\run_project.bat
```
This runs the full Python pipeline and automatically launches the Interactive Dashboard in your default web browser!

### Method 2: Command Line
Open terminal in the project directory and run:
```powershell
python main.py
```
*(Or specify the installed Python 3.12 path: `& "C:\Users\SANJOY DUTTA\AppData\Local\Programs\Python\Python312\python.exe" main.py`)*

---

## Project Structure

```
D:\Ujan Dutta\maths mini project\
│
├── main.py                         # Master orchestrator script
├── run_project.bat                 # 1-click execution & dashboard launcher
├── PROJECT_REPORT.md               # Detailed academic project report with mathematical proofs
├── README.md                       # Project documentation (this file)
│
├── data\
│   ├── airports.json               # 10 Indian airports with coordinates & categories
│   ├── routes.json                 # 21 direct bidirectional flight connections
│   ├── adjacency_matrix.csv        # 10x10 binary symmetric adjacency matrix A
│   ├── centrality_results.csv      # Complete ranked results table
│   └── spectral_analysis_summary.json  # Exported eigenvalues and metrics
│
├── src\
│   ├── __init__.py
│   ├── network_model.py            # Graph construction & adjacency matrix engine
│   ├── spectral_analysis.py        # Eigenvalues, eigenvectors, power iteration, centrality
│   ├── visualizer.py               # Publication-grade chart generation
│   └── report_generator.py         # Terminal reporting & interactive HTML generator
│
└── figures\
    ├── 01_network_graph.png        # Flight routes & centrality-scaled airport nodes
    ├── 02_adjacency_heatmap.png    # Annotated 10x10 Adjacency Matrix heatmap
    ├── 03_centrality_ranking.png   # Eigenvector vs Degree Centrality bar chart
    ├── 04_eigenvalue_spectrum.png  # Full eigenvalue spectrum & spectral gap
    ├── airport_hub_executive_dashboard.png  # Unified 4-panel presentation dashboard
    └── airport_hub_interactive_dashboard.html # Self-contained interactive web dashboard
```

---

## Visual Dashboard

Open [`figures/airport_hub_interactive_dashboard.html`](figures/airport_hub_interactive_dashboard.html) in Chrome, Edge, or Firefox to view the interactive vector network map, interactive centrality table, and matrix heatmap.

---

## Requirements

The project uses standard scientific Python libraries (already installed on this machine):
- `numpy` (>= 1.24)
- `networkx` (>= 3.0)
- `matplotlib` (>= 3.7)
- `seaborn` (>= 0.12)
- `pandas` (>= 2.0)
- `scipy` (>= 1.10)
