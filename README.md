# Airport Hub Detection using Eigenvalues & Eigenvectors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Field](https://img.shields.io/badge/Field-Applied%20Linear%20Algebra%20%7C%20MFAD-orange.svg)](#)

An applied linear algebra and spectral graph theory project that models aviation transit networks and applies **Spectral Decomposition**, the **Perron-Frobenius Theorem**, and **Flight Capacity Weighing** to identify strategic national airport hubs via **Eigenvector Centrality**.

Developed for **UE25MA242A — Mathematics for Artificial Intelligence / Data Science (MFAD)**, PES University.

---

## ✈️ Overview

When analyzing transportation networks, the most intuitive approach is to count direct connections—known as **Degree Centrality**. However, in real-world aviation corridors, **not all connections are created equal**:

> *Connecting to a premier national transit hub (such as Mumbai or Delhi) confers far greater structural importance and transit flow than connecting to multiple isolated regional airstrips.*

Furthermore, routes vary drastically in operational capacity:
- **Delhi–Mumbai Trunk:** Operates **65 daily flights**.
- **Kolkata–Guwahati:** Operates **12 daily flights**.

This project implements a complete linear algebraic pipeline to evaluate both **Topological Prestige** and **Operational Capacity Flow**.

---

## 📐 Mathematical Formulation

### 1. Dual Matrix Models
- **Binary Adjacency Matrix $A \in \{0, 1\}^{10 \times 10}$:** Models topological route existence ($A_{ij} = 1$).
- **Weighted Frequency Matrix $W \in \mathbb{R}^{10 \times 10}$:** Models total scheduled daily commercial flights ($W_{ij} = \text{daily\_flights}$).
- Both matrices are **real symmetric** ($A = A^T, W = W^T$) and **hollow** ($A_{ii} = 0, \text{Tr}(A) = 0$).

### 2. Structure of the Space & Rank-Nullity Theorem
- **Vector Space:** $\mathbb{R}^{10}$ state space.
- **Topological Matrix $A$:** $\text{Rank}(A) = 9, \text{Nullity}(A) = 1$.
- **Rank-Nullity Theorem:**
  $$\text{Rank}(A) + \text{Nullity}(A) = 9 + 1 = 10 \quad (\text{Verified})$$
- **Capacity Matrix $W$:** Full Rank ($\text{Rank} = 10, \text{Nullity} = 0$).
- **Orthonormal Eigenbasis:** $Q = [\mathbf{q}_1, \dots, \mathbf{q}_{10}]$ such that $Q^T Q = I_{10}$ and $M = Q \Lambda Q^T$.

### 3. The Central Eigenvalue Equation
The structural importance $v_i$ of airport $i$ is proportional to the sum of importance scores of its neighbors weighted by connection intensity:

$$\lambda v_i = \sum_{j \in \mathcal{N}(i)} W_{ij} v_j \iff W\mathbf{v} = \lambda \mathbf{v}$$

### 4. Perron-Frobenius Theorem
Because the route network is connected (irreducible) and non-negative ($W_{ij} \ge 0$):
1. The dominant eigenvalue $\lambda_{\max}$ is strictly positive and real.
2. The associated principal eigenvector $\mathbf{v}$ has **strictly positive components** ($v_i > 0$ for all $i$).
3. The normalized eigenvector $\mathbf{v} / \max(\mathbf{v})$ provides a canonical hub ranking.

### 5. Numerical Power Iteration
$$\mathbf{v}^{(k+1)} = \frac{W\mathbf{v}^{(k)}}{\|W\mathbf{v}^{(k)}\|_2}, \quad \lambda^{(k)} = \frac{(\mathbf{v}^{(k)})^T W \mathbf{v}^{(k)}}{(\mathbf{v}^{(k)})^T \mathbf{v}^{(k)}}$$

- **Unweighted Model $A$:** Converged in **27 iterations** ($\lambda_1 = 4.9529, \Delta \lambda = 3.1171$).
- **Weighted Model $W$:** Converged in **28 iterations** ($\lambda_1 = 145.3338, \Delta \lambda = 124.3905$).

---

## 📊 Master Hub Rankings & Empirical Insights

| Rank | Airport Code | City / Region | Routes | Daily Flights | Topological Score ($A$) | Operational Score ($W$) | Topo Rank | Volume Shift |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **DEL** | New Delhi | 7 | 223 | 0.9565 | **1.0000** | #2 | **+1** (Capital Mega Hub) |
| **2** | **BOM** | Mumbai | 7 | 211 | **1.0000** | **0.9639** | #1 | **-1** (Commercial Backbone) |
| **3** | **BLR** | Bengaluru | 6 | 165 | 0.9336 | **0.8216** | #3 | 0 (Tech & Aviation Hub) |
| **4** | **HYD** | Hyderabad | 5 | 114 | 0.8435 | **0.6277** | #5 | **+1** (Deccan Central Hub) |
| **5** | **MAA** | Chennai | 5 | 94 | 0.8681 | **0.5196** | #4 | **-1** (Southern Coastal Hub) |
| **6** | **AMD** | Ahmedabad | 2 | 42 | 0.3950 | **0.2840** | #8 | **+2** (Commercial Hub Surge) |
| **7** | **GOI** | Goa | 2 | 38 | 0.3904 | **0.2383** | #9 | **+2** (Leisure Shuttle Surge) |
| **8** | **CCU** | Kolkata | 3 | 50 | 0.4195 | **0.2375** | #7 | **-1** (Eastern Regional Hub) |
| **9** | **COK** | Kochi | 3 | 40 | 0.5657 | **0.2190** | #6 | **-3** (South Gateway) |
| **10** | **GAU** | Guwahati | 2 | 21 | 0.2778 | **0.0815** | #10 | 0 (North-East Gateway) |

### Key Mathematical & Real-World Insights:
- **DEL Surges Over BOM:** While Mumbai holds slightly higher route centrality, Delhi dominates operational capacity with over 223 daily scheduled flight departures, capturing #1 nationwide.
- **Ahmedabad (AMD) Volume Boost:** AMD connects to only 2 airports (DEL and BOM), but carries 42 heavy daily flights—pushing its capacity score past Kolkata and Kochi.
- **Goa (GOI) High-Frequency Corridors:** Frequent coastal and IT shuttles from BOM and BLR boost Goa above larger regional cities.

---

## 📁 Repository Structure

```
airport-hub-detection/
├── main.py                              # Master 5-stage orchestrator pipeline
├── run_project.bat                      # Windows one-click execution & dashboard launcher
├── PROJECT_REPORT.md                    # Academic report formatted for viva defense
├── README.md                            # Project documentation
│
├── src/                                 # Core mathematical packages
│   ├── network_model.py                 # Adjacency & Weighted matrices, Rank-Nullity, Bases
│   ├── spectral_analysis.py             # Eigendecomposition, Perron-Frobenius & Power Iteration
│   ├── visualizer.py                    # High-res publication graphics & executive dashboard
│   └── report_generator.py              # Terminal reports & Interactive HTML web dashboard
│
├── data/                                # Network datasets & exported matrices
│   ├── airports.json                    # Airport metadata & coordinates
│   ├── routes.json                      # Route definitions with daily flights & distances
│   ├── adjacency_matrix.csv             # 10x10 binary adjacency matrix A
│   ├── weighted_adjacency_matrix.csv    # 10x10 daily flight frequency matrix W
│   ├── centrality_results.csv           # Full comparative metrics output table
│   └── spectral_analysis_summary.json   # Exported eigenvalues and convergence metrics
│
└── figures/                             # High-resolution generated visual assets
    ├── 01_network_graph.png             # Geographic network scaled by flight frequency & score
    ├── 02_adjacency_heatmap.png         # Dual heatmaps: Topology A vs. Flight Frequency W
    ├── 03_centrality_ranking.png        # Comparative bar chart: Topology vs. Capacity Flow
    ├── 04_eigenvalue_spectrum.png       # Dual eigenvalue spectra & Power Iteration convergence
    ├── airport_hub_executive_dashboard.png   # 4-panel presentation dashboard
    └── airport_hub_interactive_dashboard.html # State-of-the-art interactive web dashboard
```

---

## 🚀 Getting Started

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### Installation
Install required dependencies:
```bash
pip install numpy networkx matplotlib seaborn pandas scipy
```
*(Or use virtualenv with `uv pip install -r requirements.txt`)*

### Running the Pipeline
Run the master script:
```bash
python main.py
```

### Viewing the Interactive Web Dashboard
Open `figures/airport_hub_interactive_dashboard.html` in any browser, or on Windows double-click `run_project.bat`.

---

## 📄 License
This project is licensed under the MIT License.
