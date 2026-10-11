# Airport Hub Detection using Eigenvalues & Eigenvectors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Field](https://img.shields.io/badge/Field-Applied%20Linear%20Algebra-orange.svg)](#)

An applied linear algebra and network science project that models aviation transit networks and applies **Spectral Graph Theory** and the **Perron-Frobenius Theorem** to identify strategic national airport hubs via **Eigenvector Centrality**.

---

## ✈️ Overview

When analyzing flight networks, the most intuitive approach is to count direct connections—known as **Degree Centrality**. However, in real-world transportation networks, **not all connections are created equal**:

> *Connecting to a premier national transit hub (such as Mumbai or Delhi) confers far greater structural importance and transit flow than connecting to multiple isolated regional airstrips.*

This project implements a complete linear algebraic pipeline to answer the question:
> **"If an airport's importance depends on the importance of the airports it connects to, which airports are the true structural hubs of a nation?"**

---

## 📐 Mathematical Formulation

### 1. Graph & Adjacency Matrix
A flight network with $n$ airports and $m$ bidirectional routes is modeled as an undirected graph $G = (V, E)$, represented by a real symmetric adjacency matrix $A \in \mathbb{R}^{n \times n}$:

$$A_{ij} = \begin{cases} 1 & \text{if a direct flight route exists between airport } i \text{ and } j \\ 0 & \text{otherwise} \end{cases}$$

Since there are no self-loops ($A_{ii} = 0$), the trace is zero:
$$\text{Tr}(A) = \sum_{i=1}^n \lambda_i = 0$$

### 2. Eigenvector Centrality Equation
The structural importance $v_i$ of airport $i$ is proportional to the sum of the importance scores of its neighbors $\mathcal{N}(i)$:

$$\lambda v_i = \sum_{j \in \mathcal{N}(i)} v_j \iff A\mathbf{v} = \lambda \mathbf{v}$$

### 3. Perron-Frobenius Theorem
Because the route network forms a connected graph, the adjacency matrix $A$ is **irreducible** and **non-negative**. By the **Perron-Frobenius Theorem**:
1. The dominant eigenvalue $\lambda_{\max}$ is strictly positive, real, and has multiplicity 1.
2. The associated eigenvector $\mathbf{v}$ has strictly positive entries ($v_i > 0$ for all $i$).
3. The normalized eigenvector $\mathbf{v} / \max(\mathbf{v})$ provides a unique, well-defined centrality ranking.

### 4. Power Iteration Method
The principal eigenvector is computed numerically via the Power Iteration algorithm:

$$\mathbf{v}^{(k+1)} = \frac{A\mathbf{v}^{(k)}}{\|A\mathbf{v}^{(k)}\|_2}$$

Convergence is governed by the **spectral gap** $|\lambda_1 - \lambda_2|$:
$$\left\| \mathbf{v}^{(k)} - \mathbf{v}_1 \right\| = \mathcal{O}\left( \left| \frac{\lambda_2}{\lambda_1} \right|^k \right)$$

---

## 📊 Key Results & Insights

The model was evaluated on a 10-airport, 21-route network representing India's primary domestic aviation corridors:

| Rank | Airport Code | City / Region | Routes (Degree) | Eigenvector Centrality | Strategic Classification |
|:---:|:---:|:---|:---:|:---:|:---|
| 1 | **BOM** | Mumbai | 7 | **1.0000** | Primary National Transit Hub |
| 2 | **DEL** | New Delhi | 7 | **0.9565** | Capital Mega Hub |
| 3 | **BLR** | Bengaluru | 6 | **0.9336** | Southern Tech / Aviation Hub |
| 4 | **MAA** | Chennai | 5 | **0.8681** | Southern Coastal Hub |
| 5 | **HYD** | Hyderabad | 5 | **0.8435** | Deccan Central Hub |
| 6 | **COK** | Kochi | 3 | **0.5657** | High-Prestige Regional Hub |
| 7 | **CCU** | Kolkata | 3 | **0.4195** | Eastern Regional Hub |
| 8 | **AMD** | Ahmedabad | 2 | **0.3950** | Dual-Hub Connected Airport |
| 9 | **GOI** | Goa | 2 | **0.3904** | Leisure Destination Connected to BOM & BLR |
| 10 | **GAU** | Guwahati | 2 | **0.2778** | North-Eastern Regional Gateway |

### Key Mathematical Insights:
- **Dominant Eigenvalue ($\lambda_{\max}$):** `4.952856`
- **Second Eigenvalue ($\lambda_2$):** `1.835707`
- **Spectral Gap ($\lambda_1 - \lambda_2$):** `3.117149` (rapid convergence in 27 iterations)
- **Numerical Residual $\|A\mathbf{v} - \lambda\mathbf{v}\|_2$:** `3.09 × 10⁻¹⁵` (exact to machine precision)

> **Degree vs. Eigenvector Centrality Discrepancy:**  
> **Ahmedabad (AMD)** and **Guwahati (GAU)** both have exactly **2 direct routes**. However, AMD scores **+42.2% higher** than GAU because AMD links directly into the top two national hubs (DEL and BOM), while GAU connects to lower-tier hubs.

---

## 📁 Repository Structure

```
airport-hub-detection/
├── main.py                              # Master orchestrator pipeline
├── run_project.bat                      # One-click execution & dashboard launcher
├── PROJECT_REPORT.md                    # In-depth academic report with formal proofs
├── README.md                            # Project documentation
│
├── src/                                 # Core Python package
│   ├── network_model.py                 # Graph construction & adjacency matrix builder
│   ├── spectral_analysis.py             # Eigendecomposition & Power Iteration solver
│   ├── visualizer.py                    # High-resolution chart & plot generation
│   └── report_generator.py              # CLI summary & interactive HTML dashboard builder
│
├── data/                                # Network datasets & exported matrices
│   ├── airports.json                    # 10 airports with coordinates & metadata
│   ├── routes.json                      # 21 bidirectional flight route definitions
│   ├── adjacency_matrix.csv             # 10x10 binary symmetric adjacency matrix
│   ├── centrality_results.csv           # Full ranked metrics output table
│   └── spectral_analysis_summary.json   # Exported eigenvalues and convergence metrics
│
└── figures/                             # Generated visualizations
    ├── 01_network_graph.png             # Network topology with centrality-scaled nodes
    ├── 02_adjacency_heatmap.png         # Annotated 10x10 adjacency matrix heatmap
    ├── 03_centrality_ranking.png        # Eigenvector vs. Degree comparison bar chart
    ├── 04_eigenvalue_spectrum.png       # Complete eigenvalue spectrum & spectral gap
    ├── airport_hub_executive_dashboard.png   # 4-panel presentation dashboard
    └── airport_hub_interactive_dashboard.html # Self-contained interactive web dashboard
```

---

## 🚀 Getting Started

### Prerequisites
Make sure Python 3.10+ is installed on your system.

Install required dependencies:
```bash
pip install numpy networkx matplotlib seaborn pandas scipy
```

### Running the Project

#### Option 1: Terminal / Command Line
Run the orchestrator script:
```bash
python main.py
```

#### Option 2: One-Click Execution (Windows)
Double-click `run_project.bat` to run the full pipeline and automatically launch the interactive dashboard in your default browser.

---

## 📈 Visualizations & Dashboard

The pipeline automatically produces:
1. **Interactive HTML Dashboard** (`figures/airport_hub_interactive_dashboard.html`): Self-contained SVG interactive map with hover tooltips, route connectivity inspection, and dynamic tables.
2. **Executive 4-Panel Dashboard** (`figures/airport_hub_executive_dashboard.png`): High-resolution summary showing the network graph, adjacency heatmap, centrality ranking comparison, and eigenvalue spectrum.
3. **Individual High-Resolution Figures** in `figures/` for reports and slides.

---

## 🔬 Comparison: Eigenvector Centrality vs. Google PageRank

| Feature | Eigenvector Centrality (This Project) | Google PageRank |
|:---|:---|:---|
| **Graph Type** | Undirected, symmetric | Directed web graph |
| **Matrix** | Binary symmetric adjacency matrix $A$ | Column-stochastic transition matrix with damping |
| **Perron-Frobenius Basis** | Direct graph irreducibility | Enforced via teleportation damping factor ($d = 0.85$) |
| **Application** | Structural transit hubs in flight corridors | Authority of linked webpages |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
