# Airport Hub Detection using Eigenvalues & Eigenvectors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Field](https://img.shields.io/badge/Field-Applied%20Linear%20Algebra%20%7C%20MFAD-orange.svg)](#)

A minimal, self-contained Applied Linear Algebra mini-project implementing **Spectral Graph Theory**, the **Perron-Frobenius Theorem**, and **Flight Capacity Weighing** to identify strategic national airport hubs via **Eigenvector Centrality**.

Developed for **UE25MA242A — Mathematics for Artificial Intelligence / Data Science (MFAD)**, Department of CSE, PES University.

---

## ✈️ Overview

When analyzing flight networks, counting direct connections (**Degree Centrality**) is insufficient because **not all connections are created equal**:
> *Connecting to a premier national transit hub (such as Delhi or Mumbai) confers far greater structural importance and transit flow than connecting to multiple isolated regional airstrips.*

Furthermore, corridors themselves carry vastly different operational capacities:
- **Delhi–Mumbai Trunk:** 65 daily flights
- **Kolkata–Guwahati:** 12 daily flights

This project implements a complete linear algebraic pipeline to identify true transit hubs using both **Topological Routes** and **Daily Flight Capacity**.

---

## 📐 Linear Algebra Workflow (MFAD Guidelines)

The project directly maps to the official PES University MFAD course workflow:

```
[Stage 1] Real-World Data & Matrix Representation (A: Routes, W: Flight Volume)
                           ↓
[Stage 2] Structure of the Space: Subspaces, Bases & Rank-Nullity Theorem (Rank 9 + Nullity 1 = 10)
                           ↓
[Stage 3] Spectral Decomposition: Exact Eigendecomposition (np.linalg.eigh) & Perron-Frobenius
                           ↓
[Stage 4] Final Application Output: Normalized Strategic Airport Hub Rankings
```

1. **Dual Matrix Models:**
   - Binary Adjacency Matrix $A \in \{0, 1\}^{10 \times 10}$ (topological flight connections).
   - Weighted Frequency Matrix $W \in \mathbb{R}^{10 \times 10}$ (499 total daily flights).
   - Real, symmetric ($A = A^T, W = W^T$), hollow ($\text{Tr}(A) = 0, \text{Tr}(W) = 0$).
2. **Structure of the Space & Rank-Nullity Theorem:**
   - Ambient Space: $\mathbb{R}^{10}$.
   - Binary Matrix: $\text{Rank}(A) = 9, \text{Nullity}(A) = 1$ ($\text{Col}(\text{HYD}) = \text{Col}(\text{COK}) + \text{Col}(\text{GAU})$).
   - **Rank-Nullity Theorem Verified:** $\text{Rank}(A) + \text{Nullity}(A) = 9 + 1 = 10$.
   - Weighted Matrix: Full rank ($\text{Rank} = 10, \text{Nullity} = 0$).
3. **Exact Spectral Decomposition (`np.linalg.eigh`):**
   - Diagonalizes the real symmetric matrix $W = Q \Lambda Q^T$ with orthonormal basis $Q^T Q = I_{10}$.
   - Extracts all strictly real eigenvalues directly without requiring iterative approximation.
4. **Perron-Frobenius Theorem & Score Normalization:**
   - Because $W$ is non-negative and connected (irreducible), the dominant eigenvalue $\lambda_1 = 145.3338$ is strictly positive and real.
   - Its principal eigenvector $\mathbf{v}_1$ has all strictly positive components ($v_i > 0$).
   - Hub scores are normalized by dividing by $\max(\mathbf{v}_1)$, giving the top airport a canonical score of 1.0000.

---

## 📊 Master Hub Rankings & Key Results

| Rank | Airport Code | City / Region | Routes | Daily Flights | Hub Centrality Score | Classification |
|:---:|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **DEL** | New Delhi | 7 | 223 | **1.0000** | National Mega Hub |
| **2** | **BOM** | Mumbai | 7 | 211 | **0.9639** | National Mega Hub |
| **3** | **BLR** | Bengaluru | 6 | 165 | **0.8216** | Primary Metro Hub |
| **4** | **HYD** | Hyderabad | 5 | 114 | **0.6277** | Primary Metro Hub |
| **5** | **MAA** | Chennai | 5 | 94 | **0.5196** | Primary Metro Hub |
| **6** | **AMD** | Ahmedabad | 2 | 42 | **0.2840** | Regional Hub (West) |
| **7** | **GOI** | Goa | 2 | 38 | **0.2383** | Tourist Gateway (West) |
| **8** | **CCU** | Kolkata | 3 | 50 | **0.2375** | Regional Hub (East) |
| **9** | **COK** | Kochi | 3 | 40 | **0.2190** | Regional Gateway (South) |
| **10** | **GAU** | Guwahati | 2 | 21 | **0.0815** | Regional Gateway (NE) |

### Key Insights:
- **Connection Quality vs. Quantity:** Ahmedabad (AMD) has only 2 routes, yet scores higher than Kolkata (CCU, 3 routes) and Kochi (COK, 3 routes) because AMD feeds directly into the nation's premier mega-hubs (DEL and BOM).
- **Capital Hub Surge:** New Delhi (DEL, 1.0000) captures #1 over Mumbai (BOM, 0.9639) when high-frequency trunk passenger volumes are factored in.

---

## 📁 Repository Structure

```
airport-hub-detection/
├── main.py                  # Single self-contained linear algebra pipeline
├── requirements.txt         # Minimal dependency: numpy
├── README.md                # Project documentation
├── PROJECT_REPORT.md        # Academic report for evaluation and viva
└── data/                    # Clean dataset files
    ├── airports.json        # 10 airports with coordinates & categories
    └── routes.json          # 21 route definitions with flight counts
```

---

## 🚀 Execution

### Prerequisites
Only `numpy` is required:
```bash
pip install numpy
```

### Running the Program
```bash
python main.py
```
*(Runs in under a second and prints all 4 stages of the linear algebra pipeline directly to the console).*

---

## 📄 License
This project is licensed under the MIT License.
