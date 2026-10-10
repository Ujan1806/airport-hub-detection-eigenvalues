# Applied Linear Algebra Mini-Project: Airport Hub Detection using Eigenvalues & Eigenvectors

**Project Title:** Structural Hub Identification in Flight Transit Networks via Spectral Graph Decomposition  
**Author:** Ujan Dutta  
**Discipline:** Applied Mathematics & Linear Algebra  
**Date:** Academic Year 2026  

---

## Executive Summary

Flight networks are fundamental complex systems where strategic connectivity dictates operational resilience, transit capacity, and schedule routing. A naive approach to identifying the most important airports in a country simply counts their total number of direct connections (**Degree Centrality**). However, in realistic transport networks, **not all connections are created equal**: connecting to a major national transit hub (such as Mumbai or Delhi) confers vastly greater strategic importance and passenger transit potential than connecting to multiple isolated regional airstrips.

This project implements a linear algebraic network model to solve the question:
> *"If an airport connects to other important airports, which airports are the most strategically important in the overall flight network?"*

Using **Eigenvector Centrality**, **Spectral Decomposition**, and the **Perron-Frobenius Theorem**, we construct a $10 \times 10$ symmetric adjacency matrix $A$ representing a simulated Indian domestic aviation network across 10 major cities (DEL, BOM, BLR, MAA, HYD, CCU, AMD, COK, GOI, GAU). We compute the dominant eigenvalue $\lambda_{\max} \approx 4.952856$, extract the principal eigenvector, and prove mathematically why connectivity to high-centrality hubs elevates an airport's structural score far beyond raw flight counting.

---

## 1. Mathematical Formulation

### 1.1 Network Graph Representation
Let the aviation network be modeled as an undirected, unweighted graph:
$$G = (V, E)$$
Where:
- $V = \{v_1, v_2, \dots, v_n\}$ represents the set of $n = 10$ airport vertices.
- $E = \{(v_i, v_j)\}$ represents the set of $m = 21$ bidirectional direct flight routes.

### 1.2 Adjacency Matrix Construction
The network graph is converted into an $n \times n$ binary **Adjacency Matrix** $A$:
$$A_{ij} = \begin{cases} 1 & \text{if a direct flight exists between airport } i \text{ and } j \\ 0 & \text{otherwise} \end{cases}$$

#### Key Algebraic Properties of $A$:
1. **Real and Symmetric:** Because flight corridors are modeled bidirectionally, $A_{ij} = A_{ji} \implies A = A^T$.
2. **Zero Diagonal (Hollow Matrix):** No self-loops exist (airports do not operate scheduled flights to themselves), so $A_{ii} = 0 \implies \text{Tr}(A) = \sum_{i=1}^n \lambda_i = 0$.
3. **Non-Negative Entries:** $A_{ij} \ge 0$ for all $i, j \in \{1, \dots, n\}$.
4. **Spectral Theorem Applicability:** Real symmetric matrices possess strictly real eigenvalues and an orthonormal basis of eigenvectors:
   $$A = Q \Lambda Q^T = \sum_{k=1}^n \lambda_k q_k q_k^T$$

---

## 2. Spectral Analysis & Eigenvector Centrality

### 2.1 The Central Eigenvalue Equation
The structural importance score $v_i$ of an airport $i$ is defined such that it is proportional to the sum of the importance scores of its neighbors:
$$v_i = \frac{1}{\lambda} \sum_{j \in \mathcal{N}(i)} v_j = \frac{1}{\lambda} \sum_{j=1}^n A_{ij} v_j$$

Multiplying through by $\lambda$:
$$\lambda v_i = \sum_{j=1}^n A_{ij} v_j \iff A \mathbf{v} = \lambda \mathbf{v}$$

Where:
- $A \in \mathbb{R}^{n \times n}$ is the airport adjacency matrix.
- $\mathbf{v} \in \mathbb{R}^n$ is an eigenvector.
- $\lambda \in \mathbb{R}$ is the corresponding eigenvalue.

### 2.2 The Perron-Frobenius Theorem
Why do we choose the dominant eigenvalue $\lambda_{\max}$?
By the **Perron-Frobenius Theorem** for non-negative, irreducible (connected) matrices:
1. The spectral radius $\rho(A) = \max_i |\lambda_i| = \lambda_1$ is a strictly positive, simple real eigenvalue ($\lambda_1 > |\lambda_k|$ for all $k \ge 2$).
2. The corresponding principal eigenvector $\mathbf{v}_1$ has **strictly positive components** ($v_i > 0 \ \forall i$).
3. All other eigenvectors associated with $\lambda_k$ ($k \ge 2$) have both positive and negative entries, which cannot represent physically meaningful centrality scores.

Thus, the dominant eigenvector $\mathbf{v}_1$ provides the unique, mathematically sound structural centrality score vector.

### 2.3 Normalization Schemes
To make scores easily interpretable and comparable:
1. **Max-Score Normalization (Standard Hub Score):**
   $$s_i = \frac{v_i}{\max(\mathbf{v})}$$
   This anchors the nation's premier transit hub at exactly $1.0000$.
2. **Euclidean $L_2$ Normalization:**
   $$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2} = 1.0000$$
3. **Probability Mass / Sum Normalization:**
   $$\sum_{i=1}^n v_i = 1.0000$$

---

## 3. Algorithmic Convergence: Power Iteration Method

In addition to exact closed-form diagonalization via `np.linalg.eigh`, we implement the iterative **Power Iteration Method** from scratch to demonstrate how eigenvector centrality converges numerically:

1. **Initialization:** Start with a uniform positive vector $\mathbf{v}^{(0)} = \frac{1}{\sqrt{n}} [1, 1, \dots, 1]^T$.
2. **Iteration:**
   $$\mathbf{y}^{(k)} = A \mathbf{v}^{(k-1)}$$
   $$\lambda^{(k)} = \frac{(\mathbf{v}^{(k-1)})^T \mathbf{y}^{(k)}}{(\mathbf{v}^{(k-1)})^T \mathbf{v}^{(k-1)}} \quad (\text{Rayleigh Quotient})$$
   $$\mathbf{v}^{(k)} = \frac{\mathbf{y}^{(k)}}{\|\mathbf{y}^{(k)}\|_2}$$
3. **Termination:** Stop when $\|\mathbf{v}^{(k)} - \mathbf{v}^{(k-1)}\|_2 < 10^{-9}$.

### Convergence Rate & The Spectral Gap
The error after $k$ steps decays geometrically governed by the ratio of the second-largest to largest eigenvalue:
$$\mathcal{O}\left( \left|\frac{\lambda_2}{\lambda_1}\right|^k \right)$$

In our model:
- Dominant Eigenvalue $\lambda_1 = 4.952856$
- Second Eigenvalue $\lambda_2 = 1.835707$
- Ratio $\left|\frac{\lambda_2}{\lambda_1}\right| \approx 0.3706$
- **Spectral Gap:** $\lambda_1 - \lambda_2 = 3.117149$

Because the spectral ratio $0.3706 \ll 1$, the power iteration converges in **only 27 iterations** to numerical precision ($10^{-9}$).

---

## 4. Empirical Results & Hub Rankings

### 4.1 Master Centrality Ranking Table

| Rank | Code | City Name | Direct Routes ($k_i$) | Eigenvector Score ($s_i$) | $L_2$ Norm | Degree Rank | Strategic Rank Shift |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **BOM** | Mumbai | 7 | **1.0000** | 0.4414 | 1 | 0 (Dominant Hub) |
| **2** | **DEL** | New Delhi | 7 | **0.9565** | 0.4222 | 1 | -1 |
| **3** | **BLR** | Bengaluru | 6 | **0.9336** | 0.4121 | 3 | 0 |
| **4** | **MAA** | Chennai | 5 | **0.8681** | 0.3831 | 4 | 0 |
| **5** | **HYD** | Hyderabad | 5 | **0.8435** | 0.3723 | 4 | -1 |
| **6** | **COK** | Kochi | 3 | **0.5657** | 0.2497 | 6 | 0 |
| **7** | **CCU** | Kolkata | 3 | **0.4195** | 0.1852 | 6 | -1 |
| **8** | **AMD** | Ahmedabad | 2 | **0.3950** | 0.1744 | 8 | 0 |
| **9** | **GOI** | Goa | 2 | **0.3904** | 0.1723 | 8 | -1 |
| **10** | **GAU** | Guwahati | 2 | **0.2778** | 0.1226 | 8 | -2 |

---

## 5. Mathematical Insights: Connection Quality vs. Connection Quantity

The primary thesis of this linear algebra project is that **eigenvector centrality evaluates neighbor importance, whereas degree centrality merely counts edges**. The empirical results demonstrate this phenomenon with remarkable clarity:

### Case Study 1: Ahmedabad (AMD) vs. Guwahati (GAU) — Equal Degree, Divergent Prestige
- **Ahmedabad (AMD):** Has exactly **2** routes (connected to DEL and BOM).
- **Guwahati (GAU):** Has exactly **2** routes (connected to DEL and CCU).
- **Degree Centrality:** Identical ($k_{\text{AMD}} = k_{\text{GAU}} = 2$).
- **Eigenvector Centrality:**
  - $s_{\text{AMD}} = 0.3950$
  - $s_{\text{GAU}} = 0.2778$
- **Mathematical Deduction:** Ahmedabad achieves a **+42.2% higher centrality score** than Guwahati because its two neighbors are the two undisputed national mega-hubs (DEL with score 0.9565, and BOM with score 1.0000). Guwahati, while connected to DEL, connects secondarily to Kolkata (0.4195).

### Case Study 2: Kochi (COK) vs. Kolkata (CCU) — Equal Degree, Regional Divergence
- **Kochi (COK):** Has **3** routes (BOM, BLR, MAA).
- **Kolkata (CCU):** Has **3** routes (DEL, HYD, GAU).
- **Eigenvector Centrality:**
  - $s_{\text{COK}} = 0.5657$
  - $s_{\text{CCU}} = 0.4195$
- **Mathematical Deduction:** Despite having identical flight counts ($k = 3$), Kochi outscores Kolkata by **+34.8%**. Kochi connects to three high-centrality southern and western mega-hubs ($s_{\text{BOM}} = 1.00, s_{\text{BLR}} = 0.93, s_{\text{MAA}} = 0.87$). Conversely, Kolkata's score is diluted by connecting to Guwahati ($s_{\text{GAU}} = 0.28$).

### Case Study 3: Ahmedabad (2 routes) vs. Kolkata (3 routes)
- Kolkata has **50% more routes** than Ahmedabad (3 vs 2).
- Yet, Ahmedabad's eigenvector score ($0.3950$) is within **6%** of Kolkata's score ($0.4195$).
- In pure linear algebraic terms, two connections to first-tier hubs provide nearly the exact same network prestige as three connections to mixed-tier hubs!

---

## 6. Comparison: Eigenvector Centrality vs. Google PageRank

While both algorithms belong to the family of spectral graph theory, they address fundamentally different graph structures:

| Feature | Airport Hub Detection (Our Project) | Google PageRank |
|:---|:---|:---|
| **Domain** | Physical flight route networks | World Wide Web hyperlinks |
| **Graph Topology** | Undirected & Bidirectional ($A = A^T$) | Directed ($A \neq A^T$, asymmetric web graph) |
| **Underlying Matrix** | Adjacency Matrix $A$ | Column-Stochastic Transition Matrix $M = (1-d)P + \frac{d}{N}E$ |
| **Linear Algebra Model** | Eigenvalue problem $A \mathbf{v} = \lambda_{\max} \mathbf{v}$ | Stationary distribution $\mathbf{\pi} = M \mathbf{\pi}$ ($\lambda = 1$) |
| **Damping Factor** | Not required (pure adjacency spectrum) | Required ($d \approx 0.85$) to solve dead ends & spider traps |
| **Interpretation** | Structural hub prestige based on neighbor status | Long-term probability of a random surfer visiting page |

---

## 7. Numerical Verification

To guarantee linear algebraic rigor, we verify the spectral residual norm:
$$\text{Residual} = \|A \mathbf{v} - \lambda_{\max} \mathbf{v}\|_2 = 3.09 \times 10^{-15}$$
$$\text{Relative Residual} = \frac{\|A \mathbf{v} - \lambda_{\max} \mathbf{v}\|_2}{\lambda_{\max} \|\mathbf{v}\|_2} = 6.24 \times 10^{-16}$$

This confirms machine-precision convergence to the true theoretical eigenspace.

---

## 8. Visual Artifacts Generated

The project produces five high-resolution figures in the `figures/` directory:
1. `01_network_graph.png`: Geographic network layout with nodes sized and colored by eigenvector score.
2. `02_adjacency_heatmap.png`: High-contrast annotated $10 \times 10$ matrix heatmap showing connections.
3. `03_centrality_ranking.png`: Side-by-side comparative bar chart contrasting Eigenvector Centrality vs Degree Centrality.
4. `04_eigenvalue_spectrum.png`: Full eigenvalue spectrum $(\lambda_1 \dots \lambda_{10})$ illustrating the spectral gap.
5. `airport_hub_executive_dashboard.png`: Unified 4-panel executive presentation dashboard.
6. `airport_hub_interactive_dashboard.html`: Self-contained interactive web application with hover tooltips and dynamic tables.

---

## 9. Conclusion

This project successfully proves the application of eigenvalues and eigenvectors in identifying critical infrastructure hubs. By elevating network analysis from naive frequency counting to spectral graph theory, eigenvector centrality correctly captures how mutual structural reinforcement between well-connected hubs defines the true backbone of an aviation network.
