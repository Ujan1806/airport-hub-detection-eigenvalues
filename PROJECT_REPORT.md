# Applied Linear Algebra Mini-Project: Airport Hub Detection using Eigenvalues & Eigenvectors

**Course:** UE25MA242A — Mathematics for Artificial Intelligence / Data Science (MFAD)  
**Institution:** PES University, Department of Computer Science & Engineering  
**Project Title:** Structural Hub Identification in Flight Transit Networks via Spectral Graph Decomposition & Capacity Weighing  
**Author:** Ujan Dutta  
**Academic Year:** 2026  

---

## Executive Summary

Flight networks are complex transit systems where strategic connectivity dictates operational resilience, transit capacity, and schedule routing. A naive approach to identifying the most important airports in a country simply counts their total number of direct connections (**Degree Centrality**). However, in realistic transport networks, **not all connections are created equal**:
> *Connecting to a major national transit hub (such as Mumbai or Delhi) confers vastly greater strategic importance and passenger transit potential than connecting to multiple isolated regional airstrips.*

Furthermore, corridors themselves carry vastly different operational capacities: the **Delhi–Mumbai trunk corridor** operates over **65 daily flights**, whereas a regional connector operates fewer than **10 daily flights**.

This project implements an applied linear algebraic pipeline addressing:
1. **Topological Structural Hubs:** Via the binary symmetric Adjacency Matrix $A \in \{0, 1\}^{10 \times 10}$.
2. **Operational Capacity Hubs:** Via the weighted Flight Frequency Matrix $W \in \mathbb{R}^{10 \times 10}$ representing 499 daily commercial flights.
3. **Structure of the Space:** Explicitly analyzing Vector Spaces, Subspaces, Column & Null Space Bases, and verifying the **Rank-Nullity Theorem**.
4. **Spectral Graph Theory & Perron-Frobenius Theorem:** Formally proving why the dominant eigenvector yields an all-positive, canonical hub ranking.
5. **Direct Spectral Solution:** Leveraging exact symmetric eigendecomposition (`np.linalg.eigh`) to directly extract dominant transit modes.

---

## 1. Mathematical Formulation & Matrix Representations

### 1.1 Network Graph Representation
Let the aviation transit network be modeled as an undirected graph $G = (V, E)$:
- $V = \{v_1, v_2, \dots, v_{10}\}$ represents 10 major Indian commercial airports: DEL, BOM, BLR, MAA, HYD, CCU, AMD, COK, GOI, GAU.
- $E = \{(v_i, v_j)\}$ represents 21 bidirectional flight route corridors.

### 1.2 Binary Adjacency Matrix $A$ (Topological Routes)
$$A_{ij} = \begin{cases} 1 & \text{if a direct route exists between airport } i \text{ and } j \\ 0 & \text{otherwise} \end{cases}$$

### 1.3 Weighted Frequency Matrix $W$ (Operational Flight Volume)
$$W_{ij} = \begin{cases} \text{Daily scheduled flights between airport } i \text{ and } j & \text{if a route exists} \\ 0 & \text{otherwise} \end{cases}$$

#### Key Algebraic Properties of $A$ and $W$:
1. **Real and Symmetric:** Bidirectional route scheduling implies $A = A^T$ and $W = W^T$.
2. **Zero Diagonal (Hollow Matrix):** No self-loops exist ($A_{ii} = 0, W_{ii} = 0$), so:
   $$\text{Tr}(A) = \sum_{i=1}^n \lambda_i = 0, \quad \text{Tr}(W) = \sum_{i=1}^n \lambda_i = 0$$
3. **Non-Negative Entries:** $A_{ij} \ge 0, W_{ij} \ge 0$ for all $i, j \in \{1, \dots, 10\}$.
4. **Spectral Theorem Applicability:** Real symmetric matrices possess strictly real eigenvalues and an orthonormal basis of eigenvectors $Q$:
   $$M = Q \Lambda Q^T = \sum_{k=1}^n \lambda_k \mathbf{q}_k \mathbf{q}_k^T \quad \text{with } Q^T Q = I_n$$

---

## 2. Structure of the Space: Subspaces, Bases & Rank-Nullity Theorem

In accordance with the MFAD curriculum pipeline (*Structure of the Space → Remove Redundancy*):

### 2.1 The Ambient Vector Space $\mathbb{R}^{10}$
Each airport state is represented as a 10-dimensional vector in $\mathbb{R}^{10}$.

### 2.2 Column Space, Null Space & The Rank-Nullity Theorem
- **For Binary Adjacency Matrix $A$:**
  - $\text{Rank}(A) = \dim(\text{Col}(A)) = 9$
  - $\text{Nullity}(A) = \dim(\text{Null}(A)) = 1$
  - **Rank-Nullity Theorem:**
    $$\text{Rank}(A) + \text{Nullity}(A) = 9 + 1 = 10 \quad (\text{Exact})$$
  - *Interpretation:* The route network possesses exactly 1 linearly dependent mode among its topological corridors ($\lambda_5 = 0$).
- **For Weighted Frequency Matrix $W$:**
  - $\text{Rank}(W) = 10$ (Full Rank)
  - $\text{Nullity}(W) = 0$
  - Differential flight volumes break linear redundancy across corridors.
- **Orthonormal Eigenbasis:**
  $$Q = [\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_{10}], \quad Q^T Q = I_{10}$$
  Verified numerically to tolerance $10^{-14}$.

---

## 3. Spectral Graph Theory & Perron-Frobenius Theorem

### 3.1 What is Spectral Graph Theory?
Spectral graph theory bridges discrete topology and continuous linear algebra by analyzing the **spectrum** (eigenvalues) and eigenspaces of graph-associated matrices. 
Just as a Fourier transform decomposes a time signal into harmonic frequencies, eigendecomposition decomposes the flight network into orthogonal structural vibration modes:
- **Mode 1 ($\lambda_1, \mathbf{q}_1$):** Dominant structural flow (Transit Hub Backbone).
- **Spectral Gap ($\lambda_1 - \lambda_2$):** Governs diffusion speed and synchronization across the national airspace.

### 3.2 The Central Eigenvalue Equation
$$\lambda v_i = \sum_{j \in \mathcal{N}(i)} M_{ij} v_j \iff M\mathbf{v} = \lambda \mathbf{v}$$

### 3.3 The Perron-Frobenius Theorem
Why does the dominant eigenvector provide a physically meaningful centrality score?
For any real, non-negative ($M_{ij} \ge 0$), and irreducible (connected graph) matrix:
1. **Spectral Radius:** The spectral radius $\rho(M) = \lambda_1$ is a strictly positive, simple real eigenvalue.
2. **Positivity of Eigenvector:** The associated principal eigenvector $\mathbf{v}$ has **strictly positive components** ($v_i > 0$ for all $i$).
3. **Uniqueness:** No other eigenvector has all non-negative entries (all other orthogonal eigenvectors $\mathbf{q}_k$ for $k \ge 2$ contain both positive and negative coordinates).

Hence, Perron-Frobenius guarantees that every airport receives a strictly positive, non-trivial, well-defined importance score.

---

## 4. Spectral Graph Decomposition & Dominant Vector Extraction
 
Because $W$ is real and symmetric ($W = W^T$), the **Spectral Theorem** guarantees that all 10 eigenvalues are strictly real and that the matrix is orthogonally diagonalizable:
$$W = Q \Lambda Q^T$$
 
Using standard exact symmetric eigendecomposition (`np.linalg.eigh`):
1. **Direct Decomposition:** Computes all eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{10}$ and their orthogonal eigenvectors directly in machine precision ($O(n^3)$ operations, completing in $< 0.1$ ms for $n = 10$).
2. **Dominant Mode Selection:** The principal eigenvalue $\lambda_1 = 145.3338$ captures the dominant passenger flow through the network.
3. **Perron-Frobenius Positive Mode:** Extracting $\mathbf{v}_1 = |\mathbf{q}_1|$ yields strictly positive scores across all nodes ($v_i > 0$).
4. **Max-Normalization:** Dividing by $\max(\mathbf{v}_1)$ scales Delhi (DEL) to $1.0000$, benchmarking all other airports relative to the nation's premier transit hub.

---

## 5. Master Hub Centrality Rankings & Empirical Insights

| Rank (W) | Code | City Name | Routes | Daily Flights | Topological Score ($A$) | Operational Score ($W$) | Topo Rank | Volume Shift |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **DEL** | New Delhi | 7 | 223 | 0.9565 | **1.0000** | #2 | **+1** (Surges to #1) |
| **2** | **BOM** | Mumbai | 7 | 211 | **1.0000** | 0.9639 | #1 | **-1** |
| **3** | **BLR** | Bengaluru | 6 | 165 | 0.9336 | 0.8216 | #3 | 0 |
| **4** | **HYD** | Hyderabad | 5 | 114 | 0.8435 | 0.6277 | #5 | **+1** |
| **5** | **MAA** | Chennai | 5 | 94 | 0.8681 | 0.5196 | #4 | **-1** |
| **6** | **AMD** | Ahmedabad | 2 | 42 | 0.3950 | 0.2840 | #8 | **+2** (Commercial Surge) |
| **7** | **GOI** | Goa | 2 | 38 | 0.3904 | 0.2383 | #9 | **+2** (Leisure Shuttle) |
| **8** | **CCU** | Kolkata | 3 | 50 | 0.4195 | 0.2375 | #7 | **-1** |
| **9** | **COK** | Kochi | 3 | 40 | 0.5657 | 0.2190 | #6 | **-3** |
| **10** | **GAU** | Guwahati | 2 | 21 | 0.2778 | 0.0815 | #10 | 0 |

### Key Real-World Network Findings:
1. **The Capital Shift (DEL vs BOM):** In pure topological routing ($A$), Mumbai edges out Delhi because of balanced connectivity to southern hubs. However, in operational capacity ($W$), **New Delhi captures #1** due to immense daily trunk frequencies to Bengaluru (45 flt/d), Hyderabad (32 flt/d), and Kolkata (26 flt/d).
2. **The High-Yield Regional Hub (Ahmedabad AMD):** AMD has only 2 routes, yet ranks #6 nationwide in capacity prestige—outscoring 3-route gateways like Kolkata and Kochi—because both of AMD's routes feed directly into high-frequency national mega-hubs (DEL and BOM).
3. **Leisure Corridor Impact (Goa GOI):** Frequent flights from Mumbai and Bengaluru push Goa above eastern gateway Kolkata in operational centrality.

---

## 6. Numerical Verification & Error Norms

Linear algebraic exactness was verified against machine precision:
- **Unweighted Residual:** $\|A\mathbf{v} - \lambda_1 \mathbf{v}\|_2 = 2.60 \times 10^{-15}$
- **Weighted Residual:** $\|W\mathbf{v}_w - \lambda_1 \mathbf{v}_w\|_2 = 1.38 \times 10^{-13}$
- **Orthogonality Check:** $\|Q^T Q - I_{10}\|_F = 4.21 \times 10^{-16}$
- **Diagonal Reconstruction:** $\|W - Q \Lambda Q^T\|_F = 8.52 \times 10^{-14}$

---

## 7. Viva Defense: Concept &rarr; Purpose &rarr; Outcome

| Concept | Purpose in Project | Outcome on Data |
|:---|:---|:---|
| **Matrix Representation ($A$ & $W$)** | Formulate discrete route corridors and daily flight capacity algebraically. | Constructed $10 \times 10$ symmetric matrices $A$ (routes) and $W$ (499 daily flights). |
| **Vector Space Subspaces & Rank-Nullity** | Determine linear independence and dimension of network space. | Verified $\text{Rank}(A) + \text{Nullity}(A) = 9 + 1 = 10$. Found 9 basis airports. |
| **Spectral Theorem & Diagonalization** | Decompose symmetric matrices into orthonormal eigenmodes ($M = Q \Lambda Q^T$). | Proved all 10 eigenvalues are real; verified $Q^T Q = I_{10}$ to machine precision. |
| **Perron-Frobenius Theorem** | Ensure existence of a strictly positive, non-trivial centrality score. | Guaranteed dominant eigenvalues ($\lambda_A \approx 4.95, \lambda_W \approx 145.33$) yield strictly positive scores ($v_i > 0$). |
| **Direct Spectral Decomposition (`eigh`)** | Extract eigenvalues and eigenvectors directly without iterative approximation. | Solved in $< 0.1$ ms with residual $\|W\mathbf{v} - \lambda_1 \mathbf{v}\|_2 \approx 10^{-13}$. |

---

## 8. Conclusion

By elevating transportation network analysis from naive edge counting to **Spectral Graph Theory** and **Capacity-Weighted Eigendecomposition**, this project rigorously fulfills the requirements of the PES University MFAD mini-project curriculum.
