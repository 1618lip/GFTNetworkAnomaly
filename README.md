# 📡 GSP-NetDetect: **Graph Signal Processing for Network Anomaly Detection**

---

## 📚 Theoretical Background

### 1. Graph Representation

A network can be modeled as a **graph** $G = (V, E)$ where:
- $V$ = set of nodes (routers, devices)
- $E$ = set of edges (communication links)

We define the **adjacency matrix** $A\in \mathbb{R}^{N \times N}$:

$$A_{ij} = 
\begin{cases}
1, & \text{if nodes } i \text{ and } j \text{ are connected} \\\\
0, & \text{otherwise}
\end{cases}
$$

and the **degree matrix** $D$ (diagonal matrix with degrees):

$$D_{ii} = \sum_{j} A_{ij}$$

The **(combinatorial) graph Laplacian** is:

$$L = D - A$$

It captures how each node is connected to others — it's **fundamental** in graph signal processing.

---

### 2. Graph Fourier Transform (GFT)

The GFT generalizes classical Fourier analysis to graphs:
- Solve the eigenvalue problem:
$$L u_k = \lambda_k u_k$$

where $u_k$ is the $k$-th eigenvector (basis function) and $\lambda_k$ is its eigenvalue (frequency).

- The set $\{u_1, u_2, ..., u_N\}$ forms an orthonormal basis.

Given a graph signal $x \in \mathbb{R}^N$, its **Graph Fourier Transform** is:

$$\hat{x} = U^T x$$
where $U = [u_1\ |\ u_2\ |\ \dots\ |\ u_N]$ is the matrix of eigenvectors.

Thus, $\hat{x}_k$ measures the contribution of the $k$-th graph frequency to the signal $x$.

---

### 3. Graph Spectral Energy

The **spectral energy** of the signal is defined as:

$$E_{\text{total}} = \sum_{k=1}^{N} |\hat{x}_k|^2$$

We often break it down:
- **Low-frequency energy**: sum of $|\hat{x}_k|^2$ for small $\lambda_k$ (smooth, slow variations).
- **High-frequency energy**: sum for large $\lambda_k$ (sharp, localized changes — like attacks).

Thus:

$$E_{\text{low}} = \sum_{k \in \text{low}} |\hat{x}_k|^2
\quad\quad
E_{\text{high}} = \sum_{k \in \text{high}} |\hat{x}_k|^2$$

**Energy ratio**:

$$\text{Ratio} = \frac{E_{\text{high}}}{E_{\text{low}} + \varepsilon}$$

where $\varepsilon$ is a small constant to prevent division by zero.

---

### 4. Traffic Modeling

#### Normal traffic:
Modeled as a **Poisson process**:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
where $\lambda$ is the average arrival rate.

**Motivation**: In normal conditions, packet arrivals are memoryless and well-modeled by Poisson.

---

#### Anomalous traffic:
Simulated as **bursts** modeled by:

$$x(t) = A \sin(2\pi f t + \phi)$$

where:
- $A$ = attack amplitude
- $f$ = frequency (small perturbations around 0.2 Hz)
- $\phi$ = random phase

Only positive pulses are added (since packet counts can't be negative).

---

### 5. Anomaly Detection Heuristic

At each timestep:
- If node's **instantaneous traffic** exceeds a threshold $T_{\text{attention}}$ (say, 12 packets), that node is **flagged**.
- In spectral space, large high-frequency energies can also be used to flag anomalies.

Thus, **two domains** for anomaly detection:
- **Node domain**: directly threshold the packet counts.
- **Spectral domain** (optional extension): threshold based on energy ratios.

---

## 🧠 More Mathematical Insights

- **Laplacian eigenvalues** $lambda_k$ relate to signal smoothness:
  - Small $\lambda_k$ → smooth signals
  - Large $\lambda_k$ → rapidly changing signals
- Anomalies inject **high-variation signals** localized on the graph ⇒ **high-frequency content**.
  
If you plotted $\hat{x}$ over $k$, you'd expect an **energy spike at high frequencies** during attacks.

---

# 📈 Visualization Techniques

- **Color nodes** proportional to traffic intensity.
- **Highlight nodes** exceeding threshold $T_{\text{attention}}$ in **red**.
- **Animation** over time to see how attacks evolve and spread.

---

# ⚙️ Setup and Run Instructions

1. Create and activate a virtual environment:

```bash
python3 -m venv gspenv
source gspenv/bin/activate  # Linux/Mac
gspenv\\Scripts\\activate   # Windows
```

2. Install required packages:

```bash
pip install numpy networkx matplotlib pandas scipy
```

3. (Optional) Install FFmpeg to save MP4 animations:

```bash
# Linux
sudo apt install ffmpeg
# Mac
brew install ffmpeg
# Windows
# Download from ffmpeg.org and add to PATH
```

4. Run the main script:

```bash
python highlight_multiple_anomalies.py
```

---

# 📦 Project Output

| Artifact | Description |
|:---|:---|
| `gsp_graph_traffic_multiple_anomalies.mp4` | Animation showing traffic over time with anomaly nodes highlighted. |
| `graph_traffic_log_multiple_anomalies.csv` | Full log of traffic data for every node and every timestep. |

---

# 🔥 Workshop Extensions

- **Real-time spectral analysis**: Plot live GFT energy while traffic evolves.
- **Online anomaly detector**: Use logistic regression or neural networks trained on spectral features.
- **Dynamic graphs**: Simulate link failures or topology changes.

---
