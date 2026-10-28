# Conditional VAE for Lode Runner Level Generation & Verification

A deep learning project pairing a **Conditional Variational Autoencoder (CVAE)** with a **Game-Aware Solver** to generate, verify, repair, and benchmark Lode Runner levels, visualized in an interactive **Streamlit dashboard**.

---

## 3-Line Explanation of Solver Rules
1. **Support & Gravity**: A player is supported if standing on solid ground (`B`/`b`), a ladder (`#`), a rope (`-`), or the screen bottom; unsupported players fall straight down until landing.
2. **Action Transitions**: Supported players can move left/right into non-solid cells, climb up/down ladders (including stepping out of the top into open air), drop from ropes, and step down onto ladders from above.
3. **Monotone Digging Fixpoint**: Players can dig adjacent-diagonal diggable bricks (`b`) if the adjacent tile is clear; dug holes are treated as passable air in a monotone reachability fixpoint without timers.

---

## Architecture & Project Structure
- `constants.py`: Authoritative tile tags dynamically loaded from VGLC `Loderunner.json`, dimensions ($22 \times 32$), 8 tile classes, and rendering palettes.
- `solver.py`: Game-aware discrete transition physics, monotone digging fixpoint, Tarjan's SCC completability check for one-way dead-end pits, and A* nearest-gold tour solver.
- `data.py`: One-hot tensor encoding $(8, 22, 32)$, 85/15 train/val split by level, training horizontal-flip augmentation, and 5D condition vector $c = [\text{gold}, \text{ladders}, \text{ropes}, \text{enemies}, \text{density}]$ normalized by dataset maximums.
- `model.py`: PyTorch `LodeRunnerCVAE` featuring a 13-channel condition-broadcast encoder, latent bottleneck ($z \in \mathbb{R}^{32}$), and a 3-stage transposed convolution decoder with exact $(22, 32)$ shape enforcement.
- `train.py`: Class-weighted cross-entropy (sqrt inverse frequency clipped to 20), linear KL annealing ($0 \rightarrow \beta_{\max}$ over first 30% of epochs), early stopping, and hyperparameter sweep over $\beta_{\max} \in \{0.25, 0.5, 1.0\}$.
- `generate.py`: Generation modes (`raw`, `rejection` sampling with $K=50$, and `repair` with minimal surgical edits).
- `benchmark.py`: Empirical evaluation across Real Levels, CVAE Raw, CVAE Rejection, and CVAE Repair over playability, completability, gold reachable fraction, novelty, diversity, and solver path length.
- `app.py`: Full interactive Streamlit dashboard.
- `./outputs/`: Stores all checkpoints (`best.pt`), loss curves (`loss_curve.png`), reconstruction snapshots, benchmark CSVs, and diagnostic JSON reports.

---

## Benchmark Results (N=50 Samples / Mode)

| Mode | Playable % | Completable % | Gold Reachable % | Novelty % | Diversity | Mean Path Steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Levels (Baseline)** | 22.0% | 20.0% | 59.4% | 0.0% | 0.564 | 232.5 |
| **CVAE (Raw)** | 22.0% | 22.0% | 53.0% | 100.0% | 0.264 | 75.9 |
| **CVAE (Rejection Sampling)** | 56.0% | 56.0% | 79.2% | 100.0% | 0.268 | 67.5 |
| **CVAE (Surgical Repair)** | **86.0%** | **86.0%** | **87.6%** | 100.0% | 0.279 | 87.7 |

---

## How to Run

### 1. Run Solver Diagnostics & Audit
```bash
python test_solver.py
```

### 2. Run Data and Model Verification Tests
```bash
python test_data.py
python test_model.py
python test_generate.py
```

### 3. Train CVAE with Beta Sweep
```bash
python train.py
```

### 4. Run Benchmark Suite
```bash
python benchmark.py
```

### 5. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
