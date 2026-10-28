# Research Project Report: Conditional VAE for Lode Runner Level Generation, Verification & Repair

**Author / Team:** Lode Runner CVAE Project  
**Date:** October 2026  
**Status:** Working Research Prototype & Interactive Benchmark Suite  

---

## 1. Executive Summary & Objective

This project investigates **Procedural Content Generation via Machine Learning (PCGML)** applied to the classic discrete puzzle-platformer *Lode Runner*. 

Unlike games with continuous movement, *Lode Runner* operates under strict discrete physics affordances: platforms must support players against gravity, ladders and ropes provide vertical and horizontal navigation conduits, and diggable bricks allow irreversible environment modification.

We implemented an end-to-end framework pairing:
1. A **Conditional Variational Autoencoder (CVAE)** to learn the spatial tile distribution conditioned on targeted level metrics.
2. A **Game-Aware Physics & Reachability Solver** capable of simulating player mechanics, computing monotone digging fixpoints, detecting one-way trap pits via Tarjan's Strongly Connected Components (SCC), and solving gold collection tours with $A^*$.
3. An **Interactive Web Dashboard (Streamlit)** allowing real-time level synthesis, latent exploration, reachability inspection, and surgical repair.
4. An **Empirical Benchmark Suite** comparing raw generation, rejection sampling, and repair against authentic human-designed levels from the Video Game Level Corpus (VGLC).

---

## 2. System Architecture

```
+--------------------------------------------------------------------------+
|                               DATA PIPELINE                              |
|  150 VGLC Levels (22x32) -> 8-channel One-Hot Tensors -> Augmentation   |
|  Condition Vector c = [Gold, Ladders, Ropes, Enemies, Solid Density]     |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                         CVAE MODEL ARCHITECTURE                          |
|  Encoder: 3x Conv2d(stride=2) + BatchNorm + LeakyReLU (13ch -> z in R^32)|
|  Decoder: Latent z + Condition c -> Linear -> 3x ConvTranspose2d(s=2)    |
|  Loss: Class-Weighted Cross-Entropy + Beta-Annealed KL Divergence        |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                        LEVEL GENERATION MODES                            |
|  1. Raw Sampling: z ~ N(0, I) -> argmax logits                           |
|  2. Rejection Sampling: Search z space for high-playability candidate   |
|  3. Surgical Repair: Bridge platforms & relocate trapped gold            |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                     SOLVER & VERIFICATION ENGINE                         |
|  Discrete Support & Gravity Physics | Monotone Digging Reachability      |
|  Tarjan's SCC One-Way Dead-End Analysis | A* Full Gold Tour Solver       |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                       STREAMLIT INTERACTIVE APP                          |
|  NES Retro Sprite Renderer | Reachability Overlays | Heatmaps & Tours    |
+--------------------------------------------------------------------------+
```

### 2.1 The CVAE Model (`model.py`)
- **Encoder**: Takes an 8-channel one-hot grid $(8 \times 22 \times 32)$ concatenated with a broadcasted 5D condition vector $c$ (total 13 channels). Three convolutional blocks compress the spatial representation to a 32-dimensional latent bottleneck $z \sim \mathcal{N}(\mu, \sigma^2)$.
- **Decoder**: Concatenates $z$ and $c$ (37 dimensions), projects via linear layer to $(128 \times 3 \times 4)$, and upsamples through three transposed convolutional layers back to logits of shape $(8 \times 22 \times 32)$.
- **Condition Vector ($c \in \mathbb{R}^5$)**: Controls `[gold_count, ladder_count, rope_count, enemy_count, solid_density]`, normalized by dataset maximums.

### 2.2 The Game-Aware Solver (`solver.py`)
- **Discrete Physics**: Implements gravity fall dynamics, ladder climbing (including stepping out over ladder tops into air), and horizontal handhold rope movement.
- **Monotone Digging Reachability**: Solves the monotone reachability fixpoint: if a player standing on a supported tile can dig an adjacent diagonal brick `b`, that brick is marked passable for navigation.
- **Tarjan's SCC Analysis**: Computes reachability directed graphs to identify non-invertible transitions (pits that can be fallen into but never escaped), ensuring completability.
- **$A^*$ Nearest-Gold Tour Solver**: Computes end-to-end playable solutions, proving whether a generated map can actually be beaten.

---

## 3. Major Challenges Encountered & How We Fixed Them

During development, raw machine learning outputs clashed repeatedly with strict game physics. Below are the engineering challenges identified and resolved:

### 1. The Zero-Rope Suppression Problem
- **Problem**: In the training data, ropes (`-`) make up only $3.2\%$ of total tiles, while empty air (`.`) makes up over $56\%$. Standard `argmax(logits)` completely suppressed ropes, causing the model to generate zero ropes across almost all seeds.
- **Solution**: Implemented an anchored rope placement pass in `generate.py` that identifies candidate rope-layer logits and anchors them horizontally across open platform chasms, restoring game-accurate rope structures.

### 2. The Wall-Tunneling Physics Bug
- **Problem**: In early testing, the solver's monotone digging reachability allowed the player to "tunnel" horizontally through up to 10 consecutive solid bricks because once dug, adjacent cells were treated as open air.
- **Solution**: Restricted the digging action in `solver.py` so digging is only permitted from legitimate non-solid ground, and prohibited horizontal movement within active dug pit depths.

### 3. The 13-Gold "Conveyor Belt" Clustering & The Multi-Pass Placement Trade-Off
- **Problem**: When a long horizontal ledge was generated, raw model confidence caused up to 13 adjacent gold chests to spawn directly touching (`G G G...`), turning entire platforms into solid gold strips rather than dispersed puzzle objectives.
- **Implementation & Trade-Off**:
  - **Pass 1 (Dispersed Tier Pass)**: Enforced a non-touching horizontal spacing check ($\Delta c \ge 2$, meaning at least 1 tile/brick must separate gold coins: `G . G`) and a target cap of $\le 3$ gold coins per row across round-robin vertical tiers (upper, mid, lower).
  - **Pass 2 (Reachability Fallback Relaxation)**: If the user requests a high gold count (e.g., 16 gold) but the generated map contains only 2–3 reachable platforms, Pass 1 hits the 3-per-row cap before meeting the quota. To prevent placing remaining gold in unreachable sky zones (which would destroy playability), Pass 2 relaxes the 3-per-row limit while preserving the alternating spacing ($\Delta c \ge 2$). Consequently, maps with sparse platform infrastructure will still display alternating gold lines (`G . G . G`) across the few valid ledges.

### 4. Ladder Perimeter Overcrowding & Dead Middle Air
- **Problem**: The raw model placed ladders predominantly on the outer borders (columns 0–3 and 28–31), leaving the interior center of the level as an empty, disconnected dead-air abyss.
- **Solution**: 
  - Capped perimeter ladders to $\le 20\%$ of total ladder budget, matching authentic VGLC ground-truth metrics ($16.8\%$).
  - Added interior ladder conduits (columns 6–26) that bridge floating platforms down to lower landings.

### 5. Floating Sky Bricks & Truncated Ladder Stubs
- **Problem**: Convolutions generated random 1-tile floating solid blocks in the upper sky and 1-rung ladder stubs suspended in mid-air.
- **Solution**: Built a cleanup filter in `post_process_level` with **functional immunity** (preserving blocks that touch ladders or support actors/gold, while pruning floating sky noise and extending mid-tier blocks into full 2-brick ledges).

### 6. UI State Desynchronization in Streamlit
- **Problem**: When repairing or reverting levels in the web app, the solver path and gold badges desynchronized from the displayed grid.
- **Solution**: Stored `current_order` in Streamlit session state, binding the solver's exact visit order to the grid across generate, repair, and revert workflows.

---

## 4. Benchmark Performance Comparison

Evaluation across $N=50$ generated samples per mode against authentic levels (from authoritative `outputs/benchmark_results.csv`):

| Evaluation Metric | Real Levels (Baseline) | CVAE (Raw Sampling) | CVAE (Rejection $K=50$) | CVAE (Surgical Repair) |
| :--- | :--- | :--- | :--- | :--- |
| **Playability Rate** | $22.0\%$ | $22.0\%$ | $56.0\%$ | **$86.0\%$** |
| **Completability Rate** | $20.0\%$ | $22.0\%$ | $56.0\%$ | **$86.0\%$** |
| **Gold Reachable Fraction** | $59.4\%$ | $52.9\%$ | $79.2\%$ | **$87.6\%$** |
| **Novelty Rate** | $0.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ |
| **Tile Diversity Score** | $0.564$ | $0.264$ | $0.268$ | $0.279$ |
| **Mean Solution Path (Steps)**| $232.5$ | $75.9$ | $67.5$ | $87.7$ |

*Note on Real Levels*: Human-designed levels from 1983 contain complex multi-phase puzzle mechanics (e.g. trapping guards in dug holes to walk across their heads) that static reachability graphs classify as impassable, yielding $22\%$ formal static playability. Both rejection sampling ($56\%$) and surgical repair with relocation ($86\%$) significantly outperform the raw CVAE ($22\%$).

---

## 5. Lingering Issues & Observations

Despite achieving high benchmark scores, several noticeable architectural issues remain:

1. **Lower-Floor Concentration Bias**:
   - The model heavily concentrates platforms and gold on the lower rows (rows 14–20).
   - *Cause*: In the 149 valid training levels (Level 150 has no player spawn `'M'`), row 21 is always 100% solid bedrock, and lower rows contain the bulk of the platforms. Upper rows are mostly open air. The network accurately learned this marginal distribution. Furthermore, gravity allows downward movement but requires ladder infrastructure for upward movement; if an upper ladder is broken, upper platforms become unreachable, causing reachable gold placement to fall back to the bottom.
2. **"Blobby" Platforms vs. Crisp Puzzle Architecture**:
   - Authentic levels consist of long, straight horizontal ledges (3 to 15 bricks) and straight vertical ladder shafts.
   - The CVAE outputs somewhat rounded, organic clumps of bricks due to the continuous nature of convolutional feature maps.
3. **Surgical Repair Trade-Off (High Playability vs. Platform Crowding)**:
   - Surgical repair achieves a high playability rate ($86\%$) primarily because its fallback pass relocates stranded, unreachable gold down to verified reachable platforms. 
   - *The Catch*: When a raw level generates very few reachable platforms, relocating all unreachable gold onto those few platforms forces multiple gold chests onto the same ledges, creating the alternating rows of gold (`G . G . G`) observed during testing.
4. **Gold Spacing & Density Dilemma on Sparse Maps**:
   - The placement algorithm prioritizes verified reachability. When a generated level contains only 2–3 reachable platforms but the user requests 16+ gold chests, the system faces a trilemma: (a) under-deliver on the requested gold count, (b) place gold on unreachable sky blocks (destroying playability), or (c) relax the row quota to place alternating gold (`G . G . G`) on the few available platforms. It currently chooses (c), which can visually look like rows of gold alternating with spaces or blocks.

---

## 6. Literature Comparison & Architectural Limitations

Our findings directly mirror the academic literature on machine learning for *Lode Runner*:

### 6.1 What Published Researchers Discovered
- **Thakkar et al. (IEEE CoG 2019)** (*"Autoencoder and Evolutionary Algorithm for Level Generation in Lode Runner"*):
  - Thakkar et al. trained standard AEs and VAEs on the exact same 150 VGLC levels.
  - **Crucial Finding**: They reported that **raw sampling from the VAE latent space produces severely broken levels (<15% playability)** with floating blocks and disconnected ladders.
  - **Their Solution**: They did **not** use random Gaussian sampling $z \sim \mathcal{N}(0, I)$ for their final outputs. Instead, they ran **Latent Vector Evolution (LVE)**: an Evolutionary Algorithm (CMA-ES / GA) that searched through the continuous latent space $z \in \mathbb{R}^{32}$ using fitness functions based on tile n-grams and $A^*$ playability.
- **Sorochan et al. (FDG 2021)** (*"Generating Lode Runner Levels by Learning Player Paths with LSTMs"*):
  - Concluded that feed-forward convolutional architectures lack inductive biases for long-range vertical connectivity. They advocated for recurrent or path-guided sequential generation.
- **Bhaumik et al. (2020)** (*"Lode Encoder"*):
  - Demonstrated that autoencoders excel as mixed-initiative co-creative tools (taking human sketches and projecting them into coherent levels) rather than pure unconstrained generators from noise.

### 6.2 Root Cause of the Architectural Limitation
1. **The Curse of High-Dimensional Latent Spaces**: In a 32-dimensional Gaussian latent space, the volume containing coherent, playable levels forms a very thin, manifold. Randomly drawn vectors land in intermediate interpolation zones that decode into blurry, fragmented features.
2. **Independent Softmax Argmax**: Transposed convolutions output continuous logit fields. The argmax operator decides each tile independently, lacking structural awareness of platform continuity (e.g. knowing a platform must be $\ge 3$ tiles long).

### 6.3 Shared Challenges Documented Across PCGML Research

The lingering issues observed in our implementation are not isolated implementation bugs; they are fundamental, well-documented challenges across Procedural Content Generation via Machine Learning (PCGML) in discrete platform games:

1. **The "Broken Affordance" Phenomenon (Snodgrass & Ontañón, 2017)**:
   Standard machine learning loss functions (like cross-entropy) evaluate tile predictions point-wise and independently. In game physics, however, a ladder that is 1 tile short of reaching a floor is completely useless ($0\%$ functional utility), even though cross-entropy penalizes it by less than $0.1\%$. The loss function rewards models for capturing broad statistical brushstrokes rather than respecting critical functional affordances.

2. **The "Empty Air" Dominance Dilemma (Sorochan & Guzdial, 2021)**:
   In classic 2D platformers, empty air represents $>55\%$ of all tiles, and solid bottom rows represent nearly $100\%$ of boundary ground. Generative models learn this vertical marginal probability prior effortlessly, naturally biasing mass toward the bottom and leaving the upper center sparse. Researchers repeatedly noted that models bias towards "safe" empty air predictions to minimize reconstruction loss.

3. **Latent Manifold Dead Zones in VAEs (Thakkar et al., 2019)**:
   While Variational Autoencoders enforce a continuous Gaussian prior $\mathcal{N}(0, I)$ over latent space, only an infinitesimally small volume of that hypersphere corresponds to valid, coherent game levels. The spaces in between training instances decode into "hybrid" or "interpolated" levels characterized by floating fragmented blocks, truncated ladders, and half-formed ledges.

4. **Discrete Rule Invariance vs. Continuous Convolutions**:
   Convolutional feature maps operate via smooth spatial interpolation and receptive fields. Games like *Lode Runner*, however, operate on sharp, discrete rules: a ledge must be flat and horizontal, a ladder must be straight and vertical, and player support is binary (supported or falling). Continuous convolutional decoders have no native mechanism to enforce these rigid geometric grammars without external search or constraint filters.

---

## 7. Analytical Discussion: Potential Methodologies & Trade-Offs

Below is an objective evaluation of theoretical approaches that could address these challenges, analyzing their respective advantages and trade-offs:

| Potential Methodology | Core Mechanism | Advantages (Pros) | Trade-Offs & Complications (Cons) |
| :--- | :--- | :--- | :--- |
| **1. Latent Space Search / Evolution (Thakkar et al. 2019)** | Run an optimization algorithm (CMA-ES, Genetic Algorithm, or Latent Gradient Descent) over $z \in \mathbb{R}^{32}$ using a fitness function based on solver reachability and platform continuity. | - **No retraining required**: Uses the existing frozen CVAE checkpoint.<br>- **Avoids dead zones**: Actively steers sampling toward the high-quality manifold.<br>- Directly optimizes for $100\%$ playability. | - **High inference latency**: Requires dozens or hundreds of forward passes and simulation runs per level generation.<br>- **Sensitivity to hyperparameters**: Fitness weights (playability vs. density) require careful balancing. |
| **2. Affordance & Grammar-Based Morphological Post-Processing** | Post-process decoded grids with 1D morphological filters: merge adjacent brick fragments on the same row into solid continuous ledges ($\ge 3$ tiles), and vertically snap ladder stubs to connect overhead ledges directly to supporting floors. | - **Extremely fast**: Runs in milliseconds without neural inference overhead.<br>- **Enforces clean aesthetics**: Eliminates "blobby" circular mounds and single-tile floating blocks.<br>- Deterministic and fully explainable. | - **Heuristic creep**: Relies on handcrafted rules rather than purely learned distributions.<br>- Can potentially override subtle probabilistic nuances predicted by the network. |
| **3. Temperature Scaling & Vertical Logit Re-Weighting** | Apply temperature scaling ($T < 1.0$) to sharpen logits before argmax, and apply a vertical spatial mask that softly boosts solid-tile logits in middle tiers (rows 6–14). | - **Trivial implementation**: Requires minimal lines of arithmetic during inference.<br>- **Counteracts bottom bias**: Directly corrects for the empty-air dominance in upper-middle rows. | - Does not guarantee physical affordances (e.g. might boost a platform without creating the ladder needed to reach it).<br>- Requires empirical tuning of temperature and row weights. |
| **4. Sequential / Autoregressive Path Generation (Sorochan et al. 2021)** | Transition from feed-forward convolutions to an autoregressive model (LSTM / Transformer) that generates levels sequentially along simulated player trajectories. | - **Strong physical coherence**: Naturally models contiguous platforms and vertical ladder chains.<br>- Avoids spatial blurring of convolutions. | - **Major architectural rework**: Requires full retraining and redesigning the model and training pipeline.<br>- Sequential token-by-token generation is slower than single-pass parallel convolutional decoding. |
