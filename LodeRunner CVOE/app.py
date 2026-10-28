"""Interactive Streamlit Dashboard for Lode Runner CVAE & Game-Aware Solver.

Features:
- Authentic Arcade Retro Pixel-Art Sprite Rendering (NES/Arcade running-bond bricks, steel bedrock, white ladders, cyan ropes, glowing gold chests, pixel hero & guards).
- Sidebar controls for condition vector (gold, ladders, ropes, enemies, density), mode selection, seed.
- Automatic game-aware solver verification (Playable, Completable, Reachable fraction, Path length).
- Interactive frame-by-frame path animation player.
- Real vs Generated side-by-side comparison.
- Benchmark and Training metrics tabs.
- In-place "Repair This Level" functionality with before/after comparison.
"""

import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import constants
import solver
import generate
import sprites


st.set_page_config(
    page_title="Lode Runner CVAE Level Generator & Solver",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .badge-playable {
        background-color: #1b5e20;
        color: #e8f5e9;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-unplayable {
        background-color: #b71c1c;
        color: #ffebee;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-completable {
        background-color: #0d47a1;
        color: #e3f2fd;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-repair {
        background-color: #e65100;
        color: #fff3e0;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        margin-right: 8px;
    }
    .mode-explainer {
        background-color: #1a1e27;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #00e676;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_models():
    """Load and cache trained CVAE model, conditions metadata, and real levels."""
    model, max_condition, condition_names = generate.load_cvae()
    real_files = sorted(list(constants.LEVELS_DIR.glob("*.txt")))
    real_levels = {
        f.name: solver.normalize_grid(f.read_text(encoding="utf-8"))
        for f in real_files
    }
    return (model, max_condition, condition_names), real_levels


model_tuple, real_levels = get_cached_models()
model, max_cond, cond_names = model_tuple


# Session state initialization
if "current_level" not in st.session_state:
    cond_init = np.array([16.0, 55.0, 18.0, 3.0, 0.32], dtype=np.float32)
    res_init = generate.generate(cond_init, mode="rejection", seed=42, model_tuple=model_tuple)
    st.session_state.current_level = res_init["grid"]
    st.session_state.original_level = res_init.get("original_grid", res_init["grid"])
    st.session_state.current_analysis = res_init["analysis"]
    st.session_state.current_comp = res_init["is_completable"]
    path, steps, order = solver.solve_path(res_init["grid"])
    st.session_state.current_path = path
    st.session_state.current_steps = steps
    st.session_state.current_order = order
    st.session_state.generation_time = 0.02
    st.session_state.mode_used = "rejection"
    st.session_state.repair_info = None


# --- SIDEBAR CONTROLS ---
st.sidebar.title("🎮 CVAE Level Controls")
st.sidebar.markdown("Specify gameplay attributes to condition the neural generator:")

# Quick Presets
preset = st.sidebar.selectbox(
    "Quick Preset",
    options=["Balanced Arcade (Default)", "Vertical Tower (High Ladders)", "Chamber Maze (High Density)", "Custom"],
    index=0,
)

if preset == "Balanced Arcade (Default)":
    p_gold, p_ladder, p_rope, p_enemy, p_dense = 16, 55, 18, 3, 0.32
elif preset == "Vertical Tower (High Ladders)":
    p_gold, p_ladder, p_rope, p_enemy, p_dense = 18, 75, 14, 4, 0.35
elif preset == "Chamber Maze (High Density)":
    p_gold, p_ladder, p_rope, p_enemy, p_dense = 14, 48, 22, 2, 0.40
else:
    p_gold, p_ladder, p_rope, p_enemy, p_dense = 16, 55, 18, 3, 0.32

gold_in = st.sidebar.slider("Gold Count ('G')", min_value=2, max_value=30, value=p_gold, step=1)
ladder_in = st.sidebar.slider("Ladder Count ('#')", min_value=10, max_value=90, value=p_ladder, step=1)
rope_in = st.sidebar.slider("Rope Count ('-')", min_value=0, max_value=40, value=p_rope, step=1)
enemy_in = st.sidebar.slider("Enemy Count ('E')", min_value=0, max_value=6, value=p_enemy, step=1)
density_in = st.sidebar.slider("Solid Ground Density", min_value=0.15, max_value=0.55, value=p_dense, step=0.01)

st.sidebar.markdown("---")
mode_in = st.sidebar.selectbox(
    "Generation Mode",
    options=["rejection", "repair", "raw"],
    index=0,
    help="Rejection (K=50 candidates for playability), Repair (minimal surgical ladder/block fixes), Raw (pure CVAE decoding).",
)

seed_in = st.sidebar.number_input("Random Seed", value=42, step=1)

if st.sidebar.button("✨ Generate Level", type="primary", use_container_width=True):
    t0 = time.time()
    c_user = np.array([gold_in, ladder_in, rope_in, enemy_in, density_in], dtype=np.float32)
    res = generate.generate(c_user, mode=mode_in, seed=seed_in, model_tuple=model_tuple, max_rejection=30)
    dur = time.time() - t0

    st.session_state.current_level = res["grid"]
    st.session_state.original_level = res.get("original_grid", res["grid"])
    st.session_state.current_analysis = res["analysis"]
    st.session_state.current_comp = res["is_completable"]
    path, steps, order = solver.solve_path(res["grid"])
    st.session_state.current_path = path
    st.session_state.current_steps = steps
    st.session_state.current_order = order
    st.session_state.generation_time = dur
    st.session_state.mode_used = mode_in
    st.session_state.repair_info = res.get("repair_info", None)

st.sidebar.markdown("---")
st.sidebar.markdown("### Arcade Tile Legend")
leg_cols = st.sidebar.columns(2)
with leg_cols[0]:
    st.markdown("🧱 `B`: Steel Bedrock")
    st.markdown("🟧 `b`: Red Brick")
    st.markdown("⬛ `.`: Empty Air")
    st.markdown("🔹 `-`: Cyan Rope")
with leg_cols[1]:
    st.markdown("🪜 `#`: White Ladder")
    st.markdown("🟡 `G`: Gold Chest")
    st.markdown("🔴 `E`: Enemy Guard")
    st.markdown("🟢 `M`: Hero Runner")


# --- MAIN INTERFACE TABS ---
st.title("Lode Runner Conditional VAE & Game Solver")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Generate & Verify",
    "⚖️ Real vs Generated",
    "📊 Benchmarks",
    "📈 Training & Model",
])

# ==========================================
# TAB 1: GENERATE & VERIFY
# ==========================================
with tab1:
    ana = st.session_state.current_analysis
    is_play = ana["is_playable"]
    is_comp = st.session_state.current_comp
    dur = st.session_state.generation_time
    order = st.session_state.get("current_order", [])
    collected_count = len(order)
    total_gold = ana["n_gold"]
    full_tour = (collected_count == total_gold and total_gold > 0)

    # Verification Badges & Metrics
    badge_cols = st.columns([3, 1, 1, 1])
    with badge_cols[0]:
        if full_tour:
            st.markdown('<span class="badge-playable">✅ FULL TOUR (SOLVED)</span>', unsafe_allow_html=True)
            if is_comp:
                st.markdown('<span class="badge-completable">🏆 COMPLETABLE</span>', unsafe_allow_html=True)
        elif is_play:
            st.markdown(f'<span class="badge-repair">⚠️ PARTIAL TOUR ({collected_count}/{total_gold} GOLD)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-unplayable">❌ UNPLAYABLE</span>', unsafe_allow_html=True)

        rep_info = st.session_state.get("repair_info")
        if rep_info and rep_info.get("edits", 0) > 0:
            st.markdown(f'<span class="badge-repair">🔧 REPAIRED ({rep_info["edits"]} EDITS)</span>', unsafe_allow_html=True)

    with badge_cols[1]:
        st.metric("Gold Collected", f"{collected_count}/{total_gold} ({ana['gold_reachable_fraction']*100:.0f}% reach)")
    with badge_cols[2]:
        st.metric("Solver Path Length", f"{st.session_state.current_steps} steps")
    with badge_cols[3]:
        st.metric("Compute Latency", f"{dur*1000:.1f} ms")

    # Repair Feedback Callout Banner
    if rep_info and rep_info.get("edits", 0) > 0:
        st.info(
            f"🛠️ **Repair Feedback**: {rep_info.get('summary', '')} "
            f"| Status: {'✅ 100% Solvable' if rep_info.get('final_playable') else '⚠️ Partial Reachability'}"
        )
        with st.expander("🔍 View Surgical Actions Applied to Guarantee Playability", expanded=False):
            for action in rep_info.get("actions", []):
                st.markdown(f"- {action}")

    st.markdown("---")

    col_view, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown("#### Visual Overlays")
        show_reachable = st.checkbox("Overlay Reachable Air (Cyan)", value=True)
        show_unreachable = st.checkbox("Highlight Unreachable Gold (Red Ring)", value=True)
        show_path = st.checkbox("Display Solver A* Tour (Green Line)", value=True)

        step_slider = None
        if show_path and st.session_state.current_path:
            total_frames = len(st.session_state.current_path) - 1
            st.markdown("#### Solver Step-Through")
            step_slider = st.slider("Step Index", min_value=0, max_value=max(1, total_frames), value=total_frames)

        st.markdown("---")
        st.markdown("#### Surgical Repair")
        if not is_play:
            st.caption(f"{len(ana['unreachable_gold'])} gold chest(s) unreachable. Apply automated ladder bridges and platform supports:")
            if st.button("🔧 Repair This Level", use_container_width=True, type="primary"):
                rep_grid, edits, rep_ana, rep_comp = generate.repair_level(st.session_state.current_level, max_edits=20)
                st.session_state.current_level = rep_grid
                st.session_state.current_analysis = rep_ana
                st.session_state.current_comp = rep_comp
                path, steps, order = solver.solve_path(rep_grid)
                st.session_state.current_path = path
                st.session_state.current_steps = steps
                st.session_state.current_order = order
                st.session_state.repair_info = rep_ana.get("repair_info")
                st.session_state.mode_used = "repair"
                st.rerun()
        else:
            st.success("✅ Level is 100% playable! All gold reachable.")
            if rep_info and rep_info.get("edits", 0) > 0:
                if st.button("↩️ Revert to Pre-Repair Level", use_container_width=True):
                    orig = st.session_state.get("original_level", st.session_state.current_level)
                    st.session_state.current_level = orig
                    st.session_state.current_analysis = solver.analyze(orig)
                    st.session_state.current_comp = solver.is_completable(orig) if st.session_state.current_analysis["is_playable"] else False
                    path, steps, order = solver.solve_path(orig)
                    st.session_state.current_path = path
                    st.session_state.current_steps = steps
                    st.session_state.current_order = order
                    st.session_state.repair_info = None
                    st.session_state.mode_used = "raw"
                    st.rerun()

    with col_view:
        reach_set = ana["reachable_cells"] if show_reachable else None
        unreach_list = ana["unreachable_gold"] if show_unreachable else None
        path_list = st.session_state.current_path if show_path else None

        img = sprites.render_authentic_level(
            st.session_state.current_level,
            reachable_cells=reach_set,
            unreachable_gold=unreach_list,
            path=path_list,
            current_step=step_slider,
        )
        st.image(img, use_container_width=True, caption=f"Mode: {st.session_state.mode_used.upper()} | Dimensions: 22x32 tiles (528x768px)")


# ==========================================
# TAB 2: REAL VS GENERATED
# ==========================================
with tab2:
    st.subheader("Comparative Level Inspector")
    st.markdown("Compare the CVAE generated level alongside human-authored VGLC Lode Runner levels rendered with authentic arcade sprites.")

    comp_cols = st.columns(2)
    with comp_cols[0]:
        selected_real = st.selectbox("Select Real Level to Compare", options=list(real_levels.keys()), index=0)
        real_grid = real_levels[selected_real]
        real_ana = solver.analyze(real_grid)
        real_comp = solver.is_completable(real_grid)
        real_path, real_steps, _ = solver.solve_path(real_grid)

        st.markdown(f"**Real Level: {selected_real}**")
        st.markdown(f"Status: {'✅ Playable' if real_ana['is_playable'] else '❌ Unplayable'} | Gold: {real_ana['n_gold']} | Solver Steps: {real_steps}")
        img_real = sprites.render_authentic_level(real_grid, path=real_path)
        st.image(img_real, use_container_width=True)

    with comp_cols[1]:
        st.markdown(f"**Current Generated Level ({st.session_state.mode_used.upper()})**")
        st.markdown(f"Status: {'✅ Playable' if ana['is_playable'] else '❌ Unplayable'} | Gold: {ana['n_gold']} | Solver Steps: {st.session_state.current_steps}")
        img_gen = sprites.render_authentic_level(st.session_state.current_level, path=st.session_state.current_path)
        st.image(img_gen, use_container_width=True)


# ==========================================
# TAB 3: BENCHMARK RESULTS
# ==========================================
with tab3:
    st.subheader("Empirical Benchmark Comparison")
    st.markdown("Comparison across **Real Levels** vs **CVAE (Raw)** vs **CVAE (Rejection)** vs **CVAE (Repair)**:")

    csv_path = constants.OUTPUTS_DIR / "benchmark_results.csv"
    chart_path = constants.OUTPUTS_DIR / "benchmark_comparison.png"

    if csv_path.exists():
        df_bench = pd.read_csv(csv_path)
        st.dataframe(df_bench.style.format({
            "Playability Rate (%)": "{:.1f}%",
            "Completable Rate (%)": "{:.1f}%",
            "Gold Reachable (%)": "{:.1f}%",
            "Novelty (%)": "{:.1f}%",
            "Diversity": "{:.3f}",
            "Mean Path Steps": "{:.1f}",
        }), use_container_width=True)

    if chart_path.exists():
        st.image(str(chart_path), caption="Quantitative Metric Comparison Across Generative Modes", use_container_width=True)


# ==========================================
# TAB 4: TRAINING & DIAGNOSTICS
# ==========================================
with tab4:
    st.subheader("Training Convergence & Reconstructions")

    t_cols = st.columns([2, 1])
    with t_cols[0]:
        loss_path = constants.OUTPUTS_DIR / "loss_curve.png"
        if loss_path.exists():
            st.image(str(loss_path), caption="CVAE Loss Curve (Weighted Cross-Entropy, KL Divergence, and Beta Annealing)", use_container_width=True)

    with t_cols[1]:
        config_path = constants.OUTPUTS_DIR / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            st.markdown("#### Sweep Summary")
            st.markdown(f"- **Best Beta (β_max)**: `{cfg.get('best_beta_max')}`")
            st.markdown(f"- **Best Epoch**: `{cfg.get('best_epoch')}`")
            st.markdown(f"- **Best Val Loss**: `{cfg.get('best_val_loss'):.4f}`")
            st.markdown(f"- **Raw Playability**: `{cfg.get('playability_rate')*100:.1f}%`")
            st.markdown(f"- **Latent Dimension**: `{cfg.get('latent_dim')}`")

    st.markdown("---")
    st.subheader("Visual Reconstruction Progression")
    recon_files = sorted(list(constants.OUTPUTS_DIR.glob("reconstructions_epoch_*.png")))
    if recon_files:
        chosen_recon = st.selectbox("Select Validation Reconstruction Snapshot", [f.name for f in recon_files])
        st.image(str(constants.OUTPUTS_DIR / chosen_recon), caption=f"Validation Tile Reconstructions ({chosen_recon})", use_container_width=True)
