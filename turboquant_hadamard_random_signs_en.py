import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import deck_style as ds

# ==========================================
# 1. SIMULATION SETTINGS
# ==========================================
STAGES = 5                 # Number of stages (5 stages = 32 dimensions)
D = 2**STAGES              # Vector dimensionality
FRAMES_PER_STAGE = 30      # Number of frames to transition one stage
FPS = 25                   # Frames per second
TOTAL_FRAMES = STAGES * FRAMES_PER_STAGE

# OPTION: Whether to use Random Signs (Randomized Hadamard Transform)
USE_RANDOM_SIGNS = True 

# Create initial vector: 
# Base background + slight noise + ONE massive outlier (5-6x larger)
np.random.seed(42)
base_signal = np.ones(D) * 2.0                # Systematic bias / background
noise = np.random.normal(0, 0.2, D)           # Slight noise
initial_v = base_signal + noise

outlier_idx = int(D * 0.3)
initial_v[outlier_idx] = 12.0                 # The massive outlier spike (~6x background)

# Apply random signs if the option is enabled
if USE_RANDOM_SIGNS:
    random_signs = np.random.choice([-1, 1], size=D)
    initial_v = initial_v * random_signs

V_MAX = np.max(np.abs(initial_v)) * 1.2       # Scale X-axis for the histogram

# ==========================================
# 2. HADAMARD NETWORK MATHEMATICS
# ==========================================
def get_pairs(stage):
    """Returns pairs of indices for butterflies at the current stage."""
    stride = 2**stage
    pairs = []
    for i in range(D):
        if (i // stride) % 2 == 0:
            pairs.append((i, i + stride))
    return pairs

# Precompute vector states after each stage
V_exact = np.zeros((STAGES + 1, D))
V_exact[0] = initial_v
for s in range(STAGES):
    pairs = get_pairs(s)
    V_exact[s+1] = np.copy(V_exact[s])
    for (i, j) in pairs:
        vi, vj = V_exact[s, i], V_exact[s, j]
        V_exact[s+1, i] = (vi + vj) / np.sqrt(2)
        V_exact[s+1, j] = (-vi + vj) / np.sqrt(2)

def get_vector_at(time_val):
    s = int(min(time_val, STAGES - 0.001))
    t = time_val - s
    smooth_t = 0.5 - 0.5 * np.cos(np.pi * t)
    
    v_curr = np.copy(V_exact[s])
    pairs = get_pairs(s)
    angle = smooth_t * (-np.pi / 4)
    
    for (i, j) in pairs:
        vi, vj = V_exact[s, i], V_exact[s, j]
        v_curr[i] = vi * np.cos(angle) - vj * np.sin(angle)
        v_curr[j] = vi * np.sin(angle) + vj * np.cos(angle)
    return v_curr, s, smooth_t

# ==========================================
# 3. VISUALIZATION SETUP
# ==========================================
plt.style.use('dark_background')

# CHANGED: Adjusted width_ratios to make the histogram significantly wider
# Strict 16:9 so the video fills the 10 x 5.625in slide edge to edge.
fig, (ax_hist, ax_net) = plt.subplots(
    1, 2, 
    figsize=ds.FIGSIZE_16_9, dpi=ds.DPI_16_9, 
    gridspec_kw={'width_ratios': [1.4, 2.2], 'wspace': 0.08},
    sharey=True
)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.92)
# Flat deck black so the full-bleed frame blends into the slide artwork.
ds.apply_figure_background(fig)
ds.clear_axes_background(ax_hist, ax_net)

def update(frame):
    ax_hist.clear()
    ax_net.clear()
    # clear() restores the stylesheet fill, so re-apply transparency here.
    ds.clear_axes_background(ax_hist, ax_net)
    ax_hist.axis('off')
    ax_net.axis('off')
    
    span_x = (STAGES + 0.5) - (-0.5)
    span_y = (D - 0.5) - (-0.5)
    ax_net.set_xlim(-0.5, STAGES + 0.5)
    ax_net.set_ylim(D - 0.5, -0.5)
    ax_hist.set_xlim(-V_MAX, V_MAX)
    
    ax_hist.axvline(0, color='#30363d', lw=2, zorder=1)

    time_val = frame / FRAMES_PER_STAGE
    if time_val >= STAGES:
        time_val = STAGES - 0.001
        
    v_curr, active_stage, t_progress = get_vector_at(time_val)
    current_x = time_val

    # --- PANEL 1: Histogram ---
    colors = ['#57f287' if v >= 0 else '#ff4d6d' for v in v_curr]
    ax_hist.barh(range(D), v_curr, color=colors, height=0.6, alpha=0.9, zorder=3)
    
    title_hist = "Projection Amplitude"
    if USE_RANDOM_SIGNS:
        title_hist += "\n(+ Random Signs)"
    ax_hist.set_title(title_hist, color='#8b949e', fontsize=12, pad=15)

    for idx in range(D):
        ax_hist.axhline(idx, color='#30363d', lw=0.5, alpha=0.2, zorder=0)
        ax_net.axhline(idx, color='#30363d', lw=0.5, alpha=0.2, zorder=0)

    # --- PANEL 2: Hadamard Network ---
    pos = ax_net.get_position()
    width_inches = pos.width * fig.get_figwidth()
    height_inches = pos.height * fig.get_figheight()
    
    R_y = 0.38 
    R_x = R_y * (height_inches / width_inches) * (span_x / span_y)

    for s in range(STAGES):
        pairs = get_pairs(s)
        if s < active_stage:
            line_color, line_alpha = '#30363d', 0.3
        elif s == active_stage:
            line_color, line_alpha = '#4cc9f0', 0.8
        else:
            line_color, line_alpha = '#4b5563', 0.4

        for (i, j) in pairs:
            ax_net.plot([s, s+1], [i, i], color=line_color, alpha=line_alpha, lw=1.2)
            ax_net.plot([s, s+1], [j, j], color=line_color, alpha=line_alpha, lw=1.2)
            ax_net.plot([s, s+1], [i, j], color=line_color, alpha=line_alpha, lw=1.2)
            ax_net.plot([s, s+1], [j, i], color=line_color, alpha=line_alpha, lw=1.2)
            ax_net.scatter([s, s, s+1, s+1], [i, j, i, j], color=line_color, s=10, alpha=line_alpha, zorder=3)

    if USE_RANDOM_SIGNS:
        for idx in range(D):
            sign_str = "+" if random_signs[idx] > 0 else "-"
            sign_col = "#57f287" if random_signs[idx] > 0 else "#ff4d6d"
            ax_net.text(-0.35, idx, sign_str, color=sign_col, fontsize=10, 
                        ha='center', va='center', fontweight='bold', alpha=0.8)

    ax_net.axvline(current_x, color='#ff4d6d', linestyle='-', lw=2, alpha=0.8, zorder=4)

    # Drawing NORMALIZED 2D rotations
    active_pairs = get_pairs(active_stage)
    for (i, j) in active_pairs:
        center_x = active_stage + 0.5
        center_y = (i + j) / 2
        
        circle = patches.Ellipse((center_x, center_y), width=2*R_x, height=2*R_y, 
                                 facecolor=ds.DECK_BG, edgecolor='#4cc9f0', alpha=0.95, zorder=7)
        ax_net.add_patch(circle)
        
        ax_net.plot([center_x - R_x, center_x + R_x], [center_y, center_y], color='#8b949e', lw=0.5, alpha=0.5, zorder=8)
        ax_net.plot([center_x, center_x], [center_y - R_y, center_y + R_y], color='#8b949e', lw=0.5, alpha=0.5, zorder=8)
        
        vi_start, vj_start = V_exact[active_stage, i], V_exact[active_stage, j]
        angle = t_progress * (-np.pi / 4)
        v_x_rot = vi_start * np.cos(angle) - vj_start * np.sin(angle)
        v_y_rot = vi_start * np.sin(angle) + vj_start * np.cos(angle)
        
        norm_val = np.sqrt(v_x_rot**2 + v_y_rot**2)
        if norm_val > 1e-6:
            v_x_dir = v_x_rot / norm_val
            v_y_dir = v_y_rot / norm_val
        else:
            v_x_dir, v_y_dir = 0, 0
        
        end_x = center_x + v_x_dir * (R_x * 0.9)
        end_y = center_y + v_y_dir * (R_y * 0.9)
        
        ax_net.plot([center_x, end_x], [center_y, end_y], color='#ffb86c', lw=1.8, zorder=9)

    title_net = f"Walsh-Hadamard Network (Stages: {STAGES}, D={D}) — Outlier Dispersion"
    ax_net.set_title(title_net, color='white', fontsize=14, fontweight='bold', pad=15)

# ==========================================
# 4. SAVING THE ANIMATION (H.264, see deck_style.save_animation)
# ==========================================
anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES + int(FPS*1.5), interval=1000 // FPS)
ds.save_animation(anim, "turboquant_hadamard_random_signs_en.mp4", fps=FPS)