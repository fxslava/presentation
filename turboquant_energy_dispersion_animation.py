import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import ortho_group, norm

import deck_style as ds

# 1. Simulation parameters
np.random.seed(42)
D = 128          # Feature dimension (channels)
FRAMES = 60      # Total animation frames
FPS = 20

# 2. Spike activation vector (typical transformer activation with heavy outliers)
x_initial = np.random.normal(0.0, 0.3, size=D)
outlier_indices = [15, 42, 85, 110]
x_initial[outlier_indices] = [7.5, -6.8, 8.2, -7.0]

# 3. Haar-distributed random orthogonal rotation matrix Q in O(D)
Q = ortho_group.rvs(dim=D)

# Continuous geodesic interpolation on SO(D): R(t) = exp(t * log(Q))
eigenvalues, V = np.linalg.eig(Q)
log_eigenvalues = np.log(eigenvalues)

def get_rotation_matrix(t: float) -> np.ndarray:
    """Computes smooth geodesic rotation at progress t in [0, 1]."""
    # Smoothstep acceleration profile for presentation clarity
    smooth_t = 0.5 - 0.5 * np.cos(np.pi * t)
    interpolated_diag = np.exp(smooth_t * log_eigenvalues)
    return np.real(V @ np.diag(interpolated_diag) @ np.linalg.inv(V))

# 4. Canvas configuration
plt.style.use('dark_background')
# Strict 16:9 so the video fills the 10 x 5.625in slide edge to edge.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ds.FIGSIZE_16_9, dpi=ds.DPI_16_9)
# Flat deck black so the full-bleed frame blends into the slide artwork.
ds.apply_figure_background(fig)
ds.clear_axes_background(ax1, ax2)

color_spike = np.array([1.0, 0.30, 0.43])      # Vivid crimson for outliers
color_gaussian = np.array([0.30, 0.79, 0.94])   # Cyan for Gaussianized states
grid_color = '#30363d'

y_max_coord = 9.0
x_max_hist = 9.0
bins = np.linspace(-x_max_hist, x_max_hist, 45)

def update(frame: int):
    ax1.clear()
    ax2.clear()
    # clear() restores the stylesheet fill, so re-apply transparency here.
    ds.clear_axes_background(ax1, ax2)

    t = frame / (FRAMES - 1)
    R_t = get_rotation_matrix(t)
    x_current = R_t @ x_initial

    # Linear color blend reflecting transformation progress
    current_color = (1.0 - t) * color_spike + t * color_gaussian

    # --- Left Subplot: Coordinate-wise Amplitude Profile ---
    markerline, stemlines, _ = ax1.stem(
        range(D), x_current, linefmt='-', markerfmt='o', basefmt=' '
    )
    plt.setp(stemlines, color=current_color, alpha=0.7, linewidth=1.3)
    plt.setp(markerline, color=current_color, markersize=3.5)

    curr_max = np.max(np.abs(x_current))
    ax1.axhline(curr_max, color='#fca311', linestyle='--', alpha=0.75,
                label=f'Max magnitude: {curr_max:.2f}')
    ax1.axhline(-curr_max, color='#fca311', linestyle='--', alpha=0.75)

    ax1.set_ylim(-y_max_coord, y_max_coord)
    ax1.set_xlim(-2, D + 2)
    ax1.set_title(f"Coordinate Components (D = {D})\nRotation Progress: {int(t * 100)}%",
                  fontsize=11, fontweight='bold', color='white')
    ax1.set_xlabel("Channel / Dimension Index", fontsize=10, color='#8b949e')
    ax1.set_ylabel("Activation Amplitude", fontsize=10, color='#8b949e')
    ax1.grid(True, color=grid_color, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.35, fontsize=9)

    # --- Right Subplot: Empirical Density vs. CLT Gaussian Limit ---
    ax2.hist(
        x_current, bins=bins, density=True, color=current_color,
        alpha=0.65, edgecolor=ds.DECK_BG
    )

    # Theoretical variance by norm conservation: sigma^2 = ||x||^2 / D
    theoretical_std = np.linalg.norm(x_initial) / np.sqrt(D)
    x_axis = np.linspace(-x_max_hist, x_max_hist, 250)
    gauss_curve = norm.pdf(x_axis, 0.0, theoretical_std)

    ax2.plot(
        x_axis, gauss_curve, color='#57f287', linestyle='-', linewidth=2.0,
        label=f'CLT Limit: N(0, σ² = {theoretical_std**2:.2f})'
    )

    ax2.set_ylim(0.0, 0.65)
    ax2.set_xlim(-x_max_hist, x_max_hist)
    ax2.set_title("Coordinate Value Distribution\n(Density vs. Quantization Range)",
                  fontsize=11, fontweight='bold', color='white')
    ax2.set_xlabel("Component Value", fontsize=10, color='#8b949e')
    ax2.set_ylabel("Probability Density", fontsize=10, color='#8b949e')
    ax2.grid(True, color=grid_color, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.35, fontsize=9)

    plt.suptitle("TurboQuant: Outlier Energy Smearing via Random Orthogonal Rotation",
                 fontsize=13, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout()

if __name__ == "__main__":
    anim = FuncAnimation(fig, update, frames=FRAMES, interval=1000 // FPS)
    ds.save_animation(anim, "turboquant_dispersion_en.mp4", fps=FPS)
    plt.close()