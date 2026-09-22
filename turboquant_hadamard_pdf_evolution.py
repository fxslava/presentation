import sys

# Сообщения на русском: в cp1252-консоли print() иначе падает.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from scipy.stats import ortho_group, gaussian_kde
import imageio_ffmpeg

try:
    import deck_style as ds
    HAS_DECK_STYLE = True
except ImportError:
    HAS_DECK_STYLE = False

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# ==========================================
# 1. ПАРАМЕТРЫ СИМУЛЯЦИИ 
# ==========================================
# 1 = FP4 (E2M1) vs INT4
# 2 = FP4 (E1M2) vs Lloyd-Max
# Режим можно задать аргументом: python turboquant_hadamard_pdf_evolution.py 1
COMPARISON_MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 2

# Имена под слайды колоды.
OUT_FILES = {
    1: "turboquant_fp4_vs_int4_fwht.mp4",
    2: "turboquant_e1m2_vs_lloydmax_stem.mp4",
}

np.random.seed(42)
D = 128          
FRAMES = 120     
FPS = 30         

x_initial = np.random.normal(0.0, 0.3, size=D)
outlier_indices = [15, 42, 85, 110]
x_initial[outlier_indices] = [7.5, -6.8, 8.2, -7.0]

Q = ortho_group.rvs(dim=D)
eigenvalues, V = np.linalg.eig(Q)
log_eigenvalues = np.log(eigenvalues)

def get_rotation_matrix(t: float) -> np.ndarray:
    smooth_t = 0.5 - 0.5 * np.cos(np.pi * t)
    interpolated_diag = np.exp(smooth_t * log_eigenvalues)
    return np.real(V @ np.diag(interpolated_diag) @ np.linalg.inv(V))

color_spike = np.array([1.0, 0.30, 0.43])      
color_gaussian = np.array([0.30, 0.79, 0.94])   

BG_COLOR = '#0a0a0a'
TEXT_COLOR = '#d4af37'
GOLD = '#d4af37'
GRAY = '#8b949e'
GREEN = '#57f287'
grid_color = '#30363d'

# ==========================================
# 2. НАСТРОЙКИ СЕТОК КВАНТОВАНИЯ
# ==========================================
if COMPARISON_MODE == 1:
    # Mode 1: FP4 (E2M1) vs INT4 (Uniform absmax)
    grid_left_positive = np.array([0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0])
    grid_right_positive = np.linspace(1/16, 15/16, 8)
    titles_quant = [
        "FP4 (E2M1 Log-Grid)\nOutlier Capture -> Grid Waste", 
        "INT4 (Uniform absmax/8)\nGrid Starvation -> Perfect Fit"
    ]
else:
    # Mode 2: FP4 (E1M2) vs Lloyd-Max
    # E=0: 0.0, 0.25, 0.5, 0.75 | E=1: 2.0, 2.5, 3.0, 3.5
    # Нормализовано делением на 3.5
    grid_left_positive = np.array([0.0, 0.071, 0.143, 0.214, 0.571, 0.714, 0.857, 1.0])
    
    # Lloyd-Max optimal 4-bit levels for Gaussian N(0,1)
    grid_right_positive = np.array([0.047, 0.142, 0.240, 0.345, 0.460, 0.592, 0.757, 1.0])
    
    titles_quant = [
        "FP4 (E1M2)\nGiant Gap Between Exponents", 
        "Lloyd-Max (Non-Linear Optimal)\nDensity-Matched for Gaussian"
    ]

levels_left = np.unique(np.concatenate((-grid_left_positive, grid_left_positive)))
levels_right = np.unique(np.concatenate((-grid_right_positive, grid_right_positive)))
levels_left.sort()
levels_right.sort()

def get_bin_boundaries(levels):
    bounds = (levels[:-1] + levels[1:]) / 2.0
    return np.concatenate(([-1.1], bounds, [1.1]))

bounds_left = get_bin_boundaries(levels_left)
bounds_right = get_bin_boundaries(levels_right)

def quantize_and_hist(v, levels, bounds):
    max_v = np.max(np.abs(v))
    if max_v == 0: return np.zeros(len(levels)), 0
    v_norm = v / max_v

    idx = np.abs(v_norm[:, None] - levels[None, :]).argmin(axis=1)
    v_q = levels[idx] * max_v

    noise_power = np.mean((v - v_q)**2)
    signal_power = np.mean(v**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-9 else 99.0

    hist_bins = np.concatenate(([-np.inf], bounds[1:-1], [np.inf]))
    counts, _ = np.histogram(v_norm, bins=hist_bins)
    return counts, snr


# ==========================================
# 2b. KL-ДИВЕРГЕНЦИЯ: СКОЛЬКО ИНФОРМАЦИИ СЪЕДАЕТ СЕТКА
# ==========================================
# SNR меряет энергию ошибки, но ничего не говорит о том, сохранила ли сетка
# форму распределения. KL(p || q) как раз про это: p -- непрерывная плотность
# сигнала (KDE), q -- плотность, которую способна выразить сетка: вся масса
# ячейки размазана по её ширине, потому что внутри ячейки квантователь не
# различает ничего. Узкие ячейки там, где плотность высокая -> KL мал.
# Точек на ячейку. Интегрировать по общей сетке нельзя: её узлы не попадают
# на границы ячеек, крайние срезы теряются, масса ячейки занижается -- и KL
# раздувается тем сильнее, чем уже ячейка. Отдельная сетка на ячейку ставит
# границы точно; значение сходится уже к 32 точкам.
KL_CELL_SAMPLES = 64


def signal_pdf(v_norm):
    """Непрерывная плотность сигнала (KDE). None, если оценивать нечего."""
    if np.allclose(v_norm, v_norm[0]):
        return None
    return gaussian_kde(v_norm)


def kl_divergence_bits(kde, bounds):
    """KL(p || q) в битах между плотностью сигнала и разрешением сетки.

    q кусочно-постоянна: на ячейке [b_i, b_i+1] она равна вероятностной массе
    этой ячейки, делённой на её ширину -- ровно то, что квантователь способен
    сказать о значении внутри ячейки. Узкие ячейки там, где плотность высокая,
    дают маленький KL; широкий провал между экспонентами FP4 -- большой.
    """
    if kde is None:
        return 0.0

    cells = [np.linspace(lo, hi, KL_CELL_SAMPLES)
             for lo, hi in zip(bounds[:-1], bounds[1:])]
    densities = [np.maximum(kde(cell), 0.0) for cell in cells]

    # Нормировка на носитель сетки, чтобы сумма масс ячеек была ровно 1.
    total = sum(np.trapezoid(p, cell) for p, cell in zip(densities, cells))
    if total <= 0:
        return 0.0

    kl = 0.0
    for p, cell in zip(densities, cells):
        p = p / total
        mass = np.trapezoid(p, cell)
        width = cell[-1] - cell[0]
        if mass <= 1e-12 or width <= 0:
            continue
        nz = p > 1e-12
        if not nz.any():
            continue
        kl += np.trapezoid(p[nz] * np.log2(p[nz] / (mass / width)), cell[nz])
    return max(kl, 0.0)

# ==========================================
# 3. ПОДГОТОВКА ХОЛСТА И ГРАФИКОВ
# ==========================================
plt.style.use('dark_background')
figsize = (16, 9) if not HAS_DECK_STYLE else ds.FIGSIZE_16_9
dpi = 120 if not HAS_DECK_STYLE else ds.DPI_16_9

fig = plt.figure(figsize=figsize, dpi=dpi)
if HAS_DECK_STYLE:
    ds.apply_figure_background(fig)
else:
    fig.patch.set_facecolor(BG_COLOR)

gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 2], hspace=0.4, wspace=0.25)

ax_sig = fig.add_subplot(gs[0, :])
ax_sig.set_facecolor(BG_COLOR)
y_max_coord = 9.0
ax_sig.set_ylim(-y_max_coord, y_max_coord)
ax_sig.set_xlim(-2, D + 2)
ax_sig.set_title("TurboQuant: Outlier Energy Smearing via Random Orthogonal Rotation", 
                 fontsize=14, fontweight='bold', color='white')
ax_sig.set_xlabel("Channel / Dimension Index", fontsize=10, color=GRAY)
ax_sig.set_ylabel("Activation Amplitude", fontsize=10, color=GRAY)
ax_sig.grid(True, color=grid_color, linestyle=':', alpha=0.6)

stem_container = None

ax_left = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[1, 1])

axes_quant = [ax_left, ax_right]
colors_quant = [color_spike, color_gaussian]
levels_list = [levels_left, levels_right]
bounds_list = [bounds_left, bounds_right]
snr_texts = []
hist_patches = [[], []] 

x_grid = np.linspace(-1.1, 1.1, 1000)

for i, ax in enumerate(axes_quant):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(titles_quant[i], color=TEXT_COLOR, fontsize=12, pad=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Normalized Magnitude (absmax)", color=GRAY)
    ax.tick_params(colors=GRAY)
    ax.grid(color=grid_color, linestyle='--', linewidth=0.5)
    
    levels = levels_list[i]
    bounds = bounds_list[i]
    
    y_step = levels[np.abs(x_grid[:, None] - levels[None, :]).argmin(axis=1)]
    ax.step(x_grid, y_step, color=colors_quant[i], lw=2, where='mid', zorder=3)
    
    for j in range(len(levels)):
        left = bounds[j]
        width = bounds[j+1] - bounds[j]
        base_y = levels[j]
        rect = Rectangle((left, base_y), width, 0, facecolor=GOLD, alpha=0.6, zorder=2)
        ax.add_patch(rect)
        hist_patches[i].append(rect)
    
    txt = ax.text(0.05, 0.95, "SNR: 0.00 dB\nKL:  0.000 bit", transform=ax.transAxes,
                  color=colors_quant[i], fontsize=14, fontweight='bold', va='top',
                  linespacing=1.4)
    snr_texts.append(txt)

# ==========================================
# 4. ЛОГИКА АНИМАЦИИ
# ==========================================
def update(frame: int):
    global stem_container
    
    t = frame / (FRAMES - 1)
    R_t = get_rotation_matrix(t)
    x_current = R_t @ x_initial
    
    current_color = (1.0 - t) * color_spike + t * color_gaussian

    ax_sig.clear()
    ax_sig.set_facecolor(BG_COLOR)
    ax_sig.set_ylim(-y_max_coord, y_max_coord)
    ax_sig.set_xlim(-2, D + 2)
    ax_sig.set_title(f"Coordinate Components (D = {D}) | Rotation Progress: {int(t * 100)}%", 
                     fontsize=12, fontweight='bold', color='white')
    ax_sig.set_xlabel("Channel / Dimension Index", fontsize=10, color=GRAY)
    ax_sig.set_ylabel("Activation Amplitude", fontsize=10, color=GRAY)
    ax_sig.grid(True, color=grid_color, linestyle=':', alpha=0.6)

    markerline, stemlines, _ = ax_sig.stem(
        range(D), x_current, linefmt='-', markerfmt='o', basefmt=' '
    )
    plt.setp(stemlines, color=current_color, alpha=0.7, linewidth=1.3)
    plt.setp(markerline, color=current_color, markersize=3.5)

    curr_max = np.max(np.abs(x_current))
    ax_sig.axhline(curr_max, color='#fca311', linestyle='--', alpha=0.75, 
                   label=f'Max magnitude: {curr_max:.2f}')
    ax_sig.axhline(-curr_max, color='#fca311', linestyle='--', alpha=0.75)
    ax_sig.legend(loc='upper right', framealpha=0.35, fontsize=9)

    # Плотность считается один раз: нормировка v/absmax от сетки не зависит.
    kde_current = signal_pdf(x_current / np.max(np.abs(x_current)))

    for i, levels in enumerate(levels_list):
        bounds = bounds_list[i]
        counts, snr = quantize_and_hist(x_current, levels, bounds)
        kl = kl_divergence_bits(kde_current, bounds)

        for j, rect in enumerate(hist_patches[i]):
            scaled_height = (counts[j] / D) * 0.95
            rect.set_height(scaled_height)
            rect.set_facecolor(current_color)

        snr_texts[i].set_text(f"SNR: {snr:.2f} dB\nKL:  {kl:.3f} bit")
        if t > 0.9:
            if "Lloyd" in titles_quant[i] or "INT4" in titles_quant[i]:
                snr_texts[i].set_color(GREEN)
            else:
                snr_texts[i].set_color(color_spike)

    return [] 

# ==========================================
# 5. СОХРАНЕНИЕ
# ==========================================
if __name__ == "__main__":
    mode_name = "FP4_vs_INT4" if COMPARISON_MODE == 1 else "E1M2_vs_LloydMax"
    print(f"Рендеринг анимации (Режим: {mode_name})...")
    anim = FuncAnimation(fig, update, frames=FRAMES, interval=1000 // FPS)

    out_file = OUT_FILES[COMPARISON_MODE]
    if HAS_DECK_STYLE:
        ds.save_animation(anim, out_file, fps=FPS)
    else:
        writer = FFMpegWriter(fps=FPS, bitrate=6000)
        anim.save(out_file, writer=writer)
    
    print(f"Готово! Анимация сохранена как {out_file}.")