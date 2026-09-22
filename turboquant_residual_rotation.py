import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches

# Принудительно указываем Matplotlib путь к локальному ffmpeg
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# ==========================================
# 1. ПАРАМЕТРЫ СИМУЛЯЦИИ И КВАНТОВАНИЯ
# ==========================================
D = 24  # Размерность: 24 (даст ровно 12 двумерных проекций, сетка 4x3)
FPS = 30
DURATION = 8  # Длительность в секундах
TOTAL_FRAMES = FPS * DURATION

# Цвета (Black & Gold Premium Theme)
BG_COLOR = '#0a0a0a'
TEXT_COLOR = '#d4af37'
GOLD = '#d4af37'
GRAY = '#8b949e'
GREEN = '#57f287'

# Фиксированный Query-вектор для вычисления скалярного произведения
np.random.seed(42)
query_vector = np.random.randn(D)
query_vector /= np.linalg.norm(query_vector)

# Амплитуды, фазы и скорости вращения для 12 плоскостей
amplitudes = np.random.uniform(0.6, 1.2, D//2)
phases = np.random.uniform(0, 2*np.pi, D//2)
speeds = np.random.uniform(-1.5, 1.5, D//2)

# Настройка квантования (4-bit)
QUANT_BINS = 15 
MAX_VAL = 1.5

def quantize_4bit(v, max_val=MAX_VAL):
    scale = max_val / 7.0
    v_q = np.round(np.clip(v, -max_val, max_val) / scale) * scale
    return v_q

def quantize_1bit_residual(residual):
    alpha = np.mean(np.abs(residual))
    v_corr = alpha * np.sign(residual)
    v_corr[v_corr == 0] = alpha
    return v_corr

# ==========================================
# 2. НАСТРОЙКА ВИЗУАЛИЗАЦИИ
# ==========================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(BG_COLOR)

# Сетка: 4 строки, 4 колонки. Первые три колонки для проекций, четвертая для Dot Product
gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 1.3], hspace=0.3, wspace=0.15)

axes = []
for row in range(4):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

# График Dot Product занимает всю четвертую колонку (от 0 до 4 строки)
ax_dp = fig.add_subplot(gs[:, 3]) 

for i, ax in enumerate(axes):
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-MAX_VAL, MAX_VAL)
    ax.set_ylim(-MAX_VAL, MAX_VAL)
    ax.set_aspect('equal')
    ax.set_title(f"Dims {i*2} & {i*2+1}", color=TEXT_COLOR, fontsize=9, pad=3)
    ax.set_xticks([]) # Убираем лишние цифры с осей для чистоты
    ax.set_yticks([])
    
    # Сетка квантования
    for tick in np.linspace(-MAX_VAL, MAX_VAL, 15):
        ax.axhline(tick, color='#1f2428', lw=0.5, zorder=0)
        ax.axvline(tick, color='#1f2428', lw=0.5, zorder=0)

ax_dp.set_facecolor(BG_COLOR)
# Лимиты Y немного расширены из-за увеличенной размерности вектора D=24
ax_dp.set_ylim(-2.0, 2.0) 
ax_dp.set_xlim(-0.5, 2.5)
ax_dp.set_xticks([0, 1, 2])
ax_dp.set_xticklabels(["FP16\n(Reference)", "Raw\n4-bit", "4-bit +\n1-bit"], color=TEXT_COLOR, fontsize=12)
ax_dp.set_title("Dot Product Preservation\n(Query · Key)", color=TEXT_COLOR, fontsize=14, pad=15)
ax_dp.axhline(0, color='#30363d', lw=2)

quivers_ref = []
quivers_q = []
quivers_corr = []
bars = ax_dp.bar([0, 1, 2], [0, 0, 0], color=[GOLD, GRAY, GREEN], width=0.7)

for ax in axes:
    q_ref = ax.quiver(0, 0, 0, 0, color=GOLD, angles='xy', scale_units='xy', scale=1, width=0.015, zorder=3)
    q_q = ax.quiver(0, 0, 0, 0, color=GRAY, angles='xy', scale_units='xy', scale=1, width=0.022, zorder=2, alpha=0.6)
    q_corr = ax.quiver(0, 0, 0, 0, color=GREEN, angles='xy', scale_units='xy', scale=1, width=0.012, zorder=4)
    quivers_ref.append(q_ref)
    quivers_q.append(q_q)
    quivers_corr.append(q_corr)

fig.text(0.5, 0.02, "TurboQuant Polar/Residual Quantization Simulation (D=24)", 
         ha='center', va='center', color=TEXT_COLOR, fontsize=14, fontweight='bold')

# ==========================================
# 3. АНИМАЦИЯ
# ==========================================
def update(frame):
    t = (frame / TOTAL_FRAMES) * 2 * np.pi
    
    v_ref = np.zeros(D)
    for i in range(D // 2):
        v_ref[i*2]   = amplitudes[i] * np.cos(speeds[i]*t + phases[i])
        v_ref[i*2+1] = amplitudes[i] * np.sin(speeds[i]*t + phases[i])
        
    v_q = quantize_4bit(v_ref)
    residual = v_ref - v_q
    v_corr = quantize_1bit_residual(residual)
    v_recon = v_q + v_corr

    for i in range(D // 2):
        idx_x, idx_y = i*2, i*2+1
        quivers_ref[i].set_UVC(v_ref[idx_x], v_ref[idx_y])
        quivers_q[i].set_UVC(v_q[idx_x], v_q[idx_y])
        quivers_corr[i].set_offsets(np.c_[v_q[idx_x], v_q[idx_y]])
        quivers_corr[i].set_UVC(v_corr[idx_x], v_corr[idx_y])

    dp_ref = np.dot(query_vector, v_ref)
    dp_q = np.dot(query_vector, v_q)
    dp_recon = np.dot(query_vector, v_recon)
    
    vals = [dp_ref, dp_q, dp_recon]
    for bar, val in zip(bars, vals):
        bar.set_height(val)
        
    [t.remove() for t in ax_dp.texts] 
    for i, val in enumerate(vals):
        y_pos = val + 0.08 if val >= 0 else val - 0.18
        ax_dp.text(i, y_pos, f"{val:.3f}", ha='center', color='white', fontweight='bold', fontsize=12)

    return quivers_ref + quivers_q + quivers_corr + list(bars)

# ==========================================
# 4. СОХРАНЕНИЕ
# ==========================================
print("Рендеринг видео (4x3 Grid)...")
anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000//FPS, blit=False)

writer = FFMpegWriter(fps=FPS, bitrate=6000, metadata=dict(title='TurboQuant Residual Simulation 4x3 Grid'))
output_file = "turboquant_1bit_correction.mp4"
anim.save(output_file, writer=writer)

print(f"Готово! Анимация сохранена как '{output_file}'")