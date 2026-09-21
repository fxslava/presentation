import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools

import deck_style as ds

# ==========================================
# 1. НАСТРОЙКИ СИМУЛЯЦИИ И ДАННЫХ
# ==========================================
D = 256
BATCH = 8
# Матрица: 256 строк, 128 столбцов (16 из матрицы Адамара * 8 батч)
ROWS = 256
COLS = 128 
TILE_H = 16
TILE_W = 16
TILES_R = ROWS // TILE_H # 16 тайлов по вертикали
TILES_C = COLS // TILE_W # 8 тайлов по горизонтали
TOTAL_TILES = TILES_R * TILES_C # 128 тайлов на каждую стадию

FPS = 25
FRAMES_STAGE1 = 60
FRAMES_PAUSE1 = 20
FRAMES_STAGE2 = 60
FRAMES_PAUSE2 = 20
TOTAL_FRAMES = FRAMES_STAGE1 + FRAMES_PAUSE1 + FRAMES_STAGE2 + FRAMES_PAUSE2

np.random.seed(42)
initial_v = np.ones(D) * 1.5 + np.random.normal(0, 0.2, D)
initial_v[int(D * 0.3)] = 15.0
initial_v *= np.random.choice([-1, 1], size=D)

mid_v = initial_v * 0.4 + np.random.normal(0, 1.5, D)
final_v = np.random.normal(0, 1.0, D)
V_MAX = np.max(np.abs(initial_v)) * 1.1

# Цвета в формате RGBA
C_CYAN = np.array([76, 201, 240, 255]) / 255.0
C_PINK = np.array([255, 77, 109, 255]) / 255.0
C_BG = np.array([10, 10, 10, 255]) / 255.0

# Генерация базового паттерна
H_2 = np.array([[1, 1], [1, -1]])
H_4 = np.kron(H_2, H_2)
H_16 = np.kron(H_4, H_4)
pattern = np.kron(H_16, np.ones((16, 8))) # Растягиваем до 256x128

img1_data = np.zeros((ROWS, COLS, 4))
img2_data = np.zeros((ROWS, COLS, 4))

for r in range(ROWS):
    for c in range(COLS):
        color = C_CYAN if pattern[r, c] > 0 else C_PINK
        img1_data[r, c] = color
        img1_data[r, c, 3] = 0.15 # Базовая прозрачность
        img2_data[r, c] = color
        img2_data[r, c, 3] = 0.15

# Списки невычисленных тайлов для обеих стадий (Cube Path)
unvisited_1 = list(itertools.product(range(TILES_R), range(TILES_C)))
np.random.shuffle(unvisited_1)
visited_1 = 0

unvisited_2 = list(itertools.product(range(TILES_R), range(TILES_C)))
np.random.shuffle(unvisited_2)
visited_2 = 0

# ==========================================
# 2. НАСТРОЙКА ВИЗУАЛИЗАЦИИ
# ==========================================
plt.style.use('dark_background')
# Строгий формат 16:9, чтобы видео закрывало слайд 10 x 5.625in целиком.
fig, (ax_hist, ax_net) = plt.subplots(
    1, 2, figsize=ds.FIGSIZE_16_9, dpi=ds.DPI_16_9, 
    gridspec_kw={'width_ratios': [1.2, 2.5], 'wspace': 0.1}
)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.90)
# Ровный чёрный фон слайда: кадр во весь экран сливается с подложкой.
ds.apply_figure_background(fig)
ds.clear_axes_background(ax_hist, ax_net)
ax_hist.axis('off')
ax_net.axis('off')

# --- Гистограмма ---
ax_hist.set_ylim(D - 0.5, -0.5)
ax_hist.set_xlim(-V_MAX, V_MAX)
ax_hist.axvline(0, color='#30363d', lw=2, zorder=1)
bar_rects = ax_hist.barh(range(D), initial_v, height=0.8, color='#57f287')
ax_hist.set_title("Projection Amplitude", color='#8b949e', fontsize=14, pad=15)

# --- Матрицы и пайплайн ---
# Переводим шкалу X в абсолютные координаты:
# Матрица 1: 0 - 128
# Пробел: 128 - 144 (ширина 16)
# Матрица 2: 144 - 272
ax_net.set_xlim(-10, 282)
ax_net.set_ylim(ROWS + 40, -40) # Запас сверху и снизу для надписей
ax_net.set_title(f"Sylvester Factorization: Hardware Execution Pipeline\nBatch Size V={BATCH} (256x128 Matrices, 16x16 Tiles)", 
                 color='white', fontsize=16, fontweight='bold', pad=10)

im1 = ax_net.imshow(img1_data, extent=[0, 128, ROWS, 0], aspect='auto', zorder=2)
im2 = ax_net.imshow(img2_data, extent=[144, 272, ROWS, 0], aspect='auto', zorder=2)

# Отрисовка разметки батчей (каждые 16 столбцов)
for c in range(16, 128, 16):
    ax_net.plot([c, c], [0, ROWS], color='#8b949e', lw=1.5, zorder=3, alpha=0.6)
    ax_net.plot([144+c, 144+c], [0, ROWS], color='#8b949e', lw=1.5, zorder=3, alpha=0.6)

# Отрисовка горизонтальных линий тайлов (каждые 16 строк)
for r in range(16, ROWS, 16):
    ax_net.plot([0, 128], [r, r], color='#8b949e', lw=0.5, zorder=3, alpha=0.25)
    ax_net.plot([144, 272], [r, r], color='#8b949e', lw=0.5, zorder=3, alpha=0.25)

# Каретка и Барьеры
playhead = ax_net.axvline(0, color='#ffb86c', lw=3, zorder=10)

# Барьер 1 (ровно посередине между матрицами)
ax_net.axvline(136, color='#ff4d6d', lw=2, linestyle='--', alpha=0.6, zorder=5)
ax_net.text(136, -15, "Barrier 1\n(Cube Sync)", color='#ff4d6d', ha='center', fontsize=12, fontweight='bold')

# Барьер 2 (в самом конце)
ax_net.axvline(280, color='#57f287', lw=2, linestyle='--', alpha=0.6, zorder=5)
ax_net.text(280, -15, "Barrier 2\n(Cube Sync)", color='#57f287', ha='center', fontsize=12, fontweight='bold')

# Подписи стадий
ax_net.text(64, ROWS + 20, "STAGE 1: Cube Mmad\n(16x16 Tensor Tiles)", color='#4cc9f0', ha='center', fontsize=14)
ax_net.text(208, ROWS + 20, "STAGE 2: Cube Mmad\n(Residual Pass)", color='#57f287', ha='center', fontsize=14)

def update(frame):
    global visited_1, visited_2
    
    # 1. Расчет позиции каретки в Data Coordinates
    if frame <= FRAMES_STAGE1:
        x = (frame / FRAMES_STAGE1) * 128.0
    elif frame <= FRAMES_STAGE1 + FRAMES_PAUSE1:
        x = 136.0 # Стоит на барьере 1
    elif frame <= FRAMES_STAGE1 + FRAMES_PAUSE1 + FRAMES_STAGE2:
        local_f = frame - (FRAMES_STAGE1 + FRAMES_PAUSE1)
        x = 144.0 + (local_f / FRAMES_STAGE2) * 128.0
    else:
        x = 280.0 # Стоит на барьере 2
        
    playhead.set_xdata([x, x])
    
    # 2. Обновление Матрицы 1 (Cube Stage 1)
    img1_data[..., 3] = np.clip(img1_data[..., 3] * 0.85, 0.15, 1.0)
    if x <= 128.0:
        target_visits = int((x / 128.0) * TOTAL_TILES)
        while visited_1 < target_visits and unvisited_1:
            r, c = unvisited_1.pop()
            img1_data[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W, 3] = 1.0
            visited_1 += 1
    im1.set_data(img1_data)
    
    # 3. Обновление Матрицы 2 (Cube Stage 2)
    img2_data[..., 3] = np.clip(img2_data[..., 3] * 0.85, 0.15, 1.0)
    if 144.0 <= x <= 272.0:
        target_visits = int(((x - 144.0) / 128.0) * TOTAL_TILES)
        while visited_2 < target_visits and unvisited_2:
            r, c = unvisited_2.pop()
            img2_data[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W, 3] = 1.0
            visited_2 += 1
    im2.set_data(img2_data)

    # 4. Обновление Гистограммы
    if x <= 136.0:
        p = min(x / 128.0, 1.0)
        curr_v = initial_v * (1 - p) + mid_v * p
    else:
        p = min((x - 144.0) / 128.0, 1.0)
        curr_v = mid_v * (1 - p) + final_v * p

    for bar, val in zip(bar_rects, curr_v):
        bar.set_width(val)
        bar.set_color('#57f287' if val >= 0 else '#ff4d6d')

    return im1, im2, playhead, *bar_rects

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000//FPS, blit=False)
ds.save_animation(anim, "turboquant_matrix_pipeline.mp4", fps=FPS)