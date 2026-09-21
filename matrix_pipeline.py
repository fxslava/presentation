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
# Матрица: 256 строк, 128 столбцов (8 элементов батча * 16 шагов Адамара).
ROWS = 256
COLS = 128
TILE_H = 16
TILE_W = 16
TILES_R = ROWS // TILE_H # 16 тайлов по вертикали
TILES_C = COLS // TILE_W # 8 тайлов по горизонтали
TOTAL_TILES = TILES_R * TILES_C # 128 тайлов на каждую стадию

# Геометрия дорожек по оси X (в клетках матрицы).
M1_X0, M1_X1 = 0, COLS                  # первая матрица
GAP = 16                                # зазор между матрицами
M2_X0, M2_X1 = M1_X1 + GAP, M1_X1 + GAP + COLS
BARRIER_1_X = M1_X1 + GAP / 2           # ровно посередине зазора
BARRIER_2_X = M2_X1 + 6                 # сразу за второй матрицей

FPS = 25
# Обе стадии -- это вспышки Cube Mmad, а не медленный векторный проход:
# на просчёт даём вдвое меньше кадров, чем раньше (60 -> 28).
FRAMES_STAGE1 = 28
FRAMES_PAUSE1 = 12
FRAMES_STAGE2 = 28
FRAMES_PAUSE2 = 16
TOTAL_FRAMES = FRAMES_STAGE1 + FRAMES_PAUSE1 + FRAMES_STAGE2 + FRAMES_PAUSE2

np.random.seed(42)
initial_v = np.ones(D) * 1.5 + np.random.normal(0, 0.2, D)
initial_v[int(D * 0.3)] = 15.0
initial_v *= np.random.choice([-1, 1], size=D)

mid_v = initial_v * 0.4 + np.random.normal(0, 1.5, D)
final_v = np.random.normal(0, 1.0, D)
V_MAX = np.max(np.abs(initial_v)) * 1.1

# Цвета
C_CYAN = np.array([76, 201, 240]) / 255.0       # текстура знаков +1
C_PINK = np.array([255, 77, 109]) / 255.0       # текстура знаков -1
C_CUBE = np.array([87, 242, 135]) / 255.0       # вспышка Cube Mmad (обе стадии)
CUBE_GREEN = '#57f287'

BASE_ALPHA = 0.15       # непосчитанный тайл
DONE_HEAT = 0.35        # посчитанный тайл не гаснет полностью
HEAT_DECAY = 0.72       # короткая, резкая вспышка

# Генерация базового паттерна знаков
H_2 = np.array([[1, 1], [1, -1]])
H_4 = np.kron(H_2, H_2)
H_16 = np.kron(H_4, H_4)
# По одному знаку на compute-тайл: текстура ложится ровно на сетку 16x16,
# иначе половинки 16x8 читаются как вдвое более узкие столбцы.
pattern = np.kron(H_16[:, :TILES_C], np.ones((TILE_H, TILE_W)))  # 256x128

base_rgb = np.where(pattern[..., None] > 0, C_CYAN, C_PINK)

# heat: 0 -- тайл не тронут, 1 -- только что посчитан на Cube.
heat_1 = np.zeros((ROWS, COLS))
heat_2 = np.zeros((ROWS, COLS))
floor_1 = np.zeros((ROWS, COLS))
floor_2 = np.zeros((ROWS, COLS))

img1_data = np.zeros((ROWS, COLS, 4))
img2_data = np.zeros((ROWS, COLS, 4))


def compose(img, heat):
    """Тайл тем зеленее и непрозрачнее, чем свежее его посчитал Cube."""
    h = heat[..., None]
    img[..., :3] = base_rgb * (1.0 - h) + C_CUBE * h
    img[..., 3] = BASE_ALPHA + (1.0 - BASE_ALPHA) * heat


compose(img1_data, heat_1)
compose(img2_data, heat_2)

# Списки непосчитанных тайлов для обеих стадий (обе -- Cube Path)
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
    gridspec_kw={'width_ratios': [1.3, 3.0], 'wspace': 0.06}
)
# Вертикаль отдана матрицам: чем выше бокс, тем крупнее тайл 16x16.
fig.subplots_adjust(left=0.035, right=0.99, bottom=0.075, top=0.94)
# Ровный чёрный фон слайда: кадр во весь экран сливается с подложкой.
ds.apply_figure_background(fig)
ds.clear_axes_background(ax_hist, ax_net)
ax_hist.axis('off')
ax_net.axis('off')

# Обе оси идут в клетках матрицы и делят одну шкалу строк.
Y_TOP, Y_BOTTOM = -0.5, ROWS - 0.5

# --- Гистограмма ---
ax_hist.set_ylim(Y_BOTTOM, Y_TOP)
ax_hist.set_xlim(-V_MAX, V_MAX)
ax_hist.axvline(0, color='#30363d', lw=2, zorder=1)
bar_rects = ax_hist.barh(range(D), initial_v, height=1.0, color=CUBE_GREEN)
ax_hist.set_title("Projection Amplitude", color='#8b949e', fontsize=14, pad=15)

# --- Матрицы и пайплайн ---
ax_net.set_xlim(-6, BARRIER_2_X + 12)
ax_net.set_ylim(Y_BOTTOM, Y_TOP)
# Клетка строго квадратная: 256 строк против 128 столбцов читаются как 2:1,
# а тайл 16x16 остаётся тайлом, а не приплюснутым прямоугольником.
ax_net.set_aspect('equal', adjustable='box')
ax_net.set_title(
    "Sylvester Factorization: Hardware Execution Pipeline — both stages on CUBE\n"
    f"Batch V={BATCH}: 256x128 matrices (128 cols = {BATCH} x 16 Hadamard steps), 16x16 Cube tiles",
    color='white', fontsize=15, fontweight='bold', pad=8)

im1 = ax_net.imshow(img1_data, extent=[M1_X0, M1_X1, Y_BOTTOM, Y_TOP],
                    interpolation='nearest', zorder=2)
im2 = ax_net.imshow(img2_data, extent=[M2_X0, M2_X1, Y_BOTTOM, Y_TOP],
                    interpolation='nearest', zorder=2)

# Разметка: жирные границы тайлов 16x16 -- они же границы элементов батча.
for x0 in (M1_X0, M2_X0):
    for c in range(0, COLS + 1, TILE_W):
        edge = c in (0, COLS)
        ax_net.plot([x0 + c, x0 + c], [Y_TOP, Y_BOTTOM], color='#8b949e',
                    lw=1.6 if edge else 1.1, alpha=0.85 if edge else 0.55, zorder=4)
    for r in range(0, ROWS + 1, TILE_H):
        edge = r in (0, ROWS)
        ax_net.plot([x0, x0 + COLS], [r + Y_TOP, r + Y_TOP], color='#8b949e',
                    lw=1.6 if edge else 1.1, alpha=0.85 if edge else 0.45, zorder=4)

# Каретка и барьеры. Подписи: x в клетках, y в долях оси -- их не режет ylim.
label_tr = ax_net.get_xaxis_transform()
playhead = ax_net.axvline(M1_X0, color='#ffb86c', lw=3, zorder=10)

# Подписи идут вдоль самих линий барьеров -- над матрицей их съедает заголовок.
for bx, bcolor, blabel in ((BARRIER_1_X, '#ff4d6d', "Barrier 1 — Cube Sync"),
                           (BARRIER_2_X, CUBE_GREEN, "Barrier 2 — Cube Sync")):
    ax_net.axvline(bx, color=bcolor, lw=2, linestyle='--', alpha=0.6, zorder=5)
    ax_net.text(bx, 0.5, blabel, transform=label_tr, color=bcolor, rotation=90,
                ha='center', va='center', fontsize=12, fontweight='bold')

# Обе стадии считает Cube, поэтому подписи одного цвета и одного типа.
ax_net.text((M1_X0 + M1_X1) / 2, -0.015, "STAGE 1: CUBE Mmad\n(16x16 tensor tiles)",
            transform=label_tr, color=CUBE_GREEN, ha='center', va='top', fontsize=11)
ax_net.text((M2_X0 + M2_X1) / 2, -0.015,
            f"STAGE 2: CUBE Mmad\n(residual Hadamard stages, V={BATCH} keeps tiles full)",
            transform=label_tr, color=CUBE_GREEN, ha='center', va='top', fontsize=11)

def update(frame):
    global visited_1, visited_2

    # 1. Расчет позиции каретки в Data Coordinates
    if frame <= FRAMES_STAGE1:
        x = M1_X0 + (frame / FRAMES_STAGE1) * COLS
    elif frame <= FRAMES_STAGE1 + FRAMES_PAUSE1:
        x = BARRIER_1_X # Стоит на барьере 1
    elif frame <= FRAMES_STAGE1 + FRAMES_PAUSE1 + FRAMES_STAGE2:
        local_f = frame - (FRAMES_STAGE1 + FRAMES_PAUSE1)
        x = M2_X0 + (local_f / FRAMES_STAGE2) * COLS
    else:
        x = BARRIER_2_X # Стоит на барьере 2

    playhead.set_xdata([x, x])

    # 2. Стадия 1: Cube Mmad
    heat_1[:] = np.maximum(heat_1 * HEAT_DECAY, floor_1)
    if x <= M1_X1:
        target_visits = int(((x - M1_X0) / COLS) * TOTAL_TILES)
        while visited_1 < target_visits and unvisited_1:
            r, c = unvisited_1.pop()
            sl = (slice(r*TILE_H, (r+1)*TILE_H), slice(c*TILE_W, (c+1)*TILE_W))
            heat_1[sl] = 1.0
            floor_1[sl] = DONE_HEAT     # посчитанный тайл остаётся подсвеченным
            visited_1 += 1
    compose(img1_data, heat_1)
    im1.set_data(img1_data)

    # 3. Стадия 2: тот же Cube Mmad, те же вспышки
    heat_2[:] = np.maximum(heat_2 * HEAT_DECAY, floor_2)
    if M2_X0 <= x <= M2_X1:
        target_visits = int(((x - M2_X0) / COLS) * TOTAL_TILES)
        while visited_2 < target_visits and unvisited_2:
            r, c = unvisited_2.pop()
            sl = (slice(r*TILE_H, (r+1)*TILE_H), slice(c*TILE_W, (c+1)*TILE_W))
            heat_2[sl] = 1.0
            floor_2[sl] = DONE_HEAT
            visited_2 += 1
    compose(img2_data, heat_2)
    im2.set_data(img2_data)

    # 4. Обновление Гистограммы
    if x <= BARRIER_1_X:
        p = min((x - M1_X0) / COLS, 1.0)
        curr_v = initial_v * (1 - p) + mid_v * p
    else:
        p = min(max((x - M2_X0) / COLS, 0.0), 1.0)
        curr_v = mid_v * (1 - p) + final_v * p

    for bar, val in zip(bar_rects, curr_v):
        bar.set_width(val)
        bar.set_color(CUBE_GREEN if val >= 0 else '#ff4d6d')

    return im1, im2, playhead, *bar_rects

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000//FPS, blit=False)
ds.save_animation(anim, "turboquant_matrix_pipeline.mp4", fps=FPS)
