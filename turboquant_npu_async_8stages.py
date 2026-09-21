import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import deck_style as ds

# ==========================================
# 1. НАСТРОЙКИ СИМУЛЯЦИИ NPU
# ==========================================
STAGES = 8                 # 8 ступеней = D=256
D = 2**STAGES
FRAMES_PER_STAGE = 40      # Кадров на каждую ступень
TOTAL_FRAMES = STAGES * FRAMES_PER_STAGE
FPS = 25

np.random.seed(42)
base_signal = np.ones(D) * 1.5
noise = np.random.normal(0, 0.2, D)
initial_v = base_signal + noise
initial_v[int(D * 0.3)] = 15.0  
initial_v *= np.random.choice([-1, 1], size=D) 

V_MAX = np.max(np.abs(initial_v)) * 1.1

# ==========================================
# 2. ЛОГИКА АСИНХРОННЫХ БАБОЧЕК
# ==========================================
def get_pairs(stage):
    stride = 2**stage
    pairs = []
    for i in range(D):
        if (i // stride) % 2 == 0:
            pairs.append((i, i + stride))
    return pairs

V_exact = np.zeros((STAGES + 1, D))
V_exact[0] = initial_v
for s in range(STAGES):
    pairs = get_pairs(s)
    V_exact[s+1] = np.copy(V_exact[s])
    for (i, j) in pairs:
        vi, vj = V_exact[s, i], V_exact[s, j]
        V_exact[s+1, i] = (vi + vj) / np.sqrt(2)
        V_exact[s+1, j] = (-vi + vj) / np.sqrt(2)

np.random.seed(101)
thread_delays = np.random.uniform(0, 0.4, size=(STAGES, len(get_pairs(0))))
thread_durations = np.random.uniform(0.4, 0.6, size=(STAGES, len(get_pairs(0))))

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ
# ==========================================
plt.style.use('dark_background')
# Строгий формат 16:9, чтобы видео закрывало слайд 10 x 5.625in целиком.
fig, (ax_hist, ax_net) = plt.subplots(
    1, 2, figsize=ds.FIGSIZE_16_9, dpi=ds.DPI_16_9, 
    gridspec_kw={'width_ratios': [1.2, 2.5], 'wspace': 0.05}, sharey=True
)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.92)
# Ровный чёрный фон слайда: кадр во весь экран сливается с подложкой.
ds.apply_figure_background(fig)
ds.clear_axes_background(ax_hist, ax_net)

ax_hist.axis('off')
ax_net.axis('off')
ax_net.set_xlim(-0.5, STAGES + 0.5)
ax_net.set_ylim(D - 0.5, -0.5)
ax_hist.set_xlim(-V_MAX, V_MAX)

# Фоновая сеть заменена на тонкие белые линии
bg_lines = []
for s in range(STAGES):
    for (i, j) in get_pairs(s):
        bg_lines.extend([[(s, i), (s+1, i)], [(s, j), (s+1, j)], 
                         [(s, i), (s+1, j)], [(s, j), (s+1, i)]])
lc = LineCollection(bg_lines, color='white', alpha=0.15, linewidths=0.4)
ax_net.add_collection(lc)

bar_rects = ax_hist.barh(range(D), initial_v, height=0.8, color=['#57f287' if v >= 0 else '#ff4d6d' for v in initial_v])
ax_hist.axvline(0, color='#30363d', lw=2, zorder=1)
ax_hist.set_title("Projection Amplitude (Asynchronous Update)", color='#8b949e', fontsize=12, pad=15)
ax_net.set_title(f"AIV Asynchronous Threads (D={D}) — Barrier Synchronization", 
                 color='white', fontsize=14, fontweight='bold', pad=15)

barriers = [ax_net.axvline(x=s+1, color='#ff4d6d', lw=2, alpha=0.2) for s in range(STAGES)]
scatter = ax_net.scatter([], [], color='#4cc9f0', s=8, zorder=5)

def update(frame):
    current_stage = min(frame // FRAMES_PER_STAGE, STAGES - 1)
    stage_frame = frame % FRAMES_PER_STAGE
    t_norm = stage_frame / FRAMES_PER_STAGE
    
    pairs = get_pairs(current_stage)
    v_curr = np.copy(V_exact[current_stage])
    
    points_x, points_y = [], []
    all_done = True
    
    for idx, (i, j) in enumerate(pairs):
        delay = thread_delays[current_stage, idx]
        duration = thread_durations[current_stage, idx]
        
        p = np.clip((t_norm - delay) / duration, 0, 1)
        if p < 1.0:
            all_done = False
            
        v_curr[i] = V_exact[current_stage, i] * (1-p) + V_exact[current_stage+1, i] * p
        v_curr[j] = V_exact[current_stage, j] * (1-p) + V_exact[current_stage+1, j] * p
        
        curr_x = current_stage + p
        points_x.extend([curr_x, curr_x, curr_x, curr_x])
        points_y.extend([i, j, i + p*(j-i), j + p*(i-j)])

    for bar, val in zip(bar_rects, v_curr):
        bar.set_width(val)
        bar.set_color('#57f287' if val >= 0 else '#ff4d6d')

    if points_x:
        scatter.set_offsets(np.c_[points_x, points_y])
        scatter.set_alpha(1.0)
    else:
        scatter.set_alpha(0.0)

    for s in range(STAGES):
        if s < current_stage:
            barriers[s].set_color('#57f287') 
            barriers[s].set_alpha(0.8)
        elif s == current_stage:
            barriers[s].set_color('#57f287' if all_done else '#ff4d6d') 
            barriers[s].set_alpha(0.8 if all_done else 0.5)
        else:
            barriers[s].set_color('#30363d') 
            barriers[s].set_alpha(0.2)

    return scatter, *bar_rects, *barriers

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000//FPS, blit=False)
ds.save_animation(anim, "turboquant_npu_async_8stages.mp4", fps=FPS)