import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon

# --- Общая палитра ---
BG_COLOR = '#0d1117'
CYAN = '#58a6ff'
PINK = '#ff7b72'
GREEN = '#3fb950'
YELLOW = '#e3b341'
GRAY = '#8b949e'

plt.style.use('dark_background')

# ==========================================
# STAGE 1: Low-Rank (Fan Basis & Lerp Web)
# ==========================================
def draw_stage1_low_rank():
    fig = plt.figure(figsize=(6, 5), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_COLOR)
    ax.axis('off') # Отключаем всё лишнее
    
    np.random.seed(42)
    num_basis = 5
    basis_vectors = []
    
    # Создаем базисные вектора (веер)
    angles = np.linspace(0, np.pi * 0.8, num_basis)
    for i, a in enumerate(angles):
        x = np.cos(a) * (1 + 0.2 * np.random.rand())
        y = np.sin(a) * (1 + 0.2 * np.random.rand())
        z = (i / num_basis) + 0.4 * np.random.rand()
        basis_vectors.append(np.array([x, y, z]))
        
        # Рисуем красный базисный вектор из начала координат
        ax.quiver(0, 0, 0, x, y, z, color=PINK, arrow_length_ratio=0.1, linewidth=2.5)
        
    # Рисуем "паутину" (сетку лерпов между концами векторов)
    for i in range(num_basis - 1):
        p1 = basis_vectors[i]
        p2 = basis_vectors[i+1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=CYAN, linestyle='-', lw=1.5, alpha=0.6)
        
        # Внутренние линии для создания эффекта "натянутой поверхности"
        ax.plot([0, (p1[0]+p2[0])/2], [0, (p1[1]+p2[1])/2], [0, (p1[2]+p2[2])/2], color=CYAN, linestyle='--', lw=0.5, alpha=0.3)

    # Генерируем искомые вектора (зеленые точки) как линейные комбинации на этой паутине
    for _ in range(40):
        # Случайные барицентрические координаты (выпуклая комбинация)
        weights = np.random.rand(num_basis)
        weights /= weights.sum()
        
        # Точка на паутине
        pt = sum(w * b for w, b in zip(weights, basis_vectors))
        ax.scatter(pt[0], pt[1], pt[2], color=GREEN, s=25, zorder=10, alpha=0.9)

    # Настройка камеры
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig('stage1_symbolic.png', facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved 'stage1_symbolic.png'")

# ==========================================
# STAGE 2: Decorrelation (WHT size 4)
# ==========================================
def draw_stage2_decorrelation():
    # Горизонтальное выравнивание: Гистограмма -> Сеть -> Гистограмма
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3.5), dpi=200, 
                                        gridspec_kw={'width_ratios': [1, 1.2, 1], 'wspace': 0.1})
    fig.patch.set_facecolor(BG_COLOR)
    
    D = 4
    # Спайковый вектор и его WHT трансформация
    v_in = np.array([0.2, 4.0, -0.1, 0.4])
    # WHT(4) матрица
    H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H4 = np.kron(H2, H2)
    v_out = H4 @ v_in
    
    # Единая шкала по амплитуде
    MAX_VAL = 4.5
    
    for ax in [ax1, ax2, ax3]:
        ax.axis('off')
        ax.set_ylim(D - 0.5, -0.5) # Инвертированная ось Y для выравнивания
        ax.set_facecolor(BG_COLOR)

    # 1. Левая гистограмма (Спайк)
    ax1.set_xlim(-MAX_VAL, MAX_VAL)
    ax1.axvline(0, color=GRAY, lw=1)
    for i in range(D):
        color = PINK if abs(v_in[i]) > 2 else GREEN
        ax1.barh(i, v_in[i], height=0.4, color=color, alpha=0.9)

    # 2. Сеть Адамара (2 ступени)
    stages = int(np.log2(D))
    ax2.set_xlim(-0.2, stages + 0.2)
    
    for s in range(stages):
        stride = 2**s
        for i in range(D):
            if (i // stride) % 2 == 0:
                j = i + stride
                # Линии бабочки
                ax2.plot([s, s+1], [i, i], color=CYAN, alpha=0.8, lw=2)
                ax2.plot([s, s+1], [j, j], color=CYAN, alpha=0.8, lw=2)
                ax2.plot([s, s+1], [i, j], color=CYAN, alpha=0.8, lw=2)
                ax2.plot([s, s+1], [j, i], color=CYAN, alpha=0.8, lw=2)
                # Жирные узлы
                ax2.scatter([s, s, s+1, s+1], [i, j, i, j], color=CYAN, s=50, zorder=5)

    # 3. Правая гистограмма (Размазанная энергия)
    ax3.set_xlim(-MAX_VAL, MAX_VAL)
    ax3.axvline(0, color=GRAY, lw=1)
    for i in range(D):
        ax3.barh(i, v_out[i], height=0.4, color=GREEN, alpha=0.9)

    plt.tight_layout()
    plt.savefig('stage2_symbolic.png', facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved 'stage2_symbolic.png'")

# ==========================================
# STAGE 3: Delta Compression (Multiple Clusters)
# ==========================================
def draw_stage3_delta():
    fig = plt.figure(figsize=(6, 5), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    np.random.seed(11)
    # 3 центроида в разных частях экрана
    centroids = np.array([[2, 7], [7, 8], [4, 2]])
    
    for cx, cy in centroids:
        # Рисуем сам центроид (звезду)
        ax.scatter(cx, cy, color=YELLOW, marker='*', s=500, zorder=5, edgecolor='#0d1117', lw=1)
        
        # Генерируем 7-10 точек вокруг центроида
        num_points = np.random.randint(7, 12)
        points = [np.array([cx, cy]) + np.random.normal(0, 0.8, 2) for _ in range(num_points)]
        
        for p in points:
            # Дельта (стрелка от центроида)
            ax.annotate("", xy=p, xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=PINK, alpha=0.7, lw=1.5))
            # Сама точка
            ax.scatter(p[0], p[1], color=GREEN, s=60, zorder=3)

    plt.tight_layout()
    plt.savefig('stage3_symbolic.png', facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved 'stage3_symbolic.png'")

if __name__ == "__main__":
    draw_stage1_low_rank()
    draw_stage2_decorrelation()
    draw_stage3_delta()