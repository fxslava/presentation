import numpy as np
import matplotlib.pyplot as plt

# --- Настройки премиальной палитры Black & Gold ---
BG_COLOR = '#0a0a0a'
TEXT_COLOR = '#d4af37'
GOLD = '#d4af37'
CYAN = '#4cc9f0'
PINK = '#ff4d6d'
GRAY = '#8b949e'

plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 6), dpi=150)
fig.patch.set_facecolor(BG_COLOR)

# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ (Эталонные FP16 векторы)
# ==========================================
np.random.seed(42)
N_PAIRS = 1000
D = 128  # Исходная размерность (стандартная голова Attention)

# Генерируем векторы X
X = np.random.randn(N_PAIRS, D)
X /= np.linalg.norm(X, axis=1, keepdims=True)

# Генерируем ортогональные им векторы Z
Z = np.random.randn(N_PAIRS, D)
Z = Z - np.sum(X * Z, axis=1, keepdims=True) * X
Z /= np.linalg.norm(Z, axis=1, keepdims=True)

# Создаем векторы Y так, чтобы их косинусное сходство с X было равномерно от -1 до 1
thetas = np.random.uniform(0, np.pi, N_PAIRS)
Y = X * np.cos(thetas)[:, None] + Z * np.sin(thetas)[:, None]

# Идеальное (FP16) скалярное произведение
true_dot = np.sum(X * Y, axis=1)

# ==========================================
# 2. 1-BIT КВАНТОВАНИЕ (Лемма Джонсона-Линденштрауса)
# ==========================================
K_values = [32, 64, 128, 256, 512]
errors = []

# Для левого графика сохраним данные при K=128 (сжатие в 1 бит без изменения размерности)
recon_dot_128 = None

for K in K_values:
    # Случайная матрица проекции (Gaussian Random Projection)
    R = np.random.randn(K, D)
    
    # Проекция
    X_proj = X @ R.T
    Y_proj = Y @ R.T
    
    # БУКВАЛЬНО 1-БИТНОЕ КВАНТОВАНИЕ (Только знаки)
    X_q = np.sign(X_proj)
    Y_q = np.sign(Y_proj)
    
    # Считаем скалярное произведение 1-битных векторов (нормализованное от -1 до 1)
    # На реальном железе это делается сверхбыстрой инструкцией XNOR + POPCNT
    bit_dot = np.sum(X_q * Y_q, axis=1) / K
    
    # Восстанавливаем оригинальное косинусное сходство через тригонометрию
    recon_dot = np.sin((np.pi / 2) * bit_dot)
    
    mae = np.mean(np.abs(true_dot - recon_dot))
    errors.append(mae)
    
    if K == 128:
        recon_dot_128 = recon_dot

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ ДОКАЗАТЕЛЬСТВА
# ==========================================
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

# График 1: Scatter plot (FP16 vs 1-bit) для K=128
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(BG_COLOR)
ax1.scatter(true_dot, recon_dot_128, alpha=0.5, color=CYAN, s=15, edgecolor='none')
ax1.plot([-1, 1], [-1, 1], color=PINK, lw=2, linestyle='--', label='Perfect Match (y=x)')
ax1.set_title("1-bit Sign Quantization vs FP16\n(Projection K=128 bits)", color=TEXT_COLOR, fontsize=14, pad=15)
ax1.set_xlabel("True FP16 Dot Product", color=GRAY, fontsize=12)
ax1.set_ylabel("Reconstructed 1-bit Dot Product", color=GRAY, fontsize=12)
ax1.tick_params(colors=GRAY)
ax1.legend(facecolor='#161b22', edgecolor='none', labelcolor='white')
ax1.grid(color='#1f2428', linestyle='-', linewidth=0.5)

# График 2: Сходимость ошибки при увеличении числа битов
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(BG_COLOR)
ax2.plot(K_values, errors, color=GOLD, marker='o', lw=3, markersize=8)
ax2.set_title("Quantized JL Lemma Error Convergence", color=TEXT_COLOR, fontsize=14, pad=15)
ax2.set_xlabel("Projection Dimension (Number of Bits)", color=GRAY, fontsize=12)
ax2.set_ylabel("Mean Absolute Error (MAE)", color=GRAY, fontsize=12)
ax2.set_xticks(K_values)
ax2.tick_params(colors=GRAY)
ax2.grid(color='#1f2428', linestyle='-', linewidth=0.5)

for i, txt in enumerate(errors):
    ax2.annotate(f"{txt:.3f}", (K_values[i], errors[i]), textcoords="offset points", 
                 xytext=(0, 10), ha='center', color='white', fontweight='bold')

plt.savefig('qjl_1bit_proof.png', bbox_inches='tight', facecolor=BG_COLOR, dpi=200)
print("Доказательство сгенерировано: qjl_1bit_proof.png")