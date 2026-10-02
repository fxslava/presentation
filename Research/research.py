import os
import time
import math
import torch
import numpy as np
from safetensors.torch import load_file

def unpack_awq_gemm_int4(qweight: torch.Tensor) -> torch.Tensor:
    qw = qweight.to(torch.int32).cpu()
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32)
    unpacked = (qw.unsqueeze(-1) >> shifts) & 0x0F
    M = qw.shape[0]
    unpacked = unpacked.view(M, -1).to(torch.int8)
    return unpacked - 8

def extract_target_layer(model_dir: str, target_subname: str = "layers.10.mlp.down_proj"):
    st_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    for filepath in st_files:
        tensors = load_file(filepath, device="cpu")
        for k in tensors.keys():
            if target_subname in k and "qweight" in k:
                print(f"[+] Найден целевой слой: {k}")
                return unpack_awq_gemm_int4(tensors[k])
    raise KeyError("Целевой тензор не найден")

# =====================================================================
# 1. ОЦЕНКА ВНУТРЕННЕЙ РАЗМЕРНОСТИ МНОГООБРАЗИЯ (TwoNN)
# =====================================================================
def estimate_intrinsic_dimension_twonn(vectors: torch.Tensor) -> float:
    """
    Оценка внутренней размерности (ID) методом TwoNN (Facco et al., 2017).
    d = - ln(1 - F(mu)) / ln(mu)
    """
    N, D = vectors.shape
    dist = torch.cdist(vectors, vectors, p=2.0)
    dist.fill_diagonal_(float("inf"))

    # Находим первого и второго ближайшего соседа
    top2 = torch.topk(dist, k=2, dim=1, largest=False).values
    r1 = top2[:, 0]
    r2 = top2[:, 1]

    # Исключаем нулевые расстояния для стабильности
    valid = (r1 > 1e-5) & (r2 > r1)
    if valid.sum() < 50:
        return float(D)

    mu = (r2[valid] / r1[valid]).cpu().numpy()
    
    # Оценка методом максимального правдоподобия (MLE)
    log_mu = np.log(mu)
    d_mle = len(log_mu) / np.sum(log_mu)
    return float(d_mle)

def run_twonn_analysis(W_int8: torch.Tensor):
    print(f"\n{'='*70}")
    print(f"  1. АНАЛИЗ ВНУТРЕННЕЙ РАЗМЕРНОСТИ МНОГООБРАЗИЯ (TwoNN)")
    print(f"{'='*70}")
    
    sample_pool = 4096
    for K in [16, 32, 64]:
        chunks = W_int8.view(-1, K)[:sample_pool].float()
        id_est = estimate_intrinsic_dimension_twonn(chunks)
        print(f"  Векторы длины D = {K:2d}: Внутренняя размерность ID = {id_est:5.2f} (Сжатие пространства: {(1.0 - id_est/K)*100:.1f}%)")

# =====================================================================
# 2. АНАЛИЗ БИТОВЫХ ПЛОСКОСТЕЙ (Bit-Plane Entropy & Mutual Info)
# =====================================================================
def run_bitplane_analysis(W_int8: torch.Tensor):
    print(f"\n{'='*70}")
    print(f"  2. ХИРУРГИЯ БИТОВЫХ ПЛОСКОСТЕЙ (Bit-Plane Entropy & MI)")
    print(f"{'='*70}")
    
    # Переводим [-8, 7] в диапазон без знака [0, 15]
    U = (W_int8.to(torch.int32) + 8).flatten()
    N_total = len(U)

    # Раскладываем на 4 бита: B0 (младший), B1, B2, B3 (старший/знаковый)
    b0 = (U & 1).cpu().numpy()
    b1 = ((U >> 1) & 1).cpu().numpy()
    b2 = ((U >> 2) & 1).cpu().numpy()
    b3 = ((U >> 3) & 1).cpu().numpy()
    planes = [b0, b1, b2, b3]

    entropies = []
    print("  Индивидуальная энтропия Шеннона по битовым слоям:")
    for i in range(4):
        p1 = np.mean(planes[i])
        p0 = 1.0 - p1
        if p0 == 0 or p1 == 0:
            h = 0.0
        else:
            h = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        entropies.append(h)
        print(f"    Бит B{i} ({'LSB - Младший' if i==0 else ('MSB - Знак' if i==3 else 'Средний')}): "
              f"P(1) = {p1*100:5.2f}% | H = {h:.4f} бит")

    total_bit_entropy = sum(entropies)
    print(f"\n  Теоретический минимум размера слоя (сумма H): {total_bit_entropy:.3f} bpw (из 4.000)")

    # Взаимная информация I(Bi; Bj) = H(Bi) + H(Bj) - H(Bi, Bj)
    print("\n  Матрица взаимной информации I(Bi; Bj) в миллибитах (связность битов):")
    mi_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                mi_matrix[i, j] = entropies[i]
            else:
                pair = (planes[i] << 1) | planes[j]
                counts = np.bincount(pair, minlength=4)
                probs = counts / N_total
                probs = probs[probs > 0]
                h_joint = -np.sum(probs * np.log2(probs))
                mi_matrix[i, j] = max(0.0, entropies[i] + entropies[j] - h_joint)

    print("        B0       B1       B2       B3")
    for i in range(4):
        row_str = " ".join([f"{mi_matrix[i, j]*1000:7.1f}m" for j in range(4)])
        print(f"    B{i} {row_str}")

# =====================================================================
# 3. ПРОСТРАНСТВЕННАЯ 2D АВТОКОРРЕЛЯЦИЯ (ACF)
# =====================================================================
def run_autocorrelation_analysis(W_int8: torch.Tensor, max_lag: int = 64):
    print(f"\n{'='*70}")
    print(f"  3. ПРОСТРАНСТВЕННАЯ АВТОКОРРЕЛЯЦИЯ (ACF) (Лаг до {max_lag})")
    print(f"{'='*70}")
    
    W = W_int8.float()
    W_centered = W - W.mean()
    var = W_centered.var().item()

    # 1. Автокорреляция вдоль каналов (по столбцам N=2048)
    row_acf = []
    for lag in range(1, max_lag + 1):
        c = (W_centered[:, :-lag] * W_centered[:, lag:]).mean().item() / var
        row_acf.append(c)

    # 2. Автокорреляция поперек проекций (по строкам M=11008)
    col_acf = []
    for lag in range(1, max_lag + 1):
        c = (W_centered[:-lag, :] * W_centered[lag:, :]).mean().item() / var
        col_acf.append(c)

    print("  Лаг (смещение):          1       2       4       8      16      32      64")
    sample_lags = [1, 2, 4, 8, 16, 32, 64]
    
    row_str = " ".join([f"{row_acf[l-1]*1000:6.1f}‰" for l in sample_lags])
    print(f"  ACF Вдоль каналов:  {row_str}")
    
    col_str = " ".join([f"{col_acf[l-1]*1000:6.1f}‰" for l in sample_lags])
    print(f"  ACF Поперек каналов: {col_str}")

    max_row_lag = int(np.argmax(np.abs(row_acf))) + 1
    max_col_lag = int(np.argmax(np.abs(col_acf))) + 1
    print(f"\n  Пик канальной корреляции: лаг {max_row_lag} (r = {row_acf[max_row_lag-1]:.4f})")
    print(f"  Пик межстрочной корреляции: лаг {max_col_lag} (r = {col_acf[max_col_lag-1]:.4f})")

if __name__ == "__main__":
    MODEL_DIR = r"C:\Qwen3.5-2B"
    try:
        raw_weights = extract_target_layer(MODEL_DIR, target_subname="layers.10.mlp.down_proj")
        
        # 1. Локальная геометрия (TwoNN)
        run_twonn_analysis(raw_weights)
        
        # 2. Распределение информации по битам
        run_bitplane_analysis(raw_weights)
        
        # 3. Периодичность структуры (ACF)
        run_autocorrelation_analysis(raw_weights, max_lag=64)
        
    except Exception as e:
        print(f"[!] Ошибка: {e}")