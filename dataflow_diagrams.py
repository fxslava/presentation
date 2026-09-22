"""Dataflow-диаграммы слоя внимания: фаза prefill и фаза decode.

Обе картинки рисуются по одним и тем же правилам, чтобы слайды читались как
одна пара:

* **Данные** -- прямоугольники с явной размерностью под ними. Батч (S >> 1)
  показывается «колодой» слоёв, напоминая, что по конвейеру идёт пачка задач;
  при S = 1 это одна тонкая плашка.
* **Операторы** -- только формула внутри фигуры. Никаких пояснений внутри
  блока: всё словесное живёт подписями рядом, но снаружи.
* **Поток** -- стрелки слева направо, размерность тензора подписана на ребре.

Размеры задаются в долях осей (0..1). Ось растянута на всю фигуру, поэтому
доля по X и доля по Y -- это разные физические длины; ASPECT переводит одну
в другую там, где нужна «квадратность» (смещение слоёв колоды, эллипсы).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path

import deck_style as ds

GOLD = '#d4af37'
DIM_GOLD = '#8a7320'
GREEN = '#57f287'
CYAN = '#4cc9f0'
PINK = '#ff4d6d'
ORANGE = '#ffb86c'
GRAY = '#8b949e'
FILL = '#0a0a0a'

FS_FORMULA = 10.5      # формулы в блоках операторов
FS_DIMS = 8.0          # размерности тензоров
FS_NOTE = 8.0          # подписи снаружи блоков
FS_LANE = 8.5          # имена дорожек


def _new_canvas(width_in, height_in, dpi=300):
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ds.make_transparent(fig, ax)
    ax._aspect_xy = height_in / width_in     # доля Y -> доля X
    return fig, ax


def _x_of(ax, y_frac):
    """Столько же по X, сколько ``y_frac`` по Y (для квадратных смещений)."""
    return y_frac * ax._aspect_xy


# ==========================================
# Примитивы
# ==========================================
def op_block(ax, cx, cy, w, h, formula, color, fontsize=FS_FORMULA,
             note=None, note_color=None, note_gap=0.045, dashed=False,
             note_dx=0.0, note_ha='center'):
    """Оператор: в фигуре только формула, пояснение -- подписью снизу."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.004,rounding_size=0.01',
        facecolor=FILL, alpha=0.9, edgecolor=color, linewidth=1.6,
        linestyle='--' if dashed else 'solid', zorder=4))
    ax.text(cx, cy, formula, color=color, fontsize=fontsize,
            ha='center', va='center', zorder=5)
    if note:
        ax.text(cx + note_dx, cy - h / 2 - note_gap, note,
                color=note_color or GRAY, fontsize=FS_NOTE, ha=note_ha,
                va='top', zorder=5, linespacing=1.35)
    return cx + w / 2


def tensor(ax, cx, cy, w, h, color, dims=None, layers=1, label=None,
           dims_gap=0.035, fontsize=FS_DIMS):
    """Тензор данных.

    ``layers > 1`` рисует колоду: пачка задач, которую жуёт конвейер.
    ``layers == 1`` -- одна тонкая плашка (S = 1, считать нечего).
    """
    dx = _x_of(ax, 0.028)
    dy = 0.028
    for i in range(layers - 1, 0, -1):                 # задние слои колоды
        ax.add_patch(Rectangle((cx - w / 2 + i * dx, cy - h / 2 + i * dy), w, h,
                               facecolor=FILL, alpha=0.9, edgecolor=color,
                               linewidth=1.0, zorder=3 + (layers - i) * 0.01))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           facecolor=FILL, alpha=0.95, edgecolor=color,
                           linewidth=1.6, zorder=4))
    if label:
        ax.text(cx, cy, label, color=color, fontsize=fontsize + 1.5,
                ha='center', va='center', zorder=5)
    if dims:
        top_of_stack = cy - h / 2 - dims_gap
        ax.text(cx + (layers - 1) * dx / 2, top_of_stack, dims, color=color,
                fontsize=fontsize, ha='center', va='top', zorder=5)
    return cx + w / 2 + (layers - 1) * dx


def kv_cache(ax, cx, cy, w, h, color, dims=None, layers=3, title=None,
             title_dx=0.0, title_ha='center'):
    """Хранилище KV: цилиндр, внутри -- лежащие друг на друге страницы кэша."""
    ry = _x_of(ax, 0.0) or 0.0
    ell_h = h * 0.16
    body_top, body_bottom = cy + h / 2 - ell_h / 2, cy - h / 2 + ell_h / 2

    ax.add_patch(Rectangle((cx - w / 2, body_bottom), w, body_top - body_bottom,
                           facecolor=FILL, alpha=0.9, edgecolor='none', zorder=3))
    for y in (body_bottom, body_top):
        ax.add_patch(Ellipse((cx, y), w, ell_h, facecolor=FILL, alpha=0.95,
                             edgecolor=color, linewidth=1.6, zorder=4))
    for x in (cx - w / 2, cx + w / 2):
        ax.plot([x, x], [body_bottom, body_top], color=color, lw=1.6, zorder=4)

    # Страницы кэша внутри цилиндра -- тот же «стек», что и снаружи.
    inner_w = w * 0.62
    for i in range(layers):
        y = body_bottom + (i + 0.85) * (body_top - body_bottom) / (layers + 0.7)
        ax.add_patch(Rectangle((cx - inner_w / 2, y), inner_w, 0.022,
                               facecolor=color, alpha=0.35, edgecolor=color,
                               linewidth=0.8, zorder=5))
    if title:
        ax.text(cx + title_dx, cy + h / 2 + 0.03, title, color=color,
                fontsize=FS_LANE, ha=title_ha, va='bottom', zorder=5,
                fontweight='bold')
    if dims:
        ax.text(cx, cy - h / 2 - 0.035, dims, color=color, fontsize=FS_DIMS,
                ha='center', va='top', zorder=5, linespacing=1.4)
    return cx + w / 2, ry


def arrow(ax, p0, p1, color=DIM_GOLD, dims=None, dims_side='above',
          elbow=None, lw=1.4, fontsize=FS_DIMS):
    """Ребро потока данных; ``dims`` подписывает размерность прямо на нём."""
    style = dict(arrowstyle='-|>', mutation_scale=11,
                 color=color, lw=lw, shrinkA=1, shrinkB=1, zorder=2)
    if elbow:
        style['connectionstyle'] = elbow
    ax.add_patch(FancyArrowPatch(p0, p1, **style))
    if dims:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        off = 0.035
        va = 'bottom' if dims_side == 'above' else 'top'
        ax.text(mx, my + (off if dims_side == 'above' else -off), dims,
                color=color, fontsize=fontsize, ha='center', va=va, zorder=5)


def _poly_arrow(ax, verts, color, lw=1.4):
    """Стрелка по ломаной: голова рисуется на последнем сегменте."""
    path = Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=11,
                                 color=color, lw=lw, zorder=2,
                                 joinstyle='miter', capstyle='butt'))


def hvh(ax, p0, p1, bus_x=None, color=DIM_GOLD, dims=None, dims_side='above',
        lw=1.4, fontsize=FS_DIMS):
    """Горизонталь -> вертикаль -> горизонталь.

    Последний сегмент всегда горизонтальный, поэтому стрелка входит ровно в
    бок блока, а не в угол -- иначе голова прячется под рамкой.
    """
    (x0, y0), (x1, y1) = p0, p1
    if abs(y0 - y1) < 1e-4:
        arrow(ax, p0, p1, color=color, dims=dims, dims_side=dims_side, lw=lw)
        return
    bx = bus_x if bus_x is not None else (x0 + x1) / 2
    _poly_arrow(ax, [(x0, y0), (bx, y0), (bx, y1), (x1, y1)], color, lw)
    if dims:
        ax.text((x0 + bx) / 2, y0 + (0.035 if dims_side == 'above' else -0.035),
                dims, color=color, fontsize=fontsize, ha='center',
                va='bottom' if dims_side == 'above' else 'top', zorder=5)


def hv(ax, p0, p1, color=DIM_GOLD, dims=None, lw=1.4, fontsize=FS_DIMS):
    """Горизонталь -> вертикаль: вход в верхнюю или нижнюю грань блока."""
    (x0, y0), (x1, y1) = p0, p1
    _poly_arrow(ax, [(x0, y0), (x1, y0), (x1, y1)], color, lw)
    if dims:
        ax.text((x0 + x1) / 2, y0 + 0.035, dims, color=color, fontsize=fontsize,
                ha='center', va='bottom', zorder=5)


def fan_out(ax, src, bus_x, targets, color=DIM_GOLD):
    """Раздача одного тензора по дорожкам: общая шина + врезка в каждый блок."""
    x0, y0 = src
    ys = [y for _, y in targets]
    ax.plot([x0, bus_x], [y0, y0], color=color, lw=1.4, zorder=2,
            solid_capstyle='butt')
    ax.plot([bus_x, bus_x], [min(ys + [y0]), max(ys + [y0])], color=color,
            lw=1.4, zorder=2, solid_capstyle='butt')
    for x1, y1 in targets:
        arrow(ax, (bus_x, y1), (x1, y1), color=color)


def lane_label(ax, cx, cy, w, h, text, color, gap=0.03):
    """Имя дорожки -- над её первым оператором, по левому краю блока."""
    ax.text(cx - w / 2, cy + h / 2 + gap, text, color=color, fontsize=FS_LANE,
            ha='left', va='bottom', fontweight='bold', zorder=5)


def note(ax, x, y, text, color, ha='center', va='center', fontsize=FS_NOTE):
    ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha, va=va,
            zorder=5, linespacing=1.4)


# ==========================================
# Слайд 1: PREFILL -- батч заполняет тайл, WHT выгодно слить в Cube
# ==========================================
DIM_BSD = r"$\mathbb{R}^{B \times S \times D}$"
DIM_BSH = r"$\mathbb{R}^{B \times S \times D_{\mathrm{head}}}$"
DIM_BS_INT4 = r"$\mathbb{R}^{B \times S \times (D_{\mathrm{head}}/2)}$"
DIM_BSS = r"$\mathbb{R}^{B \times S \times S}$"          # матрица внимания


def prefill_diagram(filename='dataflow_prefill.png'):
    fig, ax = _new_canvas(9.6, 3.55)

    Y_Q, Y_K, Y_V = 0.845, 0.520, 0.175
    H_OP, H_T = 0.145, 0.105
    X_IN, W_IN = 0.030, 0.038
    X_OP, W_OP = 0.268, 0.250
    X_T, W_T = 0.452, 0.038
    X_CACHE, W_CACHE = 0.592, 0.098
    # Хвост внимания -- четыре ступени, поэтому блоки ниже полосных.
    X_ATT, W_ATT, H_ATT = 0.862, 0.205, 0.100
    Y_SCORE, Y_SM, Y_PV, Y_OUT = 0.745, 0.545, 0.345, 0.145

    # --- Вход: глубокая пачка токенов ---
    tensor(ax, X_IN, Y_K, W_IN, H_T, GOLD, layers=5, label=r"$X$", dims=DIM_BSD)
    note(ax, X_IN + 0.005, Y_K + 0.245, r"$S \gg 1$", GOLD, fontsize=FS_LANE)
    fan_out(ax, (X_IN + 0.062, Y_K), 0.108,
            [(X_OP - W_OP / 2, y) for y in (Y_Q, Y_K, Y_V)], color=GOLD)

    # --- Q ---
    lane_label(ax, X_OP, Y_Q, W_OP, H_OP, "Q-LANE", CYAN)
    op_block(ax, X_OP, Y_Q, W_OP, H_OP,
             r"$\widetilde{Q} = \mathrm{RoPE}(X \cdot W_q) \cdot H$", CYAN)
    arrow(ax, (X_OP + W_OP / 2, Y_Q), (X_T - W_T / 2 - 0.008, Y_Q), color=CYAN)
    right = tensor(ax, X_T, Y_Q, W_T, H_T, CYAN, layers=4, dims=DIM_BSH)
    hv(ax, (right + 0.012, Y_Q), (X_ATT, Y_SCORE + H_ATT / 2 + 0.008), color=CYAN)

    # --- K: RoPE и поворот слиты в один Cube-оператор ---
    lane_label(ax, X_OP, Y_K, W_OP, H_OP, "K-LANE", GOLD)
    op_block(ax, X_OP, Y_K, W_OP, H_OP,
             r"$\widetilde{K} = \mathrm{RoPE}(X \cdot W_k) \cdot H$", GOLD,
             note="fused on AIC (Cube) before Attention —\n"
                  r"$B \times S$ rows fill the $16 \times 16 \times C_0$ tile",
             note_color=GREEN, note_gap=0.038)
    arrow(ax, (X_OP + W_OP / 2, Y_K), (X_T - W_T / 2 - 0.008, Y_K), color=GOLD)
    right = tensor(ax, X_T, Y_K, W_T, H_T, GOLD, layers=4, dims=DIM_BSH)
    hvh(ax, (right + 0.012, Y_K), (X_CACHE - W_CACHE / 2 - 0.004, 0.455),
        bus_x=0.520, color=GOLD)

    # --- V: поворот целиком в офлайн-весах ---
    lane_label(ax, X_OP, Y_V, W_OP, H_OP, "V-LANE", GREEN)
    op_block(ax, X_OP, Y_V, W_OP, H_OP, r"$\widetilde{V} = X \cdot \widetilde{W}_v$",
             GREEN, note=r"offline: $\widetilde{W}_v = W_v \cdot H$  —  0 runtime cost",
             note_color=GREEN, note_gap=0.038)
    arrow(ax, (X_OP + W_OP / 2, Y_V), (X_T - W_T / 2 - 0.008, Y_V), color=GREEN)
    right = tensor(ax, X_T, Y_V, W_T, H_T, GREEN, layers=4, dims=DIM_BSH)
    hvh(ax, (right + 0.012, Y_V), (X_CACHE - W_CACHE / 2 - 0.004, 0.265),
        bus_x=0.520, color=GREEN)

    # --- Кэш ---
    kv_cache(ax, X_CACHE, 0.355, W_CACHE, 0.42, ORANGE, layers=3,
             title="KV-CACHE (INT4)",
             dims=r"$\widetilde{K}_{INT4},\ \widetilde{V}_{INT4} \in$" "\n" + DIM_BS_INT4)

    # --- Внимание: Score -> softmax -> P*V -> выходная проекция ---
    # K уходит в Score, V -- в произведение после softmax: до softmax V не нужен.
    hvh(ax, (X_CACHE + W_CACHE / 2 + 0.004, 0.470),
        (X_ATT - W_ATT / 2 - 0.008, Y_SCORE), bus_x=0.700, color=ORANGE)
    hvh(ax, (X_CACHE + W_CACHE / 2 + 0.004, 0.250),
        (X_ATT - W_ATT / 2 - 0.008, Y_PV), bus_x=0.700, color=ORANGE)

    def _step(y_from, y_to, color, dims):
        """Связка двух ступеней хвоста: стрелка + размерность сбоку от неё."""
        arrow(ax, (X_ATT, y_from - H_ATT / 2 - 0.038), (X_ATT, y_to + H_ATT / 2),
              color=color)
        # Подпись уходит правее стрелки: под блоком уже стоит метка ядра.
        note(ax, X_ATT + 0.026, (y_from + y_to) / 2 - 0.026, dims, color,
             ha='left', fontsize=7.5)

    op_block(ax, X_ATT, Y_SCORE, W_ATT, H_ATT,
             r"$\mathrm{Score} = \widetilde{Q} \cdot \widetilde{K}^{T}$", CYAN,
             fontsize=10, note="AIC (Cube)", note_color=GRAY, note_gap=0.024,
             note_dx=-0.022, note_ha='right')
    _step(Y_SCORE, Y_SM, CYAN, DIM_BSS)

    op_block(ax, X_ATT, Y_SM, W_ATT, H_ATT,
             r"$P = \mathrm{softmax}(\mathrm{Score} / \sqrt{D_{\mathrm{head}}})$",
             ORANGE, fontsize=9,
             note="AIV (Vector Core)", note_color=GRAY, note_gap=0.024,
             note_dx=-0.022, note_ha='right')
    _step(Y_SM, Y_PV, ORANGE, DIM_BSS)

    op_block(ax, X_ATT, Y_PV, W_ATT, H_ATT,
             r"$\widetilde{O} = P \cdot \widetilde{V}$", CYAN, fontsize=10,
             note="AIC (Cube)", note_color=GRAY, note_gap=0.024,
             note_dx=-0.022, note_ha='right')
    _step(Y_PV, Y_OUT, CYAN, DIM_BSH)

    op_block(ax, X_ATT, Y_OUT, W_ATT, H_ATT,
             r"$O = \widetilde{O} \cdot \widetilde{W}_o$", GREEN, fontsize=10,
             note=r"$\widetilde{W}_o = H^{T} \cdot W_o$ folded into the output /"
                  "\n" r"gate projection: $\mathrm{Gate}(\cdot) \cdot H^{T}$",
             note_color=GREEN, note_gap=0.024)

    ds.save_transparent(fig, filename, pad_inches=0.03)
    return filename


# ==========================================
# Слайд 2: DECODE -- S = 1, тайл пуст, поворот уходит в веса или на AIV
# ==========================================
DIM_B1D = r"$\mathbb{R}^{B \times 1 \times D}$"
DIM_B1H = r"$\mathbb{R}^{B \times 1 \times D_{\mathrm{head}}}$"
DIM_B1_INT4 = r"$\mathbb{R}^{B \times 1 \times (D_{\mathrm{head}}/2)}$"
# Один запрос против всего накопленного контекста.
DIM_B1S = r"$\mathbb{R}^{B \times 1 \times S_{\mathrm{ctx}}}$"


def decode_diagram(filename='dataflow_decode.png'):
    fig, ax = _new_canvas(9.6, 3.95)

    Y_Q, Y_KA, Y_KB, Y_V = 0.915, 0.705, 0.425, 0.115
    H_OP, H_T = 0.115, 0.046
    X_IN, W_IN = 0.036, 0.040
    X_T, W_T = 0.645, 0.036
    X_CACHE, W_CACHE = 0.730, 0.082
    X_TAIL, W_TAIL = 0.905, 0.180
    X_OP1, W_OP1 = 0.245, 0.235

    # --- Вход: один токен ---
    tensor(ax, X_IN, 0.565, W_IN, H_T, GOLD, layers=1, label=r"$X$", dims=DIM_B1D,
           dims_gap=0.028)
    note(ax, X_IN, 0.645, r"$S = 1$", GOLD, fontsize=FS_LANE)
    # У ветки B первый блок начинается левее остальных -- иначе стрелка
    # проскакивает внутрь рамки.
    fan_out(ax, (X_IN + W_IN / 2 + 0.006, 0.565), 0.098,
            [(X_OP1 - W_OP1 / 2, Y_Q), (X_OP1 - W_OP1 / 2, Y_KA),
             (0.118, Y_KB), (X_OP1 - W_OP1 / 2, Y_V)], color=GOLD)

    # --- Q ---
    lane_label(ax, X_OP1, Y_Q, W_OP1, H_OP, "Q-LANE", CYAN, gap=0.022)
    op_block(ax, X_OP1, Y_Q, W_OP1, H_OP,
             r"$\widetilde{Q} = \mathrm{FWHT}(\mathrm{RoPE}(X \cdot W_q))$", CYAN,
             fontsize=9.5)
    hv(ax, (X_OP1 + W_OP1 / 2, Y_Q), (X_TAIL, 0.845 + 0.098 / 2 + 0.008),
       color=CYAN, dims=DIM_B1H)

    # --- K, ветка A: без RoPE поворот вплавляется целиком ---
    lane_label(ax, X_OP1, Y_KA, W_OP1, H_OP, "K-LANE / A  —  no RoPE", GREEN,
               gap=0.022)
    op_block(ax, X_OP1, Y_KA, W_OP1, H_OP, r"$\widetilde{K} = X \cdot \widetilde{W}_k$",
             GREEN, note=r"full offline folding: $\widetilde{W}_k = W_k \cdot H$"
                         "  (GPT, OPT)", note_color=GREEN, note_gap=0.03)
    arrow(ax, (X_OP1 + W_OP1 / 2, Y_KA), (X_T - W_T / 2 - 0.008, Y_KA), color=GREEN,
          dims=DIM_B1H)

    # --- K, ветка B: RoPE рушит офлайн-вплавление ---
    # Имя ветки и причина, по которой её нельзя вплавить, -- одной строкой:
    # двум подписям над дорожкой места уже не остаётся.
    ax.text(0.118, Y_KB + 0.105,
            "K-LANE / B  —  RoPE (Llama, Qwen):  offline folding blocked,  "
            r"$H \cdot \mathrm{RoPE} \neq \mathrm{RoPE} \cdot H$",
            color=PINK, fontsize=FS_LANE, ha='left', va='bottom', fontweight='bold',
            zorder=5)
    op_block(ax, 0.168, Y_KB, 0.100, H_OP, r"$K = X \cdot W_k$", PINK, fontsize=9.5)
    arrow(ax, (0.2180, Y_KB), (0.2475, Y_KB), color=PINK)
    op_block(ax, 0.330, Y_KB, 0.165, H_OP, r"$K_{rope} = \mathrm{RoPE}(K)$", PINK,
             fontsize=9.5)
    arrow(ax, (0.4125, Y_KB), (0.4350, Y_KB), color=PINK)
    op_block(ax, 0.520, Y_KB, 0.170, H_OP, r"$\widetilde{K} = \mathrm{FWHT}(K_{rope})$",
             CYAN, fontsize=9.5,
             note="AIV (Vector Core): in-place\n" r"$O(D \log D)$, entirely inside UB",
             note_color=CYAN, note_gap=0.03)
    arrow(ax, (0.6050, Y_KB), (X_T - W_T / 2 - 0.008, Y_KB), color=CYAN)

    # --- V: вплавление работает всегда ---
    lane_label(ax, X_OP1, Y_V, W_OP1, H_OP, "V-LANE", GREEN, gap=0.022)
    op_block(ax, X_OP1, Y_V, W_OP1, H_OP, r"$\widetilde{V} = X \cdot \widetilde{W}_v$",
             GREEN, note=r"100% folded: $\widetilde{W}_v = W_v \cdot H$  —  0 runtime cost",
             note_color=GREEN, note_gap=0.03)
    arrow(ax, (X_OP1 + W_OP1 / 2, Y_V), (X_T - W_T / 2 - 0.008, Y_V), color=GREEN,
          dims=DIM_B1H)

    # --- Плоские тензоры (S = 1) перед кэшем ---
    for y, color, label, y_cache in ((Y_KA, GREEN, r"$\widetilde{K}$", 0.545),
                                     (Y_KB, CYAN, r"$\widetilde{K}$", 0.455),
                                     (Y_V, GREEN, r"$\widetilde{V}$", 0.365)):
        tensor(ax, X_T, y, W_T, H_T, color, layers=1, label=label)
        hvh(ax, (X_T + W_T / 2 + 0.008, y),
            (X_CACHE - W_CACHE / 2 - 0.004, y_cache), bus_x=0.680, color=color)

    kv_cache(ax, X_CACHE, 0.455, W_CACHE, 0.34, ORANGE, layers=1,
             title="KV-CACHE (INT4)", dims=DIM_B1_INT4,
             title_dx=-0.035, title_ha='left')

    # --- Хвост: Score -> softmax -> P*V -> выходная проекция -> выход ---
    # K читается из кэша в Score, V -- в произведение уже после softmax.
    H_ATT = 0.098
    Y_SCORE, Y_SM, Y_PV, Y_OUT = 0.845, 0.660, 0.475, 0.290

    hvh(ax, (X_CACHE + W_CACHE / 2 + 0.004, 0.545),
        (X_TAIL - W_TAIL / 2 - 0.008, Y_SCORE), bus_x=0.796, color=ORANGE)
    hvh(ax, (X_CACHE + W_CACHE / 2 + 0.004, 0.395),
        (X_TAIL - W_TAIL / 2 - 0.008, Y_PV), bus_x=0.796, color=ORANGE)

    def _step(y_from, y_to, color, dims):
        arrow(ax, (X_TAIL, y_from - H_ATT / 2 - 0.034), (X_TAIL, y_to + H_ATT / 2),
              color=color)
        note(ax, X_TAIL + 0.026, (y_from + y_to) / 2 - 0.022, dims, color,
             ha='left', fontsize=7.5)

    op_block(ax, X_TAIL, Y_SCORE, W_TAIL, H_ATT,
             r"$\mathrm{Score} = \widetilde{Q} \cdot \widetilde{K}^{T}$", CYAN,
             fontsize=9.5, note="AIC (Cube)", note_color=GRAY, note_gap=0.022,
             note_dx=-0.020, note_ha='right')
    _step(Y_SCORE, Y_SM, CYAN, DIM_B1S)

    op_block(ax, X_TAIL, Y_SM, W_TAIL, H_ATT,
             r"$P = \mathrm{softmax}(\mathrm{Score} / \sqrt{D_{\mathrm{head}}})$",
             ORANGE, fontsize=8,
             note="AIV (Vector Core)", note_color=GRAY, note_gap=0.022,
             note_dx=-0.020, note_ha='right')
    _step(Y_SM, Y_PV, ORANGE, DIM_B1S)

    op_block(ax, X_TAIL, Y_PV, W_TAIL, H_ATT,
             r"$\widetilde{O} = P \cdot \widetilde{V}$", CYAN, fontsize=9.5,
             note="AIC (Cube)", note_color=GRAY, note_gap=0.022,
             note_dx=-0.020, note_ha='right')
    _step(Y_PV, Y_OUT, CYAN, DIM_B1H)

    op_block(ax, X_TAIL, Y_OUT, W_TAIL, H_ATT,
             r"$O = \widetilde{O} \cdot \widetilde{W}_o$", GREEN, fontsize=9.5)
    # Подпись про вплавление уходит под сам выход: между блоком и тензором
    # идёт стрелка, и текст она бы проткнула.
    arrow(ax, (X_TAIL, Y_OUT - H_ATT / 2), (X_TAIL, 0.150 + H_T / 2 + 0.004),
          color=GREEN)
    tensor(ax, X_TAIL, 0.150, 0.05, H_T, GOLD, layers=1, label=r"$O$", dims=DIM_B1D,
           dims_gap=0.026)
    note(ax, X_TAIL, 0.036,
         r"$\widetilde{W}_o = H^{T} \cdot W_o$ — fully folded," "\n"
         "original phase space recovered", GREEN)

    ds.save_transparent(fig, filename, pad_inches=0.03)
    return filename


if __name__ == "__main__":
    prefill_diagram()
    decode_diagram()
