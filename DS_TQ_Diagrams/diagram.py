# -*- coding: utf-8 -*-
"""DeepSeek-V4: схемы потока тензоров (ландшафтная публикационная вёрстка).

Генерирует два изображения в 300 DPI:
  * deepseek_mla_turboquant_flow.png — Rotated MLA + TurboQuant;
  * deepseek_moe_swiglu_flow.png     — Rotated MoE + SwiGLU.

Структура обеих схем: общая шапка (вход слоя + fused-ядро FWHT), затем две
параллельные колонки путей, затем нижняя полоса схождения.

Раскладка строится по реальным метрикам текста: размеры карточек, колонок и
всего холста вычисляются из габаритов набранных строк, поэтому заголовки,
формулы, пояснения, врезки и бейджи базиса не перекрываются.
Оси заданы в дюймах (1 единица данных = 1 дюйм).
"""

import math
import sys
from collections import namedtuple

import matplotlib

matplotlib.use("Agg")  # экспорт без GUI + стабильный доступ к renderer

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path

# Настройка шрифтов и Mathtext
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

COLORS = {
    'bg': '#090d16',             # Глубокий темный фон

    # Цвета базисов
    'canon_border': '#38bdf8',   # Cyan <C>
    'canon_bg': '#0b2f45',
    'rot_in_border': '#fb923c',  # Orange <R_in>
    'rot_in_bg': '#3d1608',
    'rot_kv_border': '#c084fc',  # Purple <R_kv>
    'rot_kv_bg': '#33095e',
    'kernel_border': '#f43f5e',  # Rose — fused-ядра
    'kernel_bg': '#45071a',

    'text_formula': '#ffffff',   # Яркий белый для формул
    'text_title': '#f8fafc',     # Заголовки карточек
    'text_desc': '#94a3b8',      # Курсивные пояснения
    'text_micro': '#cbd5e1',     # Подписи внутри врезок
    'arrow': '#64748b',
    'rule': '#1e2b3f',
    'grid': '#51637d',   # оси и рамки внутри врезок
}

# Стиль базиса: (цвет рамки, заливка, тип линии)
BASIS = {
    'canon': (COLORS['canon_border'], COLORS['canon_bg'], '-'),
    'rot_in': (COLORS['rot_in_border'], COLORS['rot_in_bg'], '-'),
    'rot_kv': (COLORS['rot_kv_border'], COLORS['rot_kv_bg'], '-'),
    'kernel': (COLORS['kernel_border'], COLORS['kernel_bg'], (0, (6, 3))),
}

# --- Типографика -------------------------------------------------------------
KW_TITLE = dict(fontsize=11.5, fontweight='bold')     # заголовки карточек
KW_FORMULA = dict(fontsize=12.0)                      # формулы
KW_DESC = dict(fontsize=9.2, fontstyle='italic')      # пояснения
KW_TAG = dict(fontsize=8.5, fontweight='bold')        # бейджи базиса
KW_EDGE = dict(fontsize=9.0, fontstyle='italic')      # подписи стрелок
KW_COLHEAD = dict(fontsize=10.0, fontweight='bold')   # заголовки колонок
KW_MICRO = dict(fontsize=7.0)                         # подписи внутри врезок
KW_NANO = dict(fontsize=6.5)                          # байтовые смещения
KW_CAPTION = dict(fontsize=8.0, fontstyle='italic')   # подписи под врезками
KW_LEGEND = dict(fontsize=8.8)                        # легенда
KW_H1 = dict(fontsize=17.0, fontweight='bold')        # заголовок схемы
KW_H2 = dict(fontsize=10.5, fontstyle='italic')       # подзаголовок схемы

# --- Геометрия, дюймы --------------------------------------------------------
PAD_X = 0.42           # внутренние горизонтальные поля карточки
PAD_TOP = 0.28         # поле над строкой шапки карточки
PAD_BOTTOM = 0.28      # поле под телом карточки
GAP_HEAD = 0.22        # шапка карточки -> тело
GAP_LINE = 0.17        # между строками внутри тела
BADGE_GAP = 0.30       # зазор между заголовком и бейджем
BADGE_PAD_X = 0.13     # внутренние поля бейджа
BADGE_PAD_Y = 0.075
INSET_GAP = 0.42       # текстовый блок -> врезка
INSET_CAP_GAP = 0.13   # врезка -> её подпись
MIN_CARD_W = 3.80      # минимальная ширина карточки
ROW_H_GAP = 0.85       # горизонтальный зазор внутри полосы
COL_ROW_GAP = 0.50     # вертикальный зазор между карточками колонки
COL_GUTTER = 0.95      # зазор между колонками
COL_BALANCE = 0.82     # узкая колонка тянется до этой доли широкой
COL_HEAD_H = 0.46      # высота заголовка колонки
BAND_GAP = 0.82        # вертикальный зазор между полосами
MARGIN_X = 0.55
MARGIN_TOP = 0.40
MARGIN_BOTTOM = 0.48
HEADER_GAP = 0.60      # легенда -> первая полоса
H1_H2_GAP = 0.15
LEGEND_GAP = 0.26
LEGEND_ITEM_GAP = 0.50
SWATCH_W = 0.26
SWATCH_H = 0.15

# Фигура-измеритель: метрики текста в дюймах не зависят от её размера.
_RULER = plt.figure(figsize=(8, 8), dpi=100)
_RULER_RENDERER = _RULER.canvas.get_renderer()


def ink(text, **kw):
    """Габариты набранной строки (x0, y0, x1, y1) в дюймах относительно привязки."""
    artist = _RULER.text(0.5, 0.5, text, ha='left', va='baseline', **kw)
    bbox = artist.get_window_extent(_RULER_RENDERER)
    artist.remove()
    ax0, ay0 = _RULER.transFigure.transform((0.5, 0.5))
    dpi = _RULER.dpi
    return ((bbox.x0 - ax0) / dpi, (bbox.y0 - ay0) / dpi,
            (bbox.x1 - ax0) / dpi, (bbox.y1 - ay0) / dpi)


def ink_w(box):
    return box[2] - box[0]


def ink_h(box):
    return box[3] - box[1]


def put(ax, text, x, cy, box, kw, color, zorder=6, align='center'):
    """Рисует строку, привязывая её чернильный бокс к (x, cy)."""
    x0, y0, x1, y1 = box
    if align == 'center':
        tx = x - (x0 + x1) / 2
    elif align == 'left':
        tx = x - x0
    else:
        tx = x - x1
    ax.text(tx, cy - (y0 + y1) / 2, text,
            ha='left', va='baseline', color=color, zorder=zorder, **kw)


# ==============================================================================
# ВРЕЗКИ: визуальные иллюстрации операций
# ==============================================================================
Inset = namedtuple('Inset', 'w h paint caption')


def paint_butterfly(ax, x, y, w, h, accent):
    """Сеть бабочек radix-2 для FWHT: 8 точек, 3 каскада перекрёстных связей."""
    n, stages = 8, 3
    xs = [x + i * w / stages for i in range(stages + 1)]
    ys = [y + h - j * h / (n - 1) for j in range(n)]

    cross, direct = [], []
    for s in range(stages):
        bit = 1 << (stages - 1 - s)
        for j in range(n):
            direct.append([(xs[s], ys[j]), (xs[s + 1], ys[j])])
            cross.append([(xs[s], ys[j]), (xs[s + 1], ys[j ^ bit])])
    ax.add_collection(LineCollection(direct, colors=COLORS['grid'],
                                     linewidths=0.7, alpha=0.75, zorder=4), autolim=False)
    ax.add_collection(LineCollection(cross, colors=accent, linewidths=0.6,
                                     alpha=0.55, zorder=5), autolim=False)

    nodes_x = [cx for cx in xs for _ in ys]
    nodes_y = [cy for _ in xs for cy in ys]
    ax.plot(nodes_x, nodes_y, linestyle='none', marker='o', markersize=2.6,
            markerfacecolor=accent, markeredgecolor='none', zorder=6)


def paint_cache_slot(ax, x, y, w, h, accent):
    """Физический слот страницы KV-кэша с границами байтов."""
    segments = [
        ("int4 c_rot", "256 B", 6.0, COLORS['rot_kv_border'], COLORS['rot_kv_bg']),
        ("fp16", "2 B", 1.4, COLORS['rot_in_border'], COLORS['rot_in_bg']),
        ("k_pe", "128 B", 3.0, COLORS['canon_border'], COLORS['canon_bg']),
    ]
    total = sum(s[2] for s in segments)
    bar_h = 0.42
    bar_y = y + h - bar_h

    cursor = x
    bounds = [x]
    for name, size, weight, border, fill in segments:
        seg_w = w * weight / total
        ax.add_patch(Rectangle((cursor, bar_y), seg_w, bar_h, facecolor=fill,
                               edgecolor=border, linewidth=1.0, zorder=5))
        mid = cursor + seg_w / 2
        i_name = ink(name, **KW_MICRO)
        i_size = ink(size, **KW_NANO)
        put(ax, name, mid, bar_y + bar_h * 0.66, i_name, KW_MICRO,
            COLORS['text_micro'], zorder=6)
        put(ax, size, mid, bar_y + bar_h * 0.30, i_size, KW_NANO, border, zorder=6)
        cursor += seg_w
        bounds.append(cursor)

    # Засечки и байтовые смещения под слотом
    offsets = ["0", "256", "258", "386"]
    tick_top = bar_y - 0.04
    tick_bot = bar_y - 0.13
    ax.add_collection(LineCollection(
        [[(bx, tick_top), (bx, tick_bot)] for bx in bounds],
        colors=COLORS['arrow'], linewidths=0.8, zorder=5), autolim=False)
    for bx, label in zip(bounds, offsets):
        box = ink(label, **KW_NANO)
        put(ax, label, bx, tick_bot - 0.09, box, KW_NANO, COLORS['text_desc'],
            zorder=6)


def paint_rope(ax, x, y, w, h, accent):
    """Поворот координатной пары (2i, 2i+1) на угол позиции."""
    cx, cy = x + w / 2, y + h / 2
    r = min(w, h) * 0.46

    ax.add_line(Line2D([cx - r * 1.22, cx + r * 1.22], [cy, cy],
                       color=COLORS['grid'], linewidth=0.9, zorder=4))
    ax.add_line(Line2D([cx, cx], [cy - r * 1.22, cy + r * 1.22],
                       color=COLORS['grid'], linewidth=0.9, zorder=4))
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=accent,
                        linewidth=1.2, alpha=0.8, zorder=5))

    a0, a1 = math.radians(18), math.radians(80)
    for angle, alpha in ((a0, 0.45), (a1, 1.0)):
        ax.add_patch(FancyArrowPatch(
            (cx, cy), (cx + r * math.cos(angle), cy + r * math.sin(angle)),
            arrowstyle='-|>', mutation_scale=8, linewidth=1.3, color=accent,
            alpha=alpha, shrinkA=0, shrinkB=0, zorder=6))

    ra = r * 0.58
    ax.add_patch(FancyArrowPatch(
        (cx + ra * math.cos(a0), cy + ra * math.sin(a0)),
        (cx + ra * math.cos(a1), cy + ra * math.sin(a1)),
        connectionstyle="arc3,rad=-0.30", arrowstyle='-|>', mutation_scale=7,
        linewidth=1.0, color=COLORS['text_micro'], shrinkA=0, shrinkB=0, zorder=6))

    am = (a0 + a1) / 2
    box = ink(r"$\theta_p$", **KW_MICRO)
    put(ax, r"$\theta_p$", cx + ra * 1.62 * math.cos(am),
        cy + ra * 1.62 * math.sin(am), box, KW_MICRO, COLORS['text_micro'], zorder=6)
    for label, lx, ly in (("2i", cx + r * 1.25, cy - 0.11),
                          ("2i+1", cx - 0.05, cy + r * 1.25 + 0.07)):
        box = ink(label, **KW_NANO)
        put(ax, label, lx, ly, box, KW_NANO, COLORS['text_desc'], zorder=6,
            align='right' if label == "2i" else 'center')


def paint_swiglu(ax, x, y, w, h, accent):
    """Покоординатная нелинейность SiLU с асимптотой y = x."""
    lo, hi = -0.60, 4.30
    xmin, xmax = -4.0, 4.0

    def to_px(v):
        return x + (v - xmin) / (xmax - xmin) * w

    def to_py(v):
        return y + (v - lo) / (hi - lo) * h

    ax.add_patch(Rectangle((x, y), w, h, facecolor='none',
                           edgecolor=COLORS['grid'], linewidth=0.9, zorder=4))
    ax.add_line(Line2D([x, x + w], [to_py(0.0), to_py(0.0)],
                       color=COLORS['grid'], linewidth=0.9, alpha=0.8, zorder=4))
    ax.add_line(Line2D([to_px(0.0), to_px(0.0)], [y, y + h],
                       color=COLORS['grid'], linewidth=0.9, alpha=0.8, zorder=4))

    grid = [xmin + i * (xmax - xmin) / 120.0 for i in range(121)]
    ax.add_line(Line2D([to_px(v) for v in grid],
                       [to_py(v / (1.0 + math.exp(-v))) for v in grid],
                       color=accent, linewidth=1.7, zorder=6,
                       solid_capstyle='round'))
    ident = [v for v in grid if lo <= v <= hi]
    ax.add_line(Line2D([to_px(v) for v in ident], [to_py(v) for v in ident],
                       color=COLORS['arrow'], linewidth=0.9, linestyle=(0, (3, 3)),
                       zorder=5))

    box = ink("SiLU", **KW_MICRO)
    put(ax, "SiLU", to_px(2.2), to_py(3.25), box, KW_MICRO, accent, zorder=6)
    box = ink("y = x", **KW_NANO)
    put(ax, "y = x", to_px(3.5), to_py(1.55), box, KW_NANO, COLORS['text_desc'],
        zorder=6, align='right')


# ==============================================================================
# КАРТОЧКИ И РАСКЛАДКА
# ==============================================================================
class Card:
    """Карточка: шапка с бейджем базиса, формулы, пояснение и врезка-иллюстрация."""

    def __init__(self, title, formula="", desc="", tag=None, basis='canon',
                 inset=None):
        self.title = title
        self.formulas = ([formula] if isinstance(formula, str) else list(formula))
        self.formulas = [f for f in self.formulas if f]
        self.desc = desc
        self.tag = tag
        self.inset = inset
        self.border, self.fill, self.linestyle = BASIS[basis]

        self.i_title = ink(title, **KW_TITLE) if title else None
        self.i_formulas = [ink(f, **KW_FORMULA) for f in self.formulas]
        self.i_desc = ink(desc, **KW_DESC) if desc else None
        self.i_tag = ink(tag, **KW_TAG) if tag else None

        # Бейдж — чип в правом верхнем углу: цвета совпадают с карточкой.
        if self.i_tag:
            self.badge_w = ink_w(self.i_tag) + 2 * BADGE_PAD_X
            self.badge_h = ink_h(self.i_tag) + 2 * BADGE_PAD_Y
        else:
            self.badge_w = self.badge_h = 0.0

        # Шапка: заголовок + зазор + бейдж в одной строке.
        head_w = ink_w(self.i_title) if self.i_title else 0.0
        if self.i_tag:
            head_w += BADGE_GAP + self.badge_w
        self.head_h = max(ink_h(self.i_title) if self.i_title else 0.0,
                          self.badge_h)

        # Текстовый блок тела: строки формул, затем пояснение.
        lines = list(self.i_formulas) + ([self.i_desc] if self.i_desc else [])
        self.text_w = max([ink_w(b) for b in lines], default=0.0)
        self.text_h = (sum(ink_h(b) for b in lines)
                       + GAP_LINE * max(len(lines) - 1, 0))

        if inset:
            self.i_caption = ink(inset.caption, **KW_CAPTION) if inset.caption else None
            self.inset_h = inset.h + (INSET_CAP_GAP + ink_h(self.i_caption)
                                      if self.i_caption else 0.0)
            body_w = self.text_w + INSET_GAP + inset.w
        else:
            self.i_caption = None
            self.inset_h = 0.0
            body_w = self.text_w

        self.body_h = max(self.text_h, self.inset_h)
        self.w = max(max(head_w, body_w) + 2 * PAD_X, MIN_CARD_W)
        self.h = (PAD_TOP + self.head_h + GAP_HEAD + self.body_h + PAD_BOTTOM)

        self.cx = 0.0   # задаётся раскладкой
        self.top = 0.0

    def set_height(self, h):
        """Выравнивание высоты в полосе: излишек уходит в тело карточки."""
        self.body_h = h - (PAD_TOP + self.head_h + GAP_HEAD + PAD_BOTTOM)
        self.h = h

    # --- геометрия после раскладки ---
    @property
    def x0(self):
        return self.cx - self.w / 2

    @property
    def x1(self):
        return self.cx + self.w / 2

    @property
    def bottom(self):
        return self.top - self.h

    @property
    def cy(self):
        return self.top - self.h / 2

    def draw(self, ax):
        ax.add_patch(FancyBboxPatch(
            (self.x0, self.bottom), self.w, self.h,
            boxstyle="round,pad=0,rounding_size=0.14",
            linewidth=2.0, edgecolor=self.border, facecolor=self.fill,
            linestyle=self.linestyle, zorder=2))

        head_cy = self.top - PAD_TOP - self.head_h / 2

        # Бейдж: заливка и рамка как у карточки, без непрозрачной подложки.
        if self.i_tag:
            bx1 = self.x1 - PAD_X
            bx0 = bx1 - self.badge_w
            ax.add_patch(FancyBboxPatch(
                (bx0, head_cy - self.badge_h / 2), self.badge_w, self.badge_h,
                boxstyle="round,pad=0,rounding_size=0.055",
                linewidth=1.1, edgecolor=self.border, facecolor=self.fill,
                zorder=3))
            put(ax, self.tag, (bx0 + bx1) / 2, head_cy, self.i_tag, KW_TAG,
                self.border)

        # Заголовок центрируется по полосе без бейджа — лёгкий сдвиг влево.
        if self.i_title:
            shift = (self.badge_w + BADGE_GAP) / 2 if self.i_tag else 0.0
            put(ax, self.title, self.cx - shift, head_cy, self.i_title, KW_TITLE,
                COLORS['text_title'])

        body_cy = self.top - PAD_TOP - self.head_h - GAP_HEAD - self.body_h / 2

        # Врезка прижата к правому полю, текст центрируется в оставшейся полосе.
        if self.inset:
            ix0 = self.x1 - PAD_X - self.inset.w
            itop = body_cy + self.inset_h / 2
            self.inset.paint(ax, ix0, itop - self.inset.h, self.inset.w,
                             self.inset.h, self.border)
            if self.i_caption:
                put(ax, self.inset.caption, ix0 + self.inset.w / 2,
                    itop - self.inset.h - INSET_CAP_GAP - ink_h(self.i_caption) / 2,
                    self.i_caption, KW_CAPTION, COLORS['text_desc'])
            text_cx = (self.x0 + PAD_X
                       + (self.x1 - PAD_X - self.inset.w - INSET_GAP)) / 2
        else:
            text_cx = self.cx

        cursor = body_cy + self.text_h / 2
        for text, box in zip(self.formulas, self.i_formulas):
            put(ax, text, text_cx, cursor - ink_h(box) / 2, box, KW_FORMULA,
                COLORS['text_formula'])
            cursor -= ink_h(box) + GAP_LINE
        if self.i_desc:
            put(ax, self.desc, text_cx, cursor - ink_h(self.i_desc) / 2,
                self.i_desc, KW_DESC, COLORS['text_desc'])


def _clamp_x(card, toward):
    return min(max(toward, card.x0 + 0.16 * card.w), card.x1 - 0.16 * card.w)


def _clamp_y(card, toward):
    return min(max(toward, card.bottom + 0.22 * card.h), card.top - 0.22 * card.h)


def draw_edge(ax, src, dst, label="", port=None, accent=None):
    """Стрелка потока: 'h' — горизонтальная, 'v' — вертикальная, 'elbow' — с коленом."""
    color = accent or COLORS['arrow']
    if port is None:
        port = 'h' if abs(dst.cx - src.cx) > abs(dst.cy - src.cy) else 'v'

    arrow_kw = dict(arrowstyle='-|>', mutation_scale=16, linewidth=1.8,
                    color=color, shrinkA=3, shrinkB=3, zorder=1)

    if port == 'h':
        right = dst.cx > src.cx
        start = (src.x1 if right else src.x0, _clamp_y(src, dst.cy))
        end = (dst.x0 if right else dst.x1, _clamp_y(dst, src.cy))
        ax.add_patch(FancyArrowPatch(start, end, **arrow_kw))
        anchor, align = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.16), 'center'
    elif port == 'elbow':
        start = (_clamp_x(src, dst.cx), src.bottom)
        end = (_clamp_x(dst, src.cx), dst.top)
        ymid = (start[1] + end[1]) / 2
        if abs(end[0] - start[0]) < 0.03:
            ax.add_patch(FancyArrowPatch(start, end, **arrow_kw))
            anchor, align = (start[0] + 0.16, ymid), 'left'
        else:
            verts = [start, (start[0], ymid), (end[0], ymid), end]
            ax.add_patch(FancyArrowPatch(
                path=Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO,
                                  Path.LINETO]), **arrow_kw))
            anchor, align = ((start[0] + end[0]) / 2, ymid + 0.17), 'center'
    else:
        start = (_clamp_x(src, dst.cx), src.bottom)
        end = (_clamp_x(dst, src.cx), dst.top)
        ax.add_patch(FancyArrowPatch(start, end, **arrow_kw))
        anchor, align = (start[0] + 0.16, (start[1] + end[1]) / 2), 'left'

    if label:
        box = ink(label, **KW_EDGE)
        put(ax, label, anchor[0], anchor[1], box, KW_EDGE,
            accent or COLORS['text_desc'], align=align)


def row(*cards):
    return {'kind': 'row', 'cards': list(cards)}


def columns(left_title, left_cards, left_accent, right_title, right_cards,
            right_accent):
    return {'kind': 'cols', 'left': list(left_cards), 'right': list(right_cards),
            'left_title': left_title, 'right_title': right_title,
            'left_accent': left_accent, 'right_accent': right_accent}


def _legend_width(items):
    w = 0.0
    for i, (_, _, label) in enumerate(items):
        w += SWATCH_W + 0.14 + ink_w(ink(label, **KW_LEGEND))
        if i:
            w += LEGEND_ITEM_GAP
    return w


def render(path, heading, subheading, bands, edges, legend):
    """Собирает ландшафтный холст из полос и сохраняет PNG в 300 DPI."""
    # --- Проход 1: ширины -----------------------------------------------------
    band_widths = []
    for band in bands:
        if band['kind'] == 'row':
            cards = band['cards']
            band_widths.append(sum(c.w for c in cards)
                               + ROW_H_GAP * (len(cards) - 1))
        else:
            lw = max(c.w for c in band['left'])
            rw = max(c.w for c in band['right'])
            widest = max(lw, rw)
            lw = max(lw, COL_BALANCE * widest)
            rw = max(rw, COL_BALANCE * widest)
            band['lw'], band['rw'] = lw, rw
            for c in band['left']:
                c.w = lw
            for c in band['right']:
                c.w = rw
            band_widths.append(lw + COL_GUTTER + rw)

    i_h1 = ink(heading, **KW_H1)
    i_h2 = ink(subheading, **KW_H2)
    content_w = max(band_widths + [ink_w(i_h1), ink_w(i_h2),
                                   _legend_width(legend)])
    fig_w = content_w + 2 * MARGIN_X

    # --- Проход 2: высоты -----------------------------------------------------
    band_heights = []
    for band in bands:
        if band['kind'] == 'row':
            tall = max(c.h for c in band['cards'])
            for c in band['cards']:
                c.set_height(tall)
            band_heights.append(tall)
        else:
            left, right = band['left'], band['right']
            if len(left) == len(right):     # выравниваем карточки по рядам сетки
                heights = []
                for a, b in zip(left, right):
                    tall = max(a.h, b.h)
                    a.set_height(tall)
                    b.set_height(tall)
                    heights.append(tall)
                stack = sum(heights) + COL_ROW_GAP * (len(heights) - 1)
            else:
                stack = max(
                    sum(c.h for c in col) + COL_ROW_GAP * (len(col) - 1)
                    for col in (left, right))
            band_heights.append(COL_HEAD_H + stack)

    header_h = (MARGIN_TOP + ink_h(i_h1) + H1_H2_GAP + ink_h(i_h2)
                + LEGEND_GAP + SWATCH_H + HEADER_GAP)
    fig_h = (header_h + sum(band_heights) + BAND_GAP * (len(bands) - 1)
             + MARGIN_BOTTOM)

    # --- Проход 3: отрисовка --------------------------------------------------
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COLORS['bg'])
    ax = fig.add_axes([0, 0, 1, 1])  # 1 единица данных = 1 дюйм
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_autoscale_on(False)
    ax.axis('off')

    cx = fig_w / 2
    y = fig_h - MARGIN_TOP
    put(ax, heading, cx, y - ink_h(i_h1) / 2, i_h1, KW_H1, COLORS['text_title'])
    y -= ink_h(i_h1) + H1_H2_GAP
    put(ax, subheading, cx, y - ink_h(i_h2) / 2, i_h2, KW_H2, COLORS['text_desc'])
    y -= ink_h(i_h2) + LEGEND_GAP

    # Легенда базисов
    lx = cx - _legend_width(legend) / 2
    for border, dashed, label in legend:
        ax.add_patch(FancyBboxPatch(
            (lx, y - SWATCH_H), SWATCH_W, SWATCH_H,
            boxstyle="round,pad=0,rounding_size=0.045", linewidth=1.3,
            edgecolor=border, facecolor=BASIS['kernel'][1] if dashed else 'none',
            linestyle=(0, (3, 2)) if dashed else '-', zorder=4))
        box = ink(label, **KW_LEGEND)
        put(ax, label, lx + SWATCH_W + 0.14, y - SWATCH_H / 2, box, KW_LEGEND,
            COLORS['text_desc'], align='left')
        lx += SWATCH_W + 0.14 + ink_w(box) + LEGEND_ITEM_GAP
    y -= SWATCH_H + HEADER_GAP

    for band, band_h in zip(bands, band_heights):
        if band['kind'] == 'row':
            cards = band['cards']
            total = sum(c.w for c in cards) + ROW_H_GAP * (len(cards) - 1)
            cursor = cx - total / 2
            for card in cards:
                card.cx, card.top = cursor + card.w / 2, y
                card.draw(ax)
                cursor += card.w + ROW_H_GAP
        else:
            lw, rw = band['lw'], band['rw']
            # Пара колонок центрируется как целое: при разной ширине колонок
            # смещение от центра холста обязано быть общим, иначе правая
            # колонка выходит за пределы фигуры.
            group_x0 = cx - (lw + COL_GUTTER + rw) / 2
            cx_left = group_x0 + lw / 2
            cx_right = group_x0 + lw + COL_GUTTER + rw / 2
            for title, cards, col_cx, col_w, accent in (
                    (band['left_title'], band['left'], cx_left, lw,
                     band['left_accent']),
                    (band['right_title'], band['right'], cx_right, rw,
                     band['right_accent'])):
                box = ink(title, **KW_COLHEAD)
                put(ax, title, col_cx - col_w / 2, y - ink_h(box) / 2 - 0.02,
                    box, KW_COLHEAD, accent, align='left')
                rule_y = y - COL_HEAD_H + 0.16
                ax.add_line(Line2D([col_cx - col_w / 2, col_cx + col_w / 2],
                                   [rule_y, rule_y], color=accent, linewidth=1.6,
                                   alpha=0.45, zorder=3))
                top = y - COL_HEAD_H
                for card in cards:
                    card.cx, card.top = col_cx, top
                    card.draw(ax)
                    top -= card.h + COL_ROW_GAP
        y -= band_h + BAND_GAP

    for edge in edges:
        draw_edge(ax, *edge)

    # tight_layout несовместим с ручной раскладкой: обрезка только в savefig.
    fig.savefig(path, dpi=300, facecolor=COLORS['bg'],
                bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


LEGEND_CANON = (COLORS['canon_border'], False, "⟨C⟩  canonical basis")
LEGEND_ROT_IN = (COLORS['rot_in_border'], False, "⟨R_in⟩  input rotation")
LEGEND_ROT_KV = (COLORS['rot_kv_border'], False, "⟨R_kv⟩  latent rotation")
LEGEND_KERNEL = (COLORS['kernel_border'], True, "fused kernel")


# ==============================================================================
# 1. ДИАГРАММА СЛОЯ ВНИМАНИЯ (ROTATED MLA + TURBOQUANT)
# ==============================================================================
x_in = Card(
    "Layer Input Residual",
    r"$x \in \mathbb{R}^{T \times d}$",
    "Canonical activation entering the attention block",
    tag="⟨C⟩", basis='canon')

fwht1 = Card(
    "fused_rmsnorm_fwht (Site 1/2)",
    r"$h_{in}^{R} = \frac{\gamma_{in} \odot x}{r_\varepsilon(x)} \cdot R_{in}$",
    "Single pass: SRAM reduction + in-place butterfly",
    tag="FWHT", basis='kernel',
    inset=Inset(2.80, 1.05, paint_butterfly, "radix-2 butterfly, 3 stages in SRAM"))

q_down = Card(
    "Q Down-Projection",
    r"$\widetilde{W}^{DQ} = R_{in}^{\top} W^{DQ}$",
    "R_in cancelled at GEMM entry: output is canonical",
    tag="⟨C⟩", basis='canon')

q_heads = Card(
    "Query Normalization & Heads",
    r"$q_{c} = [\, q_{nope} \mid q_{rope} \,] \in \mathbb{R}^{T \times H \times 192}$",
    "gamma_q folded offline into W_UQ columns",
    tag="⟨C⟩", basis='canon')

q_rope = Card(
    "Standard Rotary Embedding",
    r"$\mathrm{RoPE}(q_{rope},\, p)$",
    "Operates strictly on 64 canonical coords",
    tag="⟨C⟩", basis='canon',
    inset=Inset(1.55, 1.25, paint_rope, "coordinate pair rotation"))

kv_down = Card(
    "KV Latent Down-Projection",
    r"$\widetilde{W}^{DKV} = R_{in}^{\top} W^{DKV} \; \mathrm{blkdiag}(R_{KV},\, I_{64})$",
    "Simultaneously rotates latent & leaves RoPE canonical",
    tag="Split", basis='rot_kv')

kv_latent = Card(
    "Pre-Rotated Latent & RoPE",
    r"$[\, c_{rot} \in \langle R_{KV} \rangle (512) \mid k_{pe} \in \langle C \rangle (64) \,]$",
    "Normalized via scalar division (gamma_kv folded)",
    tag="⟨R_kv⟩ ∥ ⟨C⟩", basis='rot_kv')

kv_cache = Card(
    "turboquant_reshape_and_cache",
    r"$\mathrm{Pack}(\mathrm{int4}(c_{rot}),\, \mathrm{fp16}(s),\, k_{pe})$",
    "Quantization boundary: 386 B per token page slot",
    tag="KV Cache", basis='kernel',
    inset=Inset(3.50, 0.95, paint_cache_slot, "paged slot, byte offsets"))

decode = Card(
    "turboquant_paged_attention",
    r"$\mathrm{Score} = (q_{abs} \cdot c_{rot}) + (q_{rope} \cdot k_{pe})$",
    "q_abs born rotated via w_qk: zero runtime FWHT in attention",
    tag="⟨R_kv⟩", basis='rot_kv')

w_vo = Card(
    "De-Rotating Output Fold (w_vo)",
    r"$w_{vo}[h] = \widetilde{W}^{UV}_h W^{O}_h$",
    "Absorbs R_KV^T and gamma_kv: restores Canonical basis",
    tag="⟨C⟩", basis='canon')

x_out = Card(
    "Attention Residual Output",
    r"$x' = x + \mathrm{attn\_out} \in \mathbb{R}^{T \times d}$",
    "Sum of two canonical tensors",
    tag="⟨C⟩", basis='canon')

ROT_IN = COLORS['rot_in_border']

render(
    "deepseek_mla_turboquant_flow.png",
    "DeepSeek-V4 Rotated MLA + TurboQuant Pipeline",
    "Tensor Flow, Basis Decoupling & In-SRAM FWHT Architecture",
    bands=[
        row(x_in, fwht1),
        columns("QUERY PATH", [q_down, q_heads, q_rope], COLORS['canon_border'],
                "LATENT KV PATH", [kv_down, kv_latent, kv_cache],
                COLORS['rot_kv_border']),
        row(decode, w_vo, x_out),
    ],
    edges=[
        (x_in, fwht1, "", 'h'),
        (fwht1, q_down, r"$h_{in}^{R}$", 'elbow', ROT_IN),
        (fwht1, kv_down, r"$h_{in}^{R}$", 'elbow', ROT_IN),
        (q_down, q_heads, "unit norm", 'v'),
        (q_heads, q_rope, "", 'v'),
        (kv_down, kv_latent, "unit norm", 'v'),
        (kv_latent, kv_cache, "", 'v'),
        (q_rope, decode, "q_rot", 'elbow'),
        (kv_cache, decode, "cached slots", 'elbow'),
        (decode, w_vo, "", 'h'),
        (w_vo, x_out, "", 'h'),
    ],
    legend=[LEGEND_CANON, LEGEND_ROT_IN, LEGEND_ROT_KV, LEGEND_KERNEL])


# ==============================================================================
# 2. ДИАГРАММА СЛОЯ МОЕ (ROTATED ROUTER + SWIGLU EXPERTS)
# ==============================================================================
x_prime = Card(
    "Inter-Layer Residual",
    r"$x' \in \mathbb{R}^{T \times d}$",
    "Canonical stream entering the MoE block",
    tag="⟨C⟩", basis='canon')

fwht2 = Card(
    "fused_rmsnorm_fwht (Site 2/2)",
    r"$h_{moe}^{R} = \frac{\gamma_{post} \odot x'}{r_\varepsilon(x')} \cdot R_{in}$",
    "Shared R_in rotation: identical to Site 1",
    tag="FWHT", basis='kernel',
    inset=Inset(2.80, 1.05, paint_butterfly, "radix-2 butterfly, 3 stages in SRAM"))

router = Card(
    "Router Gate Projection",
    [r"$\widetilde{W}^{gate} = R_{in}^{\top} W^{gate}$",
     r"$h_{moe}^{R} \widetilde{W}^{gate} = h\, W^{gate}$"],
    "Exact canonical logits",
    tag="⟨C⟩", basis='canon')

topk = Card(
    "Softmax & Top-k Selection",
    r"$w_{topk},\, \mathrm{idx} = \mathrm{TopK}(\mathrm{Softmax}(\mathrm{logits}),\, k=8)$",
    "100% reference-identical routing decisions",
    tag="Exact", basis='canon')

e_in = Card(
    "Expert GEMM Entry Projection",
    r"$\widetilde{W}^{13}_e = R_{in}^{\top} [\, W^g_e \; ; \; W^u_e \,]$",
    "Cancels R_in via R_in * R_in^T = I before non-linearity",
    tag="⟨C⟩", basis='canon')

swiglu = Card(
    "Non-linear SwiGLU Activation",
    r"$\phi(gate_e,\, up_e) = \mathrm{SiLU}(gate_e) \odot up_e$",
    "Coordinate-wise non-linearity requires Canonical input",
    tag="⟨C⟩ Non-lin", basis='canon',
    inset=Inset(2.55, 1.20, paint_swiglu, "SiLU gate, applied per coordinate"))

e_out = Card(
    "Unmodified Down-Projection",
    r"$y_e = \phi(gate_e,\, up_e) \cdot W^d_e$",
    "Weights W^d_e remain strictly unchanged",
    tag="⟨C⟩", basis='canon')

aggregate = Card(
    "Weighted Expert Aggregation",
    r"$y = \sum_{k} w_{topk}\, y_k + y_{shared} \in \mathbb{R}^{T \times d}$",
    "Accumulation across active routed and shared experts",
    tag="⟨C⟩", basis='canon')

x_next = Card(
    "Next Layer Residual Stream",
    r"$x_{next} = x' + y \in \mathbb{R}^{T \times d}$",
    "Pure canonical tensor: ready for layer N+1",
    tag="⟨C⟩", basis='canon')

render(
    "deepseek_moe_swiglu_flow.png",
    "DeepSeek-V4 Rotated MoE & SwiGLU Pipeline",
    "Preserved Top-k Routing, De-rotation at GEMM Entry & Canonical SwiGLU",
    bands=[
        row(x_prime, fwht2),
        columns("ROUTER PATH", [router, topk], COLORS['canon_border'],
                "SWIGLU EXPERTS", [e_in, swiglu, e_out], COLORS['canon_border']),
        row(aggregate, x_next),
    ],
    edges=[
        (x_prime, fwht2, "", 'h'),
        (fwht2, router, r"$h_{moe}^{R}$", 'elbow', ROT_IN),
        (fwht2, e_in, r"$h_{moe}^{R}$", 'elbow', ROT_IN),
        (router, topk, "", 'v'),
        (e_in, swiglu, "", 'v'),
        (swiglu, e_out, "", 'v'),
        (topk, aggregate, "weights", 'elbow'),
        (e_out, aggregate, "expert outputs", 'elbow'),
        (aggregate, x_next, "", 'h'),
    ],
    legend=[LEGEND_CANON, LEGEND_ROT_IN, LEGEND_KERNEL])

plt.close(_RULER)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
except (AttributeError, ValueError):  # нестандартный stdout
    pass
print("Файлы успешно сохранены: deepseek_mla_turboquant_flow.png, "
      "deepseek_moe_swiglu_flow.png")
