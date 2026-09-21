import os
import sys

# Сообщения сборки на русском: в cp1252-консоли print() иначе падает.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml import parse_xml

import deck_style as ds

# --- Премиальная корпоративная палитра (Black & Gold) ---
TEXT_COLOR = RGBColor(212, 175, 55)
GRAY_COLOR = RGBColor(220, 220, 220)
CYAN_COLOR = RGBColor(184, 134, 11)
PINK_COLOR = RGBColor(205, 127, 50)
GREEN_COLOR = RGBColor(218, 165, 32)
ORANGE_COLOR = RGBColor(238, 232, 170)

# Фирменный логотип занимает x 0.33-1.70in, y 0.29-0.60in на обеих подложках,
# поэтому заголовки слайдов начинаются правее него.
TITLE_LEFT = 1.85
TITLE_WIDTH = 7.65

# Видео стартует само при показе слайда (как это делал GIF).
# False -> останется поведение python-pptx по умолчанию: запуск по клику.
AUTOPLAY_VIDEO = True


# ==========================================
# Генерация иллюстрации формулы Сильвестра
# ==========================================
def generate_formula_image(filename='sylvester_formula.png'):
    """Рисует факторизацию Сильвестра вручную, без матричных окружений LaTeX.

    Встроенный в matplotlib парсер mathtext не знает ни ``\\begin{bmatrix}``,
    ни ``\\matrix{}`` -- оба падают с ValueError. Поэтому скобки рисуются
    линиями, а каждая ячейка выводится отдельной mathtext-строкой: такие
    выражения (``\\mathbf``, ``\\otimes``, ``\\pm``, ``\\vdots``, ``\\ddots``)
    парсер поддерживает.
    """
    fig = plt.figure(figsize=(12, 3.6), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ds.make_transparent(fig, ax)

    gold, dim = ds.GOLD, ds.GOLD_DIM

    # Верхняя строка: само тензорное произведение.
    ax.text(0.5, 0.88, r"$\mathbf{H}_{256} = \mathbf{H}_{16} \otimes \mathbf{H}_{16}$",
            color=gold, fontsize=30, ha='center', va='center')

    # Сетка ячеек 4x4: три явные строки и строка многоточий.
    col_x = [0.34, 0.48, 0.62, 0.78]
    row_y = [0.54, 0.42, 0.30, 0.16]
    cells = [
        [r"$1 \cdot \mathbf{H}_{16}$", r"$1 \cdot \mathbf{H}_{16}$", r"$\cdots$", r"$1 \cdot \mathbf{H}_{16}$"],
        [r"$1 \cdot \mathbf{H}_{16}$", r"$-1 \cdot \mathbf{H}_{16}$", r"$\cdots$", r"$-1 \cdot \mathbf{H}_{16}$"],
        [r"$\vdots$", r"$\vdots$", r"$\ddots$", r"$\vdots$"],
        [r"$1 \cdot \mathbf{H}_{16}$", r"$-1 \cdot \mathbf{H}_{16}$", r"$\cdots$", r"$+1 \cdot \mathbf{H}_{16}$"],
    ]

    for row, y in zip(cells, row_y):
        for cell, x in zip(row, col_x):
            is_dots = cell in (r"$\cdots$", r"$\vdots$", r"$\ddots$")
            ax.text(x, y, cell, color=dim if is_dots else gold,
                    fontsize=19, ha='center', va='center')

    # Знак равенства перед скобкой.
    ax.text(0.20, 0.35, r"$=$", color=gold, fontsize=30, ha='center', va='center')

    # Скобки: вертикаль + засечки сверху и снизу (замена \left[ ... \right]).
    bracket_top, bracket_bottom = 0.62, 0.08
    serif = 0.022
    for x, direction in ((0.245, 1), (0.875, -1)):
        ax.plot([x, x], [bracket_bottom, bracket_top], color=gold, lw=2.4,
                solid_capstyle='projecting')
        for y in (bracket_bottom, bracket_top):
            ax.plot([x, x + direction * serif], [y, y], color=gold, lw=2.4,
                    solid_capstyle='projecting')

    ds.save_transparent(fig, filename, pad_inches=0.05)
    return filename


# ==========================================
# Слайды: подложка, заголовки, блоки
# ==========================================
def add_background(slide, image_path):
    """Кладёт артворк во весь слайд и уводит его в самый низ z-порядка."""
    pic = slide.shapes.add_picture(image_path, 0, 0,
                                   width=prs.slide_width, height=prs.slide_height)

    # Подложка 1.793:1, слайд 1.778:1 -- режем по центру, а не растягиваем.
    with Image.open(image_path) as img:
        left, top, right, bottom = ds.cover_crop(img.width / img.height)
    pic.crop_left, pic.crop_top = left, top
    pic.crop_right, pic.crop_bottom = right, bottom

    # spTree: [0] nvGrpSpPr, [1] grpSpPr, дальше фигуры -- подложка идёт первой.
    spTree = slide.shapes._spTree
    spTree.remove(pic._element)
    spTree.insert(2, pic._element)
    return pic


def new_slide(title_slide=False):
    """Пустой слайд с нужным артворком вместо сплошной заливки."""
    slide = prs.slides.add_slide(blank_slide_layout)
    path = ds.title_background_path() if title_slide else ds.background_path()
    if path:
        add_background(slide, path)
    return slide


def add_title(slide, text):
    title_box = slide.shapes.add_textbox(Inches(TITLE_LEFT), Inches(0.18),
                                         Inches(TITLE_WIDTH), Inches(0.72))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE      # выравнивание по логотипу
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TEXT_COLOR
    return tf

def add_block(slide, top, title, title_color, content, line_spacing=1.1, width=9.0):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(width), Inches(1.2))
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = title_color
    p.space_after = Pt(2)

    for line in content:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(11)
        p.font.color.rgb = GRAY_COLOR
        p.line_spacing = line_spacing
        p.space_before = Pt(3)

def add_side_image(slide, image_path, left, top, width=None, height=None):
    if os.path.exists(image_path):
        if width and not height:
            slide.shapes.add_picture(image_path, Inches(left), Inches(top), width=Inches(width))
        elif height and not width:
            slide.shapes.add_picture(image_path, Inches(left), Inches(top), height=Inches(height))
        else:
            slide.shapes.add_picture(image_path, Inches(left), Inches(top), width=Inches(width), height=Inches(height))


# ==========================================
# Видео-слайды (H.264 во весь экран)
# ==========================================
def _set_autoplay(slide, movie, duration_ms):
    """Дописывает в тайминг слайда mainSeq, запускающий видео при показе.

    python-pptx кладёт в ``p:timing`` только узел ``p:video`` с условием
    ``delay="indefinite"`` -- это запуск по клику. Автостарт даёт
    ``p:seq``-последовательность с ``playFrom(0.0)``, которую PowerPoint
    цепляет к началу mainSeq через ``p:cond evt="onBegin"``.
    """
    childTnLsts = slide._element.xpath('./p:timing/p:tnLst/p:par/p:cTn/p:childTnLst')
    if not childTnLsts:
        return
    childTnLst = childTnLsts[0]

    used_ids = [int(v) for v in slide._element.xpath('./p:timing//p:cTn/@id')]
    base = max(used_ids) if used_ids else 1
    seq_id, click_id, group_id, effect_id, bhvr_id = (base + n for n in range(1, 6))

    seq_xml = (
        '<p:seq %s concurrent="1" nextAc="seek">\n'
        '  <p:cTn id="%d" dur="indefinite" nodeType="mainSeq">\n'
        '    <p:childTnLst>\n'
        '      <p:par>\n'
        '        <p:cTn id="%d" fill="hold">\n'
        '          <p:stCondLst>\n'
        '            <p:cond delay="indefinite"/>\n'
        '            <p:cond evt="onBegin" delay="0"><p:tn val="%d"/></p:cond>\n'
        '          </p:stCondLst>\n'
        '          <p:childTnLst>\n'
        '            <p:par>\n'
        '              <p:cTn id="%d" fill="hold">\n'
        '                <p:stCondLst><p:cond delay="0"/></p:stCondLst>\n'
        '                <p:childTnLst>\n'
        '                  <p:par>\n'
        '                    <p:cTn id="%d" presetID="1" presetClass="mediacall"'
        ' presetSubtype="0" fill="hold" nodeType="afterEffect">\n'
        '                      <p:stCondLst><p:cond delay="0"/></p:stCondLst>\n'
        '                      <p:childTnLst>\n'
        '                        <p:cmd type="call" cmd="playFrom(0.0)">\n'
        '                          <p:cBhvr>\n'
        '                            <p:cTn id="%d" dur="%d" fill="hold"/>\n'
        '                            <p:tgtEl><p:spTgt spid="%d"/></p:tgtEl>\n'
        '                          </p:cBhvr>\n'
        '                        </p:cmd>\n'
        '                      </p:childTnLst>\n'
        '                    </p:cTn>\n'
        '                  </p:par>\n'
        '                </p:childTnLst>\n'
        '              </p:cTn>\n'
        '            </p:par>\n'
        '          </p:childTnLst>\n'
        '        </p:cTn>\n'
        '      </p:par>\n'
        '    </p:childTnLst>\n'
        '  </p:cTn>\n'
        '  <p:prevCondLst>\n'
        '    <p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>\n'
        '  </p:prevCondLst>\n'
        '  <p:nextCondLst>\n'
        '    <p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>\n'
        '  </p:nextCondLst>\n'
        '</p:seq>' % (nsdecls('p'), seq_id, click_id, seq_id, group_id,
                      effect_id, bhvr_id, duration_ms, movie.shape_id)
    )
    # p:seq должен стоять перед p:video, как это делает сам PowerPoint.
    childTnLst.insert(0, parse_xml(seq_xml))


def add_fullscreen_video_slide(video_filename):
    """Слайд с H.264-видео во весь кадр 10 x 5.625in.

    Анимации рендерятся строго в 16:9, поэтому кадрирование не нужно:
    видео совпадает со слайдом пиксель в пиксель.
    """
    slide = new_slide()
    if not os.path.exists(video_filename):
        print(f"ПРОПУЩЕНО: нет файла '{video_filename}'")
        return slide

    poster = ds.poster_frame(video_filename)
    movie = slide.shapes.add_movie(
        video_filename, 0, 0, prs.slide_width, prs.slide_height,
        poster_frame_image=poster, mime_type='video/mp4',
    )
    if AUTOPLAY_VIDEO:
        _set_autoplay(slide, movie, ds.video_duration_ms(video_filename))
    return slide


# Иллюстрация формулы собирается здесь же, перед сборкой колоды.
generate_formula_image()

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(5.625)
blank_slide_layout = prs.slide_layouts[6]

# ==========================================
# SLIDE 1: MAIN TITLE SLIDE (NEW)
# ==========================================
slide1 = new_slide(title_slide=True)
# Соты на Title.jpg занимают правый нижний угол -- текст живёт в чистой левой
# половине, под логотипом.
title_box = slide1.shapes.add_textbox(Inches(0.9), Inches(1.9), Inches(6.0), Inches(2.0))
tf = title_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "TurboQuant Integration\ninto vLLM-Ascend"
p.font.size = Pt(40)
p.font.bold = True
p.font.color.rgb = TEXT_COLOR
p.alignment = PP_ALIGN.LEFT

p2 = tf.add_paragraph()
p2.text = "Deferred & Hybrid Paged KV-Cache Acceleration"
p2.font.size = Pt(18)
p2.font.color.rgb = GRAY_COLOR
p2.alignment = PP_ALIGN.LEFT
p2.space_before = Pt(20)

# ==========================================
# SLIDE 2: Evolution
# ==========================================
slide2 = new_slide()
add_title(slide2, "Evolution of KV Cache Compression: The Path to TurboQuant")
add_block(slide2, 1.0, "2022–2023 | Direct Quantization & The Outlier Dilemma", PINK_COLOR, [
    "Key Methods: SmoothQuant, GPTQ, RTN.",
    "Essence: Early attempts to shrink memory footprint using naive INT8/INT4 quantization.",
    "Bottleneck: Severely limited by extreme activation outliers that destroyed model accuracy."
])
add_block(slide2, 2.1, "2024 | Mathematical Decorrelation (WHT)", CYAN_COLOR, [
    "Key Methods: QuaRoT, SpinQuant.",
    "Essence: Solved the outlier problem via Randomized Orthogonal Rotations (Walsh-Hadamard Transform)."
])
add_block(slide2, 3.2, "2024–2025 | System-Level Memory Architecture", ORANGE_COLOR, [
    "Key Methods: PagedAttention (vLLM core).",
    "Essence: Addressed HBM fragmentation by managing KV cache in dynamic, non-contiguous pages."
])
add_block(slide2, 4.3, "2025–2026 | Hardware-Fused Synthesis", GREEN_COLOR, [
    "Key Method: TurboQuant.",
    "Essence: Deep fusion of mathematical decorrelation and memory pagination at the bare-metal kernel level."
])

# ==========================================
# SLIDE 3: Classification
# ==========================================
slide3 = new_slide()
add_title(slide3, "From Matrices to Local Redundancy: Classification")
# Иллюстрации обрезаны по альфа-каналу, поэтому колонка справа
# разложена по фактическим пропорциям, без наложений.
add_block(slide3, 1.0, "Stage 1: Linear Low-Rank Projection (SVD / Low-Rank)", ORANGE_COLOR, [
    "Concept: Finding a global secant hyperplane. Cache is projected from 'd' to a narrow subspace 'r'.",
    "Flaw: Compresses background context well, but destroys extreme radial outliers critical for rare facts."
], width=5.2)
add_side_image(slide3, "stage1_symbolic.png", left=6.4, top=0.95, height=1.25)

add_block(slide3, 2.4, "Stage 2: Outlier Smearing (Decorrelation / WHT)", CYAN_COLOR, [
    "Concept: Smoothing giant amplitude spikes using random orthogonal rotations before low-bit quantization.",
    "Flaw: Treats INT4 hardware incompatibility symptoms but ignores the true semantic topology of the data."
], width=5.2)
add_side_image(slide3, "stage2_symbolic.png", left=5.9, top=2.35, width=3.4)

add_block(slide3, 3.8, "Stage 3: Delta-Compression & Clustering", PINK_COLOR, [
    "Concept: Context is split into windows, computing a local centroid (μ), storing tokens as offsets (Δ = k - μ)."
], width=5.2)
add_side_image(slide3, "stage3_symbolic.png", left=6.1, top=4.05, height=1.25)

# ==========================================
# SLIDE 4: TurboQuant Theory
# ==========================================
slide4 = new_slide()
add_title(slide4, "Compression via Topology: Non-linear Manifolds")
add_block(slide4, 1.0, "Main Insight: Intrinsic Dimensionality", CYAN_COLOR, [
    "The latent space of LLMs does not fill the Cartesian volume uniformly.",
    "Real semantics are folded into a ~9D non-linear manifold, artificially inflated to 128D on a hypersphere."
])
add_block(slide4, 2.1, "Mechanics of the New Representation: Polar/Rotational Quantization", ORANGE_COLOR, [
    "Abandoning Cartesian coordinates. Any key vector is packed into:",
    "  1. Scalar Norm ||k|| (preserves radial outliers).",
    "  2. ~8–9 Rotation Angles around main local axes (cascade of Givens rotations).",
    "  3. 1-bit Residual to compensate for noise in the orthogonal complement."
])

# ==========================================
# SLIDE 5 & 6: Animations (Dispersion & Signs)
# ==========================================
add_fullscreen_video_slide("turboquant_dispersion_en.mp4")
add_fullscreen_video_slide("turboquant_hadamard_random_signs_en.mp4")

# ==========================================
# SLIDE 7: Integration Outline
# ==========================================
slide7 = new_slide()
add_title(slide7, "TurboQuant Integration Strategy: Table of Contents")
add_block(slide7, 1.2, "Phase 1: vLLM-Ascend Plugin Architecture", CYAN_COLOR, [
    "Validation of end-to-end mathematical contracts on CAModel simulators and bare-metal NPU."
], line_spacing=1.3)
add_block(slide7, 2.2, "Phase 2: Operator Fusion & Task Wave Scheduling", ORANGE_COLOR, [
    "Eliminating inter-kernel scheduling overhead."
], line_spacing=1.3)
add_block(slide7, 3.2, "Phase 3: Overcoming the FWHT Bottleneck (Core Challenge)", PINK_COLOR, [
    "Diagnosing severe underutilization in vector-based Fast Walsh-Hadamard Transforms."
], line_spacing=1.3)
add_block(slide7, 4.2, "Phase 4: Silicon Validation & E2E Benchmarks", GREEN_COLOR, [
    "Confirming up to 2.46x speedups and accuracy retention on 1-Million token contexts."
], line_spacing=1.3)

# ==========================================
# SLIDE 8: The FWHT Bottleneck & Benchmarks
# ==========================================
slide8 = new_slide()
add_title(slide8, "The FWHT Bottleneck & The CUBE Solution")
add_block(slide8, 0.9, "Cube Underutilization vs. Proper Vector Mapping", PINK_COLOR, [
    "While vector cores choked on data permutations, the massive 16x16 Cube systolic arrays remained idle.",
    "Solution: The CUBE Approach (Mmad + dual AIV subcores) properly saturates the hardware."
])

rows, cols = 6, 5
table_shape = slide8.shapes.add_table(rows, cols, Inches(0.5), Inches(2.2), Inches(9.0), Inches(2.8))
table = table_shape.table
headers = ["Dim (D)", "Batch (V)", "AIV Baseline (μs)", "CUBE Approach (μs)", "Speedup"]
data = [
    ["128", "16", "8.52", "6.52", "1.31x"],
    ["128", "32", "16.15", "6.57", "2.46x"],
    ["256", "16", "12.61", "6.61", "1.91x"],
    ["256", "32", "24.32", "10.38", "2.34x"],
    ["512", "32", "39.96", "20.01", "2.00x"]
]
for col_idx, header in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.text = header
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(30, 30, 30)
    p = cell.text_frame.paragraphs[0]
    p.font.bold = True
    p.font.color.rgb = CYAN_COLOR
    p.font.size = Pt(13)
    p.alignment = PP_ALIGN.CENTER
for row_idx, row_data in enumerate(data):
    for col_idx, val in enumerate(row_data):
        cell = table.cell(row_idx + 1, col_idx)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 15, 15)
        p = cell.text_frame.paragraphs[0]
        p.font.color.rgb = GRAY_COLOR
        p.font.size = Pt(12)
        p.alignment = PP_ALIGN.CENTER
        if col_idx == 4:
            p.font.color.rgb = GREEN_COLOR
            p.font.bold = True

# ==========================================
# SLIDE 9: Why AIV is Slow (video showing 8 barriers)
# ==========================================
add_fullscreen_video_slide("turboquant_npu_async_8stages.mp4")

# ==========================================
# SLIDE 10: Mathematical Breakthrough: Sylvester
# ==========================================
slide10 = new_slide()
add_title(slide10, "Mathematical Breakthrough: Sylvester Tensor FWHT")
add_block(slide10, 1.0, "Tensor Factorization", CYAN_COLOR, [
    "For a vector of length D = R × 16, we apply the Sylvester property to shift logic to CUBE cores.",
    "The remaining stages operate on full 16-element rows offloaded to AIV sub-cores using block-aligned instructions."
], width=9.0)

# Вставляем иллюстрацию формулы Сильвестра
add_side_image(slide10, "sylvester_formula.png", left=1.5, top=2.1, width=7.0)

add_block(slide10, 4.2, "Arch35 Innovation: Hi+Lo Accumulation", GREEN_COLOR, [
    "Base FP16 Cube yields 3.58×10⁻⁴ error. Splitting input into x_hi and x_lo via two-pass Mmad into one L0C accumulator",
    "drops error to machine zero (2.38×10⁻⁷) for only 12 extra cycles (<2.3% overhead)."
])

# ==========================================
# SLIDE 11: Why CUBE is Fast (video showing 2 barriers)
# ==========================================
add_fullscreen_video_slide("turboquant_matrix_pipeline.mp4")

# ==========================================
# SLIDE 12: End-to-End Decode Performance Table (NEW)
# ==========================================
slide12 = new_slide()
add_title(slide12, "End-to-End Decode Performance (vLLM-Ascend)")

# Выборка наиболее важных строк для таблицы (10 строк)
e2e_data = [
    ["Qwen3.5-9B", "2K", "1", "23.13", "39.08", "1.69x", "16.0", "+69% vs V5"],
    ["DeepSeek-V4", "2K", "1", "43.00", "30.37", "0.71x", "18.1", "Baseline"],
    ["GLM-5.2-744B", "2K", "1", "27.23", "30.07", "1.10x", "15.2", "+10% vs V5"],
    ["Qwen3.5-9B", "32K", "4", "151.18", "109.57", "0.73x", "139.9", "HBM Peak"],
    ["DeepSeek-V4", "32K", "1", "191.72", "374.33", "1.95x", "50.9", "~2x vs V5"],
    ["GLM-5.2-744B", "32K", "4", "176.73", "109.75", "0.62x", "120.6", "120 GB/s"],
    ["DeepSeek-V4", "262K", "1", "1374.13", "2945.16", "2.14x", "55.2", "V5 Dominance"],
    ["Qwen3.5-9B", "1M", "1", "3807.65", "3441.44", "0.90x", "44.1", "Near parity"],
    ["DeepSeek-V4", "1M", "1", "5422.98", "11894.87", "2.19x", "55.7", "Record: 2.2x faster"]
]

rows, cols = len(e2e_data) + 1, 8
e2e_table_shape = slide12.shapes.add_table(rows, cols, Inches(0.2), Inches(1.2), Inches(9.6), Inches(4.0))
e2e_table = e2e_table_shape.table

e2e_headers = ["Model", "Context", "Batch", "TQ (μs)", "V5 (μs)", "Speedup", "GB/s", "Note"]

for col_idx, header in enumerate(e2e_headers):
    cell = e2e_table.cell(0, col_idx)
    cell.text = header
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(30, 30, 30)
    p = cell.text_frame.paragraphs[0]
    p.font.bold = True
    p.font.color.rgb = CYAN_COLOR
    p.font.size = Pt(11)
    p.alignment = PP_ALIGN.CENTER

for row_idx, row_data in enumerate(e2e_data):
    for col_idx, val in enumerate(row_data):
        cell = e2e_table.cell(row_idx + 1, col_idx)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 15, 15)
        p = cell.text_frame.paragraphs[0]
        p.font.color.rgb = GRAY_COLOR
        p.font.size = Pt(10)
        p.alignment = PP_ALIGN.CENTER
        if col_idx == 5: # Speedup
            p.font.color.rgb = GREEN_COLOR
            p.font.bold = True

# ==========================================
# SLIDE 13: Milestone 3 - Quality Research (NEW)
# ==========================================
slide13 = new_slide()
add_title(slide13, "Milestone 3: Model Quality & Next Steps")
add_block(slide13, 1.2, "Exploration of Delta-Encoding on Compressed Cache", CYAN_COLOR, [
    "Future Research: Investigating delta-encoding mechanisms applied directly on the compressed KV cache.",
    "Necessity: Low-rank projection is fundamentally impossible for DeepSeek's MLA (Multi-Head Latent Attention) architecture.",
    "Goal: Achieve extreme compression thresholds while preserving the underlying topological integrity of rare fact vectors."
], line_spacing=1.3)
add_block(slide13, 3.2, "The Long-Context Paradox (16k - 65k Tokens)", GREEN_COLOR, [
    "16k Context: TurboQuant outperforms FP16 baseline by +5.3%. Higher structural consistency in code (+15.5%).",
    "32k / 65k Context: Significant retention advantage over baseline (Avg Delta: +11.9%).",
    "Insight: Aggressive outlier smearing via FWHT reduces attention noise, helping the model maintain focus",
    "over extreme context spans without degradation."
], line_spacing=1.3)

# ==========================================
# SLIDE 14: Conclusion
# ==========================================
slide14 = new_slide()
add_title(slide14, "Conclusion")
add_block(slide14, 1.2, "Summary of Achievements", GREEN_COLOR, [
    "1. Hybrid Cube-Vector FWHT: Overcame the AIV bottleneck with 2.46× speedup at machine-zero error.",
    "2. Achieved up to 2.2x decode speedup and saturated HBM at 148 GB/s on Ascend NPU.",
    "3. Outperformed FP16 baseline accuracy at 65k+ contexts by suppressing attention noise."
])
add_block(slide14, 3.0, "Q&A", TEXT_COLOR, [
    "Thank you for your attention.",
    "Questions regarding Cube-Vector fusion, Sylvester factorization, or vLLM integration?"
])

# Путь можно переопределить аргументом: удобно собрать копию, пока
# основной файл открыт в PowerPoint.
output_file = sys.argv[1] if len(sys.argv) > 1 else "TurboQuant_Final_Deck.pptx"
try:
    prs.save(output_file)
except PermissionError:
    sys.exit(f"Файл '{output_file}' занят (открыт в PowerPoint?). "
             f"Закройте его или укажите другое имя: python setup.py <файл.pptx>")
print(f"Финальная презентация успешно сохранена как '{output_file}' "
      f"({os.path.getsize(output_file) / 1e6:.1f} MB)!")
