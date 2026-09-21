"""Shared look-and-feel helpers for the TurboQuant deck.

Every asset in this deck is composited over the same pair of artwork
backgrounds (``Title.jpg`` for the opening slide, ``Slide.jpg`` for the rest),
so the geometry, the black point and the render settings live here instead of
being re-typed in every generator script.

Two rules follow from that artwork:

* Stills are saved with ``transparent=True`` so they drop onto ``Slide.jpg``
  with no bounding box -- see :func:`save_transparent`.
* Animations are full-bleed 16:9 H.264 video, never GIF: palette
  quantisation dithers the dark gold gradient into visible noise, while
  H.264 keeps it clean. See :func:`save_animation`.
"""

import os
import shutil
import subprocess

import numpy as np

# --- Slide geometry (16:9) ---
SLIDE_W_IN = 10.0
SLIDE_H_IN = 5.625
SLIDE_ASPECT = SLIDE_W_IN / SLIDE_H_IN          # 1.7778

# Animations render at 1920x1080 with this pair.
FIGSIZE_16_9 = (16, 9)
DPI_16_9 = 120

# --- Palette ---
# The deck's deep black. The artwork's most common pixel is #0b0b0d; this is
# the rounded deck value, and the two are indistinguishable once composited.
DECK_BG = '#0a0a0a'
GOLD = '#d4af37'
GOLD_DIM = '#8a7320'

_HERE = os.path.dirname(os.path.abspath(__file__))

# The artwork has shipped under both names; take whichever is present.
_BACKGROUND_NAMES = ('Slide.jpg', 'Background.jpg', 'Slide.png', 'Background.png')
_TITLE_NAMES = ('Title.jpg', 'Title.png')


def _resolve(names):
    for name in names:
        path = os.path.join(_HERE, name)
        if os.path.exists(path):
            return path
    return None


def background_path():
    """Absolute path to the artwork used on every slide but the first."""
    return _resolve(_BACKGROUND_NAMES)


def title_background_path():
    """Absolute path to the artwork used on the title slide."""
    return _resolve(_TITLE_NAMES)


_bg_cache = {}


def _load_background(path):
    if path not in _bg_cache:
        import matplotlib.image as mpimg
        _bg_cache[path] = mpimg.imread(path)
    return _bg_cache[path]


# The animations are full-bleed: they cover the slide artwork completely, so
# the flat deck black is what shows at the edges and painting the artwork into
# every frame buys nothing. Flip this on only if an animation is ever inset.
USE_ARTWORK_BACKGROUND = False


def apply_figure_background(fig, path=None, force=False):
    """Paint the deck artwork across the whole figure, behind every axes.

    No-op unless :data:`USE_ARTWORK_BACKGROUND` (or ``force``) is set. Returns
    the background axes, or ``None`` when it is skipped or the artwork is
    missing; either way the figure is left on the flat deck black, which
    matches the artwork's dominant tone.
    """
    fig.patch.set_facecolor(DECK_BG)

    if not (USE_ARTWORK_BACKGROUND or force):
        return None

    path = path or background_path()
    if path is None:
        return None

    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_bg.set_axis_off()
    ax_bg.set_navigate(False)
    ax_bg.set_in_layout(False)          # keep tight_layout from reflowing it
    ax_bg.patch.set_visible(False)
    ax_bg.imshow(_load_background(path), extent=(0, 1, 0, 1),
                 aspect='auto', interpolation='bilinear', zorder=-1)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    return ax_bg


def clear_axes_background(*axes):
    """Make axes fully see-through so whatever is underneath shows through.

    ``Axes.clear()`` restores the stylesheet facecolor, so animations that
    clear on every frame must call this from inside their update function.
    """
    for ax in axes:
        ax.set_facecolor('none')
        ax.patch.set_alpha(0.0)


def make_transparent(fig, *axes):
    """Strip every opaque fill from ``fig`` and ``axes`` (incl. 3D panes).

    Companion to :func:`save_transparent` for stills that sit on top of the
    slide artwork: ``savefig(transparent=True)`` clears the figure and axes
    patches, but not the panes of a 3D axes, which carry their own fill.
    """
    fig.patch.set_alpha(0.0)
    for ax in axes:
        clear_axes_background(ax)
        for axis_name in ('xaxis', 'yaxis', 'zaxis'):
            axis = getattr(ax, axis_name, None)
            pane = getattr(axis, 'pane', None)
            if pane is not None:
                pane.set_visible(False)
                pane.set_alpha(0.0)


def save_transparent(fig, filename, dpi=None, pad_inches=0.02, trim=True):
    """Save a still with a fully transparent background.

    The stills overlay ``Slide.jpg`` directly, so any baked-in fill shows up
    as a rectangle around the artwork. ``trim`` then crops the result to the
    drawing's alpha bounds: ``bbox_inches='tight'`` still leaves a wide
    transparent margin around a 3D axes, and on an invisible background that
    margin silently shrinks the artwork inside its slide placement.
    """
    import matplotlib.pyplot as plt
    fig.savefig(filename, dpi=dpi, transparent=True,
                bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)

    if trim:
        from PIL import Image
        with Image.open(filename) as img:
            img = img.convert('RGBA')
            bbox = img.getbbox(alpha_only=True)
            if bbox and bbox != (0, 0, img.width, img.height):
                img.crop(bbox).save(filename)

    print(f"Saved '{filename}' (transparent)")
    return filename


def cover_crop(img_aspect, frame_aspect=SLIDE_ASPECT):
    """Crop fractions that make ``img_aspect`` fill ``frame_aspect`` edge to edge.

    Returns ``(left, top, right, bottom)`` fractions in the sense python-pptx
    uses for ``Picture.crop_*``: the source is centre-cropped, never letterboxed.
    """
    if img_aspect > frame_aspect:                 # too wide -> trim the sides
        frac = 1.0 - (frame_aspect / img_aspect)
        return frac / 2.0, 0.0, frac / 2.0, 0.0
    if img_aspect < frame_aspect:                 # too tall -> trim top/bottom
        frac = 1.0 - (img_aspect / frame_aspect)
        return 0.0, frac / 2.0, 0.0, frac / 2.0
    return 0.0, 0.0, 0.0, 0.0


def sample_black_point(path=None):
    """Most common colour in the artwork, as ``#rrggbb`` (diagnostic helper)."""
    path = path or background_path()
    if path is None:
        return DECK_BG
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB')).reshape(-1, 3)
    colours, counts = np.unique(arr, axis=0, return_counts=True)
    return '#%02x%02x%02x' % tuple(colours[np.argmax(counts)])


# ==========================================
# H.264 video output
# ==========================================

# Visually lossless for flat-shaded plots; the deck is not size constrained.
H264_CRF = 16
H264_PRESET = 'slow'

_EXTRA_ARGS = [
    '-crf', str(H264_CRF),
    '-preset', H264_PRESET,
    # 4:2:0 is what PowerPoint's decoder actually wants.
    '-pix_fmt', 'yuv420p',
    '-profile:v', 'high',
    '-movflags', '+faststart',
]


def ffmpeg_exe():
    """Path to an ffmpeg binary, preferring one on PATH.

    Falls back to the binary shipped with ``imageio-ffmpeg`` so the deck
    builds on a machine with no system-wide ffmpeg install.
    """
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "H.264 output needs ffmpeg: install it and put it on PATH, or "
            "`pip install imageio-ffmpeg` for a bundled binary."
        )
    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_ffmpeg():
    """Point matplotlib's FFMpegWriter at a usable ffmpeg binary."""
    import matplotlib
    exe = ffmpeg_exe()
    matplotlib.rcParams['animation.ffmpeg_path'] = exe
    return exe


def save_animation(anim, filename, fps, dpi=DPI_16_9, facecolor=DECK_BG):
    """Write ``anim`` as an H.264 MP4 sized for a full-bleed 16:9 slide.

    GIF is deliberately not an option here: quantising the deck's dark gold
    gradient to 256 colours dithers fresh noise into every frame.
    """
    from matplotlib.animation import FFMpegWriter
    ensure_ffmpeg()
    writer = FFMpegWriter(
        fps=fps,
        codec='libx264',
        bitrate=-1,                       # quality driven by -crf, not bitrate
        extra_args=list(_EXTRA_ARGS),
        metadata={'title': os.path.splitext(os.path.basename(filename))[0]},
    )
    print(f"Rendering {filename} (H.264, crf {H264_CRF})...")
    anim.save(filename, writer=writer, dpi=dpi,
              savefig_kwargs={'facecolor': facecolor})
    print(f"Saved '{filename}' ({os.path.getsize(filename) / 1e6:.1f} MB)")
    return filename


def video_duration_ms(video_path, default=1):
    """Length of ``video_path`` in milliseconds, parsed from ffmpeg's banner.

    PowerPoint's autoplay timing node wants the media length; ``default`` is
    returned if ffmpeg prints something unexpected.
    """
    out = subprocess.run(
        [ffmpeg_exe(), '-hide_banner', '-i', video_path],
        capture_output=True, text=True,
    ).stderr
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('Duration:'):
            stamp = line.split()[1].rstrip(',')
            try:
                h, m, s = stamp.split(':')
                return int(round((int(h) * 3600 + int(m) * 60 + float(s)) * 1000))
            except ValueError:
                break
    return default


def poster_frame(video_path, poster_path=None, force=False):
    """Grab frame 0 of ``video_path`` as a PNG for python-pptx's poster image.

    Without one, PowerPoint shows a grey loudspeaker placeholder on the slide
    until the video is started.
    """
    if poster_path is None:
        poster_path = os.path.splitext(video_path)[0] + '_poster.png'
    if (not force and os.path.exists(poster_path)
            and os.path.getmtime(poster_path) >= os.path.getmtime(video_path)):
        return poster_path
    subprocess.run(
        [ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_path,
         '-frames:v', '1', poster_path],
        check=True,
    )
    return poster_path
