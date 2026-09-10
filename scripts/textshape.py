"""HarfBuzz-shaped text rendering into tight PIL layers (Devanagari-safe).

Pillow's built-in basic layout cannot shape complex scripts (conjuncts,
reordered matras), so shaping is done with uharfbuzz and rasterisation with
freetype-py, then composited onto a transparent RGBA image.
"""
from __future__ import annotations

import numpy as np
import freetype
import uharfbuzz as hb
from PIL import Image

_LOAD = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL


class Shaper:
    def __init__(self, font_path: str | object, size: float):
        self.font_path, self.size = str(font_path), int(size)
        with open(self.font_path, "rb") as fh:
            blob = fh.read()
        self.hb_face = hb.Face(blob)
        self.hb_font = hb.Font(self.hb_face)
        self.hb_font.scale = (self.size * 64, self.size * 64)  # 26.6 fixed point
        hb.ot_font_set_funcs(self.hb_font)
        self.ft_face = freetype.Face(self.font_path)
        self.ft_face.set_char_size(self.size * 64)
        m = self.ft_face.size
        self.asc = m.ascender / 64.0
        self.desc = -m.descender / 64.0
        self.line_h = m.height / 64.0

    # ------------------------------------------------------------------ shaping
    def shape(self, text: str):
        """Return (glyph_list, advance_px). glyph = (gid, x_adv, y_adv, x_off, y_off) px."""
        if not text:
            return [], 0.0
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf)
        out = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            out.append((info.codepoint, pos.x_advance / 64.0, pos.y_advance / 64.0,
                        pos.x_offset / 64.0, pos.y_offset / 64.0))
        return out, sum(g[1] for g in out)

    def width(self, text: str, tracking: float = 0.0) -> float:
        glyphs, adv = self.shape(text)
        return adv + tracking * max(0, len(glyphs) - 1)

    # ----------------------------------------------------------------- wrapping
    def wrap(self, text: str, max_width: float, tracking: float = 0.0) -> list[str]:
        words = text.split()
        if not words:
            return [text]
        lines, cur = [], words[0]
        for w in words[1:]:
            trial = f"{cur} {w}"
            if self.width(trial, tracking) <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    # ---------------------------------------------------------------- rendering
    def render(self, text: str, color=(235, 226, 208, 255), tracking: float = 0.0,
               line_spacing: float = 1.32, max_width: float | None = None,
               align: str = "center", pad: int | None = None) -> Image.Image:
        """Render `text` (with \\n and optional word-wrap) to a tight RGBA image."""
        lines = []
        for raw in text.split("\n"):
            lines.extend(self.wrap(raw, max_width - 2 * self.size * 0.1, tracking) if max_width else [raw])
        shaped = [self.shape(l) for l in lines]
        pad = int(self.size * 0.55) if pad is None else pad   # headroom for marks/diacritics
        lh = self.line_h * line_spacing
        widths = [adv + tracking * max(0, len(g) - 1) for g, adv in shaped]
        Wp = int(max(widths + [1])) + pad * 2
        Hp = int(len(lines) * lh) + pad * 2
        img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
        baseline0 = pad + self.asc
        for li, (glyphs, adv) in enumerate(shaped):
            total = adv + tracking * max(0, len(glyphs) - 1)
            if align == "center":
                pen = (Wp - total) / 2.0
            elif align == "right":
                pen = Wp - pad - total
            else:
                pen = float(pad)
            by = baseline0 + li * lh
            for gid, xa, ya, xo, yo in glyphs:
                self.ft_face.load_glyph(gid, _LOAD)
                slot = self.ft_face.glyph
                bm = slot.bitmap
                if bm.width and bm.rows:
                    arr = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.width)
                    alpha = Image.fromarray(arr, "L")
                    if color[3] < 255:  # scale mask by colour alpha
                        alpha = alpha.point(lambda v, a=color[3]: v * a // 255)
                    chip = Image.merge("RGBA", [Image.new("L", alpha.size, c) for c in color[:3]] + [alpha])
                    img.alpha_composite(
                        chip, (int(round(pen + xo + slot.bitmap_left)), int(round(by - yo - slot.bitmap_top))))
                pen += xa + tracking
        return img


_cache: dict = {}


def get(path, size) -> Shaper:
    key = (str(path), int(size))
    if key not in _cache:
        _cache[key] = Shaper(*key)
    return _cache[key]
