"""Visual design: chapter palettes, procedural backgrounds, and the text stack
(Sanskrit shloka + transliteration + meaning) rendered as a transparent RGBA
layer that the video engine animates (float, glow sweeps, fades).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import (W, H, FONTS, CHAPTER_NAMES, FONT_SANSKRIT,
                    FONT_SANSKRIT_ITALIC, FONT_LATIN_HEAD, FONT_LATIN_BODY,
                    FONT_DISPLAY)
from textshape import get

ACCENT_BRAND = "GITA WISDOM"

# 18 chapter palettes: (name, top rgb, bottom rgb, nebula rgb, accent rgb)
PALETTES = [
    ("kurukshetra-dusk", (16, 14, 34), (54, 24, 20), (110, 62, 38), (232, 184, 109)),
    ("cosmic-indigo",    (8, 12, 38), (26, 18, 62), (64, 52, 140), (240, 197, 118)),
    ("karma-saffron",    (30, 14, 8), (74, 34, 10), (150, 84, 32), (252, 208, 129)),
    ("jnana-gold",       (18, 16, 6), (58, 46, 12), (128, 102, 40), (248, 214, 130)),
    ("sannyasa-ash",     (20, 20, 24), (44, 38, 40), (96, 84, 88), (226, 200, 150)),
    ("dhyana-forest",    (6, 20, 18), (14, 48, 40), (36, 104, 84), (214, 219, 148)),
    ("vijnana-river",    (6, 16, 34), (12, 42, 68), (36, 96, 134), (198, 224, 178)),
    ("brahma-void",      (10, 8, 24), (30, 20, 58), (72, 52, 124), (238, 206, 142)),
    ("rajavidya-royal",  (24, 8, 26), (58, 16, 52), (122, 46, 104), (250, 202, 138)),
    ("vibhuti-flame",    (26, 10, 6), (70, 24, 8), (156, 66, 24), (255, 214, 128)),
    ("visvarupa-nova",   (4, 6, 22), (26, 18, 70), (96, 62, 160), (255, 226, 160)),
    ("bhakti-rose",      (26, 8, 18), (62, 20, 38), (138, 54, 88), (255, 208, 158)),
    ("kshetra-earth",    (16, 18, 10), (44, 44, 20), (96, 96, 44), (232, 220, 150)),
    ("guna-tricolor",    (14, 12, 30), (36, 30, 64), (84, 74, 130), (236, 208, 148)),
    ("purusha-dawn",     (22, 10, 22), (66, 26, 42), (146, 64, 84), (255, 216, 148)),
    ("daiva-divide",     (8, 18, 26), (18, 48, 60), (44, 104, 118), (222, 226, 168)),
    ("shraddha-lamp",    (24, 14, 6), (64, 38, 12), (140, 92, 38), (255, 218, 140)),
    ("moksha-pearl",     (12, 14, 24), (34, 40, 58), (78, 92, 122), (240, 226, 178)),
]

TEXT_MAIN = (241, 232, 214, 255)      # warm cream
TEXT_SOFT = (205, 190, 160, 235)      # muted sand


def palette(chapter: int) -> dict:
    name, top, bot, neb, acc = PALETTES[(chapter - 1) % len(PALETTES)]
    return dict(name=name, top=np.array(top, np.float32), bot=np.array(bot, np.float32),
                neb=np.array(neb, np.float32), acc=np.array(acc, np.float32))


# ------------------------------------------------------------------ background
def build_background(pal: dict, seed: int, w: int = int(W * 1.14), h: int = int(H * 1.14)):
    """Oversized cosmic gradient w/ nebula blobs + starfield.
    Returns (img float32 (h,w,3), starphase float32 (h,w)) — phase map used by
    the video engine to twinkle stars. Overscan gives pan/zoom/shake headroom."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = yy[:, :, None] / max(1, h - 1)
    img = pal["top"][None, None, :] * (1 - grad) + pal["bot"][None, None, :] * grad
    # soft diagonal drift
    drift = 0.10 * np.sin(xx / w * math.pi + 0.8)[:, :, None]
    img *= 0.92 + drift
    # nebula: gaussian blobs in accent/nebula hues
    for _ in range(7):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(w * 0.12, w * 0.42), rng.uniform(h * 0.08, h * 0.30)
        amp = rng.uniform(0.10, 0.30)
        g = amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        col = pal["neb"] if rng.random() < 0.7 else pal["acc"]
        img += g[:, :, None] * col[None, None, :] * 0.55
    # starfield (sparse sparkling points, phase map for twinkle)
    n_stars = 420
    xs = rng.uniform(0, w - 1, n_stars).astype(int)
    ys = rng.uniform(0, h - 1, n_stars).astype(int)
    amps = rng.uniform(10, 70, n_stars)
    img[ys, xs] += amps[:, None]
    img[ys, xs, 1] += amps * 0.25
    starphase = np.zeros((h, w), np.float32)
    starphase[ys, xs] = rng.uniform(0, 6.283, n_stars)
    starmask = np.zeros((h, w), np.float32)
    starmask[ys, xs] = amps * rng.uniform(0.7, 1.6, n_stars)
    # gentle blur to melt things together, then restore star pinpricks
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2))
    img = np.asarray(pil).astype(np.float32)
    img[ys, xs] += amps[:, None] * 0.8
    return img, starphase, starmask


def mandala_layer(pal: dict, size: int = 900, seed: int = 0) -> np.ndarray:
    """Radial lotus/chakra glow, float32 (size,size,3), additive."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    r = np.hypot(xx - cx, yy - cy) / (size / 2)
    th = np.arctan2(yy - cy, xx - cx)
    petals = (np.abs(np.sin(th * 12)) * np.exp(-((r - 0.72) ** 2) / 0.012) +
              np.abs(np.sin(th * 8)) * np.exp(-((r - 0.50) ** 2) / 0.010))
    rings = (np.exp(-((r - 0.86) ** 2) / 0.0012) + np.exp(-((r - 0.63) ** 2) / 0.0012) +
             np.exp(-((r - 0.34) ** 2) / 0.0008))
    core = np.exp(-(r ** 2) / 0.10) * 0.8
    lum = petals * 0.20 + rings * 0.5 + core * 0.55
    return (lum[:, :, None] * pal["acc"][None, None, :] * 0.42).astype(np.float32)


# -------------------------------------------------------------------- ornaments
def _diamond(draw: ImageDraw.ImageDraw, cx, cy, s, fill):
    draw.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=fill)


def _latin_safe(shaper, text: str) -> str:
    """Replace chars missing from a Latin display font by their NFKD bases."""
    import unicodedata

    def ok(ch: str) -> bool:
        return bool(shaper.ft_face.get_char_index(ch))

    out = []
    for ch in text:
        if ok(ch):
            out.append(ch)
        else:
            out.append("".join(c for c in unicodedata.normalize("NFKD", ch) if ok(c)))
    fixed = "".join(out)
    return fixed if fixed.strip() else text


def _ghost(layer, rendered, xy, blur, color, alpha_scale, dy=0):
    """Tinted, blurred copy of `rendered` used for shadows and gold glows."""
    m = rendered.split()[3].point(lambda v: int(v * alpha_scale))
    tinted = Image.new("RGBA", rendered.size, color)
    tinted.putalpha(m)
    if blur:
        tinted = tinted.filter(ImageFilter.GaussianBlur(blur))
    layer.alpha_composite(tinted, (xy[0], xy[1] + dy))


def ornament(img: Image.Image, cx: int, y: int, half: int, color):
    """Centered diamond + tapering side rules."""
    d = ImageDraw.Draw(img, "RGBA")
    _diamond(d, cx, y, 9, color)
    _diamond(d, cx, y, 4, (color[0], color[1], color[2], max(0, color[3] - 90)))
    for sgn in (-1, 1):
        for i in range(half):
            a = int(color[3] * (1 - i / half) ** 1.6)
            x = cx + sgn * (22 + i)
            d.point((x, y), fill=(color[0], color[1], color[2], a))
            d.point((x, y + 1), fill=(color[0], color[1], color[2], a))


# ------------------------------------------------------------------ text stack
def build_text_stack(verse: dict, pal: dict, brand: str = ACCENT_BRAND,
                     show_transliteration: bool = True) -> Image.Image:
    """Full-frame (W×H) RGBA: header, shloka panel, meaning, footer."""
    acc = tuple(int(c) for c in pal["acc"])
    acc_soft = acc + (170,)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    chapter = verse["chapter"]

    # --- header ---------------------------------------------------------------
    get(FONT_SANSKRIT, 40)  # warm cache
    head_skt = get(FONT_SANSKRIT, 44).render("॥ श्रीमद्भगवद्गीता ॥", color=acc + (235,), align="center")
    y = 108
    layer.alpha_composite(head_skt, ((W - head_skt.width) // 2, y))
    y += head_skt.height + 26
    chap_line = f"CHAPTER {chapter} · {CHAPTER_NAMES.get(chapter,'').upper()} · VERSE {verse['verse']}"
    chap_line = _latin_safe(get(FONT_LATIN_HEAD, 30), chap_line)
    head = get(FONT_LATIN_HEAD, 30).render(chap_line, color=TEXT_SOFT, tracking=3.2, align="center")
    layer.alpha_composite(head, ((W - head.width) // 2, y))
    y += head.height + 18
    ornament(layer, W // 2, y, 150, acc_soft)
    y += 34

    # --- shloka panel ----------------------------------------------------------
    panel_x = 64
    panel_w = W - 2 * panel_x
    sanskrit = verse["sanskrit"].replace("\\n", "\n").strip()
    translit = None
    if show_transliteration:
        tr = verse["transliteration"].replace("\\n", " · ").replace("  ", " ").strip(" |")
        translit = get(FONT_SANSKRIT_ITALIC, 30).render(tr, color=TEXT_SOFT, max_width=panel_w - 140,
                                                        line_spacing=1.3, align="center")
    # --- fit everything to the space between header and footer ----------------
    FOOTER_RESERVE = 300
    pad_v = 76
    G1 = 74                           # shloka -> ornament -> transliteration (placement flow)
    G2 = 22                           # transliteration -> meaning
    avail = H - y - FOOTER_RESERVE - 2 * pad_v
    tr_h = translit.height if translit is not None else 0
    skt_cap = min(560, int((avail - tr_h - G1 - G2) * 0.5))
    size = 66
    while size > 40:
        skt = get(FONT_SANSKRIT, size).render(sanskrit, color=TEXT_MAIN, max_width=panel_w - 110,
                                              line_spacing=1.42, align="center")
        if skt.height <= skt_cap:
            break
        size -= 4
    msize = 46
    while msize > 27:
        mng = get(FONT_LATIN_BODY, msize).render(verse["meaning"], color=TEXT_MAIN,
                                                 max_width=panel_w - 90, line_spacing=1.5, align="center")
        if mng.height <= avail - skt.height - tr_h - G1 - G2:
            break
        msize -= 2
    inner_h = skt.height + G1 + tr_h + G2 + mng.height
    panel_h = inner_h + pad_v * 2
    region_h = avail + 2 * pad_v
    panel_top = y + max(0, (region_h - panel_h) // 2)

    # panel body: deep translucent + border + corner flourishes
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel, "RGBA")
    body_col = (10, 8, 16, 156)
    pd.rounded_rectangle([0, 0, panel_w - 1, panel_h - 1], 26, fill=body_col)
    pd.rounded_rectangle([1, 1, panel_w - 2, panel_h - 2], 26, outline=acc + (105,), width=2)
    pd.rounded_rectangle([9, 9, panel_w - 10, panel_h - 10], 20, outline=acc + (46,), width=1)
    s, L = 4, 46
    for ox, sx in ((20, 1), (panel_w - 20, -1)):
        for oy, sy in ((20, 1), (panel_h - 20, -1)):
            pd.line([ox, oy, ox + sx * L, oy], fill=acc + (200,), width=s)
            pd.line([ox, oy, ox, oy + sy * L], fill=acc + (200,), width=s)
    layer.alpha_composite(panel, (panel_x, panel_top))

    cy = panel_top + pad_v
    xy = ((W - skt.width) // 2, cy)
    _ghost(layer, skt, xy, blur=10, color=acc + (255,), alpha_scale=0.50)
    _ghost(layer, skt, xy, blur=5, color=(0, 0, 0, 255), alpha_scale=0.55, dy=5)
    layer.alpha_composite(skt, xy)
    cy += skt.height + 30
    ornament(layer, W // 2, cy, 120, acc_soft)
    cy += 44
    if translit is not None:
        layer.alpha_composite(translit, ((W - translit.width) // 2, cy - 8))
        cy += translit.height + 22
    mxy = ((W - mng.width) // 2, cy)
    _ghost(layer, mng, mxy, blur=5, color=(0, 0, 0, 255), alpha_scale=0.5, dy=4)
    layer.alpha_composite(mng, mxy)

    # --- footer ----------------------------------------------------------------
    om = get(FONT_DISPLAY, 44).render("ॐ", color=acc + (230,), align="center")
    layer.alpha_composite(om, ((W - om.width) // 2, H - 200))
    foot = get(FONT_LATIN_HEAD, 26).render(brand, color=TEXT_SOFT, tracking=5.0, align="center")
    layer.alpha_composite(foot, ((W - foot.width) // 2, H - 128))
    ornament(layer, W // 2, H - 160, 90, acc + (120,))
    return layer


if __name__ == "__main__":
    import argparse, json
    from common import find_verse
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="2.47")
    ap.add_argument("--out", default="/tmp/card.png")
    ap.add_argument("--bg", action="store_true", help="compose over generated background")
    args = ap.parse_args()
    v = find_verse(args.id)
    pal = palette(int(v["chapter"]))
    card = build_text_stack(v, pal)
    if args.bg:
        bg = Image.fromarray(np.clip(build_background(pal, seed=42)[0], 0, 255).astype(np.uint8))
        bw, bh = bg.size
        bg = bg.crop(((bw - W) // 2, (bh - H) // 2, (bw - W) // 2 + W, (bh - H) // 2 + H)).convert("RGBA")
        mand = mandala_layer(pal)
        my = (H - mand.shape[0]) // 2
        bg_np = np.asarray(bg.convert("RGB")).astype(np.float32)
        mx = (W - mand.shape[1]) // 2
        bg_np[my:my + mand.shape[0], mx:mx + mand.shape[1]] += mand
        bg = Image.fromarray(np.clip(bg_np, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        bg.alpha_composite(card)
        card = bg.convert("RGB")
    card.save(args.out)
    print("saved", args.out)
