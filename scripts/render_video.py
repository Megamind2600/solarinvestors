"""Render one beat-synced Gita short: an animated still (not a video shoot) —
Ken Burns pan/zoom on a cosmic backdrop, floating golden embers, spark bursts,
shockwave rings, screen shake, light sweeps and flash pulses — every effect
driven by the tabla beats / flute phrases / tingsha bells of the track we
synthesise for it, so audio and motion are locked together by construction.

    python scripts/render_video.py --id 2.47 --out /tmp/out.mp4
    python scripts/render_video.py --id 2.47 --sheet /tmp/sheet.jpg    # quick QA
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import W, H, FPS, DEFAULT_DURATION, TRACKS, find_verse, map_track, ffmpeg_exe, VIDEOS
from scene import palette, build_background, build_text_stack, mandala_layer
from synth_music import build_track, write_wav

# --------------------------------------------------------------------- helpers


def envelope(times: np.ndarray, acc: np.ndarray, t: float, tau: float) -> float:
    """Sum of exponentially-decaying impulses at `times` up to t."""
    if not len(times):
        return 0.0
    i1 = np.searchsorted(times, t - 5 * tau)
    i2 = np.searchsorted(times, t)
    if i1 >= i2:
        return 0.0
    seg = times[i1:i2]
    return float((acc[i1:i2] * np.exp(-(t - seg) / tau)).sum())


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Embers:
    """Golden ember particles drifting upward + beat-triggered spark bursts."""

    N = 190

    def __init__(self, seed: int, pal: dict):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.color = pal["acc"]
        self.col2 = np.array([255, 236, 190], np.float32)
        # base field
        self.x = rng.uniform(0, W, self.N)
        self.y = rng.uniform(0, H, self.N)
        self.rise = rng.uniform(9, 34, self.N)
        self.sway = rng.uniform(3, 14, self.N)
        self.phase = rng.uniform(0, 6.283, self.N)
        self.size = rng.uniform(0.5, 1.7, self.N)
        self.bright = rng.uniform(0.25, 0.9, self.N)
        self.tw = rng.uniform(0.6, 2.6, self.N)
        # burst pool
        self.B = 160
        self.bx = np.zeros(self.B); self.by = np.zeros(self.B)
        self.bvx = np.zeros(self.B); self.bvy = np.zeros(self.B)
        self.blife = np.zeros(self.B)          # seconds remaining (0 = dead)
        self.btot = np.ones(self.B)
        self.cursor = 0
        # sprite cache
        self.sprites = [self._sprite(r) for r in (2, 3, 4, 6)]

    @staticmethod
    def _sprite(r: int) -> np.ndarray:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        g = np.exp(-(xx * xx + yy * yy) / (r * r * 0.55))
        return np.stack([g, g, g], axis=-1)

    def burst(self, kind: str):
        n = {"beat": 10, "sam": 22, "bell": 34, "fill": 14}[kind]
        cx = self.rng.uniform(W * 0.3, W * 0.7)
        cy = self.rng.uniform(H * 0.55, H * 0.8) if kind != "bell" else H * 0.5
        ang = self.rng.uniform(-math.pi * 0.85, -math.pi * 0.15, n)
        spd = self.rng.uniform(60, 260 if kind != "bell" else 420, n)
        bx = cx + self.rng.normal(0, 30, n)
        by = cy + self.rng.normal(0, 12, n)
        bvx = spd * np.cos(ang) * 0.6
        bvy = spd * np.sin(ang)
        btot = self.rng.uniform(0.7, 1.4, n)
        # ring-buffer write (split at wrap point)
        first = min(n, self.B - self.cursor)
        for a, b in ((0, first), (first, n)):
            if a >= b:
                continue
            s = slice((self.cursor + a) % self.B, (self.cursor + a) % self.B + (b - a))
            self.bx[s] = bx[a:b]; self.by[s] = by[a:b]
            self.bvx[s] = bvx[a:b]; self.bvy[s] = bvy[a:b]
            self.btot[s] = btot[a:b]; self.blife[s] = btot[a:b]
        self.cursor = (self.cursor + n) % self.B

    def draw(self, frame: np.ndarray, t: float, dt: float, energy: float):
        h, w = frame.shape[:2]
        # ambient embers
        y = (self.y - (self.rise * (1 + 0.5 * energy)) * t) % h
        x = (self.x + self.sway * np.sin(t * 0.7 + self.phase)) % w
        b = np.clip(self.bright * (0.55 + 0.45 * np.sin(t * self.tw + self.phase)) * (1 + 1.4 * energy), 0, 1.6)
        for i in range(self.N):
            bi = b[i]
            if bi < 0.04:
                continue
            spr = self.sprites[min(1 + int(self.size[i] * 1.4), 3)]
            r = spr.shape[0] // 2
            xi, yi = int(x[i]), int(y[i])
            x0, x1 = max(0, xi - r), min(w, xi + r + 1)
            y0, y1 = max(0, yi - r), min(h, yi + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            sx0, sy0 = x0 - (xi - r), y0 - (yi - r)
            col = self.color * 0.6 + self.col2 * 0.4
            frame[y0:y1, x0:x1] += spr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] * col * bi * 0.55
        # bursts
        alive = self.blife > 0
        if alive.any():
            self.blife[alive] -= dt
            self.bvy[alive] += 240 * dt
            self.bx[alive] += self.bvx[alive] * dt
            self.by[alive] += self.bvy[alive] * dt
            af = np.clip(self.blife / self.btot, 0, 1)
            for j in np.where(alive)[0]:
                fade = af[j] ** 1.4
                if fade < 0.03:
                    continue
                spr = self.sprites[1]
                r = spr.shape[0] // 2
                xi, yi = int(self.bx[j]), int(self.by[j])
                x0, x1 = max(0, xi - r), min(w, xi + r + 1)
                y0, y1 = max(0, yi - r), min(h, yi + r + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                sx0, sy0 = x0 - (xi - r), y0 - (yi - r)
                frame[y0:y1, x0:x1] += spr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] * self.col2 * fade * 1.4


# --------------------------------------------------------------- frame engine
class Engine:
    def __init__(self, verse: dict, duration: float, seed: int, track_idx: int | None,
                 show_translit: bool = True):
        self.verse, self.duration, self.seed = verse, duration, seed
        self.pal = palette(int(verse["chapter"]))
        self.track_idx = map_track(verse["id"]) if track_idx is None else track_idx
        self.spec = TRACKS[self.track_idx]
        self.rng = np.random.default_rng(seed)

        # --- audio + beat map -------------------------------------------------
        self.pcm, self.beatmap = build_track(self.spec, duration + 0.35)
        beats = self.beatmap["beats"]
        self.b_times = np.array([b["t"] for b in beats], np.float32)
        self.b_acc = np.array([b["acc"] for b in beats], np.float32)
        self.sam_times = np.array([b["t"] for b in beats if b.get("sam")], np.float32)
        self.sam_acc = np.ones_like(self.sam_times)
        self.fill_times = self.b_times[(self.b_acc > 0.4) & (self.b_acc < 0.8)]
        self.fill_acc = self.b_acc[(self.b_acc > 0.4) & (self.b_acc < 0.8)]
        self.sweeps = np.array(self.beatmap["sweeps"], np.float32)
        self.bells = np.array(self.beatmap["bells"], np.float32)
        self._bells_fired: set = set()
        self._sam_fired: set = set()

        # --- static visual layers ---------------------------------------------
        bg, _sp, _sm = build_background(self.pal, seed + 5)
        self.bg_pil = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        # camera-space twinkling starfield
        self.star_phase = np.zeros((H, W), np.float32)
        self.star_amp = np.zeros((H, W), np.float32)
        n_st = 220
        xs = self.rng.uniform(0, W - 1, n_st).astype(int)
        ys = self.rng.uniform(0, H - 1, n_st).astype(int)
        self.star_phase[ys, xs] = self.rng.uniform(0, 6.283, n_st)
        self.star_amp[ys, xs] = self.rng.uniform(18, 85, n_st)
        self.bw, self.bh = self.bg_pil.size
        self.bg_off_x = (self.bw - W) / 2
        self.bg_off_y = (self.bh - H) / 2
        self.mandala = mandala_layer(self.pal, seed=seed + 6)
        self.text = build_text_stack(verse, self.pal, show_transliteration=show_translit)
        self.text_np = np.asarray(self.text).astype(np.float32)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        self.vignette = (0.78 + 0.22 * np.exp(-(rr ** 2) * 1.35))[:, :, None]
        glow_r = np.exp(-(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.52) / (H * 0.5)) ** 2))
        self.glowmap = glow_r[:, :, None] * self.pal["acc"][None, None, :] * 0.30
        self.flashmap = np.stack([np.exp(-rr ** 2 * 2.4), np.exp(-rr ** 2 * 2.4),
                                  np.exp(-rr ** 2 * 2.9)], axis=-1) * 120
        dg = ((xx - yy) * 0.7 + W * 0.5) % (W * 2.2)
        self.sweep_band = np.exp(-((dg - W * 1.1) ** 2) / (W * W * 0.016))[:, :, None] * 34.0
        self.rings_active: list[tuple[float, float]] = []   # (t_start, strength)
        self.embers = Embers(seed + 7, self.pal)
        self.grain_rng = np.random.default_rng(seed + 9)
        self.t_prev = 0.0
        self.panel_cy = H * 0.52

    # ------------------------------------------------------------------ events
    def _fire_events(self, t: float):
        for bt in self.bells:
            if abs(t - bt) < 1 / FPS * 0.9 and bt not in self._bells_fired:
                self._bells_fired.add(bt)
                self.embers.burst("bell")
                self.rings_active.append((t, 1.0))
        for st in self.sam_times:
            if abs(t - st) < 1 / FPS * 0.9 and st not in self._sam_fired:
                self._sam_fired.add(st)
                self.embers.burst("sam")
                self.rings_active.append((t, 0.65))

    # ------------------------------------------------------------------- frame
    def frame(self, t: float) -> np.ndarray:
        dt = t - self.t_prev
        self.t_prev = t
        self._fire_events(t)

        slow = envelope(self.b_times, self.b_acc, t, 0.55)          # musical swell
        fast = envelope(self.sam_times, self.sam_acc, t, 0.085)     # sharp punch
        fills = envelope(self.fill_times, self.fill_acc, t, 0.16)
        bell_e = envelope(self.bells, np.ones_like(self.bells), t, 1.5)

        # Ken Burns window + punch zoom + shake
        z_base = 1.115 - 0.085 * smoothstep(t / self.duration)
        z = z_base * (1 - 0.020 * min(fast + fills * 0.4, 1.0))
        wi, hi = W * z, H * z
        ax = (self.bw - wi) / 2
        ay = (self.bh - hi) / 2
        px = ax * (1 + 0.55 * math.sin(2 * math.pi * (t / 47.0) + 0.9))
        py = ay * (1 + 0.62 * math.sin(2 * math.pi * (t / 59.0) + 2.1))
        shake = min(1.8, fast * 2.0 + bell_e * 1.2)
        sx = shake * 3.4 * math.sin(t * 61.7)
        sy = shake * 2.6 * math.sin(t * 53.3 + 1.0)
        x0 = float(np.clip(px + sx, 0, self.bw - wi))
        y0 = float(np.clip(py + sy, 0, self.bh - hi))

        crop = self.bg_pil.crop((int(x0), int(y0), int(x0 + wi), int(y0 + hi))).resize(
            (W, H), Image.BILINEAR)
        f = np.asarray(crop).astype(np.float32)

        # twinkling stars (camera space — distant stars don't pan with the crop)
        tw = np.clip(np.sin(t * 2.2 + self.star_phase), 0, 1) ** 6
        f += (tw * self.star_amp)[:, :, None] * np.array([0.9, 0.95, 1.0], np.float32)[None, None, :]

        # ambient mandala breathing (centered behind panel)
        m = self.mandala * (0.42 + 0.14 * math.sin(2 * math.pi * t / 23.0) + 0.55 * slow + 0.5 * bell_e)
        mh, mw = m.shape[:2]
        my, mx = (H - mh) // 2, (W - mw) // 2
        f[my:my + mh, mx:mx + mw] += m

        # beat glow + sam flash
        f += self.glowmap * min(1.3, slow * 0.8 + fills * 0.25)
        f += self.flashmap * min(1.0, fast * 0.55 + bell_e * 0.8)

        # light sweep on flute phrase starts
        sw = envelope(self.sweeps, np.ones_like(self.sweeps), t, 1.15)
        if sw > 0.03:
            shift = int((((t * 0.55) % 1.6) - 0.3) * W * 1.4)
            f += np.roll(self.sweep_band, shift, axis=1) * min(1.0, sw)

        # shockwave rings
        self.rings_active = [(ts, s) for ts, s in self.rings_active if t - ts < 1.05]
        if self.rings_active:
            ring_img = Image.new("L", (W, H), 0)
            rd = ImageDraw.Draw(ring_img)
            for ts, s in self.rings_active:
                p = (t - ts) / 1.05
                r = 70 + 760 * (p ** 0.8)
                a = int(150 * s * (1 - p) ** 1.6)
                if a > 2:
                    rd.ellipse([W / 2 - r, self.panel_cy - r, W / 2 + r, self.panel_cy + r],
                               outline=a, width=4)
            ring_np = np.asarray(ring_img.filter(ImageFilter.GaussianBlur(1.4))).astype(np.float32)[:, :, None]
            f += ring_np / 255.0 * self.pal["acc"][None, None, :] * 0.85

        # particles
        self.embers.draw(f, t, dt or 1 / FPS, energy=min(1.0, slow * 0.7 + fills * 0.3))

        # vignette
        f *= self.vignette

        # text stack: float + fade in/out
        dy = int(6.5 * math.sin(2 * math.pi * t / 11.0) + 1.8 * math.sin(2 * math.pi * t / 3.7))
        fade = smoothstep(t / 1.1) * smoothstep((self.duration - t) / 2.2)
        a = (self.text_np[:, :, 3:4] * fade / 255.0)
        txt = self.text_np
        f = f * (1 - a) + txt[:, :, :3] * a if dy == 0 else self._shifted_text(f, a, txt, dy, fade)

        # film grain + fades
        grain = self.grain_rng.integers(-3, 4, (H, W, 1)).astype(np.float32)
        f += grain
        f *= smoothstep(t / 0.7) * smoothstep((self.duration + 0.05 - t) / 1.4)
        np.clip(f, 0, 255, out=f)
        return f

    def _shifted_text(self, f, a, txt, dy, fade):
        h, w = f.shape[:2]
        out = f.copy()
        y0s, y1s = max(0, -dy), min(h, h - dy)
        y0d, y1d = max(0, dy), min(h, h + dy)
        region_txt = txt[y0s:y1s]
        region_a = a[y0s:y1s]
        out[y0d:y1d] = out[y0d:y1d] * (1 - region_a) + region_txt[:, :, :3] * region_a
        return out

    # ------------------------------------------------------------------ render
    def render_mp4(self, out_path: str, crf: int = 21, preset: str = "fast"):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        write_wav(self.pcm, wav_path)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
               "-i", wav_path,
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-profile:v", "high", "-level", "4.2", "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
               "-shortest", "-t", f"{self.duration:.2f}", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = int(round(self.duration * FPS))
        t0 = time.time()
        for i in range(n):
            t = i / FPS
            proc.stdin.write(self.frame(t).astype(np.uint8).tobytes())
            if i % (FPS * 5) == 0:
                el = time.time() - t0
                print(f"  {i}/{n} frames  ({el:.0f}s elapsed)", flush=True)
        proc.stdin.close()
        rc = proc.wait()
        Path(wav_path).unlink(missing_ok=True)
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
        return out

    def contact_sheet(self, path: str, times=(2.0, 9.0, 16.0, 24.0, 33.0)):
        ims = [Image.fromarray(self.frame(t).astype(np.uint8)) for t in times]
        tw = W // 3
        th = H // 3
        sheet = Image.new("RGB", (tw * len(ims), th))
        for i, im in enumerate(ims):
            sheet.paste(im.resize((tw, th)), (i * tw, 0))
        sheet.save(path, quality=88)
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--fps", type=float, default=None, help="override FPS (QA only)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--track", type=int, default=None)
    ap.add_argument("--no-translit", action="store_true")
    ap.add_argument("--sheet", default=None, help="save a contact sheet instead of mp4")
    ap.add_argument("--crf", type=int, default=21)
    ap.add_argument("--preset", default="fast")
    args = ap.parse_args()

    global FPS
    if args.fps:
        FPS = args.fps
    verse = find_verse(args.id)
    seed = args.seed if args.seed is not None else (hash(args.id) & 0xFFFF)
    eng = Engine(verse, args.duration, seed=seed, track_idx=args.track,
                 show_translit=not args.no_translit)
    print(f"verse {args.id} | track {eng.track_idx} ({eng.spec['key']}) | "
          f"{len(eng.b_times)} beats, {len(eng.sweeps)} sweeps, {len(eng.bells)} bells")
    if args.sheet:
        eng.contact_sheet(args.sheet)
        print("sheet:", args.sheet)
        return
    out = args.out or str(VIDEOS / f"ch{int(verse['chapter']):02d}" / f"{args.id.replace('.', '_')}.mp4")
    t0 = time.time()
    eng.render_mp4(out, crf=args.crf, preset=args.preset)
    print(f"saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
