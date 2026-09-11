"""Render one reel: a still card that *feels* filmed, driven by real audio.

Audio comes from the open-licensed meditative library (scripts/fetch_music.py,
music/library/) when it exists - each shloka gets its own window into a track -
and falls back to the built-in procedural raga synth otherwise.  Whatever the
source, the motion is driven by what the audio actually does:

  * camera kick / vibration / rotation on onsets and deep swells
  * layer parallax (backdrop, mandala and the card each move differently)
  * sparks, firecracker bursts with falling trails, smoke, shockwave rings,
    spark-shuttles that streak across the frame, lens-flare stars,
    ignition glow on the shloka and light sweeps
  * continuous breathing from the energy / bass / air envelopes, so even the
    quietest drone passage is visibly alive

Everything soft (glow, flash, rim, sweep, halo) is composited at quarter
resolution and blurred up once - 4x the pixels costs nothing, and a phone
screen never sees the difference.  That keeps a 45 s reel near ~5 min of CPU.

    python scripts/render_video.py --id 2.47 --out /tmp/o.mp4
    python scripts/render_video.py --id 2.47 --fx meditative --sheet /tmp/s.jpg
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_DURATION, FPS, TRACKS, VIDEOS, W, H, ffmpeg_exe,
                    find_verse, map_track)
from fx import FX, PRESETS, bloom
from library import build_pad, ensure_map, load_pool, pick, slice_events, load_map
from scene import build_background, build_text_stack, mandala_layer, palette
from synth_music import build_track, write_wav

Q = 4                     # soft-light layers are computed at 1/Q res
W4, H4 = W // Q, H // Q


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Engine:
    def __init__(self, verse: dict, duration: float, seed: int, track_idx: int | None,
                 show_translit: bool = True, audio: str = "auto",
                 fx_preset: str = "cinematic", fx_gain: float | None = None,
                 pool: list[dict] | None = None):
        self.verse, self.duration, self.seed = verse, duration, seed
        self.pal = palette(int(verse["chapter"]))
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------- audio + event map
        self.lib_rec, self.audio_wav, ev, self.meta = None, None, None, None
        if audio in ("auto", "library"):
            rec, off, meta = pick(verse["id"], duration, pool if pool is not None
                                  else load_pool())
            if rec is not None:
                mp = ensure_map(rec)
                if mp is not None:
                    try:
                        ev = slice_events(load_map(mp), off, duration)
                        self.audio_wav = build_pad(rec, off, duration)
                        self.lib_rec, self.lib_off, self.meta = rec, off, meta
                    except (subprocess.CalledProcessError, OSError) as e:
                        print("  pad failed, using synth:", str(e)[:120])
                        ev = self.lib_rec = self.audio_wav = self.meta = None
        if ev is None:
            self.track_idx = map_track(verse["id"]) if track_idx is None else track_idx
            self.spec = TRACKS[self.track_idx]
            pcm, bm = build_track(self.spec, duration + 0.35)
            ev = dict(beats=bm["beats"], sweeps=bm["sweeps"], bells=bm["bells"],
                      env={}, env_dt=0.25, key=bm["key"])
            self.pcm = pcm
            self.source = f"synth:{self.spec['key']}"
            fx_gain = 1.0 if fx_gain is None else fx_gain
        else:
            self.pcm = None
            self.source = f"library:{self.lib_rec['id']}"
            fx_gain = float(self.meta["mood"]["gain"]) if fx_gain is None else fx_gain
        self.ev = ev
        self.fx_preset, self.fx_gain = fx_preset, float(fx_gain)
        self.fx = FX(ev, duration, FPS, W, H, self.pal, preset=fx_preset,
                     seed=seed, gain=self.fx_gain)

        # ------------------------------------------------------ static layers
        bg, _sp, _sm = build_background(self.pal, seed + 5)
        self.bg_pil = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        self.bw, self.bh = self.bg_pil.size
        self.mandala = mandala_layer(self.pal, seed=seed + 6)
        # kept at half res: per-frame rotation/zoom of the halo is then ~free
        mh0, mw0 = self.mandala.shape[:2]
        self.mand_small = Image.fromarray(np.clip(self.mandala, 0, 255).astype(np.uint8)) \
            .resize((mw0 // 2, mh0 // 2), Image.BILINEAR)
        self.text = build_text_stack(verse, self.pal, show_transliteration=show_translit)
        self.text_np = np.asarray(self.text).astype(np.float32)

        # starfield: 220 discrete sprites (a full-res twinkle map cost 60 ms/frame)
        n_st = 220
        self.st_x = self.rng.uniform(2, W - 3, n_st).astype(np.int32)
        self.st_y = self.rng.uniform(2, H - 3, n_st).astype(np.int32)
        self.st_ph = self.rng.uniform(0, 6.283, n_st).astype(np.float32)
        self.st_amp = self.rng.uniform(26, 120, n_st).astype(np.float32)
        self.st_spr = [self._sprite(r) for r in (2, 3, 4)]

        # ---- soft light maps, all at quarter res ----------------------------
        yy, xx = np.mgrid[0:H4, 0:W4].astype(np.float32)
        cx, cy = (xx * Q + Q / 2 - W / 2) / (W / 2), (yy * Q + Q / 2 - H * 0.52) / (H / 2)
        rr = np.hypot(cx, cy)
        acc = np.asarray(self.pal["acc"], np.float32)[None, None, :]
        self.glow4 = (np.exp(-(((xx * Q) / (W * 0.62)) ** 2 + ((yy * Q - H * 0.52) / (H * 0.5)) ** 2))
                      [:, :, None] * acc * 0.36)
        fl = np.exp(-rr ** 2 * 2.2)
        self.flash4 = np.stack([fl, fl ** 0.97, fl ** 0.70], axis=-1) * 130.0
        self.rim4 = (np.exp(-((rr - 1.02) ** 2) * 26.0)[:, :, None] *
                     np.array([1.0, 0.60, 0.26], np.float32)[None, None, :] * 60.0)
        dg = ((xx * Q - yy * Q * 0.9) * 0.7 + W * 0.5) % (W * 2.2)
        self.sweep4 = (np.exp(-((dg - W * 1.1) ** 2) / (W * W * 0.010))[:, :, None] * 52.0)
        ta = Image.fromarray(self.text_np[:, :, 3].astype(np.uint8)).resize(
            (W4, H4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2.2))
        self.halo4 = (np.asarray(ta, np.float32)[:, :, None] / 255.0) * acc * 1.15
        self.vig4 = (0.78 + 0.22 * np.exp(-(rr ** 2) * 1.35))[:, :, None]
        self.vigd4 = (rr ** 2)[:, :, None] * 0.11
        # film grain: 6 cached half-res frames, cycled (cheaper than drawing it)
        g_rng = np.random.default_rng(seed + 9)
        self.grains = [(g_rng.integers(-3, 4, (H // 2, W // 2, 1))).astype(np.float32)
                       for _ in range(6)]
        rows = np.where(self.text_np[:, :, 3] > 4)[0]
        self.tbox = ((max(0, int(rows.min()) - 26), min(H, int(rows.max()) + 26))
                     if len(rows) else (0, H))
        self.t_prev = 0.0
        self.panel_cy = H * 0.52

    @staticmethod
    def _sprite(r: int) -> np.ndarray:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        return np.exp(-(xx * xx + yy * yy) / (r * r * 0.5))

    # -------------------------------------------------------- backdrop + camera
    def _background(self, i: int, t: float) -> np.ndarray:
        shx, shy, rot, zp = self.fx.camera(i)
        g = self.fx.glow(i)
        z_base = 1.118 - 0.075 * smoothstep(t / self.duration)
        z = float(np.clip(z_base * (1 + 0.006 * (g["breath"] - 0.5)) / zp, 1.035, 1.131))
        wi, hi = W * z, H * z
        ax = (self.bw - wi) / 2
        ay = (self.bh - hi) / 2
        px = ax * (1 + 0.55 * math.sin(2 * math.pi * (t / 47.0) + 0.9))
        py = ay * (1 + 0.62 * math.sin(2 * math.pi * (t / 59.0) + 2.1))
        pad = int(0.035 * wi)
        x0 = float(np.clip(px + shx, 0, self.bw - wi))
        y0 = float(np.clip(py + shy, 0, self.bh - hi))
        bx0, by0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        bx1 = int(min(self.bw, x0 + wi + pad))
        by1 = int(min(self.bh, y0 + hi + pad))
        crop = self.bg_pil.crop((bx0, by0, bx1, by1))
        if abs(rot) > 0.02:
            # rotate the *padded* window so the overscan swallows the corners
            crop = crop.rotate(rot, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        cpx, cpy = int(x0) - bx0, int(y0) - by0
        crop = crop.crop((cpx, cpy, cpx + int(wi), cpy + int(hi))).resize((W, H), Image.BILINEAR)
        return np.asarray(crop).astype(np.float32)

    # ------------------------------------------------------------------- frame
    def frame(self, t: float) -> np.ndarray:
        i = int(round(t * FPS))
        dt = max(1e-4, min(0.2, t - self.t_prev))
        self.t_prev = t
        self.fx.kick(i)                                   # spawn this frame's events
        g = self.fx.glow(i)
        shx, shy = self.fx.camera(i)[0], self.fx.camera(i)[1]

        f = self._background(i, t)

        # starfield twinkle (screen space)
        tw = np.clip(np.sin(t * 2.2 + self.st_ph), 0, 1) ** 6 * self.st_amp
        for k in range(len(self.st_x)):
            b = tw[k]
            if b < 6:
                continue
            spr = self.st_spr[min(2, int(b / 45))]
            r = spr.shape[0] // 2
            xi, yi = int(self.st_x[k]), int(self.st_y[k])
            y0, y1 = max(0, yi - r), min(H, yi + r + 1)      # clamp at frame edges
            x0, x1 = max(0, xi - r), min(W, xi + r + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            sy0, sx0 = y0 - (yi - r), x0 - (xi - r)
            f[y0:y1, x0:x1] += spr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][:, :, None] \
                * np.array([0.9, 0.95, 1.0], np.float32) * b

        # mandala: slow rotation + scale breathing (continuous motion even in silence),
        # parallax kick at 1.45x the backdrop, brightness riding the envelope
        m_amp = 0.44 + 0.18 * math.sin(2 * math.pi * t / 21.0) + 0.62 * g["slow"] \
            + 0.5 * g["bell"] + 0.22 * g["air"]
        rot_m = 6.0 * math.sin(2 * math.pi * t / 53.0) + 0.42 * t          # deg, never still
        ms = 1.0 + 0.06 * math.sin(2 * math.pi * t / 17.0) + 0.05 * g["slow"]
        m_im = self.mand_small.rotate(rot_m, resample=Image.BILINEAR,
                                      fillcolor=(0, 0, 0))
        mh, mw = self.mandala.shape[:2]
        nw, nh = int(mw * ms), int(mh * ms)
        m_np = np.asarray(m_im.resize((nw, nh), Image.BILINEAR), np.float32)
        my = (H - nh) // 2 + int(shy * 1.45)
        mx = (W - nw) // 2 + int(shx * 1.45)
        y0, y1 = max(0, my), min(H, my + nh)
        x0, x1 = max(0, mx), min(W, mx + nw)
        if y0 < y1 and x0 < x1:
            f[y0:y1, x0:x1] += m_np[y0 - my:y1 - my, x0 - mx:x1 - mx] * m_amp

        # particles / rings / petals: half-res buffer, blurred once, added
        img = self.fx.layer(i, t, dt).filter(ImageFilter.GaussianBlur(1.1))
        f += np.asarray(img.resize((W, H), Image.BILINEAR), np.float32) * 1.0

        # ---- all soft light in one quarter-res composite --------------------
        tx4, ty4 = int(shx * 0.5 / Q), int(shy * 0.5 / Q)
        lay = (self.glow4 * min(1.2, g["slow"] * 0.7 + g["beat"] * 0.35)
               + self.flash4 * min(0.42, g["flash"] * 0.55)
               + self.rim4 * min(0.8, g["sam"] * 0.7 + g["bell"] * 0.5))
        if g["sweep"] > 0.02:
            shift = int((((t * 0.5) % 1.9) - 0.35) * W4 * 1.4)
            lay = lay + np.roll(self.sweep4, shift, axis=1) * min(1.0, g["sweep"] * 1.3)
        lay = lay + np.roll(np.roll(self.halo4, ty4, axis=0), tx4, axis=1) * \
            (0.08 + 0.42 * g["slow"] + 0.7 * g["sam"])
        # lay is already in 0..255 light units: quantise, upscale, add
        f += np.asarray(Image.fromarray(np.clip(lay, 0, 255).astype(np.uint8))
                        .resize((W, H), Image.BILINEAR), np.float32)

        # vignette, tightened on hits
        v = self.vig4 * (1.0 - self.vigd4 * min(1.0, g["sam"] + g["bell"]))
        f *= np.repeat(np.repeat(v, Q, axis=0), Q, axis=1)[:H, :W]

        # ---- text: float + parallax kick + beat lift, inside its own band ----
        dy = int(7.0 * math.sin(2 * math.pi * t / 11.0) + 2.2 * math.sin(2 * math.pi * t / 3.7)
                 + shy * 0.5)
        dx = int(tx4 * 2)
        fade = smoothstep(t / 1.1) * smoothstep((self.duration - t) / 2.2)
        yb0, yb1 = self.tbox[0], self.tbox[1]
        a = self.text_np[yb0:yb1, :, 3:4] * fade / 255.0
        lift = 1.0 + 0.14 * g["slow"] + 0.30 * g["sweep"] + 0.18 * g["beat"]
        txt = np.clip(self.text_np[yb0:yb1, :, :3] * lift, 0, 255)
        if dx or dy:
            a, txt = self._shift(a, txt, dx, dy)
        f[yb0:yb1] = f[yb0:yb1] * (1 - a) + txt * a

        # bloom on the finished frame, then grain + fades
        f += bloom(f, gain=0.34 * self.fx.p["bloom"] + 0.22 * g["sam"])
        d = np.clip(f - 168.0, 0, None)          # soft knee: highlights roll off
        if d.max() > 0:                           # instead of flattening to white
            f -= d * (1.0 - np.exp(-d / 130.0)) * 0.72
        f += self.grains[i % len(self.grains)].repeat(2, 0).repeat(2, 1)[:H, :W]
        f *= smoothstep(t / 0.7) * smoothstep((self.duration + 0.05 - t) / 1.4)
        np.clip(f, 0, 255, out=f)
        return f

    @staticmethod
    def _shift(a, txt, dx, dy):
        h, w = a.shape[:2]
        ys, yd = max(0, -dy), min(h, h - dy)
        xs, xd = max(0, -dx), min(w, w - dx)
        ya, yb = max(0, dy), min(h, h + dy)
        xa, xb = max(0, dx), min(w, w + dx)
        A, T = np.zeros_like(a), np.zeros_like(txt)
        if ya < yb and xa < xb and ys < yd and xs < xd:
            A[ya:yb, xa:xb] = a[ys:yd, xs:xd]
            T[ya:yb, xa:xb] = txt[ys:yd, xs:xd]
        return A, T

    # ------------------------------------------------------------------ output
    def render_mp4(self, out_path: str, crf: int = 21, preset: str = "fast"):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        if self.audio_wav is not None:
            wav = str(self.audio_wav)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = wav = tmp.name
            write_wav(self.pcm, wav)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "pipe:0", "-i", wav,
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-profile:v", "high", "-level", "4.2", "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
               "-shortest", "-t", f"{self.duration:.2f}", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = int(round(self.duration * FPS))
        t0 = time.time()
        for k in range(n):
            proc.stdin.write(self.frame(k / FPS).astype(np.uint8).tobytes())
            if k % (FPS * 5) == 0:
                el = time.time() - t0
                print(f"  {k}/{n} frames  ({el:.0f}s, {el/max(1,k):.2f}s/frame)", flush=True)
        proc.stdin.close()
        rc = proc.wait()
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
        return out

    def contact_sheet(self, path: str, times=None):
        if times is None:
            n = 5
            hot = np.argsort(-self.fx.flash)[: 80]
            cand = sorted(set(int(round(h / FPS * 10) / 10) for h in hot)) or [1.0]
            times = [cand[int(k * (len(cand) - 1) / (n - 1))] for k in range(n)] \
                if len(cand) > n else cand
        ims = [Image.fromarray(self.frame(float(t)).astype(np.uint8)) for t in times]
        tw, th = W // 3, H // 3
        sheet = Image.new("RGB", (tw * len(ims), th))
        for k, im in enumerate(ims):
            sheet.paste(im.resize((tw, th)), (k * tw, 0))
        sheet.save(path, quality=90)
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--fps", type=float, default=None, help="override FPS (QA only)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--track", type=int, default=None, help="force synth track index")
    ap.add_argument("--audio", choices=["auto", "library", "synth"], default="auto")
    ap.add_argument("--fx", choices=sorted(PRESETS), default="cinematic",
                    help="meditative = breathing + sparse embers; festive = fireworks")
    ap.add_argument("--fx-gain", type=float, default=None)
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
                 show_translit=not args.no_translit, audio=args.audio,
                 fx_preset=args.fx, fx_gain=args.fx_gain)
    ev = eng.ev
    print(f"verse {args.id} | audio {eng.source} | fx {args.fx} x{eng.fx_gain:.2f} | "
          f"{len(ev['beats'])} beats ({len([b for b in ev['beats'] if b['sam']])} accents), "
          f"{len(ev['sweeps'])} sweeps, {len(ev['bells'])} bells"
          + (f" | window {eng.lib_off:.1f}s [{eng.lib_rec.get('license')}]"
             if eng.lib_rec else ""))
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
