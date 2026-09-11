"""The "alive" layer: beat-reactive camera, sparks, crackers, rings, petals.

Everything here is driven by an Exciters object built from the audio analysis
(see analyze_music.py): a handful of per-frame float channels.  Two kinds of
signal are used on purpose:

  * *continuous* channels (energy / bass / air swells) keep the picture moving
    all the time - the frame never looks like a paused photo, which matters most
    for meditative music where nothing "drops", and
  * *impulse* channels (onsets, downbeat-like swells, bells) fire the punchy
    stuff: camera kick, sparks, cracker bursts, shockwave rings, flash.

Rendering is deliberately done in a low-res "fx buffer" (particles, rings,
smoke) which is blurred once and added back: that is what real pyro looks like
on a phone screen - soft blooming streaks, not 1-pixel dots.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# ------------------------------------------------------------------- presets
PRESETS = {
    # calm: music breathes, so the picture breathes; sparks are sparse embers
    "meditative": dict(shake=5.0, rot=0.22, zoom=0.014, flash=0.34, sparks=0.45,
                       crackers=0.0, rings=0.9, petals=1.0, dust=1.0, bloom=0.55,
                       twinkle=2.0, trail=0.5, max_fire=3.2, ember=0.75, sweep=26.0),
    # the house style: audible on every beat, still tasteful
    "cinematic": dict(shake=9.5, rot=0.45, zoom=0.026, flash=0.62, sparks=1.0,
                      crackers=0.55, rings=1.35, petals=0.35, dust=1.0, bloom=0.85,
                      twinkle=4.5, trail=0.9, max_fire=5.5, ember=1.0, sweep=42.0),
    # festival / samadhi-level: full fireworks grammar
    "festive": dict(shake=15.0, rot=0.8, zoom=0.042, flash=0.72, sparks=1.5,
                    crackers=1.0, rings=1.9, petals=0.2, dust=1.0,
                    bloom=0.95, twinkle=7.0, trail=1.35, max_fire=9.0, ember=1.3,
                    sweep=58.0),
}
GOLD = np.array([255.0, 208.0, 118.0], np.float32)
WHITE = np.array([255.0, 250.0, 235.0], np.float32)
EMBER = np.array([255.0, 116.0, 34.0], np.float32)
SMOKE = np.array([120.0, 104.0, 96.0], np.float32)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ exciters
class Exciters:
    """Per-frame scalar channels, precomputed once for the whole reel."""

    CH = ("beat", "sam", "bell", "sweep", "energy", "bass", "air", "flux")

    def __init__(self, events: dict, duration: float, fps: float):
        n = int(round(duration * fps)) + 2
        t = np.arange(n) / fps
        self.n, self.t, self.fps = n, t, fps

        beats = events.get("beats", []) or [{"t": 1.0, "acc": 0.4, "sam": True}]
        b_t = np.array([b["t"] for b in beats], np.float32)
        b_a = np.array([float(b.get("acc", 0.5)) for b in beats], np.float32)
        s_t = b_t[[bool(b.get("sam")) for b in beats]]
        bell_t = np.array(events.get("bells", []), np.float32)
        sweep_t = np.array(events.get("sweeps", []), np.float32)

        def impulses(times) -> np.ndarray:
            y = np.zeros(n, np.float32)
            if len(times):
                idx = (np.asarray(times) * fps).astype(int)
                idx = idx[(idx >= 0) & (idx < n)]
                np.add.at(y, idx, 1.0)
            return y

        def decay(y, tau: float) -> np.ndarray:
            """IIR exp decay (a decaying impulse train in one pass)."""
            a = math.exp(-1.0 / (tau * fps))
            out = np.empty_like(y)
            acc = 0.0
            for i in range(len(y)):
                acc = acc * a + y[i]
                out[i] = acc
            return out

        # weighted impulses so accent strength survives the decay
        b_imp, s_imp = np.zeros(n, np.float32), np.zeros(n, np.float32)
        bi = (b_t * fps).astype(int)
        ok = (bi >= 0) & (bi < n)
        np.add.at(b_imp, bi[ok], b_a[ok])
        if len(s_t):
            si = (s_t * fps).astype(int)
            ok = (si >= 0) & (si < n)
            np.add.at(s_imp, si[ok], 1.0)

        self.ch = np.zeros((n, len(self.CH)), np.float32)
        self.ch[:, 0] = decay(b_imp, 0.34)
        self.ch[:, 1] = decay(s_imp, 0.42)
        self.ch[:, 2] = decay(impulses(bell_t), 1.15)
        self.ch[:, 3] = decay(impulses(sweep_t), 0.75)
        env = events.get("env", {})
        dt = float(events.get("env_dt", 0.25))
        for k, name in ((4, "energy"), (5, "bass"), (6, "air"), (7, "flux")):
            v = np.asarray(env.get(name, []), np.float32) / 255.0
            if len(v) > 1:
                x = np.interp(t, np.arange(len(v)) * dt, v)
            elif len(v) == 1:
                x = np.full(n, float(v[0]))
            else:
                x = 0.55 + 0.30 * np.sin(2 * math.pi * t / 9.0)   # no analysis yet
            self.ch[:, k] = x.astype(np.float32)
        for k in range(len(self.CH)):
            m = np.percentile(self.ch[:, k], 99) or 1.0
            self.ch[:, k] /= max(m, 1e-6)
        # fast "hit" channel used for shake: sharp attack, short tail
        self.hit = decay(b_imp * 0.35 + s_imp + impulses(bell_t) * 1.4, 0.115)
        self.hit /= (np.percentile(self.hit, 99) or 1.0)
        # event -> frame buckets so nothing is missed or double-fired
        self.fire_beat: list[list[float]] = [[] for _ in range(n)]
        for tt, aa in zip(b_t, b_a):
            i = int(tt * fps)
            if 0 <= i < n:
                self.fire_beat[i].append(float(aa))
        self.fire_sam = [(int(tt * fps), tt) for tt in s_t if 0 <= int(tt * fps) < n]
        self.fire_bell = [(int(tt * fps), tt) for tt in bell_t if 0 <= int(tt * fps) < n]
        self.fire_sweep = [(int(tt * fps), tt) for tt in sweep_t if 0 <= int(tt * fps) < n]

    def get(self, i: int, k: str) -> float:
        return float(self.ch[min(i, self.n - 1), self.CH.index(k)])


# ------------------------------------------------------------------- sprites
def _springs(rng, t: np.ndarray, hit: np.ndarray, freq: float, amp: float) -> np.ndarray:
    """Damped oscillation driven by an impulse train -> shake-style wobble."""
    ph = rng.uniform(0, 6.283)
    return (amp * hit * np.sin(2 * math.pi * freq * t + ph)).astype(np.float32)


# ------------------------------------------------------------------ particles
TICK, CRACKER, BELL, STREAK = 0, 1, 2, 3


class Sparks:
    """Pooled particle system: drifting embers, cracker sparks, spark-shuttles.

    Everything is drawn into a half-res RGB buffer with PIL primitives (a hot
    1-px core + a wider halo + a motion streak), then blurred once and added -
    that is what burning metal looks like on a phone screen.
    """

    MAX = 2000

    def __init__(self, seed: int, W: int, H: int, scale: float = 0.5):
        self.W, self.H, self.s = W, H, scale
        self.w2, self.h2 = int(W * scale), int(H * scale)
        rng = np.random.default_rng(seed)
        self.rng = rng
        M = self.MAX
        self.x = np.zeros(M, np.float32)
        self.y = np.zeros(M, np.float32)
        self.vx = np.zeros(M, np.float32)
        self.vy = np.zeros(M, np.float32)
        self.life = np.zeros(M, np.float32)
        self.tot = np.ones(M, np.float32)
        self.grav = np.zeros(M, np.float32)
        self.drag = np.zeros(M, np.float32)
        self.hot = np.zeros(M, np.float32)     # 1 = white hot, 0 = dying ember
        self.size = np.ones(M, np.float32)
        self.kind = np.zeros(M, np.uint8)
        self.popped = np.zeros(M, bool)
        self.ph = rng.uniform(0, 6.283, M).astype(np.float32)
        self.tw = rng.uniform(9, 34, M).astype(np.float32)
        self.cur = 0
        # ambient embers: always rising, so the frame is never a still photo
        self.N = 240
        self.ax = rng.uniform(0, self.w2, self.N).astype(np.float32)
        self.ay = rng.uniform(0, self.h2, self.N).astype(np.float32)
        self.arise = rng.uniform(4, 22, self.N).astype(np.float32)
        self.asway = rng.uniform(1.5, 7, self.N).astype(np.float32)
        self.aph = rng.uniform(0, 6.283, self.N).astype(np.float32)
        self.asz = rng.uniform(0.7, 2.3, self.N).astype(np.float32)

    # -- spawning ------------------------------------------------------------
    def _alloc(self, n: int) -> slice:
        i0 = self.cur
        self.cur = (self.cur + n) % self.MAX
        stop = min(self.MAX, i0 + n)
        return slice(i0, stop)

    def burst(self, cx: float, cy: float, n: int, speed: float, kind,
              spread: float = 1.0, upward: float = 0.65):
        if n <= 0:
            return
        rng = self.rng
        sl = self._alloc(n)
        k = sl.stop - sl.start
        if k <= 0:
            return
        ang = rng.uniform(-math.pi, 0, k) * spread + np.where(rng.random(k) < upward, 0.0, -math.pi)
        spd = rng.uniform(0.3, 1.0, k) ** 1.4 * speed
        self.x[sl] = cx + rng.normal(0, 3, k)
        self.y[sl] = cy + rng.normal(0, 2, k)
        self.vx[sl] = np.cos(ang) * spd
        self.vy[sl] = np.sin(ang) * spd
        self.popped[sl] = False
        self.kind[sl] = kind
        if kind == CRACKER:
            self.tot[sl] = rng.uniform(0.42, 1.05, k)
            self.grav[sl] = rng.uniform(360, 620, k)
            self.drag[sl] = rng.uniform(1.1, 2.4, k)
            self.hot[sl] = rng.uniform(0.55, 0.95, k)
            self.size[sl] = rng.uniform(0.8, 1.9, k)
        elif kind == BELL:
            self.tot[sl] = rng.uniform(0.9, 1.9, k)
            self.grav[sl] = rng.uniform(-55, 55, k)         # bell sparks hang
            self.drag[sl] = rng.uniform(0.7, 1.5, k)
            self.hot[sl] = rng.uniform(0.35, 0.75, k)
            self.size[sl] = rng.uniform(0.7, 1.7, k)
        elif kind == STREAK:
            self.tot[sl] = rng.uniform(0.75, 1.15, k)
            self.grav[sl] = rng.uniform(10, 40, k)
            self.drag[sl] = rng.uniform(0.05, 0.25, k)
            self.hot[sl] = 1.0
            self.size[sl] = rng.uniform(2.2, 3.4, k)
        else:                                              # soft tick on a beat
            self.tot[sl] = rng.uniform(0.26, 0.62, k)
            self.grav[sl] = rng.uniform(300, 560, k)
            self.drag[sl] = rng.uniform(1.6, 3.0, k)
            self.hot[sl] = rng.uniform(0.3, 0.75, k)
            self.size[sl] = rng.uniform(0.6, 1.4, k)
        self.life[sl] = self.tot[sl]

    def streak(self, y_frac: float, speed: float, down: float = 0.28):
        """A spark 'shuttle': one bright head that crosses the frame shedding embers."""
        rng = self.rng
        sl = self._alloc(1)
        if sl.stop <= sl.start:
            return
        d = 1.0 if rng.random() < 0.5 else -1.0
        self.kind[sl] = STREAK
        self.x[sl] = (self.w2 + 30) * (0 if d > 0 else 1)
        self.y[sl] = self.h2 * y_frac
        self.vx[sl] = d * speed
        self.vy[sl] = speed * down * rng.uniform(0.5, 1.0)
        self.tot[sl] = self.life[sl] = np.array([self.w2 / max(1.0, abs(speed)) + 0.25], np.float32)
        self.grav[sl] = 70.0
        self.drag[sl] = 0.06
        self.hot[sl] = 1.0
        self.size[sl] = 3.0
        self.popped[sl] = False

    def update(self, dt: float, wind: float = 0.0):
        al = self.life > 0
        if not al.any():
            return
        self.life[al] -= dt
        self.vy[al] += self.grav[al] * dt
        damp = np.exp(-self.drag[al] * dt)
        self.vx[al] *= damp
        self.vy[al] *= damp
        self.vx[al] += wind * dt
        self.x[al] += self.vx[al] * dt
        self.y[al] += self.vy[al] * dt
        self.life[self.life < 0] = 0
        # crackle: a dying cracker spark pops into a couple of tiny ones
        idx = np.where(al)[0]
        live = self.life[idx]
        cand = idx[(live < 0.22) & (~self.popped[idx]) &
                   (self.kind[idx] == CRACKER) & (self.rng.random(len(idx)) < 0.4)]
        for c in cand[:14]:
            self.popped[c] = True
            self.burst(self.x[c], self.y[c], 3, 150.0, CRACKER, spread=1.0, upward=0.4)

    # -- rendering -----------------------------------------------------------
    def render(self, t: float, gain: float, twinkle: float, trail: float,
               pal_col: np.ndarray) -> Image.Image:
        img = Image.new("RGB", (self.w2, self.h2), 0)
        d = ImageDraw.Draw(img)
        s = self.s
        # ambient embers
        for i in range(self.N):
            yy = (self.ay[i] - self.arise[i] * t) % self.h2
            xx = (self.ax[i] + self.asway[i] * math.sin(t * 0.6 + self.aph[i])) % self.w2
            b = (0.22 + 0.60 * (0.5 + 0.5 * math.sin(t * 2.1 + self.aph[i]))) * gain
            if b < 0.06:
                continue
            r = self.asz[i]
            v = int(min(255, 215 * b))
            d.ellipse([xx * s - r, yy * s - r, xx * s + r, yy * s + r],
                      fill=(v, int(v * 0.72), int(v * 0.34)))
        # particles: streak + halo + white-hot core
        al = np.where(self.life > 0)[0]
        if not len(al):
            return img
        frac = np.clip(self.life[al] / self.tot[al], 0, 1)
        hot = self.hot[al] * frac ** 1.25                  # white-hot only while new
        twk = 0.60 + 0.40 * np.sin(t * self.tw[al] * twinkle * 0.22 + self.ph[al])
        br = np.clip(frac ** 1.35 * gain * twk * 1.35, 0, 1.5)
        xs, ys = self.x[al], self.y[al]
        tl = trail * np.where(self.kind[al] == STREAK, 0.22,
                       np.where(self.kind[al] == CRACKER, 0.042, 0.030))
        vx0, vy0 = self.vx[al] * tl, self.vy[al] * tl
        W_, G_, E_ = WHITE / 255.0, GOLD / 255.0, EMBER / 255.0
        col = W_[None, :] * hot[:, None] + G_[None, :] * (1 - hot)[:, None]
        cool = (1 - hot)[:, None]
        col = col * (0.55 + 0.45 * frac[:, None]) + E_[None, :] * cool * 0.55
        for j in range(len(al)):
            b = br[j]
            if b < 0.05:
                continue
            x0, y0, x1, y1 = xs[j], ys[j], xs[j] - vx0[j], ys[j] - vy0[j]
            is_streak = self.kind[al[j]] == STREAK
            r = max(0.7, float(self.size[j]) * (0.45 + 0.7 * b)) * (0.55 if is_streak else 1.0)
            c = np.clip(col[j] * b, 0, 1)
            halo = tuple(int(255 * min(1.0, c[k] * 0.85)) for k in range(3))
            glow = tuple(int(255 * min(1.0, c[k] * 0.42)) for k in range(3))
            core = tuple(int(255 * min(1.0, 0.30 + c[k] * 0.95)) for k in range(3))
            d.line([x1, y1, x0, y0], fill=halo, width=max(1, int(r * (0.7 if is_streak else 0.85))))
            if not is_streak:
                d.ellipse([x0 - r * 2.1, y0 - r * 2.1, x0 + r * 2.1, y0 + r * 2.1], fill=glow)
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=halo)
            d.ellipse([x0 - r * 0.42, y0 - r * 0.42, x0 + r * 0.42, y0 + r * 0.42], fill=core)
        return img


# ---------------------------------------------------------------------- rings
class Rings:
    """Shockwave rings + flare streaks, drawn additively in low-res."""

    def __init__(self, W: int, H: int, scale: float = 0.5, max_rings: int = 9):
        self.W, self.H, self.s = W, H, scale
        self.w2, self.h2 = int(W * scale), int(H * scale)
        self.act: list[tuple[float, float, float, float, int, float]] = []
        self.max = max_rings

    def add(self, t: float, cx: float, cy: float, strength: float, kind: int, life: float):
        self.act.append((t, cx, cy, strength, kind, life))
        if len(self.act) > self.max:
            self.act = self.act[-self.max:]

    def render(self, t: float, pal_col: np.ndarray) -> Image.Image:
        keep = []
        img = Image.new("RGB", (self.w2, self.h2), 0)
        if not self.act:
            return img
        d = ImageDraw.Draw(img)
        for (t0, cx, cy, st, kind, life) in self.act:
            p = max(0.0, (t - t0) / life)
            if p >= 1.0:
                continue
            keep.append((t0, cx, cy, st, kind, life))
            e = (1 - min(1.0, p)) ** 1.7
            r = (0.03 + 0.97 * (p ** 0.62)) * 0.30 * self.h2 * (1.5 if kind else 1.0)
            a = e * st
            a = min(1.0, a * 1.15)
            if kind == 1:                       # bell: two soft halos, wide
                for f, wsc in ((1.0, 5), (0.72, 3)):
                    rr = r * f
                    col = tuple(int(min(255, 255 * a * c / 255)) for c in pal_col)
                    d.ellipse([cx - rr, cy - rr * 0.62, cx + rr, cy + rr * 0.62],
                              outline=col, width=wsc)
            else:                               # impact: hot edge + flare star
                hotmix = 0.35 * WHITE + 0.65 * np.asarray(pal_col, np.float32)
                col = tuple(int(min(255, 255 * a * c / 255)) for c in hotmix)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
                col2 = tuple(int(min(255, 200 * a * c / 255)) for c in pal_col)
                d.ellipse([cx - r * 0.88, cy - r * 0.88, cx + r * 0.88, cy + r * 0.88],
                          outline=col2, width=5)
                n = 6
                for k in range(n):
                    ang = k * math.pi / n + t0 * 1.7
                    ln = r * (0.62 + 0.22 * (k % 2))
                    d.line([cx, cy, cx + math.cos(ang) * ln, cy + math.sin(ang) * ln * 0.85],
                           fill=col2, width=3)
        self.act = keep
        return img


# ------------------------------------------------------------------ the FX rig
class FX:
    """Owns every reactive channel + effect layer for one render."""

    def __init__(self, events: dict, duration: float, fps: float, W: int, H: int,
                 pal: dict, preset: str = "cinematic", seed: int = 0,
                 gain: float = 1.0):
        self.p = dict(PRESETS[preset] if preset in PRESETS else PRESETS["cinematic"])
        for k in list(self.p):
            self.p[k] *= gain
        # soft-light gains saturate fast: apply the chapter gain gently
        self.p["flash"] = min(1.0, self.p["flash"] * (0.55 + 0.45 * min(1.4, gain)))
        self.p["bloom"] = min(1.3, self.p["bloom"] * (0.6 + 0.4 * min(1.4, gain)))
        self.gain = gain
        self.W, self.H, self.fps, self.duration = W, H, fps, duration
        self.pal = pal
        self.col = np.asarray(pal["acc"], np.float32)
        self.ex = Exciters(events, duration, fps)
        rng = np.random.default_rng(seed + 31)
        self.rng = rng
        n = self.ex.n
        t = self.ex.t
        hit = self.ex.hit
        # camera: damped multi-frequency shake + rotation + zoom punch
        kx = _springs(rng, t, hit, 11.3, self.p["shake"]) + \
            _springs(rng, t, hit, 5.1, self.p["shake"] * 0.55)
        ky = _springs(rng, t, hit, 8.7, self.p["shake"] * 0.8) + \
            _springs(rng, t, hit, 3.3, self.p["shake"] * 0.5)
        self.cam = np.zeros((n, 4), np.float32)
        self.cam[:, 0] = kx * rng.choice([-1.0, 1.0], n)
        self.cam[:, 1] = ky
        self.cam[:, 2] = _springs(rng, t, hit, 6.3, self.p["rot"])            # degrees
        self.cam[:, 3] = 1.0 + self.p["zoom"] * self.ex.ch[:, 1]              # punch scale
        # slow "breathing" so quiet passages still move
        self.breath = (0.5 + 0.5 * np.sin(2 * math.pi * t / 12.5)) * 1.0
        self.slow = np.maximum(self.ex.ch[:, 4], self.ex.ch[:, 7] * 0.8)
        self.air = self.ex.ch[:, 6]
        self.flash = np.clip((self.ex.ch[:, 1] * 1.0 + self.ex.ch[:, 2] * 0.85 +
                             hit * 0.25) * self.p["flash"], 0, 1.6)
        # fx buffers at half res
        self.sparks = Sparks(seed + 7, W, H, 0.5)
        self.rings = Rings(W, H, 0.5)
        self.n_fired: set[int] = set()
        self.smoke: list[tuple[float, float, float, float]] = []
        self.flashes: list[tuple[float, float, float, float]] = []
        self.petals = self._make_petals(seed + 13, n)

    def _make_petals(self, seed: int, n: int):
        rng = np.random.default_rng(seed)
        m = int(30 * self.p["petals"])
        if m <= 0:
            return None
        return dict(x=rng.uniform(0, self.W, m).astype(np.float32),
                    y=(rng.uniform(-self.H * 0.4, self.H, m)).astype(np.float32),
                    fall=rng.uniform(18, 52, m).astype(np.float32),
                    sway=rng.uniform(14, 55, m).astype(np.float32),
                    spin=rng.uniform(-1.7, 1.7, m).astype(np.float32),
                    ph=rng.uniform(0, 6.283, m).astype(np.float32),
                    sz=rng.uniform(9, 21, m).astype(np.float32))

    # ------------------------------------------------------------------ events
    def kick(self, i: int):
        """Fire whatever the music does exactly on this frame."""
        for j, (fi, tt) in enumerate(self.ex.fire_sam):
            if fi == i and (900000 + j) not in self.n_fired:
                self.n_fired.add(900000 + j)
                cx = self.W * (0.5 + self.rng.normal(0, 0.17))
                cy = self.H * (0.52 + self.rng.normal(0, 0.10))
                self.sparks.burst(cx * 0.5, cy * 0.5, int(24 * self.p["sparks"]),
                                  200 * 0.5, TICK, upward=0.5)
                self.rings.add(tt, cx * 0.5, cy * 0.5, 0.85, 0, 0.72)
                self.flashes.append((tt, cx * 0.5, cy * 0.5, 0.8))
                if self.rng.random() < min(0.85, self.p["crackers"] * 0.5):
                    self.cracker(cx, cy, scale=0.9, t0=tt)
        for j, (fi, tt) in enumerate(self.ex.fire_bell):
            if fi == i and (800000 + j) not in self.n_fired:
                self.n_fired.add(800000 + j)
                cx, cy = self.W * 0.5, self.H * 0.46
                self.sparks.burst(cx * 0.5, cy * 0.5, int(34 * self.p["sparks"]),
                                  150 * 0.5, BELL, upward=0.15, spread=1.0)
                self.rings.add(tt, cx * 0.5, cy * 0.5, 1.05, 1, 1.2)
                self.flashes.append((tt, cx * 0.5, cy * 0.5, 1.0))
                if self.rng.random() < min(0.8, self.p["crackers"] * 0.45):
                    self.cracker(self.W * self.rng.uniform(0.22, 0.78),
                                 self.H * self.rng.uniform(0.14, 0.4), 1.15, tt)
        for j, (fi, tt) in enumerate(self.ex.fire_sweep):
            if fi == i and (700000 + j) not in self.n_fired:
                self.n_fired.add(700000 + j)
                # a spark that shoots across the frame, shedding embers
                self.sparks.streak(self.rng.uniform(0.24, 0.7),
                                   self.rng.uniform(430, 640) * 0.5,
                                   self.rng.uniform(-0.18, 0.34))
        # ordinary beats: a light tick of sparks, probability from the accent
        for aa in self.ex.fire_beat[i][:4]:
            if self.rng.random() < 0.45 * min(1.0, aa) * min(1.6, self.p["sparks"]):
                self.sparks.burst(self.rng.uniform(0.18, 0.82) * self.W * 0.5,
                                  self.rng.uniform(0.5, 0.88) * self.H * 0.5,
                                  int(6 * self.p["sparks"]), 95 * 0.5, TICK, upward=0.9)

    def cracker(self, cx: float, cy: float, scale: float = 1.0, t0: float = 0.0):
        """A firecracker: white pop, radial streaks, falling sparks, smoke."""
        s, g = self.sparks, self.p["crackers"]
        s.burst(cx * 0.5, cy * 0.5, min(96, int(52 * g * scale)), 430 * 0.5 * scale,
                CRACKER, spread=1.0, upward=0.5)
        s.burst(cx * 0.5, cy * 0.5, int(14 * g * scale), 120 * 0.5, TICK, upward=0.95)
        self.rings.add(t0, cx * 0.5, cy * 0.5, 0.95, 0, 0.6)
        self.flashes.append((t0, cx * 0.5, cy * 0.5, 1.15 * g * scale))
        self.smoke.append((cx * 0.5, cy * 0.5, 0.0, scale))

    # ------------------------------------------------------------------ layers
    def layer(self, i: int, t: float, dt: float) -> Image.Image:
        """One half-res additive buffer with everything that flies around."""
        j = min(i, self.ex.n - 1)
        img = self.sparks.render(t, min(1.5, 0.4 + self.slow[j] * 1.2),
                                 self.p["twinkle"], self.p["trail"], self.col)
        d = ImageDraw.Draw(img)
        s = 0.5                                              # buffer scale
        # blast flashes: the hot core of an impact, gone in a fifth of a second
        if self.flashes:
            keep = []
            for (t0, fx_, fy, st) in self.flashes:
                q = max(0.0, (t - t0) / 0.24)
                if q >= 1.0:
                    continue
                keep.append((t0, fx_, fy, st))
                a = (1 - q) ** 2.4 * st
                r = (3 + 30 * (q ** 0.55)) * st
                core = tuple(int(min(255, 255 * a * c / 255 * 2.2)) for c in WHITE)
                d.ellipse([fx_ - r * 0.32, fy - r * 0.32, fx_ + r * 0.32, fy + r * 0.32],
                          fill=core)
                halo = tuple(int(min(255, 255 * a * c / 255 * 1.25)) for c in GOLD)
                d.ellipse([fx_ - r, fy - r, fx_ + r, fy + r], outline=halo, width=3)
                for k in range(4):
                    ang = k * math.pi / 4 + t0
                    ln = r * (1.5 + 0.9 * (k % 2))
                    d.line([fx_, fy, fx_ + math.cos(ang) * ln, fy + math.sin(ang) * ln],
                           fill=halo, width=2)
            self.flashes = keep
        # shuttles shed embers as they travel
        sk = self.sparks
        al = np.where((sk.life > 0) & (sk.kind == STREAK))[0]
        for c in al[:6]:
            if self.rng.random() < 0.85:
                sk.burst(sk.x[c], sk.y[c], 2, 105.0, TICK, spread=1.2, upward=0.35)
        # marigold petals (drawn as little diamond-pod shapes)
        p = self.petals
        if p:
            for k in range(len(p["x"])):
                y = (p["y"][k] + p["fall"][k] * t) % (self.H + 200) - 100
                x = (p["x"][k] + p["sway"][k] * math.sin(t * 0.5 + p["ph"][k])) % (self.W + 120) - 60
                ang = t * p["spin"][k] + p["ph"][k]
                r = float(p["sz"][k]) * s * 1.35
                a = 0.62 + 0.30 * math.sin(t * 1.3 + p["ph"][k])
                ca, sa = math.cos(ang), math.sin(ang)
                pts = [(x + (u * ca - v * sa) * r, y + (u * sa + v * ca) * r * 0.6)
                       for u, v in ((-1, 0), (0, -0.55), (1, 0), (0, 0.55))]
                col = tuple(int(255 * a * c) for c in np.array([1.0, 0.62, 0.16], np.float32))
                d.polygon(pts, fill=col)
        # smoke puffs from crackers
        if self.smoke:
            keep = []
            for (sx, sy, age, sc) in self.smoke:
                a = age + dt
                if a > 2.6:
                    continue
                keep.append((sx, sy, a, sc))
                q = a / 2.6
                r = (7 + 46 * q) * sc
                al = 24 * (1 - q) ** 1.7
                col = tuple(int(SMOKE[c] * al / 255 * 3) for c in range(3))
                d.ellipse([sx - r, sy - r * 0.8, sx + r, sy + r * 0.8], fill=col)
            self.smoke = keep
        # sparks, rings on top (all additive: max-composite in low-res)
        rings = self.rings.render(t, self.col)
        img = ImageChops.add(img, rings)
        self.sparks.update(dt, wind=8.0 * (self.ex.ch[j, 4] - 0.5))
        return img

    # ------------------------------------------------------------------ camera
    def camera(self, i: int) -> tuple[float, float, float, float]:
        j = min(i, self.ex.n - 1)
        return (float(self.cam[j, 0]), float(self.cam[j, 1]),
                float(self.cam[j, 2]), float(self.cam[j, 3]))

    def glow(self, i: int) -> dict:
        j = min(i, self.ex.n - 1)
        return dict(slow=float(self.slow[j]), air=float(self.air[j]),
                    flash=float(self.flash[j]), beat=float(self.ex.ch[j, 0]),
                    sam=float(self.ex.ch[j, 1]), bell=float(self.ex.ch[j, 2]),
                    sweep=float(self.ex.ch[j, 3]), breath=float(self.breath[j]))


def bloom(f: np.ndarray, gain: float, thresh: float = 118.0,
          blur: int = 3, keep: float = 0.62) -> np.ndarray:
    """Cheap real bloom: bright-pass at quarter res, blur, add back.

    Returns the *additive* glow layer (same size as f) so the caller can tint
    and clamp it - this is what makes sparks read as fire instead of dots.
    """
    if gain <= 0.01:
        return np.zeros_like(f)
    h, w = f.shape[:2]
    w4, h4 = max(2, w // 4), max(2, h // 4)
    bright = np.clip(f - thresh, 0, None)
    img = Image.fromarray(np.clip(bright * keep, 0, 255).astype(np.uint8)) \
        .resize((w4, h4), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(blur)).resize((w, h), Image.BILINEAR)
    return np.asarray(img, np.float32) * gain
