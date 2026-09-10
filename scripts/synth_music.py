"""Procedural Mahabharat-ambient music: tanpura drone, bansuri flute, tabla,
tingsha bells, convolution reverb. 100 % synthesized in numpy -> license-clean,
and because *we* generate it, the beat map is exact, which is what the video
engine syncs its flashes/sparks/vibrations to.

    python scripts/synth_music.py --track 0 --duration 45
        -> music/tracks/track_0_yaman.mp3  +  music/beats/track_0.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from common import MUSIC, TRACKS, ffmpeg_exe

SR = 44100


# ------------------------------------------------------------------ utilities
def _fft_shape_noise(n: int, centers, widths, rng) -> np.ndarray:
    """Gaussian-shaped band noise (for flute breath) in one FFT shot."""
    x = rng.normal(0.0, 1.0, n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mask = np.zeros_like(freqs)
    for c, w in zip(centers, widths):
        mask += np.exp(-0.5 * ((freqs - c) / w) ** 2)
    y = np.fft.irfft(X * mask, n)
    peak = np.abs(y).max() or 1.0
    return y / peak


def _hp_filter(x: np.ndarray, fc: float = 42.0) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= f ** 2 / (f ** 2 + fc ** 2)
    return np.fft.irfft(X, len(x))


def reverb(mono: np.ndarray, decay: float = 1.9, predelay_ms: float = 18.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, m = len(mono), int(decay * SR)
    ir = rng.normal(0, 1, m) * np.exp(-4.2 * np.arange(m) / m)
    ir[int(0.01 * SR):int(0.02 * SR)] *= 0.35
    ir /= np.abs(ir).sum() or 1.0
    size = 1 << (n + m - 1).bit_length()
    Y = np.fft.irfft(np.fft.rfft(mono, size) * np.fft.rfft(ir, size), size)[:n + m]
    out = np.zeros(len(mono), dtype=np.float64)
    out[:len(Y)] = Y[:len(mono)]
    d = int(predelay_ms * SR / 1000)
    return np.concatenate([np.zeros(d), out])[:len(mono)]


# ----------------------------------------------------------------- instruments
def tanpura_note(freq: float, dur: float, rng) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for h in range(1, 9):
        f = freq * h * (1 + rng.uniform(-1, 1) * 1.2e-3)
        if f > 9000:
            break
        tau = 3.4 / (h ** 1.15)
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.283)) * np.exp(-t / tau) / (h ** 1.55)
    nz = int(0.010 * SR)
    out[:nz] += 0.30 * rng.normal(0, 1, nz) * np.hanning(nz)
    return out


def tanpura_track(root: float, pa_semis: int, duration: float, rng) -> np.ndarray:
    """Continuous Sa-Pa pluck cycle: pattern per 4 plucks, ~1.5 s apart."""
    mix = np.zeros(int(duration * SR))
    cycle = [pa_semis, 0, 0, 0]
    pluck_dur = 6.0
    t = 0.4
    i = 0
    gap = 1.52
    while t < duration - 0.5:
        f = root * 2 ** (cycle[i % 4] / 12.0)
        frag = tanpura_note(f, pluck_dur, rng)
        start = int((t + rng.uniform(-0.03, 0.03)) * SR)
        end = min(len(mix), start + len(frag))
        mix[start:end] += frag[:end - start]
        t += gap * (1.0 if i % 4 else 1.06)
        i += 1
    mix /= (np.abs(mix).max() or 1.0)
    return mix * 0.30


def flute_voice(note_events, duration: float, rng) -> np.ndarray:
    """note_events: list of (t0, t1, semitone_from_Sa, amp). Renders with meend
    glides, delayed vibrato and breath noise into the shared buffer."""
    mix = np.zeros(int(duration * SR))
    for k, (t0, t1, semi, amp) in enumerate(note_events):
        n = int((t1 - t0) * SR)
        if n < 32:
            continue
        t = np.arange(n) / SR
        ftarget = 440.0 / 2 * 2 ** ((semi + 3) / 12.0)  # register around D4-ish (root mapped below)
        # portamento from previous note (meend)
        if k > 0 and t0 - note_events[k - 1][1] < 0.05:
            fprev = 440.0 / 2 * 2 ** ((note_events[k - 1][2] + 3) / 12.0)
            glide_t = min(0.22, (t1 - t0) * 0.5)
            g = np.clip(t / glide_t, 0, 1)
            g = g * g * (3 - 2 * g)
            f0 = fprev * (ftarget / fprev) ** g
        else:
            f0 = np.full(n, ftarget)
        vib = np.where(t > 0.30, 1.0, t / 0.30) * 0.006 * np.sin(2 * np.pi * 4.7 * t)
        f0 = f0 * (1 + vib)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        a = np.minimum(1, t / 0.09) * np.minimum(1, (t1 - t0 - t) / 0.13)
        tone = (np.sin(phase) + 0.14 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)) * a
        breath = _fft_shape_noise(n, [f0.mean(), 2 * f0.mean()], [0.5 * f0.mean(), 0.7 * f0.mean()], rng) * a
        seg = amp * (tone + 0.10 * breath * (0.7 + 0.3 * np.sin(2 * np.pi * 2.2 * t)))
        start = int(t0 * SR)
        seg = seg[:max(0, min(len(seg), len(mix) - start))]
        if len(seg):
            mix[start:start + len(seg)] += seg
    return mix


def flute_melody(raga, root, bpm, duration, rng):
    """Generate raga-walk phrases and return (note_events, phrase_starts)."""
    beat = 60.0 / bpm
    events, phrase_starts = [], []
    deg = 0            # index into extended raga scale (two octaves)
    scale = raga + [r + 12 for r in raga]
    t = 2.2
    while t < duration - 2.0:
        phrase_starts.append(t)
        n_notes = int(rng.integers(4, 10))
        for _ in range(n_notes):
            dur_beats = float(rng.choice([0.5, 0.75, 1.0, 1.0, 1.5, 2.0], p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.1]))
            step = int(rng.choice([-2, -1, -1, 0, 1, 1, 2], p=[0.12, 0.28, 0.2, 0.05, 0.2, 0.12, 0.03]))
            deg = int(np.clip(deg + step, max(0, deg - 5), min(len(scale) - 1, deg + 6)))
            # gravity to Sa / Pa at phrase ends
            if _ == n_notes - 1 and rng.random() < 0.6:
                anchors = [0, 4 % len(raga), len(raga), 7 % len(raga)]
                deg = min(anchors, key=lambda a: abs(a - deg))
            semi = scale[deg]
            amp = float(rng.uniform(0.55, 0.9)) * (1.15 if semi % 12 == 0 else 1.0)
            events.append((t, t + dur_beats * beat * 0.96, semi, amp))
            t += dur_beats * beat
        t += float(rng.uniform(1.0, 2.8)) * beat
    return events, phrase_starts


# ------------------------------------------------------------------ tabla bols
def _bol_ge(rng, damp=1.0):
    n = int(0.20 * SR); t = np.arange(n) / SR
    f0, f1 = 118 * damp, 62
    f = f1 + (f0 - f1) * np.exp(-t / 0.045)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.exp(-t / 0.11)
    return (np.sin(phase) + 0.35 * np.sin(2 * phase)) * env


def _bol_na(rng, pitch=1.0):
    n = int(0.09 * SR); t = np.arange(n) / SR
    env = np.exp(-t / 0.028)
    click = np.zeros(n); click[: int(0.002 * SR)] = rng.normal(0, 1, int(0.002 * SR))
    return (np.sin(2 * np.pi * 720 * pitch * t) * env * 0.8 +
            _fft_shape_noise(n, [3400], [900], rng) * np.exp(-t / 0.018) * 0.5 + click * 0.4)


def _bol_ka(rng):
    n = int(0.06 * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * 640 * t) * np.exp(-t / 0.02) * 0.6 + \
        _fft_shape_noise(n, [4200], [1200], rng) * np.exp(-t / 0.012) * 0.4


_BOLS = None

def _make_bols(rng):
    ge, na, ka = _bol_ge(rng), _bol_na(rng), _bol_na(rng, 1.06)
    def cat(*frags):
        n = sum(len(f) for f in frags)
        out = np.zeros(n); i = 0
        for f in frags:
            out[i:i + len(f)] += f; i += len(f)
        return out
    dha = np.zeros(max(len(ge), len(na))); dha[:len(ge)] += ge; dha[:len(na)] += na
    dhi = np.zeros(len(ge)); dhi[:len(ge)] += ge * 0.9; dhi[:len(ka)] += ka * 1.2
    tiri = cat(ka, ka, ka, ka)
    return {"dha": dha, "dhi": dhi, "ge": ge * 0.8, "na": na, "ti": na * 0.75, "ka": ka, "tiri": tiri}


def tabla_track(bpm: float, duration: float, rng):
    """Keherwa 8-beat cycle with seeded fills. Returns (mix, beats)."""
    global _BOLS
    bols = _BOLS or _make_bols(rng); _BOLS = bols
    beat = 60.0 / bpm
    prob = ["dha", "ge", "na", "ti", "na", "ka", "dhi", "na"]
    accs = [1.00, 0.55, 0.45, 0.35, 0.45, 0.35, 0.80, 0.45]
    mix = np.zeros(int(duration * SR))
    beats = []
    n_cycles = int(duration / (8 * beat)) + 2
    t = 1.9
    for cyc in range(n_cycles):
        fill = (cyc % 4 == 3)
        for b in range(8):
            bt = t + b * beat
            if bt > duration - 0.4:
                break
            sam = (b == 0)
            name = prob[b]
            if fill and b >= 5:                      # little flourish before next sam
                frag = bols["tiri"]; sub = beat / 4.0
                for s in range(4):
                    st = int((bt + s * sub) * SR)
                    seg = frag if s == 0 else bols["ka"]
                    seg = seg[:max(0, min(len(seg), len(mix) - st))]
                    mix[st:st + len(seg)] += seg * (0.8 if s else 1.0) * 0.5
                beats.append({"t": round(bt, 4), "acc": 0.5, "sam": False})
                continue
            frag = bols[name] * accs[b] * (1.1 if sam else 1.0)
            st = int(bt * SR)
            seg = frag[:max(0, min(len(frag), len(mix) - st))]
            mix[st:st + len(seg)] += seg
            beats.append({"t": round(bt, 4), "acc": accs[b], "sam": sam})
        t += 8 * beat
    mix /= (np.abs(mix).max() or 1.0)
    return mix * 0.62, beats


def tingsha(duration: float, rng):
    mix, times = np.zeros(int(duration * SR)), []
    for bt in [0.55] + [t for t in np.arange(12.5, duration - 3.0, 13.7)]:
        n = int(4.5 * SR); t = np.arange(n) / SR
        f = 2600 * rng.uniform(0.98, 1.02)
        seg = sum(np.sin(2 * np.pi * f * p * t) * np.exp(-t / (2.6 / p)) / (p ** 1.3)
                  for p in (1.0, 2.76, 5.40, 8.93))
        st = int(bt * SR)
        seg = seg[:max(0, min(len(seg), len(mix) - st))]
        mix[st:st + len(seg)] += seg * 0.16
        times.append(round(float(bt), 3))
    return mix, times


# ------------------------------------------------------------------- assembly
def build_track(spec: dict, duration: float):
    """Return (stereo float32 (2,N), beatmap dict)."""
    seed = spec["seed"] * 7919 + int(duration * 10)
    rng = np.random.default_rng(seed)
    root = spec["root"]

    tan = tanpura_track(root, spec["tanpura"], duration, rng)
    events, phrase_starts = flute_melody(spec["raga"], root, spec["bpm"], duration, rng)
    flu = flute_voice([(a, b, s, amp) for a, b, s, amp in events], duration, rng)
    flu *= 0.5 / (np.abs(flu).max() or 1.0)
    tab, beats = tabla_track(spec["bpm"], duration, rng)
    bell, bell_times = tingsha(duration, rng)

    dry = tan + flu + tab + bell
    wet = (reverb(flu, 2.1, seed=seed + 1) * 0.5 + reverb(bell, 2.6, seed=seed + 2) * 0.9 +
           reverb(tab, 1.4, seed=seed + 3) * 0.22 + reverb(tan, 1.8, seed=seed + 4) * 0.3)

    left = dry + wet * 0.55
    right = dry + np.roll(wet, int(0.011 * SR)) * 0.55
    stereo = np.stack([_hp_filter(left), _hp_filter(right)])
    stereo = np.tanh(stereo * 1.15) * 0.92
    # fades
    n_in, n_out = int(0.6 * SR), int(3.0 * SR)
    stereo[:, :n_in] *= np.linspace(0, 1, n_in)
    stereo[:, -n_out:] *= np.linspace(1, 0, n_out)
    stereo = stereo.astype(np.float32)

    beatmap = {
        "key": spec["key"], "bpm": spec["bpm"], "seed": seed,
        "beats": [b for b in beats if b["t"] < duration - 0.1],
        "sweeps": [round(s, 3) for s in phrase_starts],
        "bells": bell_times,
    }
    return stereo, beatmap


def write_wav(pcm: np.ndarray, path: str | Path):
    data = np.clip(pcm.T, -1, 1)
    data = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=None, help="track index; all if omitted")
    ap.add_argument("--duration", type=float, default=45)
    args = ap.parse_args()
    idxs = [args.track] if args.track is not None else range(len(TRACKS))
    (MUSIC / "tracks").mkdir(parents=True, exist_ok=True)
    (MUSIC / "beats").mkdir(parents=True, exist_ok=True)
    for i in idxs:
        spec = TRACKS[i]
        pcm, beatmap = build_track(spec, args.duration)
        mp3 = MUSIC / "tracks" / f"track_{i}_{spec['key']}.mp3"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            write_wav(pcm, tmp.name)
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp.name,
                            "-codec:a", "libmp3lame", "-b:a", "160k", str(mp3)], check=True)
        with open(MUSIC / "beats" / f"track_{i}.json", "w") as fh:
            json.dump(beatmap, fh)
        print(f"track {i} ({spec['key']}): {mp3.name}  peaks≈{np.abs(pcm).max():.2f}  beats={len(beatmap['beats'])}")


if __name__ == "__main__":
    main()
