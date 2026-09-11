"""Turn a real recording into the event map the render engine animates from.

The visuals are only convincing if they move *with* the music, so instead of a
fake 8-beat tabla grid we measure the actual audio:

  beats    onset peaks of the spectral flux  -> accent = how strong the onset is
  sam      the strongest, bass-weighted ones (spaced apart) -> flash + shake + ring
  sweeps   slow rises of the mid band (a phrase beginning)  -> light sweep
  bells    high-band onsets that still ring 1 s later       -> big radial burst
  env      4 Hz curves of energy / bass / air / flux        -> continuous breathing

That last one matters most for meditative music: there is no steady pulse, so
the frame gets modulated by the *swell* of the drone, and only punctuated by
events.  Everything is numpy - no librosa, no model, no internet.

    python scripts/analyze_music.py --all
    python scripts/analyze_music.py --id fs153262
    python scripts/analyze_music.py --file music/library/audio/fs153262.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MUSIC

SR_EXPECT = 44100
NFFT = 2048
HOP = 512                      # ~86 Hz frame rate
ENV_DT = 0.25                  # envelope grid: 4 Hz
BLOCK = 768                    # frames per FFT batch


# ----------------------------------------------------------------------- read
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n_ch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    assert sw == 2, f"{path}: expected 16-bit PCM, got {sw*8}-bit"
    if sr != SR_EXPECT:
        raise ValueError(f"{path}: {sr} Hz - run scripts/fetch_music.py first "
                         f"(the library is normalised to {SR_EXPECT} Hz so event "
                         f"times are exact)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, n_ch)
    if n_ch == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x[: int(np.floor(len(x) / sr * SR_EXPECT))]), SR_EXPECT


def analyze_frames(mono: np.ndarray):
    """One streaming pass -> per-frame spectral flux + band energies (dB)."""
    n_fr = 1 + max(0, (len(mono) - NFFT) // HOP)
    win = np.hanning(NFFT).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR_EXPECT)
    bands = {"bass": (25, 160), "lowmid": (160, 620), "mid": (620, 2400),
             "air": (2400, 9000), "top": (9000, 18000)}
    masks = {k: (freqs >= a) & (freqs < b) for k, (a, b) in bands.items()}
    flux = np.zeros(n_fr, np.float32)
    band = {k: np.zeros(n_fr, np.float32) for k in bands}
    rms = np.zeros(n_fr, np.float32)
    prev = None
    for s in range(0, n_fr, BLOCK):
        e = min(n_fr, s + BLOCK)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(NFFT)[None, :]
        blk = mono[idx] * win
        rms[s:e] = np.sqrt((blk ** 2).mean(axis=1)) + 1e-9
        M = np.abs(np.fft.rfft(blk, axis=1)).astype(np.float32)
        M = np.log10(M + 1e-5)
        for k, m in masks.items():
            band[k][s:e] = M[:, m].mean(axis=1)
        if prev is None:
            prev = M
        df = np.maximum(M - prev, 0.0)
        flux[s:e] = df.sum(axis=1)
        prev = M[-1]
    return flux, band, rms, n_fr


def smooth(x: np.ndarray, n: float) -> np.ndarray:
    """Centred moving average over ~n frames (cheap, causal-free)."""
    k = max(1, int(round(n)))
    if k < 2:
        return x.copy()
    kern = np.hanning(2 * k + 1).astype(np.float32)
    kern /= kern.sum()
    xp = np.concatenate([np.full(k, x[0]), x, np.full(k, x[-1])])
    return np.convolve(xp, kern, mode="same")[k:-k]


def peaks(x: np.ndarray, thresh: np.ndarray | float, gap_fr: float) -> np.ndarray:
    """Local maxima above `thresh`, greedy, at least gap_fr frames apart."""
    hi = np.where(x[1:-1] >= x[:-2])[0] + 1
    hi = hi[x[hi] > (thresh[hi] if np.ndim(thresh) else thresh)]
    order = hi[np.argsort(-x[hi])]
    chosen: list[int] = []
    for i in order:
        if all(abs(i - c) >= gap_fr for c in chosen):
            chosen.append(int(i))
        if len(chosen) > 8000:
            break
    return np.array(sorted(chosen), int)


def qnorm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 3), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def envelope_grid(sig: np.ndarray, n_fr: int, sr_frames: float) -> np.ndarray:
    """Resample an analysis-rate curve onto the ENV_DT grid (0 when past end)."""
    n_env = int(round(n_fr / sr_frames / ENV_DT)) + 1
    out = np.zeros(n_env, np.float32)
    idx = (np.arange(n_fr) / sr_frames / ENV_DT).astype(int)
    np.add.at(out, idx, sig)
    cnt = np.zeros(n_env, np.float32)
    np.add.at(cnt, idx, 1.0)
    out /= np.maximum(cnt, 1)
    return out


# ------------------------------------------------------------------ the mapper
def build_map(mono: np.ndarray, meta: dict | None = None) -> dict:
    sr_frames = SR_EXPECT / HOP
    t_of = lambda fr: float(fr) * HOP / SR_EXPECT          # noqa: E731
    flux, band, rms, n_fr = analyze_frames(mono)

    flux_s = smooth(flux, 0.10 * sr_frames)
    bass = band["bass"] - band["bass"].mean()
    mid = band["mid"] - band["mid"].mean()
    air = band["air"] + band["top"]

    # --- onsets -------------------------------------------------------------
    local = smooth(flux_s, 1.4 * sr_frames)
    mad = smooth(np.abs(flux_s - local), 1.4 * sr_frames) + 1e-6
    thr = local + 1.35 * 1.4826 * mad
    ons = peaks(flux_s, thr, 0.30 * sr_frames)
    if len(ons) == 0:                                    # very ambient: fall back
        thr = np.full(n_fr, np.percentile(flux_s, 92))
        ons = peaks(flux_s, thr, 0.45 * sr_frames)

    strength = (flux_s[ons] - (local[ons] + 1.0 * mad[ons])) / (mad[ons] + 1e-6)
    acc = np.clip((strength - np.percentile(strength, 25)) /
                  (np.percentile(strength, 97) - np.percentile(strength, 25) + 1e-9), 0.12, 1.0)

    # "sam" = the deep, important moments (bass + loud + far apart)
    deep = 0.55 * flux_s[ons] / (flux_s[ons].max() + 1e-9) + 0.45 * (bass[ons] /
                                                                      (np.abs(bass).max() + 1e-9))
    order = np.argsort(-deep)
    sams: list[int] = []
    for j in order:
        if all(abs(ons[j] - ons[c]) >= 2.6 * sr_frames for c in sams):
            sams.append(int(j))
        if len(sams) >= 400:
            break
    sam_set = set(sams)
    beats = [{"t": round(t_of(int(f)), 4), "acc": round(float(a), 3),
              "sam": bool(int(i) in sam_set)}
             for i, (f, a) in enumerate(zip(ons, acc))]

    # --- bell-like: bright onset that is still audible a second later -------
    bell_idx = []
    for i, f in enumerate(ons):
        f2 = min(n_fr - 1, f + int(1.0 * sr_frames))
        if air[f] > np.percentile(air, 70) and air[f2] > 0.80 * air[f] and acc[i] > 0.35:
            if all(abs(f - g) >= 3.2 * sr_frames for g in bell_idx):
                bell_idx.append(int(f))
    bells = [round(t_of(f), 3) for f in bell_idx]

    # --- sweeps: slow mid-band rises = a phrase starting -------------------
    swell = smooth(mid, 3.0 * sr_frames)
    sw_thr = np.percentile(swell, 74)
    sw = peaks(swell, sw_thr, 5.5 * sr_frames)
    sweeps = [round(t_of(f), 3) for f in sw]

    def curve(sig: np.ndarray, sm: float) -> list[int]:
        return [int(round(v * 255)) for v in
                envelope_grid(qnorm(smooth(sig, sm * sr_frames)), n_fr, sr_frames)]

    env = {"dt": ENV_DT, "energy": curve(rms, 0.45), "bass": curve(band["bass"], 0.60),
           "air": curve(band["air"], 0.60), "flux": curve(flux_s, 0.25)}

    ioi = np.diff(ons) / sr_frames if len(ons) > 1 else np.array([1.0])
    out = {"sr": SR_EXPECT, "dur": round(len(mono) / SR_EXPECT, 3),
           "beats": beats, "sweeps": sweeps, "bells": bells, "env": env,
           "pulse_sec": round(float(np.median(ioi)), 3),
           "events_per_min": round(len(beats) / max(1e-6, len(mono) / SR_EXPECT / 60), 1),
           "analyzed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.update(meta or {})
    return out


def analyze_wav(wav: Path, out_json: Path, force: bool = False) -> dict | None:
    if out_json.exists() and not force:
        with open(out_json, encoding="utf-8") as fh:
            return json.load(fh)
    t0 = time.time()
    mono, _ = read_wav(wav)
    m = build_map(mono.mean(axis=1))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, separators=(",", ":"))
    os.replace(tmp, out_json)          # atomic: safe with 3 workers per CI job
    m["_secs"] = round(time.time() - t0, 1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="single wav to analyse")
    ap.add_argument("--id", default=None, help="library track id from manifest.json")
    ap.add_argument("--all", action="store_true", help="every track in the library")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    beats_dir = MUSIC / "beats"
    if args.file:
        wav = Path(args.file)
        m = analyze_wav(wav, beats_dir / f"lib_{wav.stem}.json", args.force)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from library import load_pool
        pool = load_pool()
        if args.id:
            pool = [r for r in pool if r["id"] == args.id]
        if not pool:
            sys.exit("library empty - run  python scripts/fetch_music.py --target 240")
        m = None
        for r in pool:
            from common import MUSIC as _M
            wav = _M / r["wav"]
            t0 = time.time()
            mm = analyze_wav(wav, beats_dir / f"lib_{r['id']}.json", args.force)
            if mm is not None:
                print(f"  {r['id']:<34} {r.get('duration',0):7.1f}s  "
                      f"beats/min={mm.get('events_per_min','?')} "
                      f"bells={len(mm.get('bells',[]))}  ({time.time()-t0:4.1f}s)", flush=True)
                m = mm
        print(f"analysed {len(pool)} tracks -> {beats_dir}/lib_*.json")
    if m:
        print(json.dumps({k: v for k, v in m.items() if k != "env"}, indent=1)[:900])


if __name__ == "__main__":
    main()
