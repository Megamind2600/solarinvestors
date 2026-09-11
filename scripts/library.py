"""Bridge between the music library and the reel renderer.

A reel never plays a whole 4-8 minute recording: every one of the 701 shlokas
gets *its own window* into a track, chosen deterministically, so
  * no two reels start at the same second of the same take, and
  * a pool of ~120 recordings already yields 701 distinct-sounding beds.

    python scripts/library.py --report
    python scripts/library.py --verse 2.47
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MUSIC, load_verses

PADS = MUSIC / "pads"
BEATS = MUSIC / "beats"
MANIFEST = MUSIC / "library" / "manifest.json"
PHI = 0.6180339887498949

# What each chapter wants to *feel* like: preferred tags (mood from
# music/meditative_music.json recipes) and how hard the visuals may hit.
CHAPTER_MOOD = {
    1:  dict(tags=["drone", "tanpura", "grave", "still"], gain=0.80, label="grief"),
    2:  dict(tags=["flute", "bansuri", "calm", "timeless"], gain=0.85, label="insight"),
    3:  dict(tags=["flute", "drone", "flowing", "raga"], gain=0.95, label="action"),
    4:  dict(tags=["drone", "sitar", "timeless", "still"], gain=0.85, label="knowledge"),
    5:  dict(tags=["harp", "cello", "still", "grounding"], gain=0.80, label="renunciation"),
    6:  dict(tags=["bowls", "drone", "still", "meditation"], gain=0.70, label="meditation"),
    7:  dict(tags=["drone", "sitar", "sustained", "grave"], gain=0.85, label="wisdom"),
    8:  dict(tags=["drone", "ambient", "timeless", "still"], gain=0.80, label="the imperishable"),
    9:  dict(tags=["flute", "krishna", "bansuri", "longing"], gain=1.00, label="royal secret"),
    10: dict(tags=["bells", "sitar", "flowing", "grand"], gain=1.05, label="divine glories"),
    11: dict(tags=["bells", "gong", "shimmer", "grand"], gain=1.35, label="the cosmic form"),
    12: dict(tags=["flute", "krishna", "calm", "longing"], gain=0.90, label="devotion"),
    13: dict(tags=["drone", "field", "still", "grounding"], gain=0.80, label="field & knower"),
    14: dict(tags=["drone", "santoor", "timeless", "still"], gain=0.85, label="the three gunas"),
    15: dict(tags=["harp", "forest", "still", "peaceful"], gain=0.85, label="the supreme person"),
    16: dict(tags=["drone", "grave", "temple", "bells"], gain=1.00, label="divine & demonic"),
    17: dict(tags=["temple", "bells", "sustained", "flowing"], gain=0.95, label="three faiths"),
    18: dict(tags=["flute", "bowls", "still", "moksha"], gain=1.10, label="release"),
}
DEFAULT_MOOD = dict(tags=["drone", "still", "calm"], gain=0.9)


# ------------------------------------------------------------------ pool access
def load_pool(analyzed_only: bool = False) -> list[dict]:
    """Library records whose audio is actually on disk (+ absolute paths)."""
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, encoding="utf-8") as fh:
        recs = json.load(fh)
    out = []
    for r in recs:
        wav = MUSIC / r["wav"] if not Path(r["wav"]).is_absolute() else Path(r["wav"])
        if not wav.exists():
            continue
        r = dict(r)
        r["wav_abs"] = str(wav)
        r["map_abs"] = str(BEATS / f"lib_{r['id']}.json")
        if analyzed_only and not Path(r["map_abs"]).exists():
            continue
        out.append(r)
    return out


def ensure_map(rec: dict) -> Path | None:
    """Path to the event map, analysing the track first if we have to."""
    mp = Path(rec["map_abs"])
    if mp.exists():
        return mp
    try:
        from analyze_music import analyze_wav
    except Exception as e:                                        # noqa: BLE001
        print("  analysis module unavailable:", e)
        return None
    t0 = __import__("time").time()
    m = analyze_wav(Path(rec["wav_abs"]), mp)
    if m:
        print(f"  analysed {rec['id']} ({m.get('events_per_min')} events/min, "
              f"{__import__('time').time()-t0:.0f}s)")
        return mp
    mp.unlink(missing_ok=True)
    return None


# ------------------------------------------------------- verse -> track choice
def _idx_of(verse_id: str) -> int:
    return {v["id"]: i for i, v in enumerate(load_verses())}[verse_id]


def ranked_pool(chapter: int, pool: list[dict]) -> list[dict]:
    """Sort the pool so the tracks that suit a chapter float to the top."""
    want = CHAPTER_MOOD.get(int(chapter), DEFAULT_MOOD)["tags"]
    def sc(r: dict) -> float:
        blob = " ".join([r.get("mood", ""), " ".join(r.get("tags", [])),
                         r.get("title", "")]).lower()
        hit = sum(2.0 for w in want if w.lower() in blob)
        return hit + float(r.get("score", 0)) * 0.5 + min(3.0, (r.get("duration") or 120) / 150)
    return sorted(pool, key=sc, reverse=True)


def pick(verse_id: str, clip: float, pool: list[dict] | None = None,
         top_k: int = 40) -> tuple[dict, float, dict] | tuple[None, None, None]:
    """Deterministic (track, start-offset) for a verse.

    Round-robin over the chapter's best matches keeps every track in use, while
    a golden-ratio offset means repeated tracks still never sound the same.
    """
    pool = pool if pool is not None else load_pool()
    if not pool:
        return None, None, None          # empty library -> caller falls back to synth
    verses = load_verses()
    v = [x for x in verses if x["id"] == verse_id]
    ch = int(v[0]["chapter"]) if v else 1
    idx = {x["id"]: i for i, x in enumerate(verses)}.get(verse_id, 0)
    cand = ranked_pool(ch, pool)[: max(top_k, min(len(pool), top_k * 2))]
    rec = cand[idx % len(cand)]
    dur = float(rec.get("duration") or 0) or _probe_dur(Path(rec["wav_abs"]))
    off = offset_for(idx, dur, clip)
    return rec, off, dict(idx=idx, track_dur=round(dur, 2), chapter=ch,
                          mood=CHAPTER_MOOD.get(ch, DEFAULT_MOOD))


def _probe_dur(path: Path) -> float:
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:                                             # noqa: BLE001
        return 0.0


def offset_for(idx: int, track_dur: float, clip: float, tail: float = 2.5) -> float:
    """Start second inside a track for clip-length audio: golden-ratio spread."""
    room = max(0.0, track_dur - clip - tail)
    frac = (idx * PHI) % 1.0
    return round(0.8 + frac * room, 3)


# --------------------------------------------------------------------- the pad
def build_pad(rec: dict, offset: float, clip: float, fade_in: float = 2.2,
              fade_out: float = 3.2, target_db: int = -17,
              keep: bool | None = None) -> Path:
    """Trim + fades + loudness match.

    Pads are usually consumed once and thrown away (701 cached WAVs would be
    ~5 GB), so they land in the temp dir unless GITA_KEEP_PADS=1 or keep=True.
    """
    import os
    import tempfile
    from common import ffmpeg_exe
    if keep is None:
        keep = os.environ.get("GITA_KEEP_PADS") == "1"
    root = PADS if keep else Path(tempfile.gettempdir()) / "gita_pads"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{rec['id']}_{int(offset*1000)}_{clip:g}s.wav"
    if out.exists() and out.stat().st_size > 100_000:
        return out
    dur = _probe_dur(Path(rec["wav_abs"]))
    start = min(max(0.0, offset), max(0.0, dur - clip)) if dur else max(0.0, offset)
    st_out = max(0.1, clip - fade_out)
    af = (f"afade=t=in:st=0:d={fade_in},afade=t=out:st={st_out:.2f}:d={fade_out},"
          f"loudnorm=I={target_db}:TP=-1.8:LRA=11")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{clip + 0.35:.3f}", "-i", rec["wav_abs"],
           "-af", af, "-ac", "2", "-ar", "44100", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


# --------------------------------------------------------------- event slicing
def load_map(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def slice_events(m: dict, offset: float, clip: float) -> dict:
    """Re-time a whole-track map onto the pad we actually use (t -= offset)."""
    if m.get("dur"):                       # keep in step with build_pad's clamp
        offset = min(offset, max(0.0, m["dur"] - clip))
    lo, hi = offset - 0.05, offset + clip + 0.35
    beats = [dict(b, t=round(b["t"] - offset, 4)) for b in m.get("beats", [])
             if lo <= b["t"] <= hi]
    env = m.get("env", {})
    dt = env.get("dt", 0.25)
    i0, i1 = int(offset / dt), int((offset + clip + 0.35) / dt) + 1
    out = {"beats": beats,
           "sweeps": [round(t - offset, 3) for t in m.get("sweeps", []) if lo <= t <= hi],
           "bells": [round(t - offset, 3) for t in m.get("bells", []) if lo <= t <= hi],
           "env": {k: env.get(k, [])[i0:i1] for k in ("energy", "bass", "air", "flux")},
           "env_dt": dt, "key": m.get("key", "field"), "bpm": m.get("bpm"),
           "offset": round(offset, 3), "events_per_min": m.get("events_per_min")}
    if not out["beats"]:                    # ultra-quiet window: give it a soft pulse
        pulse = m.get("pulse_sec") or 4.0
        out["beats"] = [{"t": round(t, 4), "acc": 0.35, "sam": i % 4 == 0}
                        for i, t in enumerate(np.arange(1.0, clip, pulse))]
    return out


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verse", default=None, help="show what a verse would use")
    args = ap.parse_args()
    pool = load_pool()
    if args.verse:
        rec, off, meta = pick(args.verse, 45.0, pool)
        if not rec:
            sys.exit("no library audio - run scripts/fetch_music.py first")
        print(f"verse {args.verse} (ch.{meta['chapter']}, #{meta['idx']}) -> {rec['id']}  "
              f"[{rec.get('license')}]  window {off:.1f}s of {meta['track_dur']:.0f}s\n"
              f"   “{rec.get('title')}” by {rec.get('artist')}")
        return
    n_an = sum(1 for r in pool if Path(r["map_abs"]).exists())
    tot = sum(r.get("duration") or 0 for r in pool)
    print(f"library: {len(pool)} tracks ({tot/3600:.1f} h audio), {n_an} analysed")
    by_lic: dict[str, int] = {}
    for r in pool:
        by_lic[r.get("license", "?")] = by_lic.get(r.get("license", "?"), 0) + 1
    for k, v in sorted(by_lic.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<16} {v}")
    if pool:
        best = ranked_pool(11, pool)[:5]
        print("top-5 for ch.11 (Vishvarupa):")
        for b in best:
            print(f"   {b['id']:<32} {b.get('duration',0):6.0f}s  {b.get('title','')[:44]}")


if __name__ == "__main__":
    main()
