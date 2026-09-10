"""Shared constants, data loading and ffmpeg resolution for the Gita Shorts pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "verses.json"
FONTS = ROOT / "assets" / "fonts"
MUSIC = ROOT / "music"
VIDEOS = ROOT / "videos"

# --- Instagram Reels / YouTube Shorts canvas --------------------------------
W, H = 1080, 1920          # 9:16 vertical full-HD
FPS = 30
DEFAULT_DURATION = 45      # seconds; overridden by CLI / config

FONT_SANSKRIT = FONTS / "TiroDevSanskrit-Regular.ttf"
FONT_SANSKRIT_ITALIC = FONTS / "TiroLatin-Italic.ttf"  # Latin italic used for IAST
FONT_LATIN_HEAD = FONTS / "Marcellus-Regular.ttf"
FONT_LATIN_BODY = FONTS / "Mukta-Regular.ttf"
FONT_LATIN_BODY_SB = FONTS / "Mukta-SemiBold.ttf"
FONT_DEV_BODY = FONTS / "Mukta-Devanagari.ttf"
FONT_DISPLAY = FONTS / "RozhaOne-Regular.ttf"

CHAPTER_NAMES = {
    1: "Arjuna Viṣāda Yoga", 2: "Sāṅkhya Yoga", 3: "Karma Yoga",
    4: "Jñāna Karma Sannyāsa Yoga", 5: "Karma Sannyāsa Yoga", 6: "Dhyāna Yoga",
    7: "Jñāna Vijñāna Yoga", 8: "Akṣara Brahma Yoga", 9: "Rāja Vidyā Rāja Guhya Yoga",
    10: "Vibhūti Yoga", 11: "Viśvarūpa Darśana Yoga", 12: "Bhakti Yoga",
    13: "Kṣetra-Kṣetrajña Vibhāga Yoga", 14: "Guṇa Traya Vibhāga Yoga",
    15: "Puruṣottama Yoga", 16: "Daivāsura Sampad Vibhāga Yoga",
    17: "Śraddhā Traya Vibhāga Yoga", 18: "Mokṣa Sannyāsa Yoga",
}

# --- Music track specs -------------------------------------------------------
# Each track is a deterministic raga-based synth; verse v uses map_track(v).
TRACKS = [
    dict(key="yaman",      raga=[0, 2, 4, 6, 7, 9, 11], bpm=72, root=138.59, seed=1101, tanpura=7),
    dict(key="bhairavi",   raga=[0, 1, 3, 5, 7, 8, 10], bpm=66, root=130.81, seed=2202, tanpura=7),
    dict(key="kafi",       raga=[0, 2, 3, 5, 7, 9, 10], bpm=76, root=146.83, seed=3303, tanpura=7),
    dict(key="bhimpalasi", raga=[0, 3, 5, 7, 10],       bpm=70, root=138.59, seed=4404, tanpura=5),
    dict(key="khamaj",     raga=[0, 2, 4, 5, 7, 9, 10], bpm=68, root=130.81, seed=5505, tanpura=7),
    dict(key="durga",      raga=[0, 2, 5, 7, 9],        bpm=74, root=146.83, seed=6606, tanpura=5),
]


def map_track(verse_id: str) -> int:
    """Deterministic verse -> music-track assignment (round-robin over sorted order)."""
    verses = load_verses()
    index = {v["id"]: i for i, v in enumerate(verses)}
    return index[verse_id] % len(TRACKS)


_verses_cache = None


def load_verses() -> list[dict]:
    global _verses_cache
    if _verses_cache is None:
        with open(DATA, encoding="utf-8") as fh:
            _verses_cache = json.load(fh)
    return _verses_cache


def find_verse(verse_id: str) -> dict:
    for v in load_verses():
        if v["id"] == verse_id:
            return v
    raise SystemExit(f"verse {verse_id!r} not found in {DATA}")


def ffmpeg_exe() -> str:
    """Static ffmpeg from imageio-ffmpeg if present, else system ffmpeg (GitHub runners)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return os.environ.get("FFMPEG", "ffmpeg")
