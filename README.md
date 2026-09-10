# Gita Wisdom Reels 🕉️

Turns all **701 shlokas** of the Bhagavad Gita (from
[gitawisdom](https://megamind2600.github.io/gitawisdom/)) into Instagram /
YouTube-style **vertical shorts (1080×1920, 30 fps)** — an animated *still*,
not a filmed video: a glowing Sanskrit card drifts over a cosmic backdrop
while **flashes, sparks, shockwave rings, screen shake and light sweeps fire
exactly on the beats** of ambient Mahabharata-style music (tanpura drone,
bansuri flute, tabla keherwa, tingsha bells).

Every sound is **synthesized by code in this repo** (see
`scripts/synth_music.py`), so the soundtrack is license-clean, and the beat
map the visuals sync to is exact — not post-detected.

## Repo layout

```
data/verses.json          701 verses: Sanskrit, IAST, English meaning (MIT gita/gita dataset)
assets/fonts/             Tiro Devanagari Sanskrit, merged-latin Marcellus/Mukta/Tiro-Italic (OFL)
music/tracks|beats/       6 procedurally generated raga tracks + exact beat maps
scripts/textshape.py      HarfBuzz + FreeType text renderer (proper Devanagari shaping)
scripts/scene.py          palettes, cosmic backgrounds, the verse "card" layer
scripts/synth_music.py    tanpura / bansuri / tabla / bells synth -> stereo + beat map
scripts/render_video.py   frame engine: Ken Burns, embers, rings, flash, shake, sweep
scripts/batch.py          parallel batch renderer (resumable)
scripts/build_fonts.py    merges Fontsource latin + latin-ext subsets (OFL)
.github/workflows/        renders ALL videos on GitHub runners and publishes a Release
videos/                   local output (gitignored)
samples/                  a few example reels (committed)
```

## Render one video locally

```bash
pip install -r requirements.txt
python scripts/render_video.py --id 2.47                  # 45 s default
python scripts/render_video.py --id 18.66 --duration 60 --out /tmp/18_66.mp4
python scripts/render_video.py --id 2.47 --sheet /tmp/sheet.jpg   # quick visual QA
```

## Render all 701 (recommended: GitHub Actions)

A git repo cannot hold ~12 GB of MP4s, so the heavy lifting happens on
GitHub's runners:

1. **Actions → "Render Gita shorts" → Run workflow**
   (defaults render indices 0–701 in 29 parallel jobs)
2. When done, videos appear as **artifacts** per chunk, and
   zipped per chapter in a **Release** ("Reels — batch N").

Or locally/on a server:

```bash
python scripts/batch.py --range 0-701 --workers 3 --duration 45
```

Each video is deterministic (seeded by verse id + track), so reruns are
resumable — existing files are skipped.

## How the beat sync works

`synth_music.build_track()` composes a raga melody over a tabla cycle and
returns, alongside the stereo PCM, the exact onsets and accent strength of
every bol (`dha ge na ti | na ka dhi na`), every flute-phrase start, and every
tingsha bell. `render_video.py` turns those into exciters:

| audio event            | visual effect                                              |
|------------------------|------------------------------------------------------------|
| sam (strong `dha`)     | flash + 2 % punch-zoom + screen shake + shockwave ring     |
| medium bols (`dhi`…)   | spark burst + glow swell                                   |
| flute phrase start     | diagonal golden light sweep across the shloka              |
| tingsha bell           | big radial flash, ring + ember fountain, longer decay      |
| every beat             | ember field brightens, background mandala breathes         |

## Instagram / Shorts specs

1080×1920 @ 30 fps, H.264 High@4.2 + AAC 160 kbps, `+faststart`, 45 s by
default (Reels allow 3–90 s). Upload the MP4 straight from the release zip.

## Licenses

- Verse data: [gita/gita](https://github.com/gita/gita) (MIT), translation by Swami Adidevananda.
- Fonts: SIL Open Font License (Tiro Devanagari Sanskrit, Marcellus, Mukta — via Fontsource).
- Music & video pipeline code: this repo (MIT); every soundtrack is generated at render time.
