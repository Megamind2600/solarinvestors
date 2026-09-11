# Gita Wisdom Reels 🕉️

Turns all **701 shlokas** of the Bhagavad Gita (from
[gitawisdom](https://megamind2600.github.io/gitawisdom/)) into Instagram /
YouTube-style **vertical shorts (1080×1920, 30 fps)** — an animated *still*,
not a filmed video: a glowing Sanskrit card drifts over a cosmic backdrop
while **camera kicks, sparks, firecracker bursts, shockwave rings,
spark-shuttles, embers and light sweeps fire on the music** — measured from
the track, never on a fake grid.

The sound is **real meditative music** (open-licence tanpura drone, bansuri
alap, bowls, temple bells, ambient instrumental releases) fetched from sources
whose licence is verified per track — see *Soundtrack* below. A procedural
raga synth (`scripts/synth_music.py`) remains as the offline fallback.

## Repo layout

```
data/verses.json            701 verses: Sanskrit, IAST, English meaning (MIT gita/gita dataset)
assets/fonts/               Tiro Devanagari Sanskrit, merged-latin Marcellus/Mukta/Tiro-Italic (OFL)
music/meditative_music.json curation file: which open sources to mine, and what is rejected
music/library/              fetched real recordings + manifest.json licence ledger (gitignored)
music/beats/                per-track event maps derived from the actual audio
music/pads/                 the 45 s window each verse actually uses (cached, gitignored)
music/tracks|beats/         6 procedural raga tracks kept as the offline fallback
scripts/fetch_music.py      downloads + licence-filters + normalises the music pool
scripts/analyze_music.py    numpy onset/energy analysis -> beats, sweeps, bells, envelopes
scripts/library.py          verse -> (track, offset) mapping, pad builder, event slicer
scripts/textshape.py        HarfBuzz + FreeType text renderer (proper Devanagari shaping)
scripts/scene.py            palettes, cosmic backgrounds, the verse "card" layer
scripts/synth_music.py      tanpura / bansuri / tabla / bells synth (fallback only)
scripts/fx.py               exciters + sparks, crackers, rings, shuttles, embers, petals, bloom
scripts/render_video.py     frame engine: camera, layers, compositing, encode
scripts/batch.py            parallel batch renderer (resumable)
scripts/build_fonts.py      merges Fontsource latin + latin-ext subsets (OFL)
.github/workflows/          fetches music, renders all videos, publishes a Release
videos/                     local output (gitignored)
samples/                    a few example reels (committed)
```

## Soundtrack: real meditative music, still licence-clean

Self-generated synth beds are fine as a fallback, but they are not what this
project should sound like. So the pipeline prefers **real recordings**, pulled
from sources that publish explicit free licences:

| source | what it gives | licence handling |
|--------|---------------|------------------|
| Freesound | acoustic tanpura drones, bansuri alap, singing bowls, tingsha, rain/river/forest beds | CC0 only; originals via a free API token, public HQ previews otherwise |
| archive.org netlabels | whole CC0 / CC-BY instrumental ambient & raga releases | filtered on the item's own `licenseurl` field |
| Wikimedia Commons | encyclopedic PD/CC solo instrumental audio (sitar ragas, cello, harp) | read from each file's `extmetadata` |
| `music/incoming/` | anything you own | needs a sidecar `NAME.LICENSE.txt` |

`music/meditative_music.json` is the curation brain. It keeps only what is
**instrumental, smooth and meditative** (min 95 s long so every verse can get its
own window) and hard-rejects chanting/voice/recitation, anything labelled dark,
noisy, trap or industrial, and **any licence containing `-NC` or `-ND`** — Reels
are public and sometimes monetised, so non-commercial or no-derivatives files
are never fetched.

```bash
python scripts/fetch_music.py --target 240 --dry-run   # what would be fetched
python scripts/fetch_music.py --target 240              # download + normalise
python scripts/analyze_music.py --all                    # build the event maps
python scripts/library.py --report                       # pool / licence summary
python scripts/library.py --verse 11.12                  # what that shloka will use
```

What you get back: `music/library/manifest.json` (every track with title, artist,
page URL, licence, duration, sha1) and a generated `ATTRIBUTIONS.md` so CC-BY
credits are actually carried. 701 verses never need 701 tracks: each verse gets a
**deterministic window** into a track (golden-ratio offset), so 240 recordings
already yield 240 × different-excerpt variety and no two reels open alike.

With no library present the renderer silently falls back to the procedural raga
synth in `scripts/synth_music.py`, so CI and fresh clones still work offline.
`samples/2_47_fx.mp4` demonstrates the effect grammar and was rendered on that
fallback bed — run `fetch_music.py` and the same reel gets real recordings.

## Render one video locally

```bash
pip install -r requirements.txt
python scripts/render_video.py --id 2.47                  # 45 s default
python scripts/render_video.py --id 18.66 --duration 60 --out /tmp/18_66.mp4
python scripts/render_video.py --id 2.47 --sheet /tmp/sheet.jpg   # quick visual QA
```

Contact sheets sample the loudest frames, so they show the bursts; they cannot
show particle *evolution*. For motion, render 8 s and tile it:

```bash
python scripts/render_video.py --id 2.47 --duration 8 --fps 15 --out /tmp/t.mp4
ffmpeg -i /tmp/t.mp4 -vf "fps=15,scale=200:356,tile=10x6" -frames:v 1 /tmp/grid.jpg
```

## Render all 701 (recommended: GitHub Actions)

The workflow has a `tracks` input (how many music beds to fetch, cached between
runs) and an `fx` input (`meditative | cinematic | festive`). A git repo cannot
hold ~12 GB of MP4s, so the heavy lifting happens on GitHub's runners:

1. **Actions → "Render Gita shorts" → Run workflow**
   (defaults render indices 0–701 in 29 parallel jobs)
2. When done, videos appear as **artifacts** per chunk, and
   zipped per chapter in a **Release** ("Reels — batch N").

Or locally/on a server:

```bash
python scripts/fetch_music.py --target 240 && python scripts/analyze_music.py --all
python scripts/batch.py --range 0-701 --workers 3 --duration 45 --fx cinematic
```

Each video is deterministic (seeded by verse id + track), so reruns are
resumable — existing files are skipped.

## How the beat sync works

For fetched recordings nothing is assumed: `analyze_music.py` measures the file
and emits an event map — onsets of the spectral flux (with accent = how strong
the onset is), the deepest bass-weighted swells (the "sam" role), high-band
onsets that are still ringing a second later (bells), slow mid-band rises
(phrase starts), plus 4 Hz `energy` / `bass` / `air` / `flux` curves.

Two kinds of signal drive two kinds of motion, which is what makes a *meditative*
track still look alive:

| signal | visual |
|--------|--------|
| continuous envelopes | backdrop breathing, mandala pulse, ember drift, slow glow |
| onsets / swells / bells | camera shake + rotation, spark and cracker bursts, shockwave rings, spark-shuttle streaks, blast flash, halo ignition on the shloka |

For the procedural fallback, `synth_music.build_track()` returns the exact onsets
and accents of every bol (`dha ge na ti | na ka dhi na`), flute phrase and tingsha
bell *by construction*. `render_video.py` turns either map into exciters:

| audio event            | visual effect                                              |
|------------------------|------------------------------------------------------------|
| deep swell / sam        | flash + punch-zoom + damped camera shake (x/y + rotation) + shockwave ring + spark fountain |
| ordinary onsets         | spark ticks, glow swell, ember updraft gust                 |
| phrase start / sweep    | diagonal golden light sweep + spark-shuttle streak across the frame |
| bell / high shimmer     | big radial flash, double ring, hanging bell sparks, smoke    |
| every beat              | ember field brightens, background mandala breathes, card halo lifts |

Presets (`--fx`): `meditative` (shake ≤ 5 px, no crackers, petals + embers only),
`cinematic` (default), `festive` (full pyro). The reel's chapter also scales it —
Dhyāna Yoga gets ×0.70, the Viśvarūpa chapter gets ×1.35.

## Instagram / Shorts specs

1080×1920 @ 30 fps, H.264 High@4.2 + AAC 160 kbps, `+faststart`, 45 s by
default (Reels allow 3–90 s). Upload the MP4 straight from the release zip.

## Licenses

- Verse data: [gita/gita](https://github.com/gita/gita) (MIT), translation by Swami Adidevananda.
- Fonts: SIL Open Font License (Tiro Devanagari Sanskrit, Marcellus, Mukta — via Fontsource).
- Music: per-track licences are recorded in `music/library/manifest.json` and credited in
  `ATTRIBUTIONS.md` (generated). Only CC0 / Public Domain / CC-BY / CC-BY-SA audio is ever
  fetched — never `-NC`, never `-ND`. The procedural fallback beds are MIT, this repo.
- Video pipeline code: this repo (MIT).
