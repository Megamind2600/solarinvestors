"""Merge Fontsource latin + latin-ext subset woff2s into complete TTFs.

Fontsource splits webfonts into unicode-range subsets; IAST transliterations
(ā ī ū ṛ ṃ ḥ ś ṣ ṅ ...) live in latin-ext while ASCII lives in latin, so we
merge the pair back into one full-coverage file. Source packages live on npm
(@fontsource/*, SIL OFL licensed); run this only if the assets are missing.

    python scripts/build_fonts.py            # writes assets/fonts/*.ttf
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fontTools.merge import Merger
from fontTools.ttLib import TTFont

from common import FONTS

F_DIR = Path("/tmp/fonts")  # npm-unpacked @fontsource packages (see README)

JOBS = [
    # (output, [woff2s in merge order (first wins on overlap)])
    ("TiroLatin-Italic.ttf", [
        F_DIR / "tiro-devanagari-sanskrit/files/tiro-devanagari-sanskrit-latin-400-italic.woff2",
        F_DIR / "tiro-devanagari-sanskrit/files/tiro-devanagari-sanskrit-latin-ext-400-italic.woff2",
    ]),
    ("Marcellus-Regular.ttf", [
        F_DIR / "marcellus/files/marcellus-latin-400-normal.woff2",
        F_DIR / "marcellus/files/marcellus-latin-ext-400-normal.woff2",
    ]),
]


def to_ttf(path: Path, dest: Path):
    f = TTFont(str(path))
    f.flavor = None
    f.save(str(dest))
    return str(dest)


def main():
    FONTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for out_name, sources in JOBS:
            missing = [str(s) for s in sources if not s.exists()]
            if missing:
                raise SystemExit(f"missing font sources for {out_name}: {missing}\n"
                                 "unpack @fontsource packages to /tmp/fonts first (see README)")
            ttfs = [to_ttf(s, td / f"{out_name}.{i}.ttf") for i, s in enumerate(sources)]
            merged = Merger().merge(ttfs)
            merged.save(str(FONTS / out_name))
            print("merged ->", FONTS / out_name)


if __name__ == "__main__":
    main()
