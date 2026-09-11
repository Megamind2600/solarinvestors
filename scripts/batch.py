"""Batch driver: render many verses with a process pool.

    python scripts/batch.py --range 0-25            # verses[0:25] by sorted order
    python scripts/batch.py --range 100-200 --workers 3 --duration 45
    python scripts/batch.py --chapter 6             # a whole chapter

Each verse renders in a subprocess (clean memory, crash isolation); existing
outputs are skipped, so re-runs resume where they stopped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import VIDEOS, load_verses

SCRIPTS = Path(__file__).resolve().parent


def out_path(verse: dict, root: Path) -> Path:
    return root / f"ch{int(verse['chapter']):02d}" / f"{verse['id'].replace('.', '_')}.mp4"


def render_one(job):
    verse, root, duration, crf, fx, audio = job
    out = out_path(verse, root)
    if out.exists() and out.stat().st_size > 200_000:
        return (verse["id"], "skip", 0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, str(SCRIPTS / "render_video.py"), "--id", verse["id"],
         "--duration", str(duration), "--crf", str(crf), "--fx", fx, "--audio", audio,
         "--out", str(out)],
        capture_output=True, text=True)
    status = "ok" if rc.returncode == 0 else "FAIL"
    if rc.returncode != 0:
        err = (rc.stderr or rc.stdout)[-600:]
        (out.parent / f"{verse['id'].replace('.', '_')}.log").write_text(err)
        return (verse["id"], status, time.time() - t0)
    return (verse["id"], status, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="0-10", help="slice of sorted verses, e.g. 0-25")
    ap.add_argument("--chapter", type=int, default=None)
    ap.add_argument("--duration", type=float, default=45)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(VIDEOS))
    ap.add_argument("--fx", default="cinematic", choices=["meditative", "cinematic", "festive"])
    ap.add_argument("--audio", default="auto", choices=["auto", "library", "synth"])
    args = ap.parse_args()

    verses = load_verses()
    if args.chapter:
        verses = [v for v in verses if int(v["chapter"]) == args.chapter]
    else:
        a, b = args.range.split("-")
        verses = verses[int(a):int(b)]
    root = Path(args.out)
    jobs = [(v, root, args.duration, args.crf, args.fx, args.audio) for v in verses]
    print(f"batch: {len(jobs)} videos, {args.workers} workers, {args.duration}s each, "
          f"fx={args.fx} audio={args.audio}")
    t0 = time.time()
    ok = skip = fail = 0
    with Pool(args.workers) as pool:
        for vid, status, secs in pool.imap_unordered(render_one, jobs):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
            print(f"[{ok + skip + fail}/{len(jobs)}] {vid}: {status} ({secs:.0f}s)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min — ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
