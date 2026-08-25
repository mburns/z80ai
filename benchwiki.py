#!/usr/bin/env python3
"""
What one query costs the Agon: instructions executed and bytes read from the card.

    python benchwiki.py --card dist/WIKI
    python benchwiki.py --card dist/WIKI --query "mount everest" "who wrote hamlet"

`bench.py` does this for the models and there has never been an equivalent for
the card, which is why `data/wikipedia/README.md` carried a per-query
instruction count measured before the accumulator was tiered, and a speedup
measured on a synthetic 100,000-article corpus rather than this one. Both were
honest about being provisional; neither could be settled without running the
real card, and running the real card needed something to run it.

## What the numbers mean

`instructions` is what the emulator retired, counted the same way `bench.py`
counts them. eZ80 T-states are deliberately not reported: its per-instruction
timings differ substantially from the Z80's, so an instruction count is the
honest cross-architecture figure and a cycle count would be a guess dressed up.

`card bytes` is what the program actually read through the MOS file API, which
`AgonHost` tallies rather than inferring - a hook that costs no T-states is not
pretended to cost some.

## Why the queries matter more than the average

The tiering makes cost depend on *which* pages a query touches, so one number
for "a query" is meaningless. A lookup naming a rare subject flags a handful of
the 1,110 pages and skips the rest of both passes over the 277KB accumulator; a
term common enough to appear on every page pays the whole-corpus scan it always
paid, plus the page table's overhead. Both are reported, because the second is
the case the tiering does not help and the one a reader should see.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from libhost import AgonHost

#: Queries chosen to sit at both ends of the tiering, not to flatter it.
#: The first three name a subject; the last two are common words that land on a
#: large share of the corpus, which is the case the page table costs rather
#: than saves.
DEFAULT_QUERIES = (
    "mount everest",
    "zilog z80",
    "jane austen",
    "the united states of america",
    "world war",
)


def run(binary: bytes, files: dict[str, bytes], query: str,
        max_cycles: int = 4_000_000_000) -> tuple[int, int, float, str]:
    """One query, and what it cost. Returns (instructions, bytes, seconds, text)."""
    host = AgonHost(stdin=[query, "!"], files=files)
    started = time.monotonic()
    text = host.run(binary, max_cycles=max_cycles)
    return (host.cpu.instructions, host.io_bytes,
            time.monotonic() - started, text)


def found(text: str, query: str) -> str:
    """What the card said, which is the line after it echoed the question.

    Reported so a run cannot quietly benchmark a miss: a query that finds
    nothing is cheap, and a table of costs with no answers beside them would
    make that look like the fastest result on the page.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        if line == f"? {query}":
            rest = [ln for ln in lines[i + 1:] if ln]
            return rest[0] if rest else "(nothing)"
    return "(no echo)"


def card_files(stem: Path) -> tuple[bytes, dict[str, bytes]]:
    """The binary and the card, read the way the machine would see them.

    `.GRF` is optional: a card built without `--relations` is a search card and
    has none, and the program is a different and much smaller one.
    """
    binary = stem.with_suffix(".bin").read_bytes()
    files: dict[str, bytes] = {}
    for suffix in (".IDX", ".DAT", ".GRF"):
        path = stem.with_suffix(suffix)
        if path.exists():
            files[path.name] = path.read_bytes()
    missing = {".IDX", ".DAT"} - {Path(n).suffix for n in files}
    if missing:
        sys.exit(f"{stem} is missing {', '.join(sorted(missing))}")
    return binary, files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--card", type=Path, default=Path("dist/WIKI"),
                    help="path stem: <stem>.bin, .IDX, .DAT and optionally .GRF")
    ap.add_argument("--query", nargs="*", default=list(DEFAULT_QUERIES))
    ap.add_argument("--clock", type=float, default=18.432,
                    help="MHz, for the seconds column")
    ap.add_argument("--read-rate", type=float, default=250.0,
                    help="KB/s the card is assumed to sustain")
    args = ap.parse_args()

    binary, files = card_files(args.card)
    sizes = "  ".join(f"{n} {len(b) / 1e6:.1f}MB" for n, b in sorted(files.items()))
    print(f"\n{args.card}.bin {len(binary):,} bytes   {sizes}\n")
    print(f"  {'query':<32}{'instructions':>14}{'card bytes':>12}"
          f"{'s @ ' + str(args.clock) + 'MHz':>16}{'s of I/O':>10}")

    for query in args.query:
        instructions, io_bytes, wall, text = run(binary, files, query)
        # One instruction is not one cycle on an eZ80, but the ratio is close
        # enough to state as an order of magnitude and the README quotes it
        # that way; the instruction count is the figure that is exact.
        seconds = instructions / (args.clock * 1e6)
        io_seconds = io_bytes / (args.read_rate * 1024)
        print(f"  {query:<32}{instructions:>14,}{io_bytes:>12,}"
              f"{seconds:>16.2f}{io_seconds:>10.2f}   {found(text, query)[:34]}"
              f"   [{wall:.0f}s wall]")
    print()


if __name__ == "__main__":
    main()
