#!/usr/bin/env python3
"""
How many trigram buckets the encoder should have, which nobody has ever asked.

    python tools/bucket_sweep.py                     # 128/256/512/1024, 3 seeds
    python tools/bucket_sweep.py --buckets 256 --seeds 1

`libinfer.NUM_BUCKETS` has been 128 since the first commit. #54 swept the
classifier's *hidden* width and found 128,96 was enough; nothing has swept its
*input* width, and the phrasebook has grown a long way since 128 was chosen.

It looks too small. The silo's twenty-one paths use 859 distinct trigrams, and
at 128 buckets **85% of them share a bucket with something else** - every
bucket occupied, 6.7 trigrams apiece. A distinguishing trigram that lands on
top of six others is a signal the model has to separate from noise it cannot
see around.

    128 buckets:   128 used,  731 trigrams sharing a bucket (85%)
    256 buckets:   249 used,  610 trigrams sharing a bucket (71%)
    512 buckets:   434 used,  425 trigrams sharing a bucket (49%)
   1024 buckets:   623 used,  236 trigrams sharing a bucket (27%)

## What this reports

**held out** is the number that matters and the one this repository has spent
months on: accuracy on three wordings per path the model never saw.

**prefix** is a diagnosis rather than a score. A quarter of held-out misses are
a path losing to its own prefix - `mother_is mother_is` answered as
`mother_is`, and so on for every two-hop path. Those paths already say
"grandmother" in five of their twelve wordings, so the word is there and the
encoder is losing it. If buckets are the reason, this column moves.

## What it costs

Buckets are the width of layer one, so they are weights, and weights are image
bytes on a card where an article costs about one byte of program. Going to
1,024 adds roughly 32 KB - which is 32,000 articles off a 502,016 ceiling, and
nothing at all on a silo card of 13,082.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import libinfer


@contextlib.contextmanager
def buckets(count: int) -> Iterator[None]:
    """Encode and build the network at `count` buckets rather than 128.

    `trigram_encode`'s bucket count is a default argument, bound when the
    module was imported, so setting `libinfer.NUM_BUCKETS` alone would widen
    the network and leave the encoder where it was - a mismatch that trains
    perfectly well and means nothing. Both are replaced together, and the
    caller checks the vector length against the weights afterwards.
    """
    original_encode = libinfer.trigram_encode
    original_count = libinfer.NUM_BUCKETS

    def encode(text: str, num_buckets: int = count,
               position_bands: int = libinfer.FLAT) -> np.ndarray:
        return original_encode(text, count, position_bands)

    libinfer.trigram_encode = encode
    libinfer.NUM_BUCKETS = count
    try:
        yield
    finally:
        libinfer.trigram_encode = original_encode
        libinfer.NUM_BUCKETS = original_count


def one_run(count: int, seed: int, held_out: int, epochs: int,
            hidden: list[int]) -> tuple[float, float, float]:
    """(held-out accuracy, macro, share of misses that are prefix confusions)."""
    import sqlite3

    import relationpaths
    from schema import SOURCE

    import classify

    repo = Path(__file__).resolve().parent.parent
    db = sqlite3.connect(f"file:{repo / 'data' / 'silo.db'}?mode=ro", uri=True)
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (SOURCE,))}

    with buckets(count):
        train, unseen = relationpaths.build(
            db, have, relationpaths.PER_TEMPLATE, held_out, seed)
        model, _o, _m = classify.train(
            train, hidden, epochs, 0.01, seed=seed, split_seed=seed,
            val_frac=0.1, accum_bits=24, position_bands=libinfer.FLAT,
            quiet=True)

        # The check the docstring promises: a network whose first layer does
        # not match the vectors it is fed would score something meaningless.
        width = model.weights[0].shape[1]
        assert width == count, f"layer one is {width} wide, buckets are {count}"

        right = 0
        misses = prefix = 0
        per_class: dict[str, list[int]] = {}
        for question, want in unseen:
            got = libinfer.classify(model, question, 24)
            hit, total = per_class.setdefault(want, [0, 0])
            per_class[want] = [hit + (got == want), total + 1]
            if got == want:
                right += 1
                continue
            misses += 1
            w, g = want.split(), got.split()
            if g == w[:len(g)] or w == g[:len(w)]:
                prefix += 1

    macro = statistics.mean(h / t for h, t in per_class.values())
    return right / len(unseen), macro, (prefix / misses if misses else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buckets", type=int, nargs="*",
                    default=[128, 256, 512, 1024])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", default="128,96")
    args = ap.parse_args()

    hidden = [int(v) for v in args.hidden.split(",")]
    print(f"{'buckets':>8}{'held out':>11}{'macro':>9}{'prefix misses':>15}"
          f"{'spread':>18}")
    print("-" * 61)

    for count in args.buckets:
        runs = [one_run(count, seed, args.held_out, args.epochs, hidden)
                for seed in range(args.seeds)]
        overall = [r[0] for r in runs]
        spread = " / ".join(f"{v:.1%}" for v in overall)
        print(f"{count:>8}{statistics.mean(overall):>10.1%}"
              f"{statistics.mean(r[1] for r in runs):>9.1%}"
              f"{statistics.mean(r[2] for r in runs):>14.1%}"
              f"{spread:>19}", flush=True)


if __name__ == "__main__":
    main()
