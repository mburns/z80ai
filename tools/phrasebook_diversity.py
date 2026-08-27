#!/usr/bin/env python3
"""
How much a phrasebook repeats itself, in the encoder's own view.

    python tools/phrasebook_diversity.py
    python tools/phrasebook_diversity.py --held-out 3 --seed 0

`data/silo/README.md` reports that the held-out-phrasing number climbs with the
number of wordings per path and was still climbing at nine. The obvious next
move is to write more wordings, and #58 wrote down the reason to distrust the
result before anybody did:

    a second dozen written by the same hand on the same afternoon will
    resemble the first dozen more than a stranger's would, and would
    therefore understate the gain

That is half of it. Near-duplicate wordings cut two ways at once and the two
pull in opposite directions:

- **the gain is understated** - a wording that repeats one the model has seen
  teaches it nothing, so adding twelve of those moves the curve less than
  twelve genuinely different ones would;
- **the score is overstated** - the held-out wordings are drawn from the same
  pool, so if everything resembles everything then what is being held out is
  not novel and getting it right proves less than it looks like.

Which dominates is not an argument to have; it is a number. This prints it.

## What is measured

Similarity here is cosine between `libinfer.trigram_encode` vectors - the
encoder's own representation, not a human's idea of paraphrase. Two wordings a
person would call quite different can land in nearly the same 128 buckets, and
those are exactly the pairs that matter, because the model cannot see any
difference the encoder did not keep.

**within** is the mean pairwise similarity among a path's wordings. Higher
means the phrasebook is saying one thing several ways.

**novelty** is the sharper one: for each held-out wording, one minus its
similarity to the *nearest* wording the model was trained on. A held-out
wording with a near-twin in training is not a test of generalisation, and a
phrasebook can be grown until this number collapses without anything else
looking wrong.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from random import Random

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import libinfer


def encode(text: str) -> np.ndarray:
    vector = libinfer.trigram_encode(text).astype(np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def similarity(a: str, b: str) -> float:
    return float(np.dot(encode(a), encode(b)))


def within(phrasings: list[str]) -> float:
    """Mean pairwise similarity among one path's wordings."""
    pairs = [similarity(phrasings[i], phrasings[j])
             for i in range(len(phrasings))
             for j in range(i + 1, len(phrasings))]
    return statistics.mean(pairs) if pairs else 0.0


def novelty(held_out: list[str], trained: list[str]) -> list[float]:
    """1 - the closest thing in training, for each held-out wording."""
    return [1.0 - max((similarity(one, other) for other in trained),
                      default=0.0)
            for one in held_out]


def split(phrasings: list[str], held_out: int, seed: int
          ) -> tuple[list[str], list[str]]:
    """The same split `relationpaths.build` makes: shuffle, then reserve."""
    order = list(phrasings)
    Random(seed).shuffle(order)
    return order[held_out:], order[:held_out]


def report(paths: dict[str, tuple[str, ...]], held_out: int, seed: int) -> None:
    print(f"{'path':<34}{'n':>4}{'within':>9}{'novelty':>9}")
    print("-" * 56)

    withins, novelties = [], []
    for name, phrasings in paths.items():
        # `{s}` stands for a name and is filled from the corpus; leaving the
        # brace in would put its trigrams in every wording of every path and
        # flatter every number here.
        wordings = [p.replace("{s}", "amanda m wilson") for p in phrasings]
        trained, reserved = split(wordings, held_out, seed)
        w = within(wordings)
        n = statistics.mean(novelty(reserved, trained)) if reserved else 0.0
        withins.append(w)
        novelties.append(n)
        print(f"{name:<34}{len(phrasings):>4}{w:>9.3f}{n:>9.3f}")

    print("-" * 56)
    print(f"{'mean':<34}{'':>4}{statistics.mean(withins):>9.3f}"
          f"{statistics.mean(novelties):>9.3f}")
    print("\nwithin  - how much a path's wordings resemble each other, in the "
          "128\n          buckets the model sees. 1.000 would be the same "
          "sentence twice.")
    print("novelty - how far each held-out wording is from the nearest one "
          "trained on.\n          Near zero means the held-out set is not "
          "testing generalisation.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import relationpaths

    report(relationpaths.PATHS, args.held_out, args.seed)


if __name__ == "__main__":
    main()
