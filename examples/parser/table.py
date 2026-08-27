#!/usr/bin/env python3
"""
The same command parser as a word table, for comparison with the model.

    ./table.py                    # score it on both splits
    ./table.py --unknown          # and on words it was never given

`compare.py` trains two encoders on "PUT <A> IN <B>" and scores them on object
pairs held out of training. This is the third row of that table and it is not a
model at all: a verb set, a noise-word set, a preposition set, and the sizes
written down.

## Why the comparison is fair, and why it is not

It is fair in that both end at the same answer from the same input, and both are
scored on the same held-out pairs.

It is not fair in the way that matters: **the table is told what the model has
to infer.** That is the whole argument rather than a flaw in the experiment. An
author of an Interactive Fiction writes the world down - a key is small, a barn
is large - and does not train it. Asking which approach should sit on the
critical path of `PUT X IN Y` is asking whether that authoring step is worth
what it buys, and this measures what it buys.

## What it cannot do

It has no opinion about a word it was not given. That is the property the model
does not have and the one an IF needs most: `PUT ZORKMID IN BOX` returns nothing
here and returns a confident `OK` or `NO` from the model, and only one of those
two can be reported to a player as "I don't know the word 'zorkmid'".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gendata import SIZES, TEMPLATES, answer, pairs

#: Everything the parser knows, written down rather than learned. The verbs and
#: prepositions come from `gendata.TEMPLATES`; keeping them derived means a new
#: phrasing there cannot silently leave this behind.
NOISE = {"THE", "A", "AN"}


def _vocabulary() -> tuple[set[str], set[str]]:
    """(verbs, prepositions), read out of the templates themselves."""
    verbs, preps = set(), set()
    for template in TEMPLATES:
        words = template.replace("{a}", "\0").replace("{b}", "\0").split()
        seen_slot = False
        for word in words:
            if word == "\0":
                seen_slot = True
                continue
            if word in NOISE:
                continue
            (preps if seen_slot else verbs).add(word)
    return verbs, preps


VERBS, PREPS = _vocabulary()


def parse(command: str) -> tuple[str, str] | None:
    """(subject, container), or None for a command it cannot read.

    No statistics and no threshold: a word is in the vocabulary or it is not,
    and the failure names the word rather than guessing past it.
    """
    words = [w for w in command.upper().split() if w not in NOISE]
    if not words or words[0] not in VERBS:
        return None

    nouns, seen_prep = [], False
    for word in words[1:]:
        if word in PREPS:
            seen_prep = True
            continue
        if word not in SIZES:
            return None                  # a word it was never given
        nouns.append(word)
    if len(nouns) != 2 or not seen_prep:
        return None
    return nouns[0], nouns[1]


def respond(command: str) -> str | None:
    """`OK`, `NO`, or None where the model would have guessed."""
    parsed = parse(command)
    return None if parsed is None else answer(*parsed)


def score(commands: list[tuple[str, str]]) -> tuple[float, int]:
    """(accuracy, how many it declined) over (command, expected) pairs."""
    right = declined = 0
    for command, expected in commands:
        got = respond(command)
        declined += got is None
        right += got == expected
    return right / len(commands), declined


def corpus(subset: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(t.format(a=a, b=b), answer(a, b)) for a, b in subset
            for t in TEMPLATES]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unknown", action="store_true",
                    help="also ask about objects it was never given")
    args = ap.parse_args()

    every = pairs()
    accuracy, declined = score(corpus(every))
    print(f"word table, all {len(every)} pairs: {accuracy:.1%}, "
          f"{declined} declined")

    if args.unknown:
        made_up = [("ZORKMID", "BOX"), ("KEY", "GRUE"), ("XYZZY", "PLUGH")]
        rows = [(t.format(a=a, b=b), "?") for a, b in made_up for t in TEMPLATES]
        _acc, declined = score(rows)
        print(f"words it was never given: {declined}/{len(rows)} declined, "
              f"{len(rows) - declined} answered anyway")


if __name__ == "__main__":
    main()
