#!/usr/bin/env python3
"""Is 55.6% a fact about the model, or about how much English it was shown?

    python tools/grammar_pilot.py                # 5 seeds, both arms
    python tools/grammar_pilot.py --seeds 10

`data/silo/README.md` draws a phrasing curve that is **still climbing at
nine** - 39.2% to 45.8% for the last three wordings, with three-seed spreads
that do not overlap - and then stops, because nine is what `relationpaths.py`
happens to contain. Everything tried against that number since has been a
change to the encoder or the architecture: masking (noise), position bands
(worse), halving the model (nothing), buckets (+7.5 and then flat). The one
lever measured to still have room is English, and nobody has been past nine.

Writing twelve more wordings for all twenty-six paths is 240 sentences. This
measures five of them first, so that the 240 are written knowing whether they
are worth writing.

## Why the wordings live outside `PATHS`

The obvious version of this experiment is wrong and the README says why: a path
given twelve more wordings while still holding out three has a held-out set
with more neighbours to learn from, so its score rises for a reason that is not
grammar. Novelty fell from 0.188 to 0.100 on `mother_is` when the six paths
were extended, and "some of 48.9% -> 40.9% is that, not the disambiguation".

`relationpaths.EXTRA` is training-only. The held-out three per path are drawn
from the original twelve in both arms and are byte-identical between them, so
the only thing that moves is how much grammar the model saw. That is the same
design the learning curve used, carried past where it stopped.

## Three arms, because two could not settle it

The first run of this measured `none` against `first` and found twelve
wordings worth **+16.9 +/- 3.9** to the five paths that got them and
**-3.2 +/- 0.9** to the twenty-one that did not, for an overall +0.5 that is
indistinguishable from nothing. Redistribution accounted for 80-88% of the
gain, it was not the prior (`--balance` did not move it) and it was not
capacity (114 of the 117 extra misses landed on the five specifically).

What that cannot say is whether the losses have a direction only because five
classes grew and twenty-one did not. `both` extends a second, disjoint five -
matched for difficulty, 38.2% held out against the first group's 39.2% - and
asks the question the first run raised:

    none  -> first    what twelve wordings are worth
    first -> both     whether the first five keep it while others grow
    none  -> both     what ten classes' worth of grammar buys the corpus

If growth is zero-sum whoever does it, the first five give back what they won
as the second five take their turn, and `overall` stays flat in all three.
If instead everybody's region can grow into the questions that currently land
nowhere, `first -> both` leaves the first five alone and lifts the second.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections.abc import Set as AbstractSet
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import libinfer
from bucket_sweep import buckets


def one_arm(db: sqlite3.Connection, have: set[str], seed: int, held_out: int,
            epochs: int, hidden: list[int], bucket_count: int,
            extra: frozenset[str], balance: bool = False,
            third: frozenset[str] = frozenset(),
            ) -> tuple[dict[str, tuple[int, int]], int]:
    """(per-class hits and totals, training rows) for one arm.

    ``balance`` weights the loss by inverse class frequency, which is what
    separates the two things the extended arm changes at once: those five
    classes gain grammar *and* gain rows, so the model also sees them twice as
    often and its prior moves. Unbalanced is what the shipped card does;
    balanced is what says which of the two an effect came from.
    """
    import relationpaths

    import classify

    with buckets(bucket_count):
        train, unseen = relationpaths.build(
            db, have, relationpaths.PER_TEMPLATE, held_out, seed, extra=extra,
            third=third)
        model, _o, _m = classify.train(
            train, hidden, epochs, 0.01, seed=seed, split_seed=seed,
            val_frac=0.1, accum_bits=24, position_bands=libinfer.FLAT,
            quiet=True, balance=balance, num_buckets=bucket_count)
        per_class: dict[str, tuple[int, int]] = {}
        for question, want in unseen:
            got = libinfer.classify(model, question, 24)
            hit, total = per_class.get(want, (0, 0))
            per_class[want] = (hit + (got == want), total + 1)
    return per_class, len(train)


def share(per_class: dict[str, tuple[int, int]],
          labels: AbstractSet[str]) -> float:
    hit = sum(h for k, (h, _) in per_class.items() if k in labels)
    total = sum(t for k, (_, t) in per_class.items() if k in labels)
    return hit / total if total else 0.0


def _paired(diffs: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(diffs)
    if len(diffs) < 2:
        return mean, 0.0, 0.0
    sem = statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, sem, (mean / sem if sem else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", default="128,96")
    ap.add_argument("--buckets", type=int, default=256)
    ap.add_argument("--balance", action="store_true",
                    help="Weight the loss by inverse class frequency, so the "
                         "extended classes do not also gain a prior")
    args = ap.parse_args()

    import relationpaths
    from schema import SOURCE

    hidden = [int(v) for v in args.hidden.split(",")]
    repo = Path(__file__).resolve().parent.parent
    db = sqlite3.connect(f"file:{repo / 'data' / 'silo.db'}?mode=ro", uri=True)
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (SOURCE,))}

    everything = set(relationpaths.PATHS)

    # Three arms, because two cannot answer the question the first run raised.
    # `none` vs `first` measures what twelve wordings are worth; `first` vs
    # `both` measures whether that gain survives somebody else growing too,
    # which is the whole difference between a redistribution that would vanish
    # if everybody grew and one that would not.
    ten = relationpaths.FIRST_FIVE | relationpaths.SECOND_FIVE
    everything_grown = ten | relationpaths.REMAINING_TEN
    # (second dozen, third dozen). `three` is the tail of the curve: everybody
    # on twenty-one wordings, and the first five on thirty-three.
    arms = {"none": (frozenset(), frozenset()),
            "first": (relationpaths.FIRST_FIVE, frozenset()),
            "both": (ten, frozenset()),
            "all": (everything_grown, frozenset()),
            "three": (everything_grown, frozenset(relationpaths.EXTRA_THIRD))}
    groups = {"first five": relationpaths.FIRST_FIVE,
              "second five": relationpaths.SECOND_FIVE,
              "last ten": relationpaths.REMAINING_TEN,
              # The six the prefix repair already took to twenty-one trained
              # wordings, plus `refuse` and its forty-eight. Not grown here
              # because there was nothing to grow - which is what makes them
              # the control for whether the others merely caught up.
              "already there +refuse":
                  everything - ten - relationpaths.REMAINING_TEN,
              "overall": everything}

    Counts = dict[str, tuple[int, int]]
    runs: dict[str, list[Counts]] = {k: [] for k in arms}
    print(f"{'seed':>5}{'grown':>8}{'rows':>9}"
          + "".join(f"{g:>19}" for g in groups))
    print("-" * (22 + 19 * len(groups)))
    for seed in range(args.seeds):
        for name, (grow, grow_more) in arms.items():
            per_class, n = one_arm(db, have, seed, args.held_out, args.epochs,
                                   hidden, args.buckets, grow, args.balance,
                                   grow_more)
            runs[name].append(per_class)
            print(f"{seed:>5}{name:>8}{n:>9,}"
                  + "".join(f"{share(per_class, w):>19.1%}"
                            for w in groups.values()), flush=True)

    for lo_name, hi_name in (("none", "first"), ("first", "both"),
                             ("both", "all"), ("none", "all"),
                             ("all", "three")):
        print(f"\n{lo_name} -> {hi_name}")
        print(f"{'':>22}{lo_name:>8}{hi_name:>8}{'paired diff':>15}{'t':>7}")
        print("-" * 60)
        for label, want in groups.items():
            lo = [share(a, want) for a in runs[lo_name]]
            hi = [share(b, want) for b in runs[hi_name]]
            mean, sem, t = _paired(
                [b - a for a, b in zip(lo, hi, strict=True)])
            print(f"{label:>22}{statistics.mean(lo):>8.1%}"
                  f"{statistics.mean(hi):>8.1%}"
                  f"{mean:>10.1%} +/-{sem:>4.1%}{t:>7.2f}")


if __name__ == "__main__":
    main()
