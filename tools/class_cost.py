#!/usr/bin/env python3
"""
What adding a class to the phrasebook costs the classes already in it.

    python tools/class_cost.py --added born_in_year died_in_year fate_is
    python tools/class_cost.py --added count_born_on --seeds 10 --watch born_on

A new path is nearly free on the card - one row in the step table - and is not
free in the classifier, which has to find a region in trigram space for it
somewhere between the regions it already has. `data/silo/README.md` measured
that twice and got two different answers: two count classes cost the refusal
class **18.6 +/- 6.9** points, and one of them cost **6.9 +/- 6.8**, which is
nothing. The difference between those two conclusions is entirely the seeds,
and three of them could not have told them apart.

So this is the instrument rather than the result. It trains two arms per seed -
the phrasebook with the new labels and the same phrasebook without them -
**paired**, so the spread that swamps a three-seed sweep cancels.

## What it reports, and why the first column is not the interesting one

**shared** is held-out accuracy over the labels present in *both* arms. It has
to be restricted that way: adding a class changes the denominator, and an
arm that answers more kinds of question is not comparable to one that answers
fewer unless the comparison is over the questions they have in common.

**refuse** is the one that has moved before. It is scored as *did it refuse at
all*, which is the only distinction the eZ80 can make, and it is the class
with no coherent region in trigram space - four unrelated question shapes under
one label - so it is where a new class's vocabulary lands when it displaces
something.

**added** is the new classes' own held-out accuracy, and a low number there is
a different problem from a low number anywhere else: it means the wordings are
short of grammar rather than that the model is short of room.

`--watch` reports named labels individually, for when a new class is expected
to collide with a specific old one. `born_in_year` against `born_on` is the
case this was written for: both say "born", and only the rest of the sentence
tells them apart.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import libgraphcard
import libinfer
from bucket_sweep import buckets


@dataclass
class Arm:
    """One trained model's held-out scores, by class."""

    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)

    def share(self, labels: set[str]) -> float:
        hit = sum(h for k, (h, _) in self.per_class.items() if k in labels)
        total = sum(t for k, (_, t) in self.per_class.items() if k in labels)
        return hit / total if total else 0.0


def one_arm(db: sqlite3.Connection, have: set[str],
            paths: dict[str, tuple[str, ...]],
            seed: int, held_out: int, epochs: int, hidden: list[int],
            bucket_count: int) -> Arm:
    """Train on `paths` and score the wordings it never saw.

    `refuse` is scored as whether the answer was the refusal class, not as
    whether it was the right one of several - the card encodes every refusal to
    the same step, so a finer reading would measure something the machine
    cannot act on.
    """
    import relationpaths

    import classify

    original = relationpaths.PATHS
    relationpaths.PATHS = paths
    try:
        with buckets(bucket_count):
            train, unseen = relationpaths.build(
                db, have, relationpaths.PER_TEMPLATE, held_out, seed)
            model, _o, _m = classify.train(
                train, hidden, epochs, 0.01, seed=seed, split_seed=seed,
                val_frac=0.1, accum_bits=24, position_bands=libinfer.FLAT,
                quiet=True, num_buckets=bucket_count)
            arm = Arm()
            for question, want in unseen:
                got = libinfer.classify(model, question, 24)
                hit, total = arm.per_class.get(want, (0, 0))
                arm.per_class[want] = (hit + (got == want), total + 1)
    finally:
        relationpaths.PATHS = original
    return arm


def _paired(diffs: list[float]) -> tuple[float, float, float]:
    """(mean, standard error, t) for a paired difference."""
    mean = statistics.mean(diffs)
    if len(diffs) < 2:
        return mean, 0.0, 0.0
    sem = statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, sem, (mean / sem if sem else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--added", nargs="+", required=True,
                    help="Labels to measure the cost of, from relationpaths' "
                         "PATHS (already shipped) or CANDIDATES (not shipped)")
    ap.add_argument("--watch", nargs="*", default=[],
                    help="Labels to report on their own, for expected collisions")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", default="128,96")
    ap.add_argument("--buckets", type=int, default=256)
    args = ap.parse_args()

    import relationpaths
    from schema import SOURCE

    hidden = [int(v) for v in args.hidden.split(",")]
    repo = Path(__file__).resolve().parent.parent
    db = sqlite3.connect(f"file:{repo / 'data' / 'silo.db'}?mode=ro", uri=True)
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (SOURCE,))}

    # A label already in `PATHS` is measured by taking it *out* of the control
    # arm; one in `CANDIDATES` is measured by putting it *into* the test arm.
    # Either way the two arms differ by exactly the labels named, which is the
    # only thing the pairing needs to be true.
    added = set(args.added)
    known = set(relationpaths.PATHS) | set(relationpaths.CANDIDATES)
    missing = added - known
    if missing:
        raise SystemExit(f"not in PATHS or CANDIDATES: {sorted(missing)}")
    without = {k: v for k, v in relationpaths.PATHS.items() if k not in added}
    with_added = dict(without) | {
        k: relationpaths.PATHS.get(k) or relationpaths.CANDIDATES[k]
        for k in args.added}
    shared = set(without) - {libgraphcard.REFUSE_PATH}

    rows: list[tuple[Arm, Arm]] = []
    print(f"{'seed':>5}{'arm':>10}{'shared':>9}{'refuse':>9}{'added':>9}"
          + "".join(f"{w:>22}" for w in args.watch))
    print("-" * (33 + 22 * len(args.watch)))
    for seed in range(args.seeds):
        pair = (one_arm(db, have, without, seed, args.held_out, args.epochs,
                        hidden, args.buckets),
                one_arm(db, have, with_added, seed, args.held_out,
                        args.epochs, hidden, args.buckets))
        rows.append(pair)
        for name, arm in (("without", pair[0]), ("with", pair[1])):
            print(f"{seed:>5}{name:>10}{arm.share(shared):>9.1%}"
                  f"{arm.share({libgraphcard.REFUSE_PATH}):>9.1%}"
                  f"{arm.share(added) if name == 'with' else 0.0:>9.1%}"
                  + "".join(f"{arm.share({w}):>22.1%}" for w in args.watch),
                  flush=True)

    print()
    print(f"{'':>18}{'without':>10}{'with':>10}{'paired diff':>16}{'t':>7}")
    print("-" * 61)
    for label, want in (("shared", shared),
                        ("refuse", {libgraphcard.REFUSE_PATH}),
                        *((f"watch {w}", {w}) for w in args.watch)):
        lo = [a.share(want) for a, _ in rows]
        hi = [b.share(want) for _, b in rows]
        mean, sem, t = _paired([b - a for a, b in zip(lo, hi, strict=True)])
        print(f"{label:>18}{statistics.mean(lo):>10.1%}"
              f"{statistics.mean(hi):>10.1%}"
              f"{mean:>11.1%} +/-{sem:>4.1%}{t:>7.2f}")
    new = [b.share(added) for _, b in rows]
    print(f"{'added':>18}{'-':>10}{statistics.mean(new):>10.1%}")


if __name__ == "__main__":
    main()
