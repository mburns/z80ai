#!/usr/bin/env python3
"""
What a hop costs the Agon, measured by asking it to climb further.

    python data/silo/benchcard.py                       # ~2 minutes
    python data/silo/benchcard.py --subjects 4          # quicker, noisier

`benchwiki.py` reports what one query costs. It cannot separate the parts,
because a single query is a search, a classifier forward pass and a graph walk
added together, and the first two do not care how many hops the third takes.

This separates them the only way a corpus can: **by asking the same question of
people who are different distances from the answer.** "Which founder is X
descended from" is a climb, and its length is X's generation - one hop for a
founder's child, five for their great-great-great-grandchild. Everything else
about the query is unchanged, so the slope of cost against generation is the
cost of a hop and the intercept is everything else.

One subject per generation is not enough to see it: a name is a set of search
terms, and two names differ by more card bytes than a hop costs. Averaging over
`--subjects` names per generation is what makes the slope legible, and the
spread is printed beside it rather than hidden.

## The sixth generation is not a data point

`libgraph.CLIMB_LIMIT` is 6 and counts hops rather than nodes, so generation 6
needs one more than it is allowed. On the machine that is not an error message:
the walk returns nothing, the program falls back to listing articles, and the
cost jumps - not because the climb was long, but because a fallback reads
article text and a graph answer does not. It is reported on its own line for
that reason, and left out of the fit.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import sys
import time
from math import ceil, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # also registers the silo's climbs into libgraph.CLIMB
from schema import SOURCE

import benchwiki

REPO = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO / "data" / "silo.db"

#: One phrasing, held fixed, so that generation is the only thing that varies.
#: `buildcard.py --skip-train` reports which phrasings the model is steady on;
#: this is one of them.
QUESTION = "which founder is {s} descended from"

#: Names are search terms, so subjects are drawn at a fixed length to keep the
#: index work as similar as the corpus allows. It does not make the noise go
#: away - see the spread column - it just stops it correlating with generation.
NAME_LENGTH = 16


def subjects(db: sqlite3.Connection, per_generation: int,
             ) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for generation in range(1, len(generate.GENERATIONS)):
        rows = [r for (r,) in db.execute(
            "SELECT name FROM person WHERE source = ? AND generation = ? "
            "AND father IS NOT NULL AND length(name) = ? "
            "ORDER BY name LIMIT ?",
            (SOURCE, generation, NAME_LENGTH, per_generation))]
        if rows:
            out[generation] = rows
    return out


def measure(stem: Path, names: list[str]) -> list[tuple[int, int, str]]:
    binary, files = benchwiki.card_files(stem)
    out: list[tuple[int, int, str]] = []
    for name in names:
        query = QUESTION.format(s=name.lower())
        instructions, io_bytes, _, text = benchwiki.run(binary, files, query)
        out.append((instructions, io_bytes, benchwiki.found(text, query)))
    return out


def entity_lookup(stem: Path, db: sqlite3.Connection, sample: int,
                  seed: int) -> tuple[int, int, int, int]:
    """(first, in the top three, missing, people sharing a first and last name).

    Stage one of the oracle, and the stage whose errors have no symptom: the
    walk answers correctly about the wrong subject, and what comes back is
    fluent and wrong. `tools/probe_entities.py` scores this for Wikipedia on
    twenty hand-written probes; here every person in the corpus is a probe, so
    it is a sample rather than a list.

    The last number is why this is worth measuring here at all. Faker draws
    from a few hundred first names and a few hundred surnames, so a corpus of
    10,000 people has many who differ only by a middle initial - and a single
    letter is a poor search term. This is the corpus being harder than
    Wikipedia at the one stage Wikipedia is good at, which is not a flaw to be
    tuned away: it is what a real registry of a closed population looks like.
    """
    import libsearch

    card = libsearch.CardSearch(stem.with_suffix(".IDX"), stem.with_suffix(".DAT"))
    people = [r for (r,) in db.execute(
        "SELECT name FROM person WHERE source = ? ORDER BY name", (SOURCE,))]
    shared = len(people) - len({" ".join(n.split()[::2]) for n in people})

    rng = random.Random(seed)
    first = top3 = missing = 0
    for name in rng.sample(people, min(sample, len(people))):
        hits = card.search(QUESTION.format(s=name.lower()), top=3)
        titles = [card.article(doc)[0] for doc, _ in hits]
        if titles[:1] == [name]:
            first += 1
        elif name in titles:
            top3 += 1
        else:
            missing += 1
    return first, top3, missing, shared


def _slope(points: list[tuple[int, float]]) -> float:
    """Least squares through (hops, cost). Two lines rather than a dependency."""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--card", type=Path, default=REPO / "dist" / "SILO")
    ap.add_argument("--subjects", type=int, default=8,
                    help="Names per generation. The spread is printed, so a "
                         "small number is visibly a small number")
    ap.add_argument("--probe", type=int, default=500,
                    help="People to test the entity lookup on. Runs against "
                         "the index directly, not the emulator, so it is cheap")
    ap.add_argument("--search-card", type=Path,
                    default=REPO / "dist" / "SILOSEARCH",
                    help="A card built without --relations, over the same "
                         "corpus. Running the same questions against it is "
                         "what turns the cost split from an argument into a "
                         "measurement. Skipped if it is not there")
    args = ap.parse_args()

    if not args.card.with_suffix(".GRF").exists():
        raise SystemExit(f"no graph card at {args.card}.GRF\n"
                         f"  python data/silo/buildcard.py")
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cohorts = subjects(db, args.subjects)

    total = args.probe
    first, top3, missing, shared = entity_lookup(args.card, db, total, seed=0)
    db.close()
    print(f"\nentity lookup, {total:,} people asked about by name")
    print(f"  first        {first / total:>6.1%}   <- the only one an oracle "
          f"can use; the walk only follows the top hit")
    print(f"  in the top 3 {(first + top3) / total:>6.1%}")
    print(f"  not found    {missing / total:>6.1%}")
    print(f"  {shared:,} of the 10,000 share a first and last name with "
          f"somebody else,\n  and differ only by a middle initial - which is "
          f"one search term of one letter.")

    import libgraph
    import libgraphcard

    limit = libgraph.CLIMB_LIMIT
    edges = libgraphcard.CardGraph(args.card.with_suffix(".GRF")).num_edges
    print(f"\n{QUESTION.format(s='<name>')}, {args.subjects} names per "
          f"generation\nhop limit {limit}, {args.card}\n")
    print(f"  {'hops':>4}{'instructions':>15}{'+/-':>9}{'card bytes':>12}"
          f"{'+/-':>7}{'answered':>10}")

    started = time.monotonic()
    fit: list[tuple[int, float]] = []
    fit_bytes: list[tuple[int, float]] = []
    for generation, names in sorted(cohorts.items()):
        rows = measure(args.card, names)
        instructions = [r[0] for r in rows]
        io_bytes = [r[1] for r in rows]
        # A climb that completed prints a founder's name and a full stop; one
        # that ran out of hops falls back to the article list, which prints a
        # title and no punctuation. That is the only difference on the screen,
        # which is rather the point - a machine with gaps has to say so.
        answered = sum(1 for _, _, said in rows if said.endswith("."))
        spread = statistics.pstdev(instructions) if len(instructions) > 1 else 0
        byte_spread = statistics.pstdev(io_bytes) if len(io_bytes) > 1 else 0
        note = "  <- past the hop limit" if generation >= limit else ""
        print(f"  {generation:>4}{statistics.mean(instructions):>15,.0f}"
              f"{spread:>9,.0f}{statistics.mean(io_bytes):>12,.0f}"
              f"{byte_spread:>7,.0f}{answered:>7}/{len(rows)}{note}")
        if generation < limit:
            fit.append((generation, statistics.mean(instructions)))
            fit_bytes.append((generation, statistics.mean(io_bytes)))

    whole = fit[-1][1]
    per_hop, per_hop_bytes = _slope(fit), _slope(fit_bytes)
    print(f"\n  a hop moves about {per_hop_bytes:,.0f} bytes off the card - a "
          f"binary search over\n  {edges:,} fixed-width records is "
          f"{ceil(log2(edges))} probes of 7 bytes, and that is what this is.")
    print(f"\n  In instructions it is {per_hop:,.0f}, which is a slope over "
          f"five hop counts and\n  smaller than the spread of any single row "
          f"above. Do not quote it as a\n  constant; quote that a hop is under "
          f"{per_hop * 2 / whole:.1%} of what a question costs.")

    if args.search_card and args.search_card.with_suffix(".IDX").exists():
        plain = measure(args.search_card, cohorts[3][:args.subjects])
        search = statistics.mean(r[0] for r in plain)
        print(f"\n  the same questions on a card built without --relations, "
              f"which searches\n  and does not classify or walk: "
              f"{search:,.0f} instructions.")
        print(f"    search      {search / whole:>6.1%}")
        print(f"    classifier  {(whole - search) / whole:>6.1%}   "
              f"one forward pass, 85,760 two-bit weights")
        print(f"    the walk    {per_hop * 4 / whole:>6.1%}   four hops")
        classifier = whole - search
        print(f"  The graph is the cheap part by a factor of "
              f"{classifier / (per_hop * 4):.0f}. What the card pays for\n"
              f"  is deciding which question it was asked, not answering it.")
    print(f"\n  {time.monotonic() - started:.0f}s in the emulator\n")


if __name__ == "__main__":
    main()
