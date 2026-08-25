#!/usr/bin/env python3
"""
Does the classifier answer the question, or answer the name in it?

    python tools/name_sensitivity.py --model relations.npz
    python tools/name_sensitivity.py --model dist/silo-relations.npz --corpus silo

Stage two of the oracle turns a question into a relation path. It is scored the
way classifiers are scored - accuracy over a question set - and that number
cannot separate two failures which need different fixes:

    a phrasing the model never learned         more phrasings, or better ones
    a phrasing whose answer depends on *who*   the encoder, and nothing else

The second is invisible in an accuracy figure and is the one this measures. A
query is hashed into 128 trigram buckets and a name is most of a short
question, so the subject is not something the model steps over on its way to
the verb - it is the bulk of the input. Hold the phrasing fixed, vary only the
name, and see whether the answer moves.

This was found on the synthetic silo corpus, where 116 of 240 phrasings the
model had been trained on gave different answers for different people. The
question that mattered was whether it is a property of that corpus - whose
names come from a small pool - or of the encoder. `--corpus simplewiki` is that
question, asked of the real thing.

## What it can and cannot cover on Wikipedia

Only the templated classes. `data/questions/relations.py` trains its one-hop
classes on SimpleQuestions - real questions written by people - and there are
no templates behind them to hold fixed. The four `CHAINS` paths are templated,
eight phrasings each, so those are what this varies. That is a smaller sample
than the silo's, and it is the part of the vocabulary the repository already
reports as the weak one.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import libdata
import libgraph
import libinfer

REPO = Path(__file__).resolve().parent.parent


def wikipedia(db: sqlite3.Connection) -> tuple[dict[str, tuple[str, ...]], str]:
    sys.path.insert(0, str(REPO / "data" / "questions"))
    import relations

    return dict(relations.CHAINS), "simplewiki"


def silo(db: sqlite3.Connection) -> tuple[dict[str, tuple[str, ...]], str]:
    sys.path.insert(0, str(REPO / "data" / "silo"))
    import generate  # noqa: F401  - registers the silo's climbs
    import relationpaths

    return dict(relationpaths.PATHS), "silo"


def picker(db: sqlite3.Connection, source: str) -> object:
    """A subject drawer that only offers entities the path applies to.

    A question about where somebody was born is not a question about a river,
    and asking it anyway measures the corpus rather than the classifier.
    """
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (source,))}
    cache: dict[str, list[str]] = {}

    def first_relation(label: str) -> tuple[str, bool]:
        word = label.split()[0]
        if word in libgraph.CLIMB:
            return libgraph.CLIMB[word][0], False
        if word in have:
            return word, False
        if word.endswith("_of") and word[:-3] in have:
            return word[:-3], True
        return word, False

    def draw(label: str, wanted: int) -> list[str]:
        if label not in cache:
            relation, inverse = first_relation(label)
            column = "object" if inverse else "subject"
            cache[label] = [r for (r,) in db.execute(
                f"SELECT DISTINCT {column} FROM edge "
                "WHERE source = ? AND relation = ? LIMIT 4000",
                (source, relation))]
        rows = cache[label]
        return [rows[i % len(rows)].lower() for i in range(wanted)] if rows else []

    return draw


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="A phrasebook classifier over relation paths")
    ap.add_argument("--corpus", choices=("simplewiki", "silo"),
                    default="simplewiki")
    ap.add_argument("--db", type=Path,
                    help="Defaults to the corpus's database")
    ap.add_argument("--per-template", type=int, default=40,
                    help="Subjects per phrasing")
    ap.add_argument("--accum-bits", type=int, default=24, choices=(16, 24))
    args = ap.parse_args()

    default_db = (REPO / "data" / "silo.db" if args.corpus == "silo"
                  else REPO / "data" / "simple_english_wikipedia.db")
    db_path = args.db or default_db
    if not db_path.exists():
        raise SystemExit(f"no database at {db_path}")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    templates, source = (silo(db) if args.corpus == "silo" else wikipedia(db))
    model = libinfer.Model.load(str(args.model))
    draw = picker(db, source)

    result = libdata.name_sensitivity(
        templates, draw,  # type: ignore[arg-type]
        lambda q: libinfer.classify(model, q, args.accum_bits),
        per_template=args.per_template)
    db.close()

    print(f"\n{args.model}  over {source}, {len(templates)} paths, "
          f"{result.phrasings} phrasings, {args.per_template} subjects each\n")
    print(f"  {result.accuracy:>6.1%}  of {result.asked:,} questions route to "
          f"the right path")
    print(f"  {result.steadiness:>6.1%}  of phrasings ({result.steady}/"
          f"{result.phrasings}) answer the same way whatever the subject")
    if result.worst:
        print("\n  least steady:")
        for label, phrasing, share, instead in result.worst[:8]:
            print(f"    {share:>6.1%}  {phrasing.format(s='X'):<46} "
                  f"{label} -> mostly {instead}")
    print()


if __name__ == "__main__":
    main()
