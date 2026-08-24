#!/usr/bin/env python3
"""
Ingest MetaQA into the same database, as a second source.

    python data/wikipedia/metaqa.py ~/Downloads/MetaQA

MetaQA is a movie knowledge base with question sets at one, two and three hops.
It is here for two things Simple English Wikipedia cannot supply:

  a graph that was built as a graph.  Its 134,741 triples use entity names on
  both sides, so a hop always lands on something with edges of its own.
  Wikipedia's infobox values are display strings written for a reader, and even
  after collapsing the property synonyms only about 45% of two-hop chains
  complete.  MetaQA is the control that says how much of that is the corpus.

  question sets that are explicitly multi-hop, with the path each question
  follows.  That is the supervision a chain-classifier needs and nothing else
  here has.

## Read the question data with care

MetaQA's questions are **generated from templates** - the paper describes 21
question types at two hops and 15 at three, with ten text templates each. A
classifier trained on them is learning ten phrasings per class, and will score
far above what it would manage on questions people actually wrote.

This project has been burned by exactly that: `examples/parser` scored 99.8% on
generated command phrasings and 44.7% when held out by verb. So the honest use
of MetaQA here is its **knowledge base** and its **path taxonomy** - both real -
while accuracy figures for relation classification come from SimpleQuestions,
whose questions are human-written (85.6% macro over 44 relations).

## Getting it

The dataset is a Google Drive folder linked from
https://github.com/yuyuz/MetaQA, and is not fetchable without a browser: the
folder listing refuses anonymous access, and the WikiMovies archive it derives
from returns 404. Download it by hand, unpack it, and point this at the
directory. It expects the layout the README describes:

    MetaQA/
      kb.txt                     subject|relation|object, one per line
      1-hop/vanilla/qa_train.txt question<TAB>answer1|answer2
      2-hop/vanilla/qa_train.txt
      3-hop/vanilla/qa_train.txt
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ingest import DB_PATH, connect

SOURCE = "metaqa"

#: MetaQA's relations are already canonical - it was built as a graph - so they
#: pass through untouched rather than going via libgraph.CANONICAL, which
#: exists to repair Wikipedia's folksonomy.
KB_NAME = "kb.txt"
HOPS = (1, 2, 3)


def read_kb(root: Path) -> list[tuple[str, str, str]]:
    path = root / KB_NAME
    if not path.exists():
        raise SystemExit(f"no {KB_NAME} in {root}\n{__doc__.split('## Getting it')[1]}")

    triples = []
    for line in path.open(encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("|")
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def read_questions(root: Path, hops: int) -> list[tuple[str, list[str]]]:
    """(question, answers) for one hop count, if that split is present."""
    path = root / f"{hops}-hop" / "vanilla" / "qa_train.txt"
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8", errors="replace"):
        question, _, answers = line.rstrip("\n").partition("\t")
        if question and answers:
            out.append((question, answers.split("|")))
    return out


def ingest(db: sqlite3.Connection, root: Path) -> dict[str, Any]:
    triples = read_kb(root)
    if not triples:
        raise SystemExit(f"{root / KB_NAME} held no triples")

    subjects = {s for s, _, _ in triples}
    objects = {o for _, _, o in triples}
    entities = subjects | objects

    with db:
        # Every entity is an article, so the graph and the search index agree
        # about what exists. MetaQA has no prose, so the lead stays empty.
        db.execute("DELETE FROM article WHERE source = ?", (SOURCE,))
        db.executemany("INSERT INTO article (source, title, lead) VALUES (?, ?, '')",
                       [(SOURCE, e) for e in sorted(entities)])

        db.execute("DELETE FROM fact WHERE source = ?", (SOURCE,))
        db.executemany(
            "INSERT OR REPLACE INTO fact (source, subject, property, value) "
            "VALUES (?, ?, ?, ?)",
            [(SOURCE, s, r, o) for s, r, o in triples])

        # Straight to edges: these relations need no canonicalising, and every
        # object is an entity, so nothing is dropped for naming no article.
        db.execute("DELETE FROM edge WHERE source = ?", (SOURCE,))
        db.executemany("INSERT OR REPLACE INTO edge VALUES (?, ?, ?, ?)",
                       [(SOURCE, s, r, o) for s, r, o in triples])

    linkable = sum(1 for _, _, o in triples if o in subjects)
    stats = {
        "triples": len(triples),
        "entities": len(entities),
        "relations": len({r for _, r, _ in triples}),
        "linkable": linkable / len(triples),
        "questions": {h: len(read_questions(root, h)) for h in HOPS},
    }

    with db:
        for key, value in (
            (f"{SOURCE}.triples", str(stats["triples"])),
            (f"{SOURCE}.entities", str(stats["entities"])),
            (f"{SOURCE}.relations", str(stats["relations"])),
            (f"{SOURCE}.source", "https://github.com/yuyuz/MetaQA"),
        ):
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Unpacked MetaQA directory")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    db = connect(args.db, migrate=True)
    stats = ingest(db, args.root)

    print(f"{stats['triples']:,} triples, {stats['entities']:,} entities, "
          f"{stats['relations']} relations")
    print(f"{stats['linkable']:.1%} of objects are also subjects "
          f"(Simple English Wikipedia manages 19%)")
    for hops, n in stats["questions"].items():
        if n:
            print(f"  {hops}-hop: {n:,} questions")
    print("\nQuestions are template-generated - ten phrasings per type - so "
          "train\nrelation classifiers on SimpleQuestions and use these for the "
          "path taxonomy.")

    db.execute("ANALYZE")
    db.commit()


if __name__ == "__main__":
    main()
