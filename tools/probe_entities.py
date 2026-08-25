#!/usr/bin/env python3
"""
Score the entity lookup on questions, against a built card.

    python tools/probe_entities.py --card dist/WIKI

Stage one of the oracle is "which article is this question about", and its
errors are the ones with no symptom: the walk answers correctly about the wrong
subject, and what comes back is fluent and wrong. The graph cannot be blamed
and the output looks fine.

So the probes are *questions*, not search terms. A search engine is judged on
whether the right article is somewhere in the top three; an oracle needs it
first, because only the first is walked.

The README has quoted "eleven of thirteen probe queries" since it was written,
from a measurement that lived nowhere. This is that measurement as code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import libsearch

#: (question, the article it is about). Chosen to cover the shapes that go
#: wrong rather than the ones that work: an entity whose name is a prefix of a
#: longer title, one that shares a word with somewhere unrelated, and one whose
#: article is long enough for BM25 to penalise.
PROBES = [
    ("who wrote hamlet", "Hamlet"),
    ("where was marie curie born", "Marie Curie"),
    ("where was napoleon born", "Napoleon"),
    ("who was born in edinburgh", "Edinburgh"),
    ("what did jane austen write", "Jane Austen"),
    ("who directed jaws", "Jaws (movie)"),
    ("what country is warsaw in", "Warsaw"),
    ("where did napoleon die", "Napoleon"),
    ("what language is spoken in brazil", "Brazil"),
    ("who was albert einstein", "Albert Einstein"),
    ("where is mount everest", "Mount Everest"),
    ("what is photosynthesis", "Photosynthesis"),
    ("who invented the telephone", "Telephone"),
    ("what is the zilog z80", "Z80"),
    ("where is the eiffel tower", "Eiffel Tower"),
    ("who painted the mona lisa", "Mona Lisa"),
    ("what is a black hole", "Black hole"),
    ("where was william shakespeare born", "William Shakespeare"),
    ("what country is berlin in", "Berlin"),
    ("who was nelson mandela", "Nelson Mandela"),
]


def probe(card: libsearch.CardSearch, verbose: bool) -> tuple[int, int, int]:
    """(first, top three, missing) over the probe set."""
    first = top3 = missing = 0
    for question, wanted in PROBES:
        hits = card.search(question, top=3)
        titles = [card.article(doc)[0] for doc, _score in hits]
        if titles[:1] == [wanted]:
            first += 1
            mark = "  "
        elif wanted in titles:
            top3 += 1
            mark = "~ "
        else:
            missing += 1
            mark = "X "
        if verbose:
            print(f"  {mark}{question:<38}{' | '.join(titles[:3])}")
    return first, top3, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", default="dist/WIKI",
                        help="Card stem; .IDX and .DAT are read")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stem = Path(args.card)
    card = libsearch.CardSearch(stem.with_suffix(".IDX"), stem.with_suffix(".DAT"))
    print(f"{card.num_docs:,} documents, {len(PROBES)} probes\n")

    first, top3, missing = probe(card, not args.quiet)
    total = len(PROBES)
    print(f"\n  first          {first:>3}/{total}  {first / total:>6.0%}"
          f"   <- the only one an oracle can use")
    print(f"  in the top 3   {first + top3:>3}/{total}  "
          f"{(first + top3) / total:>6.0%}")
    print(f"  not found      {missing:>3}/{total}  {missing / total:>6.0%}")


if __name__ == "__main__":
    main()
