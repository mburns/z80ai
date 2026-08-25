#!/usr/bin/env python3
"""
Ask the archive a question.

    python oracle.py --relations relations.npz "where was jane austen born"
    python oracle.py --relations relations.npz          # interactive
    python oracle.py --relations relations.npz --evaluate data/questions/held-out.txt

Three parts, each doing only what it measurably wins at:

    which entity      the BM25 search index over titles and redirects
    which relation    the phrasebook classifier, over crowdsourced questions
                      and a handful of written multi-hop ones - see
                      data/questions/relations.py for which is which
    the answer        a walk over libgraph's edges - lookups, no reading

It is wrong often, in two different ways, and both are worth knowing about.

A two-hop question has to survive the classifier recognising that it *is* two
hops (about 50% on phrasings the model never saw) and then a graph that can
actually complete the walk (77%, since the graph learned to read value
templates and categories). Those compound. This is not a machine that answers
chained questions reliably; it is a machine that sometimes does, and can tell
you when it did.

"About 50%" is deliberately vague: over five seeds the same measurement ranges
from 43.1% to 56.2% on its 320 held-out questions, so a decimal place here
would be reporting the seed. `data/questions/relations.py` has the numbers.

What it does with the failures is the point. A broken walk reports what it
*did* learn, so the machine says "Edinburgh. The archive does not record what
contains it" rather than "I don't know" - the difference between a machine with
gaps and one that is merely unreliable.

Run it with --plain to see the mechanism instead of the voice.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import libgraph
import libinfer
import liboracle
import libsearch

DB_PATH = Path(__file__).resolve().parent / "data" / "simple_english_wikipedia.db"


def load(db_path: Path, relations: Path | None, card: Path | None,
         source: str) -> liboracle.Oracle:
    if not db_path.exists():
        raise SystemExit(
            f"no database at {db_path}\n"
            f"  python data/wikipedia/ingest.py <pages-articles.xml.bz2>")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    model = libinfer.Model.load(str(relations)) if relations else None
    if model is not None and model.phrases is None:
        raise SystemExit(f"{relations} is not a phrasebook model; train one "
                         f"with classify.py on data/questions/relations.py")

    search: libsearch.CardSearch | _DatabaseSearch
    if card:
        search = libsearch.CardSearch(card.with_suffix(".IDX"),
                                      card.with_suffix(".DAT"))
    else:
        search = _DatabaseSearch(db, source)
    return liboracle.Oracle(db, source=source, relations=model, search=search)


#: Words that name a relation rather than an entity. "What is the capital of
#: France" contains two article titles - `Capital`, a newspaper, and `France` -
#: and the positional match below would take the first. BM25 has no such
#: problem, because idf discounts a word that appears in the frame of thousands
#: of questions; this list is a poor stand-in for idf that happens to be exact
#: in this domain, since the frame words *are* the relation vocabulary.
FRAME = {part
         for fields in libgraph.CANONICAL.values()
         for field in fields
         for part in field.split("_")} | {
    "what", "which", "who", "where", "when", "the", "was", "were", "did",
    "does", "born", "died", "made", "wrote", "from", "and", "for", "with",
}


class _DatabaseSearch:
    """Entity lookup straight from the database, for when no card is built.

    The card's index is what an Agon would use; this is the same idea over
    SQL, so the oracle can be exercised without building 38MB of index first.
    Title match through redirects rather than BM25 - enough to find an entity
    a question names, and honest about being less than the real thing.
    """

    def __init__(self, db: sqlite3.Connection, source: str) -> None:
        self.db = db
        self.source = source

    def search(self, question: str, top: int = 3) -> list[tuple[int, int]]:
        words = [w for w in question.replace("?", " ").split() if len(w) > 2]
        # Longest phrases first: "jane austen" should beat "jane".
        for size in range(min(6, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                span = words[start:start + size]
                # Rejected as a whole rather than word by word, so "pride and
                # prejudice" survives the "and" in the middle of it.
                if all(w.lower() in FRAME for w in span):
                    continue
                phrase = " ".join(span)
                row = self.db.execute(
                    "SELECT id, title FROM article WHERE source = ? "
                    "AND title = ? COLLATE NOCASE", (self.source, phrase)
                ).fetchone()
                if row is None:
                    row = self.db.execute(
                        "SELECT a.id, a.title FROM redirect r JOIN article a "
                        "ON a.source = r.source AND a.title = r.target "
                        "WHERE r.source = ? AND r.title = ? COLLATE NOCASE",
                        (self.source, phrase)).fetchone()
                if row:
                    return [(row[0], size)]
        return []

    def article(self, doc: int) -> tuple[str, str]:
        row = self.db.execute(
            "SELECT title, lead FROM article WHERE id = ?", (doc,)).fetchone()
        return (row[0], row[1]) if row else ("", "")


def answer(oracle: liboracle.Oracle, question: str, plain: bool) -> None:
    response = oracle.ask(question)
    if plain:
        print(f"  subject   {response.subject}")
        print(f"  relations {' -> '.join(response.relations) or '-'}")
        print(f"  path      {' -> '.join(response.path) or '-'}")
        print(f"  kind      {response.kind}"
              + (f", stopped at {response.missing}" if response.missing else ""))
        print(f"  value     {response.value}")
    else:
        print(liboracle.speak(response))


def evaluate(oracle: liboracle.Oracle, path: Path) -> None:
    """Score `question|expected` pairs, reporting how each kind of answer did.

    Separated by kind, because an oracle that answers 30% with facts and falls
    back the rest is a different machine from one that answers 30% and guesses.
    """
    import libdata

    pairs = libdata.read_files([str(path)])
    kinds: dict[str, list[bool]] = {}
    for question, expected in pairs:
        response = oracle.ask(question)
        got = (response.value or "").lower()
        kinds.setdefault(response.kind, []).append(expected.lower() in got)

    total = sum(len(v) for v in kinds.values())
    print(f"{total:,} questions\n")
    print(f"  {'kind':<10}{'share':>8}{'right':>8}")
    for kind in (liboracle.FACT, liboracle.PARTIAL, liboracle.SEARCH,
                 liboracle.UNKNOWN):
        hits = kinds.get(kind, [])
        if hits:
            print(f"  {kind:<10}{len(hits) / total:>7.1%}"
                  f"{sum(hits) / len(hits):>8.1%}")
    right = sum(sum(v) for v in kinds.values())
    print(f"\n  overall   {right / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="Ask once and exit")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="simplewiki")
    parser.add_argument("--relations", type=Path,
                        help="Phrasebook model mapping questions to relations")
    parser.add_argument("--card", type=Path,
                        help="Card stem, to use the real BM25 index")
    parser.add_argument("--plain", action="store_true",
                        help="Show the mechanism rather than the voice")
    parser.add_argument("--evaluate", type=Path,
                        help="Score a question|answer file instead")
    args = parser.parse_args()

    oracle = load(args.db, args.relations, args.card, args.source)

    if args.evaluate:
        evaluate(oracle, args.evaluate)
        return
    if args.question:
        answer(oracle, " ".join(args.question), args.plain)
        return

    print("Ask the archive. Blank line to leave.\n")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        answer(oracle, question, args.plain)
        print()


if __name__ == "__main__":
    main()
