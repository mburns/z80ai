#!/usr/bin/env python3
"""
Build a question -> relation training set from SimpleQuestions.

    python data/questions/relations.py > relations.txt
    python classify.py --file relations.txt -o relations.npz

The oracle has to decide *which relation a question is asking about* before it
can walk the graph. That is a classification problem the phrasebook head
already handles, and this is the only part of an oracle for which real,
human-written training data exists.

## Why not MetaQA

MetaQA has the multi-hop question sets, which is what this most obviously
wants. But its questions are generated - the paper describes 21 question types
at two hops and 15 at three, with ten text templates each - so a classifier
trained on them learns ten phrasings per class and reports an accuracy nobody
should believe. `examples/parser` scored 99.8% on generated command phrasings
and 44.7% when held out by verb; this project has paid that price once.

SimpleQuestions is crowdsourced, so its phrasings vary the way real ones do:
"where in germany was rudi ball born in?" and "irvin shapiro passed away in
which city?" are the same relation asked two ways nobody would template.

## The mapping

SimpleQuestions labels Wikidata properties; libgraph names relations after what
an infobox calls them. Several properties mean one relation - a director, a
composer and an author are all `created_by` as far as a walk is concerned - so
the mapping collapses them, which also lifts a handful of thin classes over the
~150-example floor that every measurement in this project runs into.

Prefixed `R` means the inverse: `R19` asks "who was born in Berlin" where `P19`
asks "where was X born". Inverses are kept as their own classes, because the
walk they need is the inverse index rather than the forward one.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import libgraph

#: The Wikidata mapping of SimpleQuestions, which is a plain TSV in a public
#: repository - unlike SimpleQuestions itself, whose Freebase ids would need a
#: separate dump to interpret.
SOURCE_URL = ("https://raw.githubusercontent.com/askplatypus/"
              "wikidata-simplequestions/master/annotated_wd_data_%s.txt")

#: Wikidata property -> the relation libgraph walks. Several map onto one:
#: a director, a composer and an author are all `created_by` to a traversal.
PROPERTY_RELATION = {
    "P19": "born_in",
    "P20": "died_in",
    "P17": "located_in",
    "P131": "located_in",
    "P36": "capital_is",
    "P26": "spouse_of",
    "P136": "genre_is",
    "P364": "language_is",
    "P57": "created_by",     # director
    "P50": "created_by",     # author
    "P86": "created_by",     # composer
    "P162": "created_by",    # producer
    "P170": "created_by",    # creator
    "P676": "created_by",    # lyricist
    "P84": "created_by",     # architect
    "P178": "created_by",    # developer
}

#: Below this a class has nothing to generalise from - the floor every
#: measurement in this project has run into.
MIN_EXAMPLES = 150

#: And a ceiling, so one relation does not become the model's default answer.
MAX_EXAMPLES = 1200

#: Two-hop paths worth asking about, and the phrasings that ask for them.
#:
#: These are templates, and templates are exactly what the module docstring
#: warns about. They are here because no crowdsourced corpus of multi-hop
#: questions over *Wikipedia* exists - MetaQA's are over a movie KB and are
#: themselves generated - and without path classes the classifier can only ever
#: emit one relation, so `libgraph.follow` never gets a chain to walk.
#:
#: The mitigation is to never report a number that a template can flatter.
#: `--held-out-templates` trains on some phrasings and scores on phrasings the
#: model has not seen, which is the only question worth asking of generated
#: data: not "can it recognise what we wrote" but "does what we wrote teach it
#: anything". `examples/parser` learned this the expensive way - 99.8% on
#: generated command phrasings, 44.7% held out by verb.
#:
#: Measured here, training on six phrasings per path and scoring on the other
#: two: **92.5% macro on a random split, 43.8% on unseen phrasings.** The same
#: collapse, to within a point of the same number. Per path:
#:
#:     located_in located_in   86.2%
#:     born_in located_in      41.2%
#:     died_in located_in      28.7%
#:     created_by born_in      18.8%
#:
#: The failure mode is specific and worth knowing: it drops the *second hop*.
#: `created_by born_in` becomes `created_by`, `died_in located_in` becomes
#: `died_in`. The model recognises the relation and loses the instruction to
#: keep going, because the words carrying "and then keep going" are the short,
#: templated part of the phrase - exactly the part a new phrasing changes.
#: `located_in located_in` survives because its eight phrasings are all near
#: neighbours of "what country is X in".
#:
#: So chain questions work, and they work about as well as one coin flip. That
#: is enough to be interesting on a machine like this and not enough to trust.
#:
#: Entity names are drawn from the corpus rather than invented, so the parts of
#: the question that carry the rare words are real even where the frame is not.
CHAINS: dict[str, tuple[str, ...]] = {
    "born_in located_in": (
        "what country was {s} born in",
        "which country was {s} born in",
        "what nation was {s} born in",
        "{s} was born in which country",
        "what country is {s} originally from",
        "in what country was {s} born",
        "which country did {s} come from",
        "what country was {s} a native of",
    ),
    "died_in located_in": (
        "what country did {s} die in",
        "which country did {s} die in",
        "what nation did {s} die in",
        "{s} died in which country",
        "in what country did {s} die",
        "what country was {s} in when they died",
        "which country did {s} pass away in",
        "what country did {s} spend their last days in",
    ),
    "located_in located_in": (
        "what country is {s} in",
        "which country is {s} part of",
        "what nation contains {s}",
        "{s} belongs to which country",
        "in what country would i find {s}",
        "what larger country is {s} within",
        "which country does {s} sit in",
        "what country is {s} located in",
    ),
    "created_by born_in": (
        "where was the writer of {s} born",
        "where was the creator of {s} born",
        "what place was the author of {s} born in",
        "who made {s} and where were they born",
        "the person behind {s} was born where",
        "where was the maker of {s} born",
        "birthplace of whoever created {s}",
        "where did the author of {s} come from",
    ),
}


def fetch(split: str, cache: Path) -> list[list[str]]:
    path = cache / f"annotated_wd_data_{split}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = SOURCE_URL % split
        print(f"fetching {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    rows = [line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8")]
    return [r for r in rows if len(r) == 4]


def build(rows: list[list[str]]) -> list[tuple[str, str]]:
    """(question, relation) pairs, balanced and deduplicated."""
    by_relation: dict[str, set[str]] = defaultdict(set)
    for _subject, prop, _obj, question in rows:
        inverse = prop.startswith("R")
        relation = PROPERTY_RELATION.get("P" + prop[1:] if inverse else prop)
        if relation is None:
            continue
        # An inverse asks the same relation from the other end, and needs the
        # inverse index rather than the forward one, so it is its own class.
        by_relation[f"{relation}_of" if inverse else relation].add(question)

    rng = random.Random(0)
    pairs: list[tuple[str, str]] = []
    for relation in sorted(by_relation):            # sorted: reproducible
        questions = sorted(by_relation[relation])
        if len(questions) < MIN_EXAMPLES:
            continue
        if len(questions) > MAX_EXAMPLES:
            questions = rng.sample(questions, MAX_EXAMPLES)
        pairs.extend((q, relation) for q in questions)

    rng.shuffle(pairs)
    return pairs


def subjects(db_path: Path, source: str, relation: str,
             limit: int) -> list[str]:
    """Article titles that actually have ``relation``, for grounding a frame.

    A chain question about an entity the corpus has never heard of teaches the
    classifier nothing useful, so the entity half of every generated question
    comes from the graph it will be asked to walk.
    """
    if not db_path.exists():
        return []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [t for (t,) in db.execute(
            "SELECT subject FROM edge WHERE source = ? AND relation = ? "
            "ORDER BY subject LIMIT ?", (source, relation, limit))]
    finally:
        db.close()


def chains(db_path: Path, source: str, per_template: int,
           hold_out: int = 0) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Generated multi-hop questions, split by *template* rather than by row.

    Holding out whole phrasings is the only split that says anything: a random
    split leaves every template represented in training, so the score measures
    memorisation of eight frames. Returns (train, held-out).
    """
    rng = random.Random(0)
    train: list[tuple[str, str]] = []
    unseen: list[tuple[str, str]] = []
    for path, templates in CHAINS.items():
        first = path.split()[0]
        names = subjects(db_path, source, first, per_template * len(templates))
        if not names:
            continue
        kept, held = templates[:len(templates) - hold_out], templates[len(templates) - hold_out:]
        for group, sink in ((kept, train), (held, unseen)):
            for template in group:
                for name in rng.sample(names, min(per_template, len(names))):
                    sink.append((template.format(s=name.lower()), path))
    rng.shuffle(train)
    rng.shuffle(unseen)
    return train, unseen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path,
                        default=Path(__file__).resolve().parent / "cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve()
                        .parent.parent / "simple_english_wikipedia.db",
                        help="Corpus to draw entity names for chain questions from")
    parser.add_argument("--source", default="simplewiki")
    parser.add_argument("--chains", type=int, default=40, metavar="N",
                        help="Chain questions per template (0 for none)")
    parser.add_argument("--held-out-templates", type=int, default=0, metavar="N",
                        help="Reserve N chain phrasings per path, unseen in training")
    parser.add_argument("--emit", choices=("train", "held-out"), default="train",
                        help="'held-out' emits only the reserved phrasings, to "
                             "score generalisation rather than recall")
    args = parser.parse_args()

    if args.emit == "held-out" and not args.held_out_templates:
        parser.error("--emit held-out needs --held-out-templates N")

    rows = fetch(args.split, args.cache)
    pairs = build(rows)

    if args.chains:
        train, unseen = chains(args.db, args.source, args.chains,
                               hold_out=args.held_out_templates)
        # The held-out set is chain questions only: it exists to answer "does
        # a phrasing we never wrote still route to the right path", and mixing
        # the one-hop classes back in would dilute that into a general score.
        pairs = unseen if args.emit == "held-out" else pairs + train
        random.Random(1).shuffle(pairs)

    counts = Counter(r for _, r in pairs)

    print("# SimpleQuestions mapped to Wikidata, then onto the relations "
          "libgraph walks.")
    print(f"# {len(pairs)} questions over {len(counts)} relations, "
          f"capped at {MAX_EXAMPLES} each.")
    print("# Source: https://github.com/askplatypus/wikidata-simplequestions")
    print("# Questions are crowdsourced, not templated - see the module "
          "docstring for why that matters.")
    for question, relation in pairs:
        clean = question.replace("|", " ").strip()
        if clean:
            print(f"{clean}|{relation}")

    walkable = set(libgraph.CANONICAL) | {
        f"{r}_of" for r in libgraph.CANONICAL} | set(CHAINS)
    unknown = set(counts) - walkable
    assert not unknown, f"relations libgraph cannot walk: {unknown}"

    for relation, n in counts.most_common():
        print(f"  {relation:<16} {n:>5,}", file=sys.stderr)


if __name__ == "__main__":
    main()
