#!/usr/bin/env python3
"""
Build a question -> relation training set from SimpleQuestions.

    python data/questions/relations.py > relations.txt
    python classify.py --file relations.txt -o relations.npz --balance

`--balance` is not optional in spirit. The multi-hop classes are five times
rarer than the one-hop ones, and without it the model answers them from the
prior - see the note on CHAINS below, which is the whole story of this file.

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
#: two, the model at first dropped the *second hop*: `created_by born_in`
#: became `created_by`, `died_in in_country` became `died_in`. It recognised
#: the relation and lost the instruction to keep going.
#:
#: The cause was not the templates. Sweeping the number of training phrasings
#: from one to six moved the score 20.3% -> 39.4%, so writing more of them
#: would have bought a few points at best. The cause was the class prior: four
#: chain classes bringing 240 examples each against 1,200 for every one-hop
#: class, so dropping the second hop was simply the better bet. Weighting the
#: loss by inverse class frequency - `classify.py --balance` - fixes it:
#:
#:     unweighted              40.3%      one-hop macro 92.4%
#:     class-weighted          64.7%      one-hop macro 90.3%
#:
#: ## How much of that is the seed
#:
#: More than is comfortable, and it is worth knowing before any of the numbers
#: on this page are read as precise. Held-out chain accuracy over five seeds,
#: same data, same flags, `--seed N --split-seed N`:
#:
#:     50.3%  43.1%  56.2%  48.4%  52.5%      mean 50.1%, spread 13.1 points
#:
#: 320 questions over four classes, so one class getting four more right moves
#: the figure a point and a quarter. Every documented value of this metric in
#: the repository - 43.8% in `oracle.py`, 51.6% in `liboracle.py` - sits inside
#: that spread, and 51.6% reproduces to the decimal at seed 0 on the corpus it
#: was taken from. They were never in conflict; they were one number quoted to
#: a precision it does not have.
#:
#: So read these as one-decimal figures at seed 0 and compare them only with
#: each other. A change worth believing has to move the mean of several seeds,
#: which is also why the comparisons below - all single-seed - say which
#: repairs lose and not by how much.
#:
#: Three other repairs were tried and are not used, each rejected by
#: measurement rather than by taste:
#:
#:     duplicating chain rows x4          50.6%   loses to weighting the loss
#:     order-sensitive encoding (5 bands) 46.6%   costs 7.6pts of one-hop macro
#:     anchor head x continuation head    52.5%   two models, still loses
#:
#: Masking the entity was tried too, on the theory that a name says nothing
#: about which relation is being asked. It scored 36.6% and cost 8.8 points of
#: one-hop macro, because the theory is wrong: the entity says a great deal.
#: "What country is X in" is only a place question because X is a place, and
#: with X removed `in_country` collapsed to 0%.
#:
#: The band row above is the short version of a longer look, which reached the
#: same place by a different route. The hypothesis was that seeding each
#: trigram's hash with where it appeared (ENCODING.md) would keep the hops in
#: order, but the class set has no reversed paths - no `located_in born_in`
#: class exists - so word order carries no signal these classes need, while
#: the banding tax on the buckets still applies. Measured under this protocol,
#: two seeds: flat 20.6% / 24.7% macro, 8 bands 11.9% / 17.5%, and bands lose
#: on the random split too (95.2% -> 89.3%). (Stand-in entity names, not the
#: corpus's - the mechanism under test is the encoder, and the deltas are
#: paired, but re-run against the real database before quoting.) The second
#: hop is dropped because it is *quiet*, not because it is out of order.
#:
#: That re-run has since happened, on `data/silo/` - a corpus with real entity
#: names, twenty paths and the same property of having no reversed class - and
#: under the multi-seed protocol this note asks for. Bands lose monotonically:
#: 45.8% flat, 35.6% at two bands, 29.5% at four, 21.5% at eight, with
#: three-seed spreads that do not overlap. It also makes the classifier *more*
#: sensitive to which entity is being asked about, not less.
#:
#: So chain questions work about as well as one coin flip. That is enough to be
#: interesting on a machine like this and not enough to trust.
#:
#: Entity names are drawn from the corpus rather than invented, so the parts of
#: the question that carry the rare words are real even where the frame is not.
CHAINS: dict[str, tuple[str, ...]] = {
    "born_in in_country": (
        "what country was {s} born in",
        "which country was {s} born in",
        "what nation was {s} born in",
        "{s} was born in which country",
        "what country is {s} originally from",
        "in what country was {s} born",
        "which country did {s} come from",
        "what country was {s} a native of",
    ),
    "died_in in_country": (
        "what country did {s} die in",
        "which country did {s} die in",
        "what nation did {s} die in",
        "{s} died in which country",
        "in what country did {s} die",
        "what country was {s} in when they died",
        "which country did {s} pass away in",
        "what country did {s} spend their last days in",
    ),
    "in_country": (
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
        # A climb names the type it wants, not the edge it walks, so the
        # entities worth asking about are the ones with the underlying edge.
        first = path.split()[0]
        first = libgraph.CLIMB[first][0] if first in libgraph.CLIMB else first
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
        f"{r}_of" for r in libgraph.CANONICAL} | set(CHAINS) | set(libgraph.CLIMB)
    unknown = set(counts) - walkable
    assert not unknown, f"relations libgraph cannot walk: {unknown}"

    for relation, n in counts.most_common():
        print(f"  {relation:<16} {n:>5,}", file=sys.stderr)


if __name__ == "__main__":
    main()
