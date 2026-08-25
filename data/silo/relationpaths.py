#!/usr/bin/env python3
"""
Questions about the silo, and the path each one is asking for.

    python data/silo/relationpaths.py > silo-relations.txt
    python data/silo/relationpaths.py --emit held-out --held-out-templates 3 \
           > silo-held-out.txt

The card's oracle is three parts, and this is the middle one. The search index
turns a name into a document; a **phrasebook classifier** turns a question into
a path; the graph walks it. `classify.py` trains the classifier on what this
prints, and `buildwikisearch.py --relations` bakes it into the binary along
with the path table `buildwikigraph.paths_for` derives from these labels.

A label *is* a path: `father_is works_in` is two hops, `child_of_of` is
`child_of` read backwards, and `founding_father` is a climb. Anything this file
emits that the corpus has no edges for becomes an inert row in the card's path
table, so `main()` asserts against the database rather than trusting the list.

## These questions are templated, and that is a problem to be measured

`data/README.md` says it plainly: a `--val-frac` split holds out unique
*queries*, so with templated data a held-out "who is X's father" still has
"who is Y's father" in the training half, and the score measures interpolation
inside a grammar this file wrote. The Wikipedia classifier avoids this for its
one-hop classes by training on SimpleQuestions - real questions, written by
people who had never heard of `libgraph`. There is no crowdsourced question set
about a silo that does not exist.

So the honest number here comes from holding out whole *phrasings*:
`--held-out-templates 3` reserves three wordings per path, and `--emit
held-out` prints only those. A model trained without them and scored on them is
being asked whether it learned the question or the vocabulary. Both numbers are
in `data/silo/README.md`, and the difference between them is the point.

Twelve phrasings per path is few enough that the held-out three are a quarter
of the grammar rather than a rounding error, which is deliberate: the failure
this is trying to catch is a classifier that has memorised "grandfather" and
falls over on "father's father".
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: F401  - registers the silo's climbs into libgraph.CLIMB
from schema import SOURCE

import libgraph

DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"

#: path -> the ways somebody might ask for it. `{s}` is the subject, filled
#: from the corpus so the rare words in a question are real even where the
#: frame is not.
#:
#: Ordered longest-path-last, because that is the order they get hard in: one
#: hop is a lookup with a synonym problem, and three hops is a question whose
#: surface form shares almost no words with the path it means.
PATHS: dict[str, tuple[str, ...]] = {
    "father_is": (
        "who is {s}'s father", "who was {s}'s father", "name {s}'s father",
        "who fathered {s}", "who is the father of {s}", "{s}'s dad is who",
        "which man is {s}'s father", "who sired {s}",
        "tell me {s}'s father", "{s} is the child of which man",
        "whose son or daughter is {s} on the father's side",
        "who is the male parent of {s}",
    ),
    "mother_is": (
        "who is {s}'s mother", "who was {s}'s mother", "name {s}'s mother",
        "who is the mother of {s}", "{s}'s mum is who",
        "which woman is {s}'s mother", "who gave birth to {s}",
        "tell me {s}'s mother", "{s} is the child of which woman",
        "who bore {s}", "who is the female parent of {s}",
        "{s} was mothered by whom",
    ),
    "spouse_of": (
        "who did {s} marry", "who is {s} married to", "who is {s}'s wife",
        "who is {s}'s husband", "name {s}'s spouse", "who is the partner of {s}",
        "{s} is married to whom", "who did {s} wed",
        "tell me who {s} married", "whose husband or wife is {s}",
        "who shares a flat and a name with {s}",
        "who is {s} paired with",
    ),
    "works_in": (
        "which department does {s} work in", "where does {s} work",
        "what department is {s} in", "who does {s} work for",
        "which section employs {s}", "{s} works in what department",
        "name the department of {s}", "which office is {s} attached to",
        "tell me where {s} is employed", "{s} is on the books of which department",
        "what outfit does {s} belong to", "which department has {s}",
    ),
    "job_is": (
        "what does {s} do", "what is {s}'s job", "what is {s}'s trade",
        "what does {s} do for a living", "name {s}'s occupation",
        "what work does {s} do", "{s} is employed as what",
        "what is the title of {s}", "tell me {s}'s trade",
        "what does {s} do all day", "what post does {s} hold",
        "{s} works as a what",
    ),
    "shift_is": (
        "which shift does {s} work", "what shift is {s} on",
        "when does {s} work", "name {s}'s shift", "{s} works which shift",
        "what hours does {s} keep", "which rotation is {s} on",
        "tell me {s}'s shift", "is {s} first second or third shift",
        "what watch does {s} stand", "which shift has {s}",
        "{s} is rostered to which shift",
    ),
    "lives_at": (
        "where does {s} live", "what is {s}'s address",
        "which flat is {s} in", "where is {s}'s apartment",
        "name the address of {s}", "{s} lives where",
        "what is the apartment number of {s}", "tell me where {s} sleeps",
        "which dwelling belongs to {s}", "where would i find {s} at night",
        "what door is {s} behind", "{s} is quartered where",
    ),
    "born_on": (
        "which level was {s} born on", "where was {s} born",
        "what floor was {s} born on", "name the level {s} was born on",
        "{s} was born on which level", "on what level did {s} arrive",
        "tell me where {s} was born", "which floor did {s} come from",
        "what level is {s} native to", "{s}'s birth level is what",
        "where did {s} first draw breath", "which deck was {s} born on",
    ),
    "class_is": (
        "which class was {s} in", "what class did {s} attend",
        "name {s}'s class", "{s} was in which class",
        "which school class did {s} belong to", "what year group was {s} in",
        "tell me {s}'s class", "which class list has {s} on it",
        "what class did {s} sit in", "{s} was schooled with which class",
        "which nursery class was {s} in", "what cohort did {s} study with",
    ),
    "crew_is": (
        "which crew is {s} on", "what crew does {s} serve on",
        "name {s}'s crew", "{s} is on which crew",
        "which work crew has {s}", "what team does {s} work with",
        "tell me {s}'s crew", "which gang does {s} belong to",
        "what crew is {s} rostered to", "{s} works alongside which crew",
        "which crew list has {s}", "what shift team is {s} part of",
    ),
    "father_is father_is": (
        "who is {s}'s grandfather", "who is {s}'s paternal grandfather",
        "who is the father of {s}'s father", "name {s}'s grandfather",
        "{s}'s father's father is who", "who fathered {s}'s father",
        "tell me {s}'s grandfather on the father's side",
        "which man is two generations above {s} on the father's line",
        "who is grandad to {s}", "{s} is the grandchild of which man",
        "who is the father of the father of {s}",
        "name the man whose son fathered {s}",
    ),
    "mother_is mother_is": (
        "who is {s}'s grandmother", "who is {s}'s maternal grandmother",
        "who is the mother of {s}'s mother", "name {s}'s grandmother",
        "{s}'s mother's mother is who", "who bore {s}'s mother",
        "tell me {s}'s grandmother on the mother's side",
        "which woman is two generations above {s} on the mother's line",
        "who is granny to {s}", "{s} is the grandchild of which woman",
        "who is the mother of the mother of {s}",
        "name the woman whose daughter bore {s}",
    ),
    "father_is works_in": (
        "which department does {s}'s father work in",
        "where does {s}'s father work", "what department is {s}'s dad in",
        "name the department of {s}'s father",
        "{s}'s father works where", "who employs {s}'s father",
        "tell me where {s}'s father is employed",
        "which office does the father of {s} report to",
        "what outfit does {s}'s father belong to",
        "the man who fathered {s} works in which department",
        "which department has {s}'s father on its books",
        "what does {s}'s father's department call itself",
    ),
    "spouse_of job_is": (
        "what does {s}'s wife do", "what does {s}'s husband do",
        "what is the trade of {s}'s spouse",
        "name the occupation of whoever {s} married",
        "{s}'s partner works as what",
        "what job does {s}'s spouse hold",
        "tell me the trade of {s}'s husband or wife",
        "the person {s} married does what for a living",
        "what is the title of {s}'s spouse",
        "what work does {s}'s partner do",
        "whoever {s} wed is employed as what",
        "what post does the spouse of {s} hold",
    ),
    "lives_at in_section": (
        "which section does {s} live in", "is {s} up top or down deep",
        "what part of the silo does {s} live in",
        "name the section {s} lives in", "{s} lives in which section",
        "which third of the silo is {s} in",
        "tell me the section of {s}'s home",
        "whereabouts in the silo does {s} sleep",
        "which stretch of levels is {s}'s flat on",
        "{s}'s address falls in which section",
        "what section contains the home of {s}",
        "which run of the silo does {s} live along",
    ),
    "works_in located_in": (
        "which level is {s}'s department on",
        "what floor does {s} work on", "where is {s}'s department",
        "name the level {s} works on", "{s} works on which level",
        "which floor houses the department of {s}",
        "tell me the level of {s}'s workplace",
        "what level would i climb to to find {s} working",
        "the department that employs {s} sits on which floor",
        "which deck is {s}'s office on",
        "what floor is {s}'s workplace on",
        "{s}'s department is headquartered where",
    ),
    "founding_father": (
        "which founder is {s} descended from",
        "who is {s}'s founding ancestor",
        "trace {s} back to a founder",
        "name the founder in {s}'s male line",
        "{s} descends from which founder",
        "which original settler fathered {s}'s line",
        "tell me the founder {s} comes from",
        "who started the line that ends at {s}",
        "which of the first generation is {s} descended from",
        "{s}'s family goes back to which founder",
        "name the founding man behind {s}",
        "which founder began {s}'s father's line",
    ),
    "child_of_of": (
        "who are {s}'s children", "name a child of {s}",
        "who did {s} raise", "{s} is the parent of whom",
        "which children does {s} have", "who calls {s} their parent",
        "tell me a child of {s}", "who was born to {s}",
        "name the offspring of {s}", "{s} has which children",
        "who are the sons and daughters of {s}",
        "which people list {s} as a parent",
    ),
    "class_is class_is_of": (
        "who was in {s}'s class", "name a classmate of {s}",
        "who was schooled with {s}", "who sat in class with {s}",
        "which pupils shared {s}'s class",
        "tell me somebody from {s}'s class",
        "who studied alongside {s}", "who else was in the class of {s}",
        "name another pupil from {s}'s year group",
        "which children learned beside {s}",
        "who shared a class list with {s}",
        "{s} went to school with whom",
    ),
    "lives_at next_along lives_at_of": (
        "who lives next door to {s}", "name a neighbour of {s}",
        "who lives beside {s}", "who is {s}'s neighbour",
        "which people live next to {s}",
        "tell me who lives along from {s}",
        "who is the next flat round from {s}",
        "name somebody living next to {s}",
        "who shares a wall with {s}",
        "which neighbour does {s} have",
        "who lives thirty minutes round the ring from {s}",
        "who is {s}'s nearest neighbour along the ring",
    ),
}

#: How many questions to write per phrasing. Twelve phrasings times this is the
#: class size, and `classify.py --balance` levels them anyway.
PER_TEMPLATE = 40


def resolve(word: str, have: set[str]) -> tuple[str, bool]:
    """(relation, is it read backwards) - the same reading `paths_for` makes.

    Written once and used twice, by the subject picker and by the assertion in
    `main`, because two readings of a path vocabulary that drift apart produce
    a card whose path table is silently inert.
    """
    if word in libgraph.CLIMB:
        return libgraph.CLIMB[word][0], False
    if word in have:
        return word, False
    if word.endswith("_of") and word[:-3] in have:
        return word[:-3], True
    raise SystemExit(f"no edges for {word!r}; the card would ignore that path")


def subjects(db: sqlite3.Connection, path: str, have: set[str], wanted: int,
             rng: Random) -> list[str]:
    """Names the question can sensibly be asked about, drawn from the corpus.

    Asking who is on somebody's crew when they died two centuries ago and have
    no crew teaches the classifier a phrasing and teaches the graph nothing, so
    a subject is drawn from the entities that actually carry the path's first
    relation.
    """
    relation, inverse = resolve(path.split()[0], have)
    column = "object" if inverse else "subject"
    rows = [r for (r,) in db.execute(
        f"SELECT DISTINCT {column} FROM edge "
        "WHERE source = ? AND relation = ?", (SOURCE, relation))]
    rng.shuffle(rows)
    return [rows[i % len(rows)] for i in range(wanted)]


def build(db: sqlite3.Connection, have: set[str], per_template: int,
          hold_out: int,
          seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(training pairs, held-out pairs), split by phrasing rather than by row."""
    rng = Random(seed)
    train: list[tuple[str, str]] = []
    unseen: list[tuple[str, str]] = []
    for path, templates in PATHS.items():
        order = list(templates)
        rng.shuffle(order)
        reserved = set(order[:hold_out]) if hold_out else set()
        names = subjects(db, path, have, per_template * len(templates), rng)
        for i, template in enumerate(templates):
            block = names[i * per_template:(i + 1) * per_template]
            rows = [(template.format(s=name.lower()), path) for name in block]
            (unseen if template in reserved else train).extend(rows)
    rng.shuffle(train)
    rng.shuffle(unseen)
    return train, unseen


@dataclass
class Audit:
    """What a trained classifier does to a phrasing it *was* trained on."""

    #: questions asked, and how many routed to the right path
    asked: int = 0
    right: int = 0
    #: phrasings where every subject got the same answer, and how many there are
    steady: int = 0
    phrasings: int = 0
    #: (path, phrasing, share right, the class it mostly gave instead)
    worst: list[tuple[str, str, float, str]] = field(default_factory=list)


def audit(model_path: str, db: sqlite3.Connection, have: set[str],
          per_template: int, seed: int) -> Audit:
    """Ask each trained phrasing about many different people, and compare.

    This exists because of one observation that a per-question accuracy hides:
    `who is alexander e wong's father` and `who is corey w wong's father` do
    not classify the same way. The encoder hashes the *whole* question into 128
    trigram buckets, and a name is most of a short question - so the subject is
    not something the model ignores on its way to the verb, it is the bulk of
    the signal.

    "Steady" counts phrasings where changing only the name never changes the
    answer. It is the number that says whether a card can be relied on for a
    question it has already been taught.
    """
    import libinfer

    model = libinfer.Model.load(model_path)
    rng = Random(seed)
    out = Audit()
    for path, templates in PATHS.items():
        names = subjects(db, path, have, per_template * len(templates), rng)
        for i, template in enumerate(templates):
            block = names[i * per_template:(i + 1) * per_template]
            got = [libinfer.classify(model, template.format(s=n.lower()), 24)
                   for n in block]
            hits = sum(1 for g in got if g.lower() == path)
            out.asked += len(got)
            out.right += hits
            out.phrasings += 1
            out.steady += len(set(got)) == 1
            if hits < len(got):
                instead = Counter(g.lower() for g in got if g.lower() != path)
                out.worst.append((path, template, hits / len(got),
                                  instead.most_common(1)[0][0]))
    out.worst.sort(key=lambda row: row[2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--per-template", type=int, default=PER_TEMPLATE)
    ap.add_argument("--held-out-templates", type=int, default=0, metavar="N",
                    help="Reserve N phrasings per path, unseen in training")
    ap.add_argument("--emit", choices=("train", "held-out"), default="train",
                    help="'held-out' prints only the reserved phrasings, to "
                         "score generalisation rather than recall")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.emit == "held-out" and not args.held_out_templates:
        ap.error("--emit held-out needs --held-out-templates N")
    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}\n"
                         f"  python data/silo/generate.py")

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (SOURCE,))}
    # Every word of every path, checked against the corpus before anything is
    # printed. A label the card cannot walk is not an error at build time - it
    # becomes an empty row in the path table and a question the machine
    # classifies correctly and then answers with silence.
    for path in PATHS:
        for word in path.split():
            resolve(word, have)

    train, unseen = build(db, have, args.per_template, args.held_out_templates,
                          args.seed)
    pairs = unseen if args.emit == "held-out" else train
    counts = Counter(path for _, path in pairs)

    print(f"# Templated questions about the silo, over {len(counts)} paths.")
    print(f"# {len(pairs):,} questions, {args.per_template} per phrasing, "
          f"subjects drawn from {args.db.name}.")
    if args.emit == "held-out":
        print(f"# Held-out phrasings only: {args.held_out_templates} of "
              f"{len(next(iter(PATHS.values())))} wordings per path, which the "
              f"training half never saw.")
    else:
        print("# Templated - a --val-frac split flatters this. See the module "
              "docstring, and score on --emit held-out.")
    for question, path in pairs:
        print(f"{question.replace('|', ' ').strip()}|{path}")

    for path, n in counts.most_common():
        print(f"  {path:<34} {n:>6,}", file=sys.stderr)


if __name__ == "__main__":
    main()
