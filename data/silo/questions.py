#!/usr/bin/env python3
"""
What can be worked out from the silo's facts, and by which machine.

    python data/silo/questions.py                    # the whole report
    python data/silo/questions.py --sample 500       # quicker, noisier
    python data/silo/questions.py --show 3           # print worked examples

Three machines are asked the same questions:

    a graph walk    `libgraph.follow` - one hop is a binary search over sorted
                    fixed-width records, which is all an eZ80 with a card can do
    a SQL view      `schema.py`'s joins, which an eZ80 cannot run at all
    a guess         the best rule that ignores the question - "the most common
                    answer", "somebody with the same surname", "whatever the
                    asker does for a living"

The first is the one this repository cares about. The second is the ceiling.
The third is here because a synthetic corpus is the easiest place in the world
to publish a flattering number, and a walk that scores 94% on a question a
coin-flip scores 92% on has demonstrated nothing.

## Ground truth does not come from the graph

Answers are computed in Python from `fact`, `residence` and `membership` - the
tables. The walk reads `edge`. Those are written by different functions in
`generate.py` from the same simulation, so this is not two independent sources
of truth and it would be dishonest to call it one. What it does catch is every
way the *graph* can be wrong while the corpus is right: a relation written
backwards, a missing inverse, a climb that stops on the wrong type.

Whether the corpus itself is coherent is a different question, and
`tests/test_silo.py` answers it - nobody is their own ancestor, no parent died
before their child was born, no address disagrees with the generated column.

## The three kinds of answer

**A value.** "Who is X's father." One forward path, one answer. `follow`
returns it, and this is what the eZ80 walk is for.

**A set.** "Who are X's siblings." A path reaches the set only through an
inverse step, and `follow` returns the *first* member rather than the set, so
these are scored on whether the value it returns is in the true set - which is
a weaker claim than the number looks, and is why it is reported separately.

**Neither.** "How many cousins does X have." "Who is the oldest person on X's
crew." An aggregate or a set difference, which no path expresses at any length.
These are counted and named rather than scored, because reporting 0% for them
would suggest the walk got them wrong; it cannot be asked.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate
import schema
from generate import DEPARTMENTS, NOW, section_of
from schema import SOURCE

import libgraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"


@dataclass
class Archive:
    """The corpus as Python, read from the fact tables and never from `edge`."""

    born: dict[str, int] = field(default_factory=dict)
    died: dict[str, int] = field(default_factory=dict)
    generation: dict[str, int] = field(default_factory=dict)
    male: dict[str, bool] = field(default_factory=dict)
    father: dict[str, str] = field(default_factory=dict)
    mother: dict[str, str] = field(default_factory=dict)
    spouse: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    children: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    department: dict[str, str] = field(default_factory=dict)
    job: dict[str, str] = field(default_factory=dict)
    shift: dict[str, str] = field(default_factory=dict)
    address: dict[str, str] = field(default_factory=dict)
    birth_level: dict[str, str] = field(default_factory=dict)
    school: dict[str, str] = field(default_factory=dict)
    crew: dict[str, str] = field(default_factory=dict)
    home: dict[str, tuple[int, int, str, int, int]] = field(default_factory=dict)
    cohort: dict[str, list[tuple[str, int, int]]] = field(
        default_factory=lambda: defaultdict(list))
    by_surname: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @property
    def people(self) -> list[str]:
        return sorted(self.born)

    def alive(self, name: str) -> bool:
        return name not in self.died

    def surname(self, name: str) -> str:
        return name.rsplit(" ", 1)[1]

    def parents(self, name: str) -> list[str]:
        return [p for p in (self.father.get(name), self.mother.get(name)) if p]

    def siblings(self, name: str) -> set[str]:
        out = {c for p in self.parents(name) for c in self.children[p]}
        return out - {name}


def load(db: sqlite3.Connection) -> Archive:
    """Read the archive out of `fact`, `residence` and `membership`."""
    a = Archive()
    numeric = {"born": a.born, "died": a.died, "generation": a.generation}
    textual = {"father": a.father, "mother": a.mother, "department": a.department,
               "occupation": a.job, "shift": a.shift, "address": a.address,
               "birth_level": a.birth_level, "class": a.school, "crew": a.crew}
    for subject, prop, value, num in db.execute(
            "SELECT subject, property, value, num FROM fact WHERE source = ?",
            (SOURCE,)):
        if prop in numeric and num is not None:
            numeric[prop][subject] = int(num)
        elif prop in textual:
            textual[prop][subject] = value
        elif prop == "sex":
            a.male[subject] = value == "male"
        elif prop == "spouse":
            a.spouse[subject].append(value)
    for child, parent in ((c, p) for c, p in a.father.items()):
        a.children[parent].append(child)
    for child, parent in ((c, p) for c, p in a.mother.items()):
        a.children[parent].append(child)
    for name in a.born:
        a.by_surname[a.surname(name)].append(name)
    for person, floor, bearing, ring, since, until in db.execute(
            "SELECT person, floor, bearing, ring, since, until FROM residence "
            "WHERE source = ?", (SOURCE,)):
        a.home[person] = (floor, bearing, ring, since,
                          NOW if until is None else until)
    for person, name, joined, until in db.execute(
            "SELECT person, cohort, joined, until FROM membership WHERE source = ?",
            (SOURCE,)):
        a.cohort[name].append((person, joined, NOW if until is None else until))
    return a


# --- ground truth -------------------------------------------------------------
#
# One function per question, each computing the answer from the Archive above.
# They are short on purpose: a truth function complicated enough to be wrong is
# not a truth function.


def _grandfather(a: Archive, x: str) -> set[str]:
    father = a.father.get(x)
    return {a.father[father]} if father and father in a.father else set()


def _grandmother(a: Archive, x: str) -> set[str]:
    mother = a.mother.get(x)
    return {a.mother[mother]} if mother and mother in a.mother else set()


def _founder(a: Archive, x: str) -> set[str]:
    here = x
    while here in a.father:
        here = a.father[here]
    return {here} if a.generation.get(here) == 0 and here != x else set()


def _neighbours(a: Archive, x: str) -> set[str]:
    """Everyone next door, by the geometry rather than by the adjacency edges.

    Computed from the bearings so that it is a genuine second opinion on
    `next_along`/`next_out`: if the generator laid the ring out wrongly, this
    disagrees with the walk rather than repeating its mistake.
    """
    if x not in a.home:
        return set()
    floor, bearing, ring, since, until = a.home[x]
    want = {(floor, (bearing + s) % schema.CLOCK, ring)
            for s in (schema.BEARING_STEP, -schema.BEARING_STEP)}
    index = schema.RINGS.index(ring)
    want |= {(floor, bearing, schema.RINGS[i])
             for i in (index - 1, index + 1) if 0 <= i < len(schema.RINGS)}
    return {other for other, (f, b, r, s, u) in a.home.items()
            if (f, b, r) in want and other != x and s <= until and since <= u}


def _living_neighbours(a: Archive, x: str) -> set[str]:
    """The subset the graph can be right about: `lives_at` is for the living."""
    if not a.alive(x):
        return set()
    return {n for n in _neighbours(a, x) if a.alive(n)}


# --- the questions ------------------------------------------------------------


@dataclass
class Question:
    """One question, three answers, and what it costs to ask."""

    label: str
    #: 'value' - a forward path returns the whole answer
    #: 'set'   - the answer is a set and a path returns one member of it
    #: 'none'  - no path expresses it; the reason is in `note`
    kind: str
    truth: Callable[[Archive, str], set[str]]
    path: list[str] = field(default_factory=list)
    #: Applies to a subject only if this says so - "who is X's grandfather" is
    #: not a question about somebody whose father is a founder.
    asks: Callable[[Archive, str], bool] = lambda a, x: True
    guess: Callable[[Archive, str], set[str]] = lambda a, x: set()
    guess_label: str = "-"
    note: str = ""


def _commonest(values: dict[str, str]) -> str:
    return Counter(values.values()).most_common(1)[0][0]


def questions(a: Archive) -> list[Question]:
    """Twenty-two questions, ordered by how far the walk has to go."""
    top_department = _commonest(a.department)
    top_shift = _commonest(a.shift)
    top_section = Counter(section_of(h[0]) for h in a.home.values()).most_common(1)[0][0]

    def same_surname_elder(a: Archive, x: str, male: bool, gap: int) -> set[str]:
        """The best guess a surname alone supports.

        Children take their father's name here, so this is not a straw man: it
        is the rule a person would use, and on "who is X's father" it is right
        often enough to be worth beating rather than worth reporting.
        """
        want = a.born[x] - gap
        pool = [p for p in a.by_surname[a.surname(x)]
                if a.male.get(p) is male and abs(a.born[p] - want) <= 8]
        return {min(pool, key=lambda p: abs(a.born[p] - want))} if pool else set()

    return [
        Question("who is X's father", "value", lambda a, x: {a.father[x]},
                 ["father_is"], asks=lambda a, x: x in a.father,
                 guess=lambda a, x: same_surname_elder(a, x, True, 28),
                 guess_label="nearest man of the same surname, 28 years older"),
        Question("which department does X work in", "value",
                 lambda a, x: {a.department[x]}, ["works_in"],
                 guess=lambda a, x: {top_department},
                 guess_label=f"always {top_department!r}"),
        Question("which shift does X work", "value", lambda a, x: {a.shift[x]},
                 ["shift_is"], guess=lambda a, x: {top_shift},
                 guess_label=f"always {top_shift!r}"),
        Question("which level was X born on", "value",
                 lambda a, x: {a.birth_level[x]}, ["born_on"]),

        Question("who is X's paternal grandfather", "value", _grandfather,
                 ["father_is", "father_is"],
                 asks=lambda a, x: bool(_grandfather(a, x)),
                 guess=lambda a, x: same_surname_elder(a, x, True, 56),
                 guess_label="nearest man of the same surname, 56 years older"),
        Question("who is X's maternal grandmother", "value", _grandmother,
                 ["mother_is", "mother_is"],
                 asks=lambda a, x: bool(_grandmother(a, x))),
        Question("which department does X's father work in", "value",
                 lambda a, x: {a.department[a.father[x]]},
                 ["father_is", "works_in"], asks=lambda a, x: x in a.father,
                 guess=lambda a, x: {a.department[x]},
                 guess_label="whatever X does - 45% follow a parent"),
        Question("what is X's spouse's trade", "value",
                 lambda a, x: {a.job[a.spouse[x][0]]}, ["spouse_of", "job_is"],
                 asks=lambda a, x: bool(a.spouse.get(x)),
                 note="`spouse_of` has two edges for the remarried and `follow` "
                      "takes the first, which is not always the one asked about"),
        Question("which section does X live in", "value",
                 lambda a, x: {section_of(a.home[x][0])},
                 ["lives_at", "in_section"],
                 asks=lambda a, x: a.alive(x) and x in a.home,
                 guess=lambda a, x: {top_section},
                 guess_label=f"always {top_section!r}"),
        Question("which level is X's department on", "value",
                 lambda a, x: {generate.level(DEPARTMENTS[a.department[x]][0])},
                 ["works_in", "located_in"]),

        Question("which section does X's father's department sit in", "value",
                 lambda a, x: {section_of(DEPARTMENTS[a.department[a.father[x]]][0])},
                 ["father_is", "works_in", "located_in", "in_section"],
                 asks=lambda a, x: x in a.father),
        Question("which shift does X's paternal grandfather work", "value",
                 lambda a, x: {a.shift[next(iter(_grandfather(a, x)))]},
                 ["father_is", "father_is", "shift_is"],
                 asks=lambda a, x: bool(_grandfather(a, x)),
                 guess=lambda a, x: {top_shift},
                 guess_label=f"always {top_shift!r}"),
        Question("which level was X's mother-in-law born on", "value",
                 lambda a, x: {a.birth_level[a.mother[a.spouse[x][0]]]},
                 ["spouse_of", "mother_is", "born_on"],
                 asks=lambda a, x: bool(a.spouse.get(x))
                 and a.spouse[x][0] in a.mother),

        Question("which founder is X descended from, father to father", "value",
                 _founder, ["founding_father"],
                 asks=lambda a, x: bool(_founder(a, x)),
                 note="a climb, not a path: the depth is the generation, and "
                      "a CLIMB_LIMIT of 6 buys five hops"),

        Question("who lives next door to X", "set", _living_neighbours,
                 ["lives_at", "next_along"],
                 asks=lambda a, x: bool(_living_neighbours(a, x)),
                 note="four walks, one per direction, unioned by the caller"),
        Question("who is X's sibling", "set",
                 lambda a, x: a.siblings(x), ["child_of", "child_of"],
                 asks=lambda a, x: bool(a.siblings(x)),
                 note="`follow` takes LIMIT 1 on `child_of`, so the walk goes "
                      "up through one parent and every half-sibling is on the "
                      "other one. Measured rather than reasoned: 0 of 1,914 "
                      "full siblings missed, 254 of 540 half-siblings missed"),
        Question("who are X's children", "set",
                 lambda a, x: set(a.children[x]), ["child_of"],
                 asks=lambda a, x: bool(a.children[x]),
                 note="one inverse hop"),
        Question("who was in X's class", "set",
                 lambda a, x: {p for p, _, _ in a.cohort[a.school[x]]} - {x},
                 ["class_is", "class_is"],
                 asks=lambda a, x: len(a.cohort[a.school.get(x, "")]) > 1,
                 note="forward to the class, then the same relation backwards - "
                      "and a class of 26 is the one answer set here big enough "
                      "for the reverse read to be worth its own table"),

        Question("how many cousins does X have", "none",
                 lambda a, x: set(), asks=lambda a, x: True,
                 note="four hops of which two are inverses, then a count - and "
                      "a count is an aggregate, not a value at the end of a path"),
        Question("who is the oldest person on X's crew", "none",
                 lambda a, x: set(), asks=lambda a, x: x in a.crew,
                 note="a maximum over a set the walk can enumerate but not rank"),
        Question("is X related to Y", "none", lambda a, x: set(),
                 note="an intersection of two ancestor sets; the `ancestor` view "
                      "is a recursive CTE and there is no such thing as a "
                      "recursive path"),
        Question("how many people live on X's floor", "none",
                 lambda a, x: set(), asks=lambda a, x: x in a.home,
                 note="an aggregate over a ring the walk can circle in 24 hops, "
                      "counting as it goes - which is a program, not a query"),
    ]


# --- asking -------------------------------------------------------------------


def _walk(db: sqlite3.Connection, subject: str, q: Question) -> set[str]:
    """What the graph walk returns, as a set so the two kinds compare.

    The neighbour question is the one place a caller has to do more than call
    `follow`: next door is four directions, two relations each read forwards
    and backwards, so it is four walks and a union. That is still four binary
    searches and no arithmetic, which is the property that matters.
    """
    if q.label.startswith("who lives next door"):
        answer = libgraph.follow(db, SOURCE, subject, ["lives_at"])
        if answer.value is None:
            return set()
        here = answer.value
        out: set[str] = set()
        for relation in ("next_along", "next_out"):
            for door in (libgraph.follow(db, SOURCE, here, [relation]).value,
                         *libgraph.inverse(db, SOURCE, here, relation, limit=4)):
                if door:
                    out.update(libgraph.inverse(db, SOURCE, door, "lives_at",
                                                limit=8))
        return out - {subject}
    if q.kind == "set":
        first = q.path[0]
        if len(q.path) == 1:
            return set(libgraph.inverse(db, SOURCE, subject, first, limit=64))
        answer = libgraph.follow(db, SOURCE, subject, [first])
        if answer.value is None:
            return set()
        return set(libgraph.inverse(db, SOURCE, answer.value, q.path[1],
                                    limit=64)) - {subject}
    answer = libgraph.follow(db, SOURCE, subject, q.path)
    return {answer.value} if answer.value is not None else set()


@dataclass
class Score:
    asked: int = 0
    walked: int = 0
    guessed: int = 0
    hops: int = 0
    #: For a set question: how much of the true set came back, and how much of
    #: what came back was true. "Did it return *a* sibling" and "did it return
    #: *the* siblings" are different claims and the second is the one a person
    #: asking the question meant.
    found: int = 0
    wanted: int = 0
    returned: int = 0

    def rate(self, hits: int) -> str:
        return f"{100 * hits / self.asked:5.1f}%" if self.asked else "    -"

    @property
    def recall(self) -> str:
        return f"{100 * self.found / self.wanted:5.1f}%" if self.wanted else "    -"

    @property
    def precision(self) -> str:
        return f"{100 * self.found / self.returned:5.1f}%" if self.returned else "    -"


def run(db: sqlite3.Connection, a: Archive, subjects: list[str],
        show: int) -> list[tuple[Question, Score]]:
    out: list[tuple[Question, Score]] = []
    for q in questions(a):
        score = Score()
        shown = 0
        for x in subjects:
            if not q.asks(a, x):
                continue
            score.asked += 1
            if q.kind == "none":
                continue
            truth = q.truth(a, x)
            got = _walk(db, x, q)
            score.hops += len(q.path)
            hit = bool(got & truth) if q.kind == "set" else got == truth
            score.walked += bool(hit)
            score.guessed += bool(q.guess(a, x) & truth)
            if q.kind == "set":
                score.found += len(got & truth)
                score.wanted += len(truth)
                score.returned += len(got)
            if show and shown < show:
                shown += 1
                mark = "ok " if hit else "MISS"
                print(f"    {mark} {x}: walk {sorted(got) or '--'} "
                      f"/ true {sorted(truth) or '--'}")
        out.append((q, score))
    return out


def report(db: sqlite3.Connection, a: Archive, subjects: list[str],
           show: int) -> None:
    scored = run(db, a, subjects, show)
    print(f"\n{len(subjects):,} subjects drawn from {len(a.people):,} people")
    _coverage(db, a)

    print("\nA value at the end of a forward path - what the eZ80 walk is for")
    print(f"  {'':<52} {'hops':>4} {'walk':>7} {'guess':>7}")
    for q, s in scored:
        if q.kind != "value":
            continue
        hops = s.hops // s.asked if s.asked else 0
        guess = s.rate(s.guessed) if q.guess_label != "-" else "      -"
        print(f"  {q.label:<52} {hops:>4} {s.rate(s.walked):>7} {guess:>7}"
              f"   n={s.asked:,}")
        if q.guess_label != "-":
            print(f"      guessing: {q.guess_label}")
        if q.note:
            print(f"      note: {q.note}")

    print("\nA set, reached through an inverse hop. `any` is whether the walk")
    print("returned a right answer; `recall` is whether it returned them all,")
    print("and the gap between the two columns is the honest reading")
    print(f"  {'':<52} {'any':>7} {'recall':>7} {'prec':>7}")
    for q, s in scored:
        if q.kind != "set":
            continue
        print(f"  {q.label:<52} {s.rate(s.walked):>7} {s.recall:>7} "
              f"{s.precision:>7}   n={s.asked:,}")
        print(f"      {q.note}")

    print("\nNot a path at any length")
    for q, s in scored:
        if q.kind == "none":
            print(f"  {q.label:<52}          n={s.asked:,}")
            print(f"      {q.note}")

    _climb_by_generation(db, a, subjects)


def _coverage(db: sqlite3.Connection, a: Archive) -> None:
    """Which relations reach everybody, and which reach only the present.

    The Wikipedia corpus has this problem by accident - 46% of articles carry
    an infobox and a chain that lands on one of the rest stops. Here it is a
    decision with a reason, and printing it beside the scores is the difference
    between a question the walk got wrong and one it was never able to reach.
    """
    print()
    for relation, note in (("child_of", "everyone with a parent"),
                           ("works_in", "everyone"),
                           ("class_is", "everyone who reached six"),
                           ("lives_at", "the living only - the graph carries "
                                        "the present, `residence` carries all "
                                        f"{NOW} years"),
                           ("crew_is", "the living of working age"),
                           ("sits_on", "committee members, past and present")):
        n = db.execute("SELECT COUNT(DISTINCT subject) FROM edge WHERE source = ? "
                       "AND relation = ?", (SOURCE, relation)).fetchone()[0]
        print(f"  {relation:<10} {n:>6,} subjects  {100 * n / len(a.people):5.1f}%"
              f"  {note}")


def _climb_by_generation(db: sqlite3.Connection, a: Archive,
                         subjects: list[str]) -> None:
    """Where `CLIMB_LIMIT` bites, which is the one number this corpus is for.

    Seven generations, and a limit that counts the values a climb may
    *examine* rather than the hops it may take - the type is tested at the top
    of the loop, so the value the last hop reached is never tested at all. Six
    examinations buy five hops. Generation g is exactly g hops from its
    founder, so generation 5 is the deepest that answers and generation 6 falls
    one short. Nothing about that is visible on a corpus where the true answer
    is unknown.
    """
    print("\nThe climb, by generation - `founding_father` against a hop limit "
          f"of {libgraph.CLIMB_LIMIT}")
    print(f"  {'generation':<12} {'asked':>8} {'hops needed':>12} {'reached':>9}")
    by_gen: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for x in subjects:
        generation = a.generation.get(x)
        founder = _founder(a, x)
        if generation is None or not founder:
            continue
        depth, here = 0, x
        while here in a.father:
            here, depth = a.father[here], depth + 1
        got = libgraph.follow(db, SOURCE, x, ["founding_father"]).value
        by_gen[generation].append((depth, got == next(iter(founder))))
    for generation in sorted(by_gen):
        rows = by_gen[generation]
        depths = {d for d, _ in rows}
        span = (f"{min(depths)}" if len(depths) == 1
                else f"{min(depths)}-{max(depths)}")
        hit = 100 * sum(1 for _, ok in rows if ok) / len(rows)
        print(f"  {generation:<12} {len(rows):>8} {span:>12} {hit:>8.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many worked examples per question")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}\n"
                         f"  python data/silo/generate.py")
    db = schema.connect(args.db)
    archive = load(db)
    rng = Random(args.seed)
    people = archive.people
    subjects = rng.sample(people, min(args.sample, len(people)))
    report(db, archive, subjects, args.show)


if __name__ == "__main__":
    main()
