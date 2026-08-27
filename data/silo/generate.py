#!/usr/bin/env python3
"""
A silo, synthesized: 10,000 people and the facts that relate them.

    pip install -r data/silo/requirements.txt
    python data/silo/generate.py                       # data/silo.db
    python data/silo/generate.py --people 2000 --seed 7
    python data/silo/generate.py --stats

Seven generations under one lid. Everyone has two parents, a dwelling given as
`FLOOR TIME RING`, a job, a shift, a school class, and - if they are still
alive - a work crew and a set of neighbours. Names come from Faker, seeded, so
the corpus is believable and reproducible at once.

The corpus tables are the ones `data/wikipedia/ingest.py` writes, so everything
downstream reads it unchanged: `libgraph` walks it, `oracle.py` answers from
it, `buildwikisearch.py` turns it into a card. See `schema.py` for what this
database adds on top, and why it is a separate file.

## Why a made-up corpus is worth having

The Wikipedia oracle is limited by **coverage**, not by the walk. 46% of
articles carry an infobox, so a chain that hops onto one of the other 54%
cannot continue, and `coverage.py` spends its time measuring where the road
ends. That makes it a poor instrument for the question underneath: given facts
that *are* all there, how much can a machine that does nothing but compare
24-bit numbers actually work out?

This corpus answers that. It has no gaps, and the answers are known by
construction rather than by a second query over the same edges.

**A synthetic corpus is also the easiest place in the world to publish a
flattering number.** So `questions.py` scores against the pedigree rather than
against the graph, and prints a trivial baseline beside every result. Two of
the baselines are deliberately strong: children take their father's surname,
and 45% follow a parent into their department, so "guess from the surname" and
"guess the asker's own department" are real competitors on the questions
somebody would actually ask.

## Stored, derivable, and walkable are three different things

Stored as edges: `father_is`, `mother_is`, `child_of`, `spouse_of`,
`lives_at`, `born_on`, `works_in`, `job_is`, `shift_is`, `crew_is`,
`class_is`, `sits_on`, plus the geography - dwelling to level to section to
silo, and `next_along`/`next_out` around the ring.

Derivable in SQL, and deliberately not stored: sibling, half-sibling,
grandparent, aunt, cousin, ancestor at any depth, housemate, neighbour,
coworker, classmate, committee colleague. `schema.py` has them as views.

Walkable on an eZ80 is a third, smaller set, and finding its edge is the point
of `questions.py`. Neighbours are in it, because ring adjacency is stored as
edges rather than computed from bearings - the machine has no modulo. Siblings
are *most of the way* in it: an inverse hop finds every full sibling and
misses half the half-siblings, because the walk goes up through one parent and
the other half are on the other one. Cousin counts are not in it at all - a
count is an aggregate, and a path ends in a value.

## The hop limit, made visible

`libgraph.CLIMB` repeats a relation until the value has a given type - what
"what country was X born in" really asks, since the number of hops is a
property of the graph and not of the question. This corpus adds three:

    in_section        located_in, until a section
    in_silo           located_in, until a silo
    founding_father   father_is, until a founder

The last one runs into `CLIMB_LIMIT`, which is 6 and counts the values a climb
may *examine* - one more than the hops it may take. Generation g is exactly g
hops from its founder, so a limit of 6 buys generations 1 to 5 and generation 6
falls one short. That is not a bug to be worked around; it is what the limit
costs, on a corpus where the true answer is known for all 10,000 - and
`buildcard.py --climb-limit 7` is what buys the seventh.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plant
import schema
from schema import BEARING_STEP, BEARINGS, RINGS, SOURCE

import libgraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Sequence

DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"

# --- geography ----------------------------------------------------------------

SILO = "Silo 18"
LEVELS = 144

#: (name, first level, last level), top to bottom.
SECTIONS: tuple[tuple[str, int, int], ...] = (
    ("Up Top", 1, 20),
    ("The Mids", 21, 120),
    ("Down Deep", 121, 144),
)

#: department -> (home level, job titles). Spread across all three sections on
#: purpose: if every department sat in the Mids, "which section does X work in"
#: would have one answer and scoring it would measure nothing.
DEPARTMENTS: dict[str, tuple[int, tuple[str, ...]]] = {
    "Cafeteria": (1, ("Cook", "Server", "Dishwasher", "Pantry Keeper")),
    "Sheriff's Office": (3, ("Sheriff", "Deputy", "Jailer", "Dispatcher")),
    "Judicial": (5, ("Judge", "Law Clerk", "Bailiff", "Archivist", "Investigator")),
    "Nursery": (12, ("Midwife", "Nurse", "Teacher", "Warden of Records")),
    "IT": (34, ("Technician", "Server Warden", "Cable Runner", "Analyst",
                "Screen Fitter")),
    "Supply": (61, ("Quartermaster", "Runner", "Ledger Clerk", "Sorter")),
    "Farms": (70, ("Grower", "Irrigator", "Seed Keeper", "Harvester",
                   "Soil Tender")),
    "Medical": (82, ("Doctor", "Nurse Practitioner", "Apothecary", "Surgeon",
                     "Orderly")),
    "Chemical": (96, ("Chemist", "Batch Mixer", "Solvent Handler", "Fume Warden")),
    "Recycling": (108, ("Reclaimer", "Reclaim Sorter", "Melt Operator", "Scrapper")),
    "Water Treatment": (118, ("Filtration Tech", "Pump Operator",
                              "Cistern Warden", "Sampler")),
    "Electrical": (126, ("Electrician", "Line Runner", "Breaker Tech",
                         "Lamp Keeper")),
    "Mining": (138, ("Miner", "Shaft Foreman", "Drill Hand", "Hauler")),
    "Mechanical": (140, ("Mechanic", "Generator Technician", "Machinist",
                         "Welder", "Shadow", "Oiler")),
}

SHIFTS: tuple[str, ...] = ("First Shift", "Second Shift", "Third Shift")

#: The climbs this corpus adds to `libgraph`: repeat a relation until the value
#: has a given type, because how many hops that takes is a property of the
#: graph and not of the question.
#:
#: Registered here, at import, rather than declared in `libgraph` - the types
#: they stop on exist only in this corpus, so a Wikipedia caller cannot reach
#: them by accident. The hazard is the mirror image and worth naming: a caller
#: who opens `silo.db` *without* importing this module gets `in_section`
#: treated as an ordinary relation, finds no edge, and is told there is no
#: answer. `tests/test_silo.py` asserts the registration for that reason.
CLIMBS: dict[str, tuple[str, str]] = {
    "in_section": ("located_in", "section"),
    "in_silo": ("located_in", "silo"),
    "founding_father": ("father_is", "founder"),
}
libgraph.CLIMB.update(CLIMBS)

COMMITTEES: tuple[str, ...] = (
    "Pact Review Committee", "Birth Lottery Board", "Water Rationing Committee",
    "Air Quality Board", "Seed Vault Trust", "Stairwell Safety Committee",
    "Power Allocation Board", "Cleaning Oversight Committee",
    "Sanitation Council", "Levy and Chit Committee", "Nursery Admissions Board",
    "Records Preservation Trust", "Generator Overhaul Committee",
    "Mine Depth Review Board", "Lighting Schedule Committee",
    "Grievance Panel", "Founders' Day Committee", "Shaft Inspection Board",
    "Ration Appeals Panel", "Relic Disposal Committee",
)

# --- demography ---------------------------------------------------------------

#: How many people are born into each generation. Seven cohorts, near enough
#: flat: a silo with a birth lottery does not grow, and a growing one would put
#: most of the corpus in the youngest generation, where nobody has children.
GENERATIONS: tuple[int, ...] = (1200, 1300, 1400, 1450, 1500, 1550, 1600)

#: Share of adults who pair off. The rest are the corpus's dead ends, and they
#: are here because a graph where every question has an answer teaches nothing
#: about what a missing edge does to a walk.
MARRIAGE_RATE = 0.88

#: Share of couples where one partner later takes a second. The only source of
#: half-siblings, and half-siblings are why `sibling` cannot be a fixed path:
#: sharing one parent and sharing two are different answers to one question.
REMARRIAGE_RATE = 0.09

#: Share of children who follow a parent into their department. Nepotism as a
#: baseline: `questions.py` reports what guessing the asker's own department
#: scores on "which department does X's father work in", and this is the knob
#: that decides whether that number is embarrassing.
INHERITS_TRADE = 0.45

#: The silo's present year, counted from the day the doors shut.
NOW = 220

BIRTH_AGE = (22, 34)
LIFESPAN = (58, 88)
#: Share who die before thirty, so "is X alive" is not a function of birth year
#: alone and the pedigree has widows in it.
EARLY_DEATH = 0.04
#: Share of deaths that were a cleaning rather than an ending.
CLEANING_RATE = 0.015

SCHOOL = (6, 16)
WORKING_AGE = 16
CLASS_SIZE = 26
CREW_SIZE = 11
COMMITTEE_SEATS = (6, 9)
TERM = (5, 15)


@dataclass
class Person:
    """One inhabitant. Indices, not names, until the corpus is written."""

    index: int
    given: str
    initial: str
    surname: str
    male: bool
    generation: int
    born: int
    father: int | None = None
    mother: int | None = None
    spouses: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    died: int | None = None
    fate: str | None = None
    birth_level: int = 0
    department: str = ""
    job: str = ""
    shift: str = ""
    home: tuple[int, int, str] | None = None
    moved: int = 0
    school: str = ""
    crew: str = ""
    seats: list[tuple[str, int, int | None]] = field(default_factory=list)
    #: Set only by `plant.py`, and only for a handful of people: the name the
    #: `fact` table gives as this person's father, when it is not the man the
    #: `father_is` edge leads to. Everywhere else the two are written from one
    #: pass and cannot disagree, which is exactly why a disagreement is worth
    #: being able to plant.
    recorded_father: str | None = None

    @property
    def name(self) -> str:
        return f"{self.given} {self.initial}. {self.surname}"

    @property
    def alive(self) -> bool:
        return self.died is None

    @property
    def address(self) -> str:
        if self.home is None:  # pragma: no cover - every person gets a home
            raise ValueError(f"{self.name} has no dwelling")
        return schema.address(*self.home)


@dataclass(frozen=True)
class Cohort:
    """A class, a crew or a committee: a named set with a date and a place."""

    name: str
    kind: str
    formed: int
    floor: int | None


class World:
    """The pedigree, and the only thing that knows the truth about it.

    `questions.py` scores the graph against this rather than against a second
    query over the same edges, so a wrong walk and a wrong expectation cannot
    quietly agree with each other.
    """

    def __init__(self, people: list[Person]) -> None:
        self.people = people
        self.dwellings: list[tuple[int, int, str]] = []
        self.cohorts: dict[str, Cohort] = {}
        self.by_name: dict[str, Person] = {p.name: p for p in people}

    def __len__(self) -> int:
        return len(self.people)

    def parents(self, p: Person) -> list[Person]:
        return [self.people[i] for i in (p.father, p.mother) if i is not None]

    def siblings(self, p: Person) -> set[int]:
        """Anyone sharing at least one parent, half-siblings included."""
        out: set[int] = set()
        for parent in self.parents(p):
            out.update(parent.children)
        out.discard(p.index)
        return out


def section_of(level: int) -> str:
    for name, low, high in SECTIONS:
        if low <= level <= high:
            return name
    raise ValueError(f"level {level} is outside the silo")


# --- the simulation -----------------------------------------------------------


class Names:
    """Faker, plus the uniqueness the database needs and Faker does not give.

    `article.title` is the primary key every edge points at, so two people
    called Jonas T. Nichols would silently be one person and every walk through
    either would be about the other half the time. A middle initial is enough
    slack: Faker's pools are large, and the initial multiplies them by 26.
    """

    def __init__(self, seed: int) -> None:
        from faker import Faker

        self.fake = Faker("en_US")
        self.fake.seed_instance(seed)
        self.taken: set[str] = set()

    def surname(self) -> str:
        return str(self.fake.last_name())

    def person(self, rng: Random, index: int, male: bool, surname: str,
               generation: int, born: int) -> Person:
        for _ in range(500):
            given = (self.fake.first_name_male() if male
                     else self.fake.first_name_female())
            person = Person(index, str(given), rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                            surname, male, generation, born)
            if person.name not in self.taken:
                self.taken.add(person.name)
                return person
        raise RuntimeError(f"no unused name left for the {surname} line")


def _assign_work(rng: Random, person: Person, parents: Sequence[Person]) -> None:
    """Department, job and shift.

    Trade inheritance is applied here rather than at pairing time because it is
    a fact about the child, and because one place for it means the baseline in
    `questions.py` and the edge in the database cannot disagree about how
    strong it is.
    """
    inherited = [p.department for p in parents if p.department]
    if inherited and rng.random() < INHERITS_TRADE:
        person.department = rng.choice(inherited)
    else:
        person.department = rng.choice(list(DEPARTMENTS))
    person.job = rng.choice(DEPARTMENTS[person.department][1])
    person.shift = rng.choice(SHIFTS)


def _related(a: Person, b: Person, people: Sequence[Person]) -> bool:
    """True if they share a parent or a grandparent.

    First cousins are refused as well as siblings, which is not squeamishness:
    a cousin marriage collapses two lines into one and makes "is X related to
    Y" true for very nearly everybody, which would flatter every kinship
    question in `questions.py`.
    """
    mine = {i for i in (a.father, a.mother) if i is not None}
    theirs = {i for i in (b.father, b.mother) if i is not None}
    if mine & theirs:
        return True
    grand = {g for i in mine
             for g in (people[i].father, people[i].mother) if g is not None}
    other = {g for i in theirs
             for g in (people[i].father, people[i].mother) if g is not None}
    return bool(grand & other)


def _pairs(rng: Random, cohort: list[Person],
           people: Sequence[Person]) -> list[tuple[Person, Person]]:
    """Pair a cohort off, refusing anyone too close or too far apart in age.

    Sorted by birth year and matched greedily against the nearest unpaired
    person of the other sex, so couples are contemporaries without anybody
    having to search.
    """
    men = sorted((p for p in cohort if p.male), key=lambda p: p.born)
    women = sorted((p for p in cohort if not p.male), key=lambda p: p.born)
    couples: list[tuple[Person, Person]] = []
    used: set[int] = set()
    for man in men:
        if rng.random() > MARRIAGE_RATE:
            continue
        for woman in women:
            if woman.index in used or abs(woman.born - man.born) > 7:
                continue
            if _related(man, woman, people):
                continue
            used.add(woman.index)
            couples.append((man, woman))
            break
    return couples


def _remarriages(rng: Random, cohort: list[Person],
                 couples: list[tuple[Person, Person]],
                 people: Sequence[Person]) -> list[tuple[Person, Person]]:
    """Second partnerships, which are where half-siblings come from."""
    paired = {p.index for couple in couples for p in couple}
    spare = [p for p in cohort if p.index not in paired]
    extra: list[tuple[Person, Person]] = []
    for man, woman in couples:
        if rng.random() >= REMARRIAGE_RATE:
            continue
        widowed = man if rng.random() < 0.5 else woman
        for other in spare:
            if other.male == widowed.male or _related(widowed, other, people):
                continue
            spare.remove(other)
            extra.append((widowed, other) if widowed.male else (other, widowed))
            break
    return extra


def _children(rng: Random, names: Names, couples: list[tuple[Person, Person]],
              quota: int, generation: int, people: list[Person]) -> list[Person]:
    """Fill a generation's quota, one child at a time across every couple.

    Round-robin rather than family-by-family, so a quota that runs out shortens
    the largest families instead of leaving the last few hundred couples
    childless - a cohort of couples who happen to be sterile would be a strange
    thing to have to explain in the data.
    """
    if not couples:
        return []
    for man, woman in couples:
        if woman.index not in man.spouses:
            man.spouses.append(woman.index)
            woman.spouses.append(man.index)

    wanted = [rng.choices((1, 2, 3, 4), weights=(20, 38, 30, 12))[0]
              for _ in couples]
    order = list(range(len(couples)))
    rng.shuffle(order)
    born: list[Person] = []
    round_ = 0
    while len(born) < quota:
        progressed = False
        for slot in order:
            if len(born) >= quota:
                break
            if wanted[slot] <= round_:
                continue
            progressed = True
            born.append(_child(rng, names, couples[slot], generation,
                               len(people) + len(born)))
        if not progressed:
            # Every couple has had all the children they wanted and the quota
            # is still short. Widening families is the honest repair; inventing
            # parentless people would put orphans in the graph that no kinship
            # question could reach. The round does *not* advance here - raising
            # the ceiling and the floor together is how this loop first managed
            # to never finish.
            wanted = [n + 1 for n in wanted]
            continue
        round_ += 1
    return born


def _child(rng: Random, names: Names, couple: tuple[Person, Person],
           generation: int, index: int) -> Person:
    father, mother = couple
    born = max(father.born, mother.born) + rng.randint(*BIRTH_AGE)
    child = names.person(rng, index, rng.random() < 0.5, father.surname,
                         generation, born)
    child.father, child.mother = father.index, mother.index
    father.children.append(index)
    mother.children.append(index)
    _assign_work(rng, child, couple)
    return child


def _die(rng: Random, people: list[Person]) -> None:
    """Kill everyone the calendar says is dead, and nobody it does not.

    One pass at the end, because a death has to come after the last child: a
    corpus where a father predeceases his son's birth is a corpus where every
    consistency check anyone writes is a false positive.
    """
    for person in people:
        span = (rng.randint(1, 30) if rng.random() < EARLY_DEATH
                else rng.randint(*LIFESPAN))
        last = max((people[i].born for i in person.children), default=person.born)
        death = max(person.born + span, last + 1)
        if death > NOW:
            continue
        person.died = death
        person.fate = ("Cleaning" if rng.random() < CLEANING_RATE
                       else "Natural causes")


# --- dwellings ----------------------------------------------------------------
#
# Addresses are allocated by reuse rather than by expansion: a household takes a
# dwelling nobody currently lives in, and the silo only opens another floor when
# there is genuinely none. That keeps occupancy high, which is what makes
# `neighbour` a question with an answer - a sparsely filled ring has none.


class Silo:
    """Who lives where, over 220 years, in as few dwellings as that needs."""

    def __init__(self, rng: Random) -> None:
        self.rng = rng
        self.floors: list[int] = []
        self.free: list[tuple[int, int, str]] = []
        #: dwelling -> year it next becomes vacant, or None while occupied
        self.until: dict[tuple[int, int, str], int | None] = {}
        self._open_floor()

    def _open_floor(self) -> None:
        """Bring another residential floor into use, chosen at random.

        Random rather than in order, so that a floor number does not encode how
        early it was settled - which would make "which generation" answerable
        from an address and quietly flatter half the questions.
        """
        remaining = [n for n in range(1, LEVELS + 1) if n not in self.floors]
        if not remaining:
            raise RuntimeError("the silo is full; lower --people")
        floor = self.rng.choice(remaining)
        self.floors.append(floor)
        fresh = [(floor, b * BEARING_STEP, ring)
                 for b in range(BEARINGS) for ring in RINGS]
        self.rng.shuffle(fresh)
        self.free.extend(fresh)

    def take(self, year: int, leaves: int | None) -> tuple[int, int, str]:
        """A dwelling standing empty in ``year``, held until ``leaves``.

        The scan rotates: a dwelling that is taken goes to the back of the
        list, so the front of it is where the long-empty ones collect and the
        search is short however many floors are open.
        """
        for i, home in enumerate(self.free):
            vacant = self.until.get(home, 0)
            if vacant is not None and vacant <= year:
                self.free.pop(i)
                self.until[home] = leaves
                self.free.append(home)
                return home
        self._open_floor()
        return self.take(year, leaves)

    def every(self) -> list[tuple[int, int, str]]:
        """Every dwelling on an opened floor, lived in or not.

        Vacant ones are in the corpus because `next_along` has to reach across
        them: a ring with holes in it is still a ring, and adjacency that
        skipped the empty flats would put two people next door to each other
        who are not.
        """
        return sorted((floor, b * BEARING_STEP, ring) for floor in self.floors
                      for b in range(BEARINGS) for ring in RINGS)


def _house(rng: Random, world: World) -> None:
    """Give everyone one dwelling: the household they headed, or grew up in.

    One residence each is a simplification worth naming. Somebody who was born,
    raised a family and died in three different apartments is recorded at the
    middle one, because `residence` is keyed by person - and a table keyed by
    (person, dwelling, year) would make every neighbour query a temporal join,
    for a corpus whose questions are not about time.
    """
    silo = Silo(rng)
    for p in sorted(world.people, key=lambda q: q.born):
        if p.home is not None:
            continue
        adult = p.born + rng.randint(18, 26)
        # Anyone who has not reached that age is still at their parents' - which
        # includes the youngest generation, who are alive and are *not* yet
        # heads of household. Giving them their own flat dated in the future
        # put tenancies beginning after the present year into the archive, and
        # made the graph's neighbours a superset of the database's.
        if adult > NOW or (p.died is not None and p.died < adult):
            parent = next((q for q in world.parents(p) if q.home is not None), None)
            if parent is not None:
                p.home, p.moved = parent.home, p.born
                continue
            adult = min(adult, p.born)
        settled = next((world.people[i] for i in p.spouses
                        if world.people[i].home is not None), None)
        if settled is not None:
            p.home = settled.home
            p.moved = min(max(adult, settled.moved), NOW)
            continue
        partner = next((world.people[i] for i in p.spouses), None)
        household = [p] + ([partner] if partner is not None else [])
        # A dwelling falls vacant when its last occupant dies, and never while
        # one of them is alive - which is what the `None` is for. Taking the
        # max over the death years would empty a flat the widow still lives in.
        leaves = (None if any(q.alive for q in household)
                  else max(q.died or 0 for q in household))
        home = silo.take(adult, leaves)
        p.home, p.moved = home, adult
        if partner is not None:
            partner.home = home
            partner.moved = max(adult, partner.born + 18)
    world.dwellings = silo.every()
    for p in world.people:
        mother = world.people[p.mother] if p.mother is not None else None
        birthplace = mother.home if mother and mother.home else p.home
        p.birth_level = birthplace[0] if birthplace else 1


# --- classes, crews, committees -----------------------------------------------


def _school(rng: Random, world: World) -> dict[str, Cohort]:
    """One class per birth year, split when a year is too big to be a class.

    Classes are the one cohort everybody gets, living or dead, which makes "who
    was at school with X" the only group question the whole corpus can answer -
    the others are about the present, and two-thirds of these people are not in
    it.
    """
    classes: dict[str, Cohort] = {}
    nursery = DEPARTMENTS["Nursery"][0]
    by_year: dict[int, list[Person]] = {}
    for p in world.people:
        # Anyone who died before six never started. Enrolling them anyway put a
        # `class_is` edge in the graph with no `membership` row behind it: the
        # row failed its own `until >= joined` check and `INSERT OR IGNORE`
        # threw it away without a word, so the walk had classmates the table
        # did not. That is why the insert below it is no longer forgiving.
        if p.died is not None and p.died < p.born + SCHOOL[0]:
            continue
        by_year.setdefault(p.born, []).append(p)
    for year, cohort in sorted(by_year.items()):
        rng.shuffle(cohort)
        for i in range(0, len(cohort), CLASS_SIZE):
            block = i // CLASS_SIZE
            suffix = "" if block == 0 else f" ({chr(ord('B') + block - 1)})"
            name = f"Class of {year}{suffix}"
            classes[name] = Cohort(name, "class", year + SCHOOL[0], nursery)
            for p in cohort[i:i + CLASS_SIZE]:
                p.school = name
    return classes


def _crews(rng: Random, world: World) -> dict[str, Cohort]:
    """Work crews, for the living only.

    A department and a shift is two hundred people, which is a payroll rather
    than a set of colleagues. A crew is the eleven or so who actually work
    together, and it is what `coworker` means here. The dead keep their
    department and lose their crew, which is a coverage gap by construction
    rather than by accident - `questions.py` reports it as one.
    """
    crews: dict[str, Cohort] = {}
    buckets: dict[tuple[str, str], list[Person]] = {}
    for p in world.people:
        if p.alive and NOW - p.born >= WORKING_AGE:
            buckets.setdefault((p.department, p.shift), []).append(p)
    for (dept, shift), staff in sorted(buckets.items()):
        rng.shuffle(staff)
        for i in range(0, len(staff), CREW_SIZE):
            name = f"{dept} {shift.split()[0]} Crew {i // CREW_SIZE + 1}"
            crews[name] = Cohort(name, "crew", NOW - 40, DEPARTMENTS[dept][0])
            for p in staff[i:i + CREW_SIZE]:
                p.crew = name
    return crews


def _committees(rng: Random, world: World) -> dict[str, Cohort]:
    """Seats with terms, refilled as they fall vacant, for 220 years.

    Terms are why `committee_mate` needs an overlap test rather than a join:
    two people on the Pact Review Committee a century apart are not colleagues,
    and every other group here is small enough that the distinction would never
    come up.
    """
    rows: dict[str, Cohort] = {}
    adults: dict[int, list[Person]] = {}
    for p in world.people:
        end = p.died if p.died is not None else NOW
        for year in range(p.born + 25, end + 1):
            adults.setdefault(year, []).append(p)
    for name in COMMITTEES:
        formed = rng.randint(5, 40)
        rows[name] = Cohort(name, "committee", formed, None)
        for _ in range(rng.randint(*COMMITTEE_SEATS)):
            year = formed
            while year < NOW:
                pool = adults.get(year, [])
                if not pool:
                    break
                person = rng.choice(pool)
                ends = min(year + rng.randint(*TERM),
                           person.died if person.died is not None else NOW)
                if ends > year and all(s[0] != name for s in person.seats):
                    person.seats.append((name, year, None if ends >= NOW else ends))
                year = max(ends, year) + 1
    return rows


def populate(rng: Random, seed: int, target: int) -> World:
    """Seven cohorts, each the children of the one before, then everything else.

    Generation sizes are scaled to whatever ``target`` asks for, so a small
    corpus is the same shape as the full one rather than a prefix of it - a
    prefix would be all founders and would answer no kinship question at all.
    """
    scale = target / sum(GENERATIONS)
    sizes = [max(2, round(n * scale)) for n in GENERATIONS]
    sizes[-1] += target - sum(sizes)

    names = Names(seed)
    people: list[Person] = []
    for index in range(sizes[0]):
        founder = names.person(rng, index, index % 2 == 0, names.surname(), 0,
                               rng.randint(0, 22))
        _assign_work(rng, founder, ())
        people.append(founder)
    for generation, quota in enumerate(sizes[1:], start=1):
        cohort = [p for p in people if p.generation == generation - 1]
        couples = _pairs(rng, cohort, people)
        couples += _remarriages(rng, cohort, couples, people)
        people.extend(_children(rng, names, couples, quota, generation, people))
    _die(rng, people)
    latest = max(p.born for p in people)
    if latest >= NOW:
        # Seven generations at 22-34 years apart run to about year 216, so NOW
        # is not a free parameter - it is a consequence of the birth ages and
        # the generation count. Getting it wrong is silent: everyone born after
        # the present year is alive, has a dwelling dated in the future, and
        # turns up as a neighbour of people who have never met them.
        raise SystemExit(
            f"the last person is born in year {latest} and the archive is "
            f"dated {NOW}; raise NOW or shorten BIRTH_AGE/GENERATIONS")

    world = World(people)
    _house(rng, world)
    world.cohorts.update(_school(rng, world))
    world.cohorts.update(_crews(rng, world))
    world.cohorts.update(_committees(rng, world))
    return world


# --- writing ------------------------------------------------------------------
#
# Facts and edges come out of the same pass over the same people, so the `fact`
# table and the `edge` table cannot disagree about what the corpus says. In the
# Wikipedia pipeline `libgraph.build` derives one from the other and the
# derivation is where the interesting failures live; here there is nothing to
# derive, and simulating that would be simulating a bug rather than a silo.


def level(n: int) -> str:
    return f"Level {n}"


def _lead(world: World, p: Person) -> str:
    """A paragraph of prose per person, for the search index to chew on.

    Every fact in it is also an edge, which makes this a fair place to ask what
    the graph buys over full-text search of the leads: the answer is in the
    text for a reader, and getting it out of the text is comprehension.
    """
    who = "He" if p.male else "She"
    out = [f"{p.name} was born in year {p.born} on {level(p.birth_level)} "
           f"of {SILO}."]
    parents = [q.name for q in world.parents(p)]
    if parents:
        out.append(f"{who} is the child of {' and '.join(parents)}.")
    out.append(f"{who} works as a {p.job} in {p.department} on {p.shift}, "
               f"and lives at {p.address} in {section_of(p.home[0] if p.home else 1)}.")
    if p.spouses:
        married = " and later ".join(world.people[i].name for i in p.spouses)
        out.append(f"{who} married {married}.")
    if p.school:
        out.append(f"{who} was schooled with the {p.school}.")
    if p.crew:
        out.append(f"{who} serves on {p.crew}.")
    if p.died is not None:
        end = "was sent to clean" if p.fate == "Cleaning" else "died"
        out.append(f"{who} {end} in year {p.died}, aged {p.died - p.born}.")
    return " ".join(s for s in out if s)


def _facts(world: World, p: Person) -> Iterator[tuple[str, int, str, str, float | None]]:
    """(property, ordinal, value, kind, num) for one person."""
    yield "born", 0, str(p.born), "number", float(p.born)
    yield "sex", 0, "male" if p.male else "female", "text", None
    yield "generation", 0, str(p.generation), "number", float(p.generation)
    yield "birth_level", 0, level(p.birth_level), "text", None
    yield "address", 0, p.address, "text", None
    yield "occupation", 0, p.job, "text", None
    yield "department", 0, p.department, "text", None
    yield "shift", 0, p.shift, "text", None
    if p.father is not None:
        # `recorded_father` is None for everybody the planter did not touch,
        # so this reads as `world.people[p.father].name` in the ordinary case.
        yield ("father", 0, p.recorded_father or world.people[p.father].name,
               "text", None)
    if p.mother is not None:
        yield "mother", 0, world.people[p.mother].name, "text", None
    for ordinal, spouse in enumerate(p.spouses):
        yield "spouse", ordinal, world.people[spouse].name, "text", None
    if p.school:
        yield "class", 0, p.school, "text", None
    if p.crew:
        yield "crew", 0, p.crew, "text", None
    for ordinal, (seat, _, _) in enumerate(p.seats):
        yield "committee", ordinal, seat, "text", None
    if p.died is not None:
        yield "died", 0, str(p.died), "number", float(p.died)
        yield "fate", 0, p.fate or "", "text", None


def _person_edges(world: World, p: Person) -> Iterator[tuple[str, str]]:
    """(relation, object) for one person - base facts, nothing derived.

    `child_of` duplicates `father_is` and `mother_is` on purpose. It is the
    relation a question about "a parent" needs, and keeping it separate is what
    lets `questions.py` show the difference between a walk that names which
    parent it wants and one that takes whichever edge comes back first.

    `lives_at` is written only for the living. The table keeps 220 years of
    tenancies; the graph carries the present, because a graph has no notion of
    time and "who lives next door" answered across three centuries of occupants
    is a wrong answer rather than a broad one.
    """
    if p.father is not None:
        yield "father_is", world.people[p.father].name
        yield "child_of", world.people[p.father].name
    if p.mother is not None:
        yield "mother_is", world.people[p.mother].name
        yield "child_of", world.people[p.mother].name
    for spouse in p.spouses:
        yield "spouse_of", world.people[spouse].name
    yield "born_on", level(p.birth_level)
    yield "works_in", p.department
    yield "job_is", p.job
    yield "shift_is", p.shift
    if p.alive:
        yield "lives_at", p.address
    if p.school:
        yield "class_is", p.school
    if p.crew:
        yield "crew_is", p.crew
    for seat, _, _ in p.seats:
        yield "sits_on", seat


def _categories(p: Person) -> Iterator[str]:
    yield f"People of {p.department}"
    yield f"Born on {level(p.birth_level)}"
    yield f"Generation {p.generation}"
    yield "Living" if p.alive else "Dead"
    if p.fate == "Cleaning":
        yield "Sent to clean"


#: property -> the relation its edge carries, or None where there is nothing to
#: point at. Recorded in the `property` table so that "what does this corpus
#: know that nothing understands" stays a query rather than an afternoon.
PROPERTY_RELATION: dict[str, str | None] = {
    "father": "father_is", "mother": "mother_is", "spouse": "spouse_of",
    "address": "lives_at", "birth_level": "born_on", "department": "works_in",
    "occupation": "job_is", "shift": "shift_is", "class": "class_is",
    "crew": "crew_is", "committee": "sits_on",
    "born": None, "died": None, "generation": None, "fate": None, "sex": None,
}


def write(db: sqlite3.Connection, world: World, seed: int,
          planted: int = 0) -> dict[str, int]:
    """Everything, in one transaction."""
    cohorts = world.cohorts
    for table in ("residence", "membership", "apartment", "cohort", "edge",
                  "entity_type", "property", "category", "fact", "article"):
        db.execute(f"DELETE FROM {table} WHERE source = ?", (SOURCE,))

    articles: list[tuple[str, str, str]] = []
    edges: list[tuple[str, str, str, str]] = []
    types: list[tuple[str, str, str]] = []

    articles.append((SOURCE, SILO,
                     f"{SILO} holds {LEVELS} levels and everyone in this archive."))
    types.append((SOURCE, "silo", SILO))
    for name, low, high in SECTIONS:
        articles.append((SOURCE, name, f"{name} is the run of {SILO} from "
                                       f"{level(low)} to {level(high)}."))
        edges.append((SOURCE, name, "located_in", SILO))
        types.append((SOURCE, "section", name))
    for n in range(1, LEVELS + 1):
        articles.append((SOURCE, level(n),
                         f"{level(n)} of {SILO} lies in {section_of(n)}."))
        edges.append((SOURCE, level(n), "located_in", section_of(n)))
        types.append((SOURCE, "level", level(n)))
    for dept, (home, jobs) in DEPARTMENTS.items():
        articles.append((SOURCE, dept,
                         f"{dept} is headquartered on {level(home)} of {SILO} "
                         f"and employs {', '.join(jobs).lower()}."))
        edges.append((SOURCE, dept, "located_in", level(home)))
        types.append((SOURCE, "department", dept))
        for job in jobs:
            articles.append((SOURCE, job, f"A {job} works in {dept}."))
            edges.append((SOURCE, job, "part_of", dept))
            types.append((SOURCE, "occupation", job))
    for shift in SHIFTS:
        articles.append((SOURCE, shift, f"{shift} is one of three worked in {SILO}."))
        types.append((SOURCE, "shift", shift))
    for name, group in sorted(cohorts.items()):
        articles.append((SOURCE, name, f"{name} is a {group.kind} of {SILO}, "
                                       f"formed in year {group.formed}."))
        types.append((SOURCE, group.kind, name))

    articles += _dwelling_articles(world)
    edges += _dwelling_edges(world)
    types += [(SOURCE, "apartment", schema.address(*d)) for d in world.dwellings]

    facts: list[tuple[str, str, str, int, str, str, float | None]] = []
    filings: list[tuple[str, str, str]] = []
    for p in world.people:
        articles.append((SOURCE, p.name, _lead(world, p)))
        types.append((SOURCE, "person", p.name))
        types.append((SOURCE, "man" if p.male else "woman", p.name))
        if p.generation == 0:
            types.append((SOURCE, "founder", p.name))
        if p.alive:
            types.append((SOURCE, "living", p.name))
        facts += [(SOURCE, p.name, *row) for row in _facts(world, p)]
        edges += [(SOURCE, p.name, *row) for row in _person_edges(world, p)]
        filings += [(SOURCE, p.name, name) for name in _categories(p)]

    db.executemany("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                   articles)
    db.executemany("INSERT INTO fact (source, subject, property, ordinal, value, "
                   "kind, num) VALUES (?, ?, ?, ?, ?, ?, ?)", facts)
    db.executemany("INSERT INTO edge (source, subject, relation, object) "
                   "VALUES (?, ?, ?, ?)", edges)
    db.executemany("INSERT INTO entity_type (source, kind, entity) "
                   "VALUES (?, ?, ?)", types)
    db.executemany("INSERT INTO category (source, title, name) "
                   "VALUES (?, ?, ?)", filings)
    db.executemany("INSERT INTO apartment (source, floor, bearing, ring) "
                   "VALUES (?, ?, ?, ?)",
                   [(SOURCE, *d) for d in world.dwellings])
    db.executemany("INSERT INTO residence (source, person, floor, bearing, ring, "
                   "since, until) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   [(SOURCE, p.name, *p.home, p.moved,
                     max(p.died, p.moved) if p.died is not None else None)
                    for p in world.people if p.home])
    db.executemany("INSERT INTO cohort (source, name, kind, formed, floor) "
                   "VALUES (?, ?, ?, ?, ?)",
                   [(SOURCE, g.name, g.kind, g.formed, g.floor)
                    for _, g in sorted(cohorts.items())])
    # Not `INSERT OR IGNORE`. A membership that violates its own date check is
    # a bug in the simulation, and forgiving it here is how a class ended up
    # with a pupil the class list had never heard of.
    db.executemany("INSERT INTO membership (source, person, cohort, "
                   "role, joined, until) VALUES (?, ?, ?, ?, ?, ?)",
                   list(_memberships(world, cohorts)))
    _write_properties(db)
    db.execute("INSERT INTO article_fts(article_fts) VALUES('rebuild')")
    # Not housekeeping. Every kinship view is a three- or four-way self-join of
    # `edge`, and without statistics the planner has no way to tell a lookup
    # that returns 2 rows from one that returns 20,000: it joined `p` to `u` as
    # a cross product and `cousin` had not finished after three minutes. With
    # `sqlite_stat1` in front of it the same query takes 0.09s. Four hundred
    # milliseconds, once, at the end of a build.
    db.execute("ANALYZE")
    db.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [(f"{SOURCE}.generated", time.strftime("%Y-%m-%dT%H:%M:%S")),
         (f"{SOURCE}.seed", str(seed)),
         (f"{SOURCE}.people", str(len(world))),
         (f"{SOURCE}.now", str(NOW)),
         (f"{SOURCE}.schema_version", str(schema.SCHEMA_VERSION)),
         (f"{SOURCE}.generations", str(len(GENERATIONS))),
         (f"{SOURCE}.planted", str(planted))])
    db.commit()
    return {"articles": len(articles), "facts": len(facts), "edges": len(edges),
            "categories": len(filings), "dwellings": len(world.dwellings)}


def _memberships(world: World, cohorts: dict[str, Cohort],
                 ) -> Iterator[tuple[str, str, str, str, int, int | None]]:
    """One row per person per group, with the years they were in it.

    School leaves early for anyone who died at school, which is the only reason
    `until` is not simply the leaving age: a class list that keeps a dead child
    on it until they would have been sixteen makes `classmate` true of people
    who never met.
    """
    for p in world.people:
        if p.school:
            leaves = p.born + SCHOOL[1]
            yield (SOURCE, p.name, p.school, "pupil", p.born + SCHOOL[0],
                   min(leaves, p.died) if p.died is not None else leaves)
        if p.crew:
            joined = max(p.born + WORKING_AGE, cohorts[p.crew].formed)
            yield SOURCE, p.name, p.crew, "crew", joined, None
        for seat, joined, until in p.seats:
            yield SOURCE, p.name, seat, "seat", joined, until


def _dwelling_articles(world: World) -> list[tuple[str, str, str]]:
    return [(SOURCE, schema.address(floor, bearing, ring),
             f"Apartment {schema.address(floor, bearing, ring)} is a dwelling on "
             f"{level(floor)} of {SILO}, ring {ring}, in {section_of(floor)}.")
            for floor, bearing, ring in world.dwellings]


def _dwelling_edges(world: World) -> list[tuple[str, str, str, str]]:
    """Containment, and the two adjacencies that make a ring walkable.

    The neighbour relation is geometry, and geometry is arithmetic: `(bearing +
    30) % 720`. An eZ80 walking a card has no modulo and no multiply, so the
    arithmetic is done once, here, and shipped as edges. `next_along` is thirty
    minutes clockwise and wraps; `next_out` is one ring outward and does not.
    Their inverses come free - the card stores the reverse table anyway.
    """
    known = set(world.dwellings)
    rows: list[tuple[str, str, str, str]] = []
    for floor, bearing, ring in world.dwellings:
        here = schema.address(floor, bearing, ring)
        rows.append((SOURCE, here, "located_in", level(floor)))
        along = (floor, (bearing + BEARING_STEP) % (BEARINGS * BEARING_STEP), ring)
        if along in known:
            rows.append((SOURCE, here, "next_along", schema.address(*along)))
        if ring != RINGS[-1]:
            out = (floor, bearing, RINGS[RINGS.index(ring) + 1])
            if out in known:
                rows.append((SOURCE, here, "next_out", schema.address(*out)))
    return rows


def _write_properties(db: sqlite3.Connection) -> None:
    rows = db.execute(
        "SELECT property, COUNT(*), COUNT(DISTINCT subject) FROM fact "
        "WHERE source = ? GROUP BY property", (SOURCE,)).fetchall()
    db.executemany(
        "INSERT OR REPLACE INTO property (source, name, uses, subjects, relation) "
        "VALUES (?, ?, ?, ?, ?)",
        [(SOURCE, name, uses, subjects, PROPERTY_RELATION.get(name))
         for name, uses, subjects in rows])


# --- reporting ----------------------------------------------------------------


def stats(db: sqlite3.Connection) -> None:
    for key, value in db.execute(
            "SELECT key, value FROM meta WHERE key LIKE ? ORDER BY key",
            (f"{SOURCE}.%",)):
        print(f"  {key:<26} {value}")
    print()
    for label, sql in (
            ("articles", "SELECT COUNT(*) FROM article WHERE source = ?"),
            ("facts", "SELECT COUNT(*) FROM fact WHERE source = ?"),
            ("edges", "SELECT COUNT(*) FROM edge WHERE source = ?"),
            ("typed entities", "SELECT COUNT(*) FROM entity_type WHERE source = ?"),
            ("category filings", "SELECT COUNT(*) FROM category WHERE source = ?"),
            ("dwellings", "SELECT COUNT(*) FROM apartment WHERE source = ?"),
            ("occupied", "SELECT COUNT(DISTINCT floor || bearing || ring) "
                         "FROM residence WHERE source = ?"),
            ("cohorts", "SELECT COUNT(*) FROM cohort WHERE source = ?"),
            ("memberships", "SELECT COUNT(*) FROM membership WHERE source = ?")):
        print(f"  {label:<26} {db.execute(sql, (SOURCE,)).fetchone()[0]:,}")
    print()
    print("  stored, by relation")
    for relation, n in db.execute(
            "SELECT relation, COUNT(*) FROM edge WHERE source = ? "
            "GROUP BY relation ORDER BY COUNT(*) DESC", (SOURCE,)):
        print(f"    {relation:<24} {n:,}")
    print()
    print("  derivable, by view")
    for view in ("sibling", "grandparent", "aunt_or_uncle", "cousin", "housemate",
                 "neighbour", "coworker", "classmate", "committee_mate"):
        n = db.execute(f"SELECT COUNT(*) FROM {view} WHERE source = ?",
                       (SOURCE,)).fetchone()[0]
        print(f"    {view:<24} {n:,}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--people", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--stats", action="store_true",
                    help="report on an existing database and exit")
    ap.add_argument("--plant", type=int, default=0, metavar="N",
                    help="Plant N contradictions and write the key beside the "
                         "database. Off by default: every number in the README "
                         "was measured without them, and a flag that quietly "
                         "changed the data under a measurement would be worse "
                         "than no flag. See plant.py")
    ap.add_argument("--key", type=Path,
                    help="Where the answers go (default: <db>.key.json)")
    args = ap.parse_args()

    if args.stats:
        if not args.db.exists():
            raise SystemExit(f"no database at {args.db}")
        stats(schema.connect(args.db))
        return

    started = time.monotonic()
    rng = Random(args.seed)
    world = populate(rng, args.seed, args.people)
    anomalies: list[plant.Anomaly] = []
    if args.plant:
        # After the simulation and before it is written, so the anomalies are
        # in the corpus rather than a layer over it - a player querying the
        # card must not be able to tell which rows were edited.
        anomalies = plant.plant(rng, world, args.plant)
    db = schema.connect(args.db, migrate=True)
    counts = write(db, world, args.seed, planted=len(anomalies))
    print(f"{args.db}: {len(world):,} people, {counts['articles']:,} articles, "
          f"{counts['facts']:,} facts, {counts['edges']:,} edges, "
          f"{counts['dwellings']:,} dwellings in {time.monotonic() - started:.1f}s")
    if anomalies:
        key = args.key or args.db.with_suffix(".key.json")
        plant.write_key(key, anomalies, args.seed)
        kinds = Counter(a.kind for a in anomalies)
        print(f"{key}: {len(anomalies)} planted - "
              + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))


if __name__ == "__main__":
    main()
