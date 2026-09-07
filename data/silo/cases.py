#!/usr/bin/env python3
"""
A murder nobody wrote, from the silo, provably fair.

    python data/silo/cases.py --seed 7              # the case sheet
    python data/silo/cases.py --seed 7 -o CASE.bin  # the whole silo, with it in
    python data/silo/cases.py --seeds 100           # the baseline

A Golden Age puzzle is a victim, a closed circle, motive, opportunity, and
one lie. The corpus already holds every part, so this picks rather than
invents:

    victim        a death this year the archive calls natural causes
    the circle    spouse, household, neighbours, colleagues on the same
                  shift, a committee - the views, living only
    motive        one a relation: the flat, the room, the water, the
                  promotion, the vote
    opportunity   the duty list for the night; whoever was on it has an
                  alibi
    the lie       the killer is on the duty list, and a gate log says
                  where they really were

The clues are documents: things with a sentence on them, placed across the
silo so that finding them is the game, each with a rule that marks it
found. Nothing about the death is put in the archive, because the archive
is the Voice and the Voice says natural causes - that is the premise, and
`LOOKUP` will say it to your face.

## The proof, which is the point

A case ships only if the clues eliminate every suspect but one. `Case.
survivors` reads the clues the way a player would - a name on the duty
list is an alibi unless the gate log breaks it, a household's note is an
alibi - and `generate` refuses a case where two survive, naming them. Then
`libplan` finds a walkthrough on the actual world, so every clue is
reachable before the deadline, and the deadline is set from it. No Golden
Age author could run either check.

## What version one is not

One lie shape, alibis by the duty list or a neighbour's note. Documents
from templates with holes, not prose anybody would call writing. Nobody is
ever out, because the corpus has no clock. And the circle is small - three
to eight - so "accuse the spouse" is right about one time in six, which
`--seeds` measures and prints beside everything else, because a generator
that made the spouse guilty half the time would be a generator of one
story.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import buildworld
import schema

import buildif
import libplan
import libworld
from libworld import Rule, Thing, World

DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"

#: Fewer suspects than this is not a circle; more than this is a crowd the
#: parser could not name (a clue a suspect, and one word each).
MIN_CIRCLE, MAX_CIRCLE = 3, 8

#: What each kind of suspect wanted. The relation is the motive, and every
#: suspect has one, so the puzzle turns on opportunity - which is version
#: one, and the thing to widen next.
MOTIVES: dict[str, str] = {
    "spouse": "stood to keep the flat, which a widow keeps and a divorcee "
              "does not",
    "housemate": "had been refused the second room, twice, in writing",
    "neighbour": "had the water dispute, the one the ring still talks about",
    "colleague": "was next in line for the post the deceased was not going "
                 "to leave",
    "committee": "lost the vote the deceased carried, and said so",
}

#: How a clue got into the world. `roster` and `note` are alibis; `log` breaks
#: one; `notice` and `file` are the premise and the motives.
NOTICE, FILE, ROSTER, LOG, NOTE = "notice", "file", "roster", "log", "note"

#: Where each kind of document lies. A department name, or `RING` for the
#: ring of the flat the document is about.
RING = "\x00ring"
PLACES: dict[str, str] = {NOTICE: "Cafeteria", FILE: "Judicial",
                          ROSTER: "Sheriff's Office", LOG: "IT", NOTE: RING}

#: The deadline, in turns, as a multiple of the walkthrough the planner
#: found. Three is generous; one would be a speedrun.
SLACK = 3

#: The clock is a byte. A silo is 144 levels tall, so a walkthrough that
#: crosses it twice cannot be given even twice its length before the clock
#: saturates - and a deadline tighter than that is a speedrun, not a
#: mystery. Below this slack the case has no deadline, and says so.
LEAST_SLACK = 2


def deadline_for(commands: int) -> int | None:
    """Turns before the file closes, or None when the clock cannot hold a
    fair one. A two-byte clock is the fix, and this is where it would go."""
    if LEAST_SLACK * commands > 255:
        return None
    return min(255, SLACK * commands)


class Unfair(ValueError):
    """A case the generator will not ship, and why."""


@dataclass(frozen=True)
class Suspect:
    name: str
    how: str                 # a key of MOTIVES
    flat: str | None         # `2 600 A`, or None if unhoused
    floor: int | None


@dataclass
class Clue:
    kind: str
    word: str                # the thing's name, one word
    text: str
    subject: str | None      # what CONSULT and ASK hold up
    place: str               # a department, or a ring key `Ring N`
    #: Who this clue alibis, if anyone, and who it accuses.
    alibis: str | None = None
    accuses: str | None = None


@dataclass
class Case:
    seed: int
    victim: str
    victim_flat: str
    victim_floor: int
    department: str
    shift: str
    day: int
    circle: list[Suspect]
    killer: str
    clues: list[Clue] = field(default_factory=list)
    walkthrough: list[str] = field(default_factory=list)
    #: Turns before the file closes, or None when the clock cannot hold one.
    deadline: int | None = None

    def survivors(self) -> set[str]:
        """Who the clues do not eliminate. Read the way a player reads them:
        the duty list is an alibi unless the gate log breaks it, and a
        household's note is an alibi. The killer had better be alone here."""
        alibied = {c.alibis for c in self.clues if c.alibis is not None}
        broken = {c.accuses for c in self.clues if c.accuses is not None}
        return {s.name for s in self.circle} - (alibied - broken)

    def sheet(self) -> str:
        lines = [f"seed {self.seed}: the death of {self.victim}, "
                 f"{self.department} ({self.shift}), {self.victim_flat}",
                 f"  recorded as natural causes on the {self.day}th; it was not.",
                 f"  the circle ({len(self.circle)}):"]
        for s in self.circle:
            mark = "  <- the killer" if s.name == self.killer else ""
            lines.append(f"    {s.name:28} {s.how:10} {MOTIVES[s.how]}{mark}")
        documents = [c for c in self.clues if c.text]
        lines.append(f"  the clues ({len(documents)}):")
        lines.extend(f"    {c.word:8} in {c.place:20} {c.text}" for c in documents)
        lines.append(f"  survivors after every clue: "
                     f"{', '.join(sorted(self.survivors()))}")
        when = (f"deadline turn {self.deadline}" if self.deadline is not None
                else "no deadline - the silo is taller than a one-byte clock")
        lines.append(f"  walkthrough: {len(self.walkthrough)} commands, {when}")
        return "\n".join(lines)


# --- reading the corpus ---------------------------------------------------------


def _victims(db: sqlite3.Connection) -> list[tuple[str, int, str, str, str]]:
    """Deaths in the latest year, natural causes, housed: (name, died,
    address, department, shift), by name."""
    latest = db.execute(
        "SELECT MAX(died) FROM person WHERE source = ?", (schema.SOURCE,)
    ).fetchone()[0]
    if latest is None:
        raise Unfair("nobody in this corpus has died")
    rows = db.execute(
        "SELECT name, died, address, department, shift FROM person "
        "WHERE source = ? AND died = ? AND fate = 'Natural causes' "
        "AND address IS NOT NULL AND department IS NOT NULL "
        "ORDER BY name", (schema.SOURCE, latest)).fetchall()
    if not rows:
        raise Unfair(f"no natural death in year {latest} to make a murder of")
    return rows


def _alive(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute("SELECT died FROM person WHERE source = ? AND name = ?",
                     (schema.SOURCE, name)).fetchone()
    return row is not None and row[0] is None


def _flat(db: sqlite3.Connection, name: str) -> tuple[str, int] | None:
    row = db.execute(
        "SELECT a.address, a.floor FROM residence r JOIN apartment a "
        "ON a.source = r.source AND a.floor = r.floor AND a.bearing = r.bearing "
        "AND a.ring = r.ring WHERE r.source = ? AND r.person = ? "
        "AND r.until IS NULL", (schema.SOURCE, name)).fetchone()
    return None if row is None else (row[0], row[1])


def _circle(db: sqlite3.Connection, rng: Random, victim: str,
            department: str, shift: str) -> list[Suspect]:
    """The living who were close, each once, nearest kind first."""
    found: dict[str, str] = {}

    def add(names: list[str], how: str, limit: int) -> None:
        picked = [n for n in names if n != victim and n not in found
                  and _alive(db, n)]
        rng.shuffle(picked)
        found.update((name, how) for name in picked[:limit])

    add([r for (r,) in db.execute(
        "SELECT object FROM edge WHERE source = ? AND subject = ? "
        "AND relation = 'spouse_of'", (schema.SOURCE, victim))], "spouse", 1)
    add([r for (r,) in db.execute(
        "SELECT housemate FROM housemate WHERE source = ? AND person = ?",
        (schema.SOURCE, victim))], "housemate", 2)
    add([r for (r,) in db.execute(
        "SELECT neighbour FROM neighbour WHERE source = ? AND person = ?",
        (schema.SOURCE, victim))], "neighbour", 2)
    add([r for (r,) in db.execute(
        "SELECT name FROM person WHERE source = ? AND department = ? "
        "AND shift = ? AND died IS NULL ORDER BY name",
        (schema.SOURCE, department, shift))], "colleague", 2)
    add([r for (r,) in db.execute(
        "SELECT committee_mate FROM committee_mate WHERE source = ? "
        "AND person = ?", (schema.SOURCE, victim))], "committee", 1)

    suspects = [_suspect(db, name, how) for name, how in found.items()]
    suspects.sort(key=lambda s: s.name)
    return suspects[:MAX_CIRCLE]


def _suspect(db: sqlite3.Connection, name: str, how: str) -> Suspect:
    flat = _flat(db, name)
    if flat is None:
        return Suspect(name, how, None, None)
    return Suspect(name, how, flat[0], flat[1])


# --- making the case --------------------------------------------------------------


def generate(db: sqlite3.Connection, seed: int) -> Case:
    """A case from this corpus and this seed, or `Unfair` with the reason.

    The same corpus and seed give the same case, because a mystery two
    builds disagreed about would be two mysteries.
    """
    rng = Random(seed)
    victims = _victims(db)
    rng.shuffle(victims)
    for name, _died, address, department, shift in victims:
        circle = _circle(db, rng, name, department, shift)
        if len(circle) < MIN_CIRCLE:
            continue
        housed = [s for s in circle if s.flat is not None]
        if not housed:
            continue
        killer = rng.choice(housed)          # accused at their own door
        floor = int(address.split()[0])
        case = Case(seed=seed, victim=name, victim_flat=address,
                    victim_floor=floor, department=department, shift=shift,
                    day=rng.randint(2, 28), circle=circle, killer=killer.name)
        _write_clues(rng, case)
        survivors = case.survivors()
        if survivors != {killer.name}:
            raise Unfair(f"the clues leave {', '.join(sorted(survivors))} "
                         f"standing; a case needs one")
        return case
    raise Unfair(f"no death this year has a circle of {MIN_CIRCLE}")


def _write_clues(rng: Random, case: Case) -> None:
    """The documents. Every innocent gets one alibi - the duty list or a
    neighbour's note, by the seed - and the killer gets the lie and the
    log that breaks it. `survivors` is what checks this did what it says."""
    victim = case.victim
    circle = case.circle
    killer = next(s for s in circle if s.name == case.killer)

    case.clues.append(Clue(
        NOTICE, "notice",
        f"Death notice, {case.day}th. {victim}, {case.department}, {case.shift}, "
        f"of {case.victim_flat}: found at 0500, natural causes per Judicial. "
        f"The door was not forced.", victim, PLACES[NOTICE]))

    quarrels = "; ".join(f"{s.name} {MOTIVES[s.how]}" for s in circle)
    case.clues.append(Clue(
        FILE, "file",
        f"Judicial file on {victim}, persons of interest: {quarrels}.",
        victim, PLACES[FILE]))

    on_duty = [killer.name]
    noted: list[Suspect] = []
    for s in circle:
        if s.name == killer.name:
            continue
        if s.flat is not None and rng.random() < 0.5:
            noted.append(s)
        else:
            on_duty.append(s.name)
    rng.shuffle(on_duty)
    case.clues.append(Clue(
        ROSTER, "roster",
        f"Duty list, night of the {case.day}th, all departments. Logged on "
        f"post from 2200 to 0600: {', '.join(on_duty)}. Signed by the "
        f"dispatcher.", None, PLACES[ROSTER]))
    # One thing carries the whole list. These rows are how `survivors` reads
    # it, one alibi a name, and they have no text so nothing emits them.
    case.clues.extend(Clue(ROSTER, "duty", "", None, PLACES[ROSTER], alibis=n)
                      for n in on_duty)

    case.clues.append(Clue(
        LOG, "log",
        f"Stair gate log, Level {case.victim_floor}, night of the {case.day}th. "
        f"0310: {killer.name}, through, down. No return logged before 0600.",
        killer.name, PLACES[LOG], accuses=killer.name))

    for index, s in enumerate(noted):
        assert s.flat is not None and s.floor is not None
        case.clues.append(Clue(
            NOTE, f"note{index + 1}",
            f"A note in a neighbour's hand: {s.name} of {s.flat} was home all "
            f"night on the {case.day}th. The light was on and the door shut. "
            f"Signed, the household next along.",
            s.name, f"Ring {s.floor}", alibis=s.name))


# --- the world -------------------------------------------------------------------


def build_world(db: sqlite3.Connection, case: Case) -> World:
    """The whole silo with the case in it, planned, with the deadline set.

    Raises `Unfair` if the planner cannot reach every clue and the
    accusation, because a clue nobody can reach is the author cheating.
    """
    world = buildworld.build(db, buildworld.ALL, seeded=False)
    rooms = {room.name: i for i, room in enumerate(world.rooms)}

    # Every suspect answers at their own door, so that the door is who is
    # accused - the first-named tenant otherwise.
    subjects = {s.flat: s.name for s in case.circle if s.flat is not None}
    for door in world.doors:
        title = _door_flat(world, door)
        if title in subjects:
            door.subject = subjects[title]

    # The goal is met in order, so the clues go into it by how far down the
    # stair they lie: the planner collecting them in file order walked the
    # silo four times on the first real case, and a walkthrough that long
    # is a deadline nobody can be given.
    depth = _depths(world)
    emitted = sorted(
        (c for c in case.clues if c.text),
        key=lambda c: depth.get(rooms.get(_room_for(rooms, c.place), -1), 0))
    flags_from = world.flags - len(emitted) - 2
    if flags_from < 0:
        raise Unfair(f"{len(emitted)} clues want more flags than the world "
                     f"reserves")
    world.won, world.lost = world.flags - 2, world.flags - 1
    found: list[int] = []
    for index, clue in enumerate(emitted):
        key = _room_for(rooms, clue.place)
        if key not in rooms:
            raise Unfair(f"{clue.word} belongs in {clue.place}, which is not a "
                         f"room")
        world.things.append(Thing(clue.word, clue.text, rooms[key],
                                  subject=clue.subject))
        thing = len(world.things) - 1
        flag = flags_from + index
        found.append(flag)
        world.rules.append(Rule(when=[(libworld.C_HAVE, thing)],
                                then=[(libworld.A_SET, flag, 0)]))

    world.culprit = case.killer
    world.win_text = (f"There is a long silence at the door of {case.killer}. "
                      f"Then: 'The gate log. I thought nobody read the gate "
                      f"log.' The deputy is already on the stair.")
    world.lose_text = ("The door of the accused stays shut, and the deputy "
                       "looks at his boots. One accusation is what the Pact "
                       "allows, and it was the wrong one.")
    world.goal = [(libworld.C_FLAG, f) for f in found] + [
        (libworld.C_FLAG, world.won), (libworld.C_NFLAG, world.lost)]
    world.messages.append(
        f"The {case.day + 3}th. Judicial closes the file on {case.victim}: "
        f"natural causes, as recorded. Whatever you found stays found, and "
        f"unheard.")
    world.check()

    walkthrough = libplan.plan(world)
    if walkthrough is None:
        raise Unfair("the planner cannot reach the clues and the accusation: "
                     + libplan.explain(world))
    case.walkthrough = walkthrough
    case.deadline = deadline_for(len(walkthrough))
    if case.deadline is not None:
        world.rules.append(Rule(
            when=[(libworld.C_TURN, case.deadline)],
            then=[(libworld.A_PRINT, len(world.messages) - 1, 0),
                  (libworld.A_SET, world.lost, 0)]))
        world.check()
        if libplan.plan(world) is None:
            raise Unfair("the deadline cut the walkthrough: "
                         + libplan.explain(world))
    return world


def _room_for(rooms: dict[str, int], place: str) -> str:
    """A department name as it is, or `Ring N` as the ring room's name."""
    return place if place in rooms else f"Level {place.split()[1]}, the ring"


def _depths(world: World) -> dict[int, int]:
    """Every room's distance from the start, which on a silo is how far down
    the stair it is."""
    depth = {world.start: 0}
    queue = [world.start]
    while queue:
        here = queue.pop(0)
        for target in world.rooms[here].exits.values():
            if target not in depth:
                depth[target] = depth[here] + 1
                queue.append(target)
    return depth


def _door_flat(world: World, door: libworld.Door) -> str | None:
    """`600A` on `Level 2, the ring` -> `2 600 A`, the apartment's address."""
    name = world.rooms[door.room].name
    if not name.endswith(", the ring"):
        return None
    floor = name.split()[1].rstrip(",")
    bearing, ring = door.name[:-1], door.name[-1]
    return f"{floor} {bearing} {ring}"


# --- the baseline ----------------------------------------------------------------


def baseline(db: sqlite3.Connection, seeds: int) -> str:
    """Over `seeds` cases: how often the spouse did it, how big a circle is,
    how long a walkthrough is. The first is the number that would make this
    a generator of one story."""
    spouse = made = refused = 0
    sizes: list[int] = []
    for seed in range(seeds):
        try:
            case = generate(db, seed)
        except Unfair:
            refused += 1
            continue
        made += 1
        sizes.append(len(case.circle))
        killer = next(s for s in case.circle if s.name == case.killer)
        spouse += killer.how == "spouse"
    if not made:
        return f"{refused} of {seeds} seeds refused, none made"
    return (f"{made} cases from {seeds} seeds ({refused} refused)\n"
            f"  circle {statistics.mean(sizes):.1f} suspects, "
            f"{min(sizes)} to {max(sizes)}\n"
            f"  the spouse did it {spouse / made:.0%} of the time - "
            f"a guess, against 1 in {statistics.mean(sizes):.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=0,
                    help="run this many seeds and print the baseline instead")
    ap.add_argument("-o", "--out", type=Path, help="write the eZ80 binary")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        if args.seeds:
            print(baseline(db, args.seeds))
            return 0
        try:
            case = generate(db, args.seed)
            world = build_world(db, case)
        except Unfair as why:
            print(f"refused: {why}", file=sys.stderr)
            return 1
    finally:
        db.close()

    print(case.sheet())
    print("  " + " / ".join(case.walkthrough))
    if args.out:
        image = buildif.build(world).build()
        args.out.write_bytes(image)
        print(f"  wrote {args.out}: {len(image):,} bytes, "
              f"{len(world.rooms)} rooms, {len(world.doors)} doors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
