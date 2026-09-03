#!/usr/bin/env python3
"""
The silo you can walk, compiled out of the silo you can ask about.

    python data/silo/buildworld.py                       # the stair and the departments
    python data/silo/buildworld.py --floors 2 -o SILO.bin
    python data/silo/buildworld.py --floors all -o SILO.bin   # the whole silo

`worlds.py` hand-authors six rooms, and says so: it exists because `buildif`
needs a world to emit, not because six rooms is an Interactive Fiction. This
reads the world out of `data/silo.db` instead, and the reason is that the map
is already in there and has been since the schema was written:

    apartment(floor, bearing, ring)     10,368 possible, 2,088 opened
    next_along                          thirty minutes clockwise, and it wraps
    next_out                            one ring outward, and it does not
    located_in                          a dwelling's level, a department's level

`data/silo/schema.py` stores those two adjacencies as **edges rather than
arithmetic**, because the machine that has to walk them has no modulo. The
card walks them to answer "who lives next door". The world used to walk
them too - every dwelling was a room, `next_along` was `EAST` - and that is
the design this file replaced, for a reason that was never the geometry.

## A dwelling is a door, not a room

`libworld.NOWHERE` is `0xFF` and a room id is one byte, so 255 is the
ceiling. 144 landings and 14 departments are 158 rooms, one residential
floor of 72 dwellings is 72 more, and **one floor fit and two did not**. The
silo has twenty-nine opened floors and 2,088 dwellings.

What a player wants from a dwelling is the name beside the door, and a
one-sentence room they cannot enter is not a room. So an opened floor is
*one* room - the corridor round the stair - and its 72 dwellings are 72
doors on it, knocked on by their number:

    > west
    Level 42, the ring
    > knock 600A
    Apartment 42 600 A is a dwelling. The name beside the door is
    Alexander E. Wong.

A door is image and nothing else - `libworld.Door` has no overlay byte,
because nothing about a door ever changes - and `KNOCK` scans only the
doors of the room the player is in, which is why `600A` on every floor is
the design rather than a collision. That is:

    144 landings + 14 departments + 29 rings    187 rooms, of 255
    2,088 doors                                 no overlay at all

The whole silo fits, with room ids to spare for whatever an author adds.

What it gives up is walking the ring. `next_along` and `next_out` are still
the card's business - "who lives next door" is a question - but a world no
longer turns them into `EAST` and `NORTH`, and the stair is the only thing a
floor has a direction to. The doors still come out of the `apartment` table
in bearing order and the names beside them out of `residence`, so nothing
here invents geography, and nothing here invents prose beyond the one
sentence that says what a corridor is.

`build` still refuses rather than truncating past 255, and names the number.
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import items
import schema

import buildif
from libworld import NOWHERE, Door, Room, Thing, World

#: The database the rest of `data/silo/` defaults to.
DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"

#: Every floor that has dwellings, for `build(db, ALL)`.
ALL = "all"

#: The ring is off the landing to the west; the stair is east of the ring.
#: The one exit here the database does not contain, because `next_out` stops
#: at ring A rather than pointing inward at the stair.
ONTO, OFF = "WEST", "EAST"


class TooManyRooms(ValueError):
    """More rooms than a one-byte room id can name."""


def _articles(db: sqlite3.Connection) -> dict[str, str]:
    return dict(db.execute(
        "SELECT title, lead FROM article WHERE source = ?", (schema.SOURCE,)))


def _levels(db: sqlite3.Connection) -> list[int]:
    """Every level with an article, which is every level in the silo."""
    rows = db.execute(
        "SELECT entity FROM entity_type WHERE source = ? AND kind = 'level'",
        (schema.SOURCE,)).fetchall()
    return sorted(int(name.rsplit(" ", 1)[1]) for (name,) in rows)


def _departments(db: sqlite3.Connection) -> dict[str, int]:
    """Department -> the level it is headquartered on, from `located_in`."""
    rows = db.execute(
        "SELECT e.subject, e.object FROM edge e JOIN entity_type t "
        "  ON t.source = e.source AND t.entity = e.subject "
        "WHERE e.source = ? AND e.relation = 'located_in' "
        "  AND t.kind = 'department'", (schema.SOURCE,)).fetchall()
    return {dept: int(level.rsplit(" ", 1)[1]) for dept, level in rows}


def _dwellings(db: sqlite3.Connection, floor: int) -> list[str]:
    rows = db.execute(
        "SELECT address FROM apartment WHERE source = ? AND floor = ? "
        "ORDER BY bearing, ring", (schema.SOURCE, floor))
    return [address for (address,) in rows]


def _opened(db: sqlite3.Connection) -> tuple[int, ...]:
    """Every floor with a dwelling on it, which is what `ALL` means."""
    rows = db.execute(
        "SELECT DISTINCT floor FROM apartment WHERE source = ? ORDER BY floor",
        (schema.SOURCE,))
    return tuple(floor for (floor,) in rows)


def door_word(address: str) -> str:
    """`42 600 A` -> `600A`: the bearing and ring, which is what is painted
    on the door. The floor is the room the door is on."""
    _floor, bearing, ring = address.split()
    return f"{bearing}{ring}"


def _occupants(db: sqlite3.Connection, floor: int) -> dict[str, list[str]]:
    """address -> who lives there now, for the names on the door.

    `until IS NULL` is the living tenancy. Everybody else who ever had the
    flat is still in `residence`, and is the card's business rather than the
    world's - which is the division this whole pair of programs is about.

    A list, because a flat is a household: the shipped corpus puts three
    Butlers behind `2 600 A`, and a door that named one of them would be
    quietly wrong about the other two.
    """
    rows = db.execute(
        "SELECT a.address, r.person FROM residence r "
        "JOIN apartment a ON a.source = r.source AND a.floor = r.floor "
        "  AND a.bearing = r.bearing AND a.ring = r.ring "
        "WHERE r.source = ? AND r.floor = ? AND r.until IS NULL "
        "ORDER BY r.person", (schema.SOURCE, floor))
    household: dict[str, list[str]] = {}
    for address, person in rows:
        household.setdefault(address, []).append(person)
    return household


def beside_the_door(names: list[str]) -> str:
    """The sentence a door says about who lives there."""
    if not names:
        return "Nobody has the key to this one."
    if len(names) == 1:
        return f"The name beside the door is {names[0]}."
    return (f"The names beside the door are {', '.join(names[:-1])} and "
            f"{names[-1]}.")


def build(db: sqlite3.Connection, floors: tuple[int, ...] | str = (),
          seeded: bool = True, notes: list[str] | None = None) -> World:
    """The stair, the departments off it, and a ring of doors on `floors`.

    `floors` is a tuple of levels, or `ALL` for every level with dwellings.
    Raises `TooManyRooms` rather than dropping the tail, because a world that
    is quietly missing its bottom forty levels walks perfectly well.

    With `seeded`, `data/silo/items.py` is placed as well - ten things, nine
    of which hang off one cleaning and name the next place to stand.

    `notes` is the build log, and it exists for one reason: an item can fail
    to be placed for two legitimate causes - a corpus with no cleaning in it,
    and a floor that was not opened with `--floors` - and a chain with a link
    missing is worse than a chain that says which link. Nine things out of ten
    is indistinguishable from a world that was always meant to hold nine.
    """
    leads = _articles(db)
    levels = _levels(db)
    if not levels:
        raise ValueError(f"no levels in this database; is source "
                         f"{schema.SOURCE!r} written?")
    departments = _departments(db)

    rooms: list[Room] = []
    index: dict[str, int] = {}

    def room(key: str, name: str, description: str) -> int:
        index[key] = len(rooms)
        rooms.append(Room(name, description))
        return index[key]

    for number in levels:
        title = f"Level {number}"
        room(title, title, leads.get(title, f"{title} of the silo."))

    walkable = frozenset(levels)
    for dept, level in sorted(departments.items()):
        if level in walkable:
            room(dept, dept, leads.get(dept, f"{dept} of the silo."))

    if floors == ALL:
        floors = _opened(db)
    assert not isinstance(floors, str)
    doors: list[Door] = []
    for floor in floors:
        if floor not in walkable:
            raise ValueError(f"floor {floor} is not a level of this silo")
        addresses = _dwellings(db, floor)
        if not addresses:
            raise ValueError(f"level {floor} has no dwellings; it was never "
                             f"opened. `--floors` wants one that was.")
        ring = room(f"Ring {floor}", f"Level {floor}, the ring",
                    f"The corridor runs all the way round the stair: "
                    f"{len(addresses)} doors on {len(schema.RINGS)} rings, "
                    f"a number painted on every one. The stair is east.")
        occupied = _occupants(db, floor)
        doors.extend(
            Door(ring, door_word(address),
                 leads.get(address, f"Apartment {address}.") + " "
                 + beside_the_door(occupied.get(address, [])))
            for address in addresses)

    if len(rooms) >= NOWHERE:
        raise TooManyRooms(
            f"{len(rooms)} rooms - {len(levels)} landings, "
            f"{len(departments)} departments and "
            f"{len(rooms) - len(levels) - len(departments)} rings - and a "
            f"room id is one byte with {NOWHERE:#x} reserved for 'no exit'. "
            f"Ask for fewer floors.")

    _stair(rooms, index, levels, departments)
    for floor in floors:
        _corridor(rooms, index, floor, len(_dwellings(db, floor)))

    things = _place(db, index, notes if notes is not None else []) \
        if seeded else []
    return World(rooms=rooms, things=things, doors=doors,
                 start=index[f"Level {levels[0]}"],
                 terminal=index.get("IT"))


def _place(db: sqlite3.Connection, index: dict[str, int],
           notes: list[str]) -> list[Thing]:
    """`items.py` against the rooms this world actually has.

    Two ways an item legitimately has nowhere to go, and both are recorded
    rather than swallowed: the department it belongs in is not a room, and
    `FLAT` when the case's floor was never opened with `--floors`.
    """
    placed, dropped = items.seed(db)
    notes.extend(f"{name}: no cleaning in this corpus to hang it on"
                 for name in dropped)

    things: list[Thing] = []
    for item, description, subject in placed:
        where = item.where
        if where == items.FLAT:
            found = items.case(db)
            # A dwelling is a door on its floor's ring, so a thing that
            # belongs in the flat lies in the corridor outside it. The
            # description still names the flat, and the door still names
            # who had it.
            where = "" if found is None else f"Ring {found.flat.split()[0]}"
        if where not in index:
            notes.append(f"{item.name}: nowhere to put it - "
                         f"{where or 'the flat'} is not a room in this world")
            continue
        things.append(Thing(item.name, description, index[where],
                            subject=subject))
    return things


def _stair(rooms: list[Room], index: dict[str, int], levels: list[int],
           departments: dict[str, int]) -> None:
    """UP and DOWN along the levels, EAST into whatever is on this one."""
    for above, below in itertools.pairwise(levels):
        rooms[index[f"Level {above}"]].exits["DOWN"] = index[f"Level {below}"]
        rooms[index[f"Level {below}"]].exits["UP"] = index[f"Level {above}"]
    shared: dict[int, str] = {}
    for dept, level in departments.items():
        if dept not in index:
            continue
        # A landing has one EAST, so two departments on one level would be a
        # silent overwrite: the second door emitted wins and the first
        # department becomes a room nothing leads to. `generate.py` gives all
        # fourteen distinct levels, so this has never fired - which is exactly
        # when a check is worth having.
        if level in shared:
            raise ValueError(
                f"{shared[level]} and {dept} are both on level {level}, and a "
                f"landing has one door east")
        shared[level] = dept
        landing = rooms[index[f"Level {level}"]]
        landing.exits["EAST"] = index[dept]
        landing.description += f" A door east is marked {dept}."
        rooms[index[dept]].exits["WEST"] = index[f"Level {level}"]


def _corridor(rooms: list[Room], index: dict[str, int], floor: int,
              dwellings: int) -> None:
    """The one join the database does not carry: the ring off the landing."""
    landing, ring = index[f"Level {floor}"], index[f"Ring {floor}"]
    rooms[landing].exits[ONTO] = ring
    rooms[ring].exits[OFF] = landing
    rooms[landing].description += (
        f" West of the stair the corridor runs all the way round: "
        f"{dwellings} dwellings on {len(schema.RINGS)} rings.")


def report(world: World, image: bytes, notes: list[str]) -> str:
    stranded = len(world.rooms) - len(world.reachable())
    named = sum(1 for t in world.things if t.subject is not None)
    lines = [
        f"{len(world.rooms)} rooms, {len(world.doors)} doors, "
        f"{len(world.things)} things "
        f"({named} of them name something on the card)",
        f"  {len(image):,} bytes of image, {world.overlay_bytes} of overlay",
        f"  {stranded} rooms nothing leads to",
        f"  room ids left: {NOWHERE - len(world.rooms)}",
    ]
    lines += [f"  not placed - {note}" for note in notes]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--floors", nargs="*", default=[],
                        help="residential levels to open up as rings of "
                             "doors, or 'all' for every one")
    parser.add_argument("--bare", action="store_true",
                        help="geography only: place none of data/silo/items.py")
    parser.add_argument("-o", "--out", type=Path,
                        help="write the eZ80 binary here")
    args = parser.parse_args()

    notes: list[str] = []
    floors: tuple[int, ...] | str = (
        ALL if args.floors == [ALL] else tuple(int(f) for f in args.floors))
    db = sqlite3.connect(args.db)
    try:
        world = build(db, floors, seeded=not args.bare, notes=notes)
    except ValueError as refused:      # TooManyRooms is one of these
        print(refused, file=sys.stderr)
        return 1
    finally:
        db.close()

    image = buildif.build(world).build()
    print(report(world, image, notes))
    for number, why in world.dead_rules():
        print(f"  rule {number} can never fire: {why}")
    if args.out:
        args.out.write_bytes(image)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
