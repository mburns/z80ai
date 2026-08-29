#!/usr/bin/env python3
"""
The silo you can walk, compiled out of the silo you can ask about.

    python data/silo/buildworld.py                       # what fits, and why not more
    python data/silo/buildworld.py --floors 2 -o SILO.bin

`worlds.py` hand-authors six rooms, and says so: it exists because `buildif`
needs a world to emit, not because six rooms is an Interactive Fiction. This
reads the world out of `data/silo.db` instead, and the reason is that the map
is already in there and has been since the schema was written:

    apartment(floor, bearing, ring)     10,368 possible, 2,088 opened
    next_along                          thirty minutes clockwise, and it wraps
    next_out                            one ring outward, and it does not
    located_in                          a dwelling's level, a department's level

`data/silo/schema.py` stores those two adjacencies as **edges rather than
arithmetic**, for a reason that turns out to be the same reason this file is
short: the machine that has to walk them has no modulo. A card walks them to
answer "who lives next door". A world walks them to go east.

So nothing here invents geography, and nothing here invents prose either -
every description is the `article` lead the corpus already carries. The
compiler's whole job is to decide which entities become rooms and which edge
becomes which direction.

## Which edge becomes which direction

`libworld.DIRECTIONS` is six because a silo needs six: four around and two
along the stair. A floor is a circle around a stairwell, so:

    EAST / WEST     next_along and its inverse - around the ring
    NORTH / SOUTH   next_out and its inverse - outward and inward
    UP / DOWN       the stair, which is the level number

with ring A's inward side opening onto the landing, because the stair is what
the innermost ring is inside of. That is the only exit here the database does
not contain: `next_out` stops at ring A rather than pointing at the stairwell,
so the stairwell has to be joined on.

## The wall is the room id, not the memory

`IF.md` measured 505 KB of SRAM free in a world binary and observed that at 12
bytes a room that is more rooms than anybody will write. It is - and it is not
what stops you, because `libworld.NOWHERE` is `0xFF` and a room id is one byte:

    144 landings + 14 departments        158 rooms
    one residential floor                 72 rooms
                                         ---
                                         230 rooms, and 255 is the ceiling

**One floor fits and two do not**, and no amount of SRAM changes that. The
silo has twenty-nine opened floors, so a world that wants all of them wants a
two-byte room id - which costs every exit a byte in the image and buys 65,535
rooms - or a world that streams floors off the card, which costs a turn the one
thing `IF.md` says a turn must never cost.

`build` refuses rather than truncating, and names the number.
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
from libworld import NOWHERE, Room, Thing, World

#: The database the rest of `data/silo/` defaults to.
DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"

#: `next_along` is thirty minutes clockwise, and clockwise on a floor plan is
#: east. Its inverse is the way back.
ALONG, AGAINST = "EAST", "WEST"

#: `next_out` is one ring outward, away from the stair at the centre.
OUT, IN = "NORTH", "SOUTH"

#: Which ring the stairwell opens onto. `next_out` stops at ring A rather than
#: pointing inward at the stair, so this is the one join the database does not
#: carry.
INNERMOST = schema.RINGS[0]


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


def _rings(db: sqlite3.Connection, floor: int) -> dict[str, tuple[str, str]]:
    """address -> (what is clockwise of it, what is outward of it).

    Read rather than computed. The bearings are right there in the table and
    `(bearing + 30) % 720` is two lines, which is exactly the temptation the
    schema's docstring warns about: the corpus and the world would then have
    two implementations of one circle, and they would agree until one of them
    was edited.
    """
    along: dict[str, str] = {}
    out: dict[str, str] = {}
    rows = db.execute(
        "SELECT e.subject, e.relation, e.object FROM edge e "
        "JOIN apartment a ON a.source = e.source AND a.address = e.subject "
        "WHERE e.source = ? AND a.floor = ? "
        "  AND e.relation IN ('next_along', 'next_out')",
        (schema.SOURCE, floor))
    for subject, relation, obj in rows:
        (along if relation == "next_along" else out)[subject] = obj
    return {address: (along.get(address, ""), out.get(address, ""))
            for address in set(along) | set(out)}


def _occupants(db: sqlite3.Connection, floor: int) -> dict[str, str]:
    """address -> who lives there now, for the name on the door.

    `until IS NULL` is the living tenancy. Everybody else who ever had the
    flat is still in `residence`, and is the card's business rather than the
    world's - which is the division this whole pair of programs is about.
    """
    rows = db.execute(
        "SELECT a.address, r.person FROM residence r "
        "JOIN apartment a ON a.source = r.source AND a.floor = r.floor "
        "  AND a.bearing = r.bearing AND a.ring = r.ring "
        "WHERE r.source = ? AND r.floor = ? AND r.until IS NULL",
        (schema.SOURCE, floor))
    return dict(rows)


def build(db: sqlite3.Connection, floors: tuple[int, ...] = (),
          seeded: bool = True, notes: list[str] | None = None) -> World:
    """The stair, the departments off it, and the rings on `floors`.

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

    for floor in floors:
        if floor not in walkable:
            raise ValueError(f"floor {floor} is not a level of this silo")
        addresses = _dwellings(db, floor)
        if not addresses:
            raise ValueError(f"level {floor} has no dwellings; it was never "
                             f"opened. `--floors` wants one that was.")
        occupied = _occupants(db, floor)
        for address in addresses:
            door = occupied.get(address)
            room(address, f"Apartment {address}",
                 leads.get(address, f"Apartment {address}.") + " "
                 + (f"The name beside the door is {door}."
                    if door else "Nobody has the key to this one."))

    if len(rooms) >= NOWHERE:
        raise TooManyRooms(
            f"{len(rooms)} rooms - {len(levels)} landings, "
            f"{len(departments)} departments and "
            f"{len(rooms) - len(levels) - len(departments)} dwellings - and a "
            f"room id is one byte with {NOWHERE:#x} reserved for 'no exit'. "
            f"Ask for fewer floors.")

    _stair(rooms, index, levels, departments)
    for floor in floors:
        _ring(db, rooms, index, floor)

    things = _place(db, index, notes if notes is not None else []) \
        if seeded else []
    return World(rooms=rooms, things=things,
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
            # A dwelling's room key is its bare address; `build` registers it
            # that way and prints it as "Apartment 107 800 A".
            where = "" if found is None else found.flat
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


def _ring(db: sqlite3.Connection, rooms: list[Room], index: dict[str, int],
          floor: int) -> None:
    """The two stored adjacencies, and the one join that is not stored."""
    landing = index[f"Level {floor}"]
    for address, (along, out) in _rings(db, floor).items():
        room = rooms[index[address]]
        if along in index:
            room.exits[ALONG] = index[along]
            rooms[index[along]].exits[AGAINST] = index[address]
        if out in index:
            room.exits[OUT] = index[out]
            rooms[index[out]].exits[IN] = index[address]

    inner = [a for a in _dwellings(db, floor) if a.endswith(f" {INNERMOST}")]
    for address in inner:
        rooms[index[address]].exits[IN] = landing
        rooms[index[address]].description += (
            " The stair is one door in from here.")
    if inner:
        rooms[landing].exits["WEST"] = index[inner[0]]
        rooms[landing].description += (
            f" West of the stair the corridor runs all the way round: "
            f"{len(_dwellings(db, floor))} dwellings on "
            f"{len(schema.RINGS)} rings.")


def report(world: World, image: bytes, notes: list[str]) -> str:
    stranded = len(world.rooms) - len(world.reachable())
    named = sum(1 for t in world.things if t.subject is not None)
    lines = [
        f"{len(world.rooms)} rooms, {len(world.things)} things "
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
    parser.add_argument("--floors", type=int, nargs="*", default=[],
                        help="residential levels to open up, ring by ring")
    parser.add_argument("--bare", action="store_true",
                        help="geography only: place none of data/silo/items.py")
    parser.add_argument("-o", "--out", type=Path,
                        help="write the eZ80 binary here")
    args = parser.parse_args()

    notes: list[str] = []
    db = sqlite3.connect(args.db)
    try:
        world = build(db, tuple(args.floors), seeded=not args.bare,
                      notes=notes)
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
