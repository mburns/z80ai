"""The silo you can walk, against the silo you can ask about.

`data/silo/buildworld.py` claims the map was already in the database. These
tests are mostly about the two halves of that claim being separable: the
geometry is *read* rather than recomputed, and the prose is quoted rather than
written. Both are easy to get right by accident - a compiler that recomputed
`(bearing + 30) % 720` would produce the same world on this data - so the tests
that matter are the ones that edit the database and check the world moved.

The database here is built by hand rather than by `generate.py`, which wants
Faker and eight seconds. Six levels and one small floor is enough to have a
ring, a stair and a department, and everything about the geometry is a property
of one floor rather than of nine thousand people.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import buildworld
import schema

import buildif
from libhost import AgonHost

SOURCE = schema.SOURCE
LEVELS = 6
#: One department, on a level that also has flats, so the landing has to use
#: EAST and WEST for different things.
DEPARTMENT, DEPARTMENT_LEVEL = "IT", 3
#: Four bearings rather than twenty-four. A ring is a ring.
BEARINGS = (0, 30, 60, 90)
FLOOR = 2


def _apartments(db: sqlite3.Connection) -> None:
    rows = [(SOURCE, FLOOR, bearing, ring)
            for bearing in BEARINGS for ring in schema.RINGS]
    db.executemany("INSERT INTO apartment (source, floor, bearing, ring) "
                   "VALUES (?, ?, ?, ?)", rows)
    for bearing in BEARINGS:
        for index, ring in enumerate(schema.RINGS):
            here = schema.address(FLOOR, bearing, ring)
            along = schema.address(
                FLOOR, BEARINGS[(BEARINGS.index(bearing) + 1) % len(BEARINGS)],
                ring)
            db.execute("INSERT INTO edge (source, subject, relation, object) "
                       "VALUES (?, ?, 'next_along', ?)", (SOURCE, here, along))
            if index + 1 < len(schema.RINGS):
                out = schema.address(FLOOR, bearing, schema.RINGS[index + 1])
                db.execute(
                    "INSERT INTO edge (source, subject, relation, object) "
                    "VALUES (?, ?, 'next_out', ?)", (SOURCE, here, out))
            db.execute("INSERT INTO article (source, title, lead) "
                       "VALUES (?, ?, ?)",
                       (SOURCE, here, f"Apartment {here} is a dwelling."))


@pytest.fixture
def db(tmp_path):
    """Six levels, one department and one floor of twelve flats."""
    conn = schema.connect(tmp_path / "silo.db", migrate=True)
    for number in range(1, LEVELS + 1):
        name = f"Level {number}"
        conn.execute("INSERT INTO article (source, title, lead) "
                     "VALUES (?, ?, ?)",
                     (SOURCE, name, f"{name} of the silo lies in Up Top."))
        conn.execute("INSERT INTO entity_type (source, kind, entity) "
                     "VALUES (?, 'level', ?)", (SOURCE, name))
    conn.execute("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                 (SOURCE, DEPARTMENT, f"{DEPARTMENT} is headquartered here."))
    conn.execute("INSERT INTO entity_type (source, kind, entity) "
                 "VALUES (?, 'department', ?)", (SOURCE, DEPARTMENT))
    conn.execute("INSERT INTO edge (source, subject, relation, object) "
                 "VALUES (?, ?, 'located_in', ?)",
                 (SOURCE, DEPARTMENT, f"Level {DEPARTMENT_LEVEL}"))
    _apartments(conn)
    conn.execute("INSERT INTO residence (source, person, floor, bearing, "
                 "ring, since, until) VALUES (?, 'Juliette Nichols', ?, 0, "
                 "'A', 200, NULL)", (SOURCE, FLOOR))
    conn.execute("INSERT INTO residence (source, person, floor, bearing, "
                 "ring, since, until) VALUES (?, 'Holston Becker', ?, 30, "
                 "'A', 100, 180)", (SOURCE, FLOOR))
    conn.commit()
    yield conn
    conn.close()


def exits(world, name: str) -> dict[str, str]:
    """A room's exits as `direction -> the name of where it goes`."""
    room = next(r for r in world.rooms if r.name == name)
    return {way: world.rooms[to].name for way, to in room.exits.items()}


# --- the stair ----------------------------------------------------------------


def test_the_stair_joins_each_level_to_the_next(db):
    world = buildworld.build(db)
    assert exits(world, "Level 1") == {"DOWN": "Level 2"}
    assert exits(world, f"Level {LEVELS}") == {"UP": f"Level {LEVELS - 1}"}
    assert exits(world, "Level 4")["UP"] == "Level 3"


def test_a_department_is_a_door_off_its_own_landing(db):
    world = buildworld.build(db)
    assert exits(world, f"Level {DEPARTMENT_LEVEL}")["EAST"] == DEPARTMENT
    assert exits(world, DEPARTMENT)["WEST"] == f"Level {DEPARTMENT_LEVEL}"


def test_the_landing_says_what_opens_off_it(db):
    """A door nothing mentions is a door nobody opens."""
    world = buildworld.build(db, (FLOOR,))
    landing = next(r for r in world.rooms
                   if r.name == f"Level {DEPARTMENT_LEVEL}")
    assert DEPARTMENT in landing.description
    ring = next(r for r in world.rooms if r.name == f"Level {FLOOR}")
    assert "12 dwellings" in ring.description


def test_the_start_is_the_top_of_the_stair(db):
    world = buildworld.build(db)
    assert world.rooms[world.start].name == "Level 1"


def test_the_terminal_stands_in_it(db):
    """`buildwikibin.build(world=...)` puts the card behind one room."""
    world = buildworld.build(db)
    assert world.rooms[world.terminal].name == "IT"


# --- the ring, which is the part that was already stored ----------------------


def test_next_along_becomes_east_and_its_inverse_west(db):
    world = buildworld.build(db, (FLOOR,))
    first = f"Apartment {schema.address(FLOOR, 0, 'A')}"
    second = f"Apartment {schema.address(FLOOR, 30, 'A')}"
    assert exits(world, first)[buildworld.ALONG] == second
    assert exits(world, second)[buildworld.AGAINST] == first


def test_next_out_becomes_outward_and_its_inverse_inward(db):
    world = buildworld.build(db, (FLOOR,))
    inner = f"Apartment {schema.address(FLOOR, 0, 'A')}"
    middle = f"Apartment {schema.address(FLOOR, 0, 'B')}"
    assert exits(world, inner)[buildworld.OUT] == middle
    assert exits(world, middle)[buildworld.IN] == inner


def test_the_ring_wraps_because_the_edge_table_wraps(db):
    """`(bearing + 30) % 720` is done once, in the generator, and shipped."""
    world = buildworld.build(db, (FLOOR,))
    last = f"Apartment {schema.address(FLOOR, BEARINGS[-1], 'A')}"
    assert exits(world, last)[buildworld.ALONG] == \
        f"Apartment {schema.address(FLOOR, 0, 'A')}"


def test_the_geometry_is_read_rather_than_recomputed(db):
    """The test that separates this compiler from one that looks the same.

    A compiler doing its own arithmetic over `bearing` would produce exactly
    the world the tests above assert, on this data and on the real corpus.
    Deleting one edge is the only way to tell the two apart - and a corpus
    where a flat has been walled off is a thing the generator can produce.
    """
    here = schema.address(FLOOR, 0, "A")
    db.execute("DELETE FROM edge WHERE source = ? AND subject = ? "
               "AND relation = 'next_along'", (SOURCE, here))
    world = buildworld.build(db, (FLOOR,))
    assert buildworld.ALONG not in exits(world, f"Apartment {here}")
    assert buildworld.OUT in exits(world, f"Apartment {here}")


def test_the_innermost_ring_opens_onto_the_stair(db):
    """The one join the database does not carry, and says it does not.

    `next_out` stops at ring A rather than pointing inward at the stairwell,
    so without this a floor is a ring nobody can get onto.
    """
    world = buildworld.build(db, (FLOOR,))
    inner = f"Apartment {schema.address(FLOOR, 0, 'A')}"
    assert exits(world, inner)[buildworld.IN] == f"Level {FLOOR}"
    assert exits(world, f"Level {FLOOR}")["WEST"] == f"Apartment {inner[10:]}"


def test_every_room_can_be_walked_to(db):
    """A world compiled out of a database is past reading by hand."""
    world = buildworld.build(db, (FLOOR,))
    assert len(world.reachable()) == len(world.rooms)


# --- the prose is quoted, not written -----------------------------------------


def test_a_room_says_what_the_article_says(db):
    """Every description is a lead the corpus already carries.

    Editing the article moves the room, which is the whole of the claim that
    this compiler writes no prose.
    """
    db.execute("UPDATE article SET lead = ? WHERE source = ? AND title = ?",
               ("The stair is out at this level and the dark is total.",
                SOURCE, "Level 4"))
    world = buildworld.build(db)
    room = next(r for r in world.rooms if r.name == "Level 4")
    assert room.description.startswith("The stair is out at this level")


def test_the_name_beside_the_door_is_whoever_lives_there_now(db):
    world = buildworld.build(db, (FLOOR,))
    lived_in = next(r for r in world.rooms
                    if r.name == f"Apartment {schema.address(FLOOR, 0, 'A')}")
    assert "Juliette Nichols" in lived_in.description


def test_a_flat_whose_tenant_has_moved_out_is_empty(db):
    """`until IS NULL` is the living tenancy; the rest is the card's business.

    Holston Becker had this flat for eighty years and the archive still knows
    it. The door does not, and that difference is the division the two
    programs exist to draw.
    """
    world = buildworld.build(db, (FLOOR,))
    empty = next(r for r in world.rooms
                 if r.name == f"Apartment {schema.address(FLOOR, 30, 'A')}")
    assert "Holston Becker" not in empty.description
    assert "Nobody has the key" in empty.description


# --- the wall, which is not memory --------------------------------------------


def test_more_rooms_than_a_room_id_can_name_is_refused(db):
    """`NOWHERE` is `0xFF` and a room id is one byte.

    `IF.md` measured 505 KB of SRAM free and observed that at 12 bytes a room
    it is more rooms than anybody will write. It is, and it is not what stops
    you. Refusing beats truncating: a world quietly missing its bottom forty
    levels walks perfectly well.
    """
    for number in range(LEVELS + 1, 300):
        name = f"Level {number}"
        db.execute("INSERT INTO article (source, title, lead) "
                   "VALUES (?, ?, 'A level.')", (SOURCE, name))
        db.execute("INSERT INTO entity_type (source, kind, entity) "
                   "VALUES (?, 'level', ?)", (SOURCE, name))
    with pytest.raises(buildworld.TooManyRooms, match="300 rooms"):
        buildworld.build(db)


def test_a_floor_that_was_never_opened_is_refused(db):
    with pytest.raises(ValueError, match="never opened"):
        buildworld.build(db, (5,))


def test_a_floor_outside_the_silo_is_refused(db):
    with pytest.raises(ValueError, match="not a level"):
        buildworld.build(db, (99,))


# --- and it is still a world --------------------------------------------------


def test_the_compiled_world_passes_the_checks_a_written_one_does(db):
    world = buildworld.build(db, (FLOOR,))
    world.check()
    assert world.dead_rules() == []


def test_a_turn_in_the_compiled_world_still_reads_nothing(db):
    """The claim `IF.md` is built on, against a world nobody wrote.

    Two hundred rooms out of a database is where a resident world would start
    to want paging, and it does not: the tables are in the image and a move is
    a table lookup whatever the size of the table.
    """
    world = buildworld.build(db, (FLOOR,))
    game = buildif.build(world).build()
    host = AgonHost(stdin=["down", "west", "east", "north", "south", "quit"],
                    files={})
    out = host.run(game, max_cycles=100_000_000)
    assert host.io_bytes == 0
    # `PRWRAP` decides where the lines break, which is its business and not
    # this assertion's - the same normalisation `tests/test_if.py` uses.
    assert "Juliette Nichols" in " ".join(out.split())
