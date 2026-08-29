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
LEVELS = 8
#: One department, on a level that also has flats, so the landing has to use
#: EAST and WEST for different things.
DEPARTMENT, DEPARTMENT_LEVEL = "IT", 3
#: The rest, one to a level, because a landing has one door east. Enough of
#: `data/silo/items.py`'s addresses to place the whole seed.
DEPARTMENTS = {"Cafeteria": 1, "Mechanical": 2, DEPARTMENT: DEPARTMENT_LEVEL,
               "Sheriff's Office": 4, "Judicial": 5, "Supply": 6,
               "Nursery": 7}
#: Four bearings rather than twenty-four. A ring is a ring.
BEARINGS = (0, 30, 60, 90)
FLOOR = 2

#: The cleaning `data/silo/items.py` hangs nine of its ten things off. Its
#: address is a flat on `FLOOR`, so the photograph has somewhere to be once
#: that floor is opened.
CLEANED = "Alexandra Anderson"
CLEANED_FACTS = {"fate": "Cleaning", "died": "148", "class": "Class of 76 (B)",
                 "spouse": "Ronald Gordon"}


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
    for name, level in DEPARTMENTS.items():
        conn.execute("INSERT INTO article (source, title, lead) "
                     "VALUES (?, ?, ?)",
                     (SOURCE, name, f"{name} is headquartered here."))
        conn.execute("INSERT INTO entity_type (source, kind, entity) "
                     "VALUES (?, 'department', ?)", (SOURCE, name))
        conn.execute("INSERT INTO edge (source, subject, relation, object) "
                     "VALUES (?, ?, 'located_in', ?)",
                     (SOURCE, name, f"Level {level}"))
    _apartments(conn)
    for prop, value in CLEANED_FACTS.items():
        conn.execute("INSERT INTO fact (source, subject, property, ordinal, "
                     "value, kind) VALUES (?, ?, ?, 0, ?, 'text')",
                     (SOURCE, CLEANED, prop, value))
    conn.execute("INSERT INTO fact (source, subject, property, ordinal, "
                 "value, kind) VALUES (?, ?, 'address', 0, ?, 'text')",
                 (SOURCE, CLEANED, schema.address(FLOOR, 60, "A")))
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
    assert exits(world, "Level 1")["DOWN"] == "Level 2"
    assert "UP" not in exits(world, "Level 1")
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
    with pytest.raises(buildworld.TooManyRooms, match="299 landings"):
        buildworld.build(db)


def test_a_floor_that_was_never_opened_is_refused(db):
    with pytest.raises(ValueError, match="never opened"):
        buildworld.build(db, (5,))


def test_a_floor_outside_the_silo_is_refused(db):
    with pytest.raises(ValueError, match="not a level"):
        buildworld.build(db, (99,))


# --- ten things, and nine of them are one case --------------------------------
#
# The corpus has no objects, so a seed cannot be derived - only placed, and
# derived *about*. `data/silo/items.py` is hand-written sentences with holes in
# them and the corpus fills the holes, which makes these tests about the holes.


def carried(world, name: str):
    return next(t for t in world.things if t.name == name)


def test_the_seed_is_placed_where_it_belongs(db):
    world = buildworld.build(db)
    assert world.rooms[carried(world, "notice").at].name == "Judicial"
    assert world.rooms[carried(world, "key").at].name == "Sheriff's Office"
    assert world.rooms[carried(world, "wrench").at].name == "Mechanical"


def test_the_case_fills_the_holes_from_the_corpus(db):
    """The notice names who was sent out, and it is not written in the file."""
    world = buildworld.build(db)
    assert CLEANED in carried(world, "notice").description
    assert carried(world, "notice").subject == CLEANED
    assert carried(world, "slate").subject == CLEANED_FACTS["class"]
    assert carried(world, "key").subject == schema.address(FLOOR, 60, "A")


def test_each_thing_names_the_next_place_to_stand(db):
    """The chain, which is the point of the seed being one case rather than
    ten props: the notice names a person, the key names their flat, and the
    photograph in it names who they married."""
    world = buildworld.build(db, (FLOOR,))
    flat = carried(world, "key").subject
    assert world.rooms[carried(world, "photo").at].name == f"Apartment {flat}"
    assert carried(world, "photo").subject == CLEANED_FACTS["spouse"]


def test_a_thing_with_nowhere_to_go_is_reported_rather_than_dropped(db):
    """Nine things out of ten is indistinguishable from a world meant to hold
    nine, which is why the build log exists."""
    notes: list[str] = []
    world = buildworld.build(db, notes=notes)
    assert not any(t.name == "photo" for t in world.things)
    assert any("photo" in note and "not a room" in note for note in notes)


def test_a_corpus_with_no_cleaning_seeds_only_what_it_can(db):
    """`generate.py` sends 1.5% of deaths out to clean, and a small enough
    corpus rounds that to none. A notice about nobody is worse than no
    notice, and the log says which nine went."""
    db.execute("DELETE FROM fact WHERE source = ? AND subject = ?",
               (SOURCE, CLEANED))
    notes: list[str] = []
    world = buildworld.build(db, notes=notes)
    names = {t.name for t in world.things}
    assert "notice" not in names and "key" not in names
    assert "wrench" in names and "ledger" in names
    assert any("no cleaning in this corpus" in note for note in notes)


def test_the_tool_that_names_nothing_stays_nameless(db):
    """`CONSULT WRENCH` has to be able to say the thing means nothing, so one
    of the ten has to mean nothing."""
    world = buildworld.build(db)
    assert carried(world, "wrench").subject is None
    assert any(t.subject is not None for t in world.things)


def test_the_seed_can_be_left_out(db):
    assert buildworld.build(db, seeded=False).things == []


def test_the_same_database_seeds_the_same_silo(db):
    """`ORDER BY subject` rather than a sample: two builds of one card must
    not disagree about what is in the drawer."""
    first = buildworld.build(db, (FLOOR,))
    second = buildworld.build(db, (FLOOR,))
    assert [(t.name, t.at, t.subject) for t in first.things] == \
        [(t.name, t.at, t.subject) for t in second.things]


def test_two_departments_on_one_level_are_refused(db):
    """A landing has one door east, so the second would win silently and the
    first department become a room nothing leads to. `generate.py` gives all
    fourteen distinct levels, which is when a check is worth having."""
    db.execute("INSERT INTO article (source, title, lead) "
               "VALUES (?, 'Farms', 'Farms is here.')", (SOURCE,))
    db.execute("INSERT INTO entity_type (source, kind, entity) "
               "VALUES (?, 'department', 'Farms')", (SOURCE,))
    db.execute("INSERT INTO edge (source, subject, relation, object) "
               "VALUES (?, 'Farms', 'located_in', ?)",
               (SOURCE, f"Level {DEPARTMENT_LEVEL}"))
    with pytest.raises(ValueError, match="one door east"):
        buildworld.build(db)


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
