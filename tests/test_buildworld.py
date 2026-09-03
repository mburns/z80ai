"""The silo you can walk, against the silo you can ask about.

`data/silo/buildworld.py` claims the map was already in the database. These
tests are mostly about the two halves of that claim being separable: the
geometry is *read* rather than recomputed, and the prose is quoted rather than
written. Both are easy to get right by accident - a compiler that recomputed
`(bearing + 30) % 720` would produce the same world on this data - so the tests
that matter are the ones that edit the database and check the world moved.

The database here is built by hand rather than by `generate.py`, which wants
Faker and eight seconds. Eight levels, seven departments and one small floor is
enough to have a ring, a stair, somewhere to put every seeded thing and one
cleaning to hang them off - and everything about the geometry is a property of
one floor rather than of nine thousand people.

The last section is the one that could not be written until all the others
existed: it builds the compiled world, an oracle binary and a card, and plays
the case from the notice through to the photograph. Every test above holds one
link and none of them says the links join, which is the arrangement that passes
while the thing it describes does not work.
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
    conn = _database(tmp_path)
    yield conn
    conn.close()


def _database(where: Path) -> sqlite3.Connection:
    """Eight levels, seven departments, one ring, and one cleaning.

    A function as well as a fixture because the end-to-end walk at the bottom
    of this file is module-scoped - it builds an oracle binary, which is not
    something to do once per test.
    """
    conn = schema.connect(where / "silo.db", migrate=True)
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
    return conn


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
    landing = next(r for r in world.rooms if r.name == f"Level {FLOOR}")
    assert "12 dwellings" in landing.description


def test_the_start_is_the_top_of_the_stair(db):
    world = buildworld.build(db)
    assert world.rooms[world.start].name == "Level 1"


def test_the_terminal_stands_in_it(db):
    """`buildwikibin.build(world=...)` puts the card behind one room."""
    world = buildworld.build(db)
    assert world.rooms[world.terminal].name == "IT"


# --- the ring, which is one room and a door a dwelling ---------------------------


def doors_on(world, floor: int) -> dict[str, str]:
    """The doors on a floor's ring, as `word -> what knocking says`."""
    ring = next(i for i, r in enumerate(world.rooms)
                if r.name == f"Level {floor}, the ring")
    return {d.name: d.text for d in world.doors if d.room == ring}


def test_a_floor_is_one_room_with_a_door_a_dwelling(db):
    world = buildworld.build(db, (FLOOR,))
    assert sum(1 for r in world.rooms if "the ring" in r.name) == 1
    assert set(doors_on(world, FLOOR)) == {
        buildworld.door_word(schema.address(FLOOR, b, r))
        for b in BEARINGS for r in schema.RINGS}
    assert len(world.doors) == len(BEARINGS) * len(schema.RINGS)


def test_a_door_is_its_bearing_and_ring_and_not_its_floor(db):
    """`42 600 A` is painted `600A`: every floor has one, and `KNOCK` looks
    only at the doors of the ring the player is on."""
    assert buildworld.door_word("42 600 A") == "600A"
    assert buildworld.door_word("2 1230 C") == "1230C"


def test_the_ring_is_off_the_landing_and_the_stair_is_off_the_ring(db):
    """The one join the database does not carry, and says it does not."""
    world = buildworld.build(db, (FLOOR,))
    ring = f"Level {FLOOR}, the ring"
    assert exits(world, f"Level {FLOOR}")[buildworld.ONTO] == ring
    assert exits(world, ring) == {buildworld.OFF: f"Level {FLOOR}"}


def test_the_doors_are_read_from_the_apartment_table(db):
    """A flat walled off is a door that is not there. The compiler reads the
    dwellings and does not count bearings, which is the only way to tell it
    from one that does."""
    db.execute("DELETE FROM apartment WHERE source = ? AND floor = ? "
               "AND bearing = 90 AND ring = 'C'", (SOURCE, FLOOR))
    world = buildworld.build(db, (FLOOR,))
    assert "130C" not in doors_on(world, FLOOR)
    assert "1230A" in doors_on(world, FLOOR)
    assert len(world.doors) == len(BEARINGS) * len(schema.RINGS) - 1


def test_every_room_can_be_walked_to(db):
    """A world compiled out of a database is past reading by hand."""
    world = buildworld.build(db, (FLOOR,))
    assert len(world.reachable()) == len(world.rooms)


def test_all_opens_every_floor_with_a_dwelling(db):
    world = buildworld.build(db, buildworld.ALL)
    assert [r.name for r in world.rooms if "the ring" in r.name] == \
        [f"Level {FLOOR}, the ring"]


def test_the_whole_silo_fits_in_one_byte():
    """The arithmetic the issue was about, on the shipped corpus's shape:
    144 landings, 14 departments and 29 rings of 72 doors is 187 rooms."""
    assert buildworld.NOWHERE > 144 + 14 + 29


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
    lived_in = doors_on(world, FLOOR)["1200A"]
    assert "Juliette Nichols" in lived_in
    assert lived_in.startswith(f"Apartment {schema.address(FLOOR, 0, 'A')} is")


def test_a_household_is_every_name_beside_the_door(db):
    """The shipped corpus puts three Butlers behind one door. A door that
    named the last one written would be quietly wrong about the others."""
    db.execute("INSERT INTO residence (source, person, floor, bearing, "
               "ring, since, until) VALUES (?, 'Lukas Kyle', ?, 0, 'A', "
               "210, NULL)", (SOURCE, FLOOR))
    world = buildworld.build(db, (FLOOR,))
    assert doors_on(world, FLOOR)["1200A"].endswith(
        "The names beside the door are Juliette Nichols and Lukas Kyle.")
    assert buildworld.beside_the_door(["A", "B", "C"]) == \
        "The names beside the door are A, B and C."


def test_a_flat_whose_tenant_has_moved_out_is_empty(db):
    """`until IS NULL` is the living tenancy; the rest is the card's business.

    Holston Becker had this flat for eighty years and the archive still knows
    it. The door does not, and that difference is the division the two
    programs exist to draw.
    """
    world = buildworld.build(db, (FLOOR,))
    empty = doors_on(world, FLOOR)["1230A"]
    assert "Holston Becker" not in empty
    assert "Nobody has the key" in empty


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
    photograph outside it names who they married. The flat is a door on the
    ring, and the photograph lies in the corridor beside it."""
    world = buildworld.build(db, (FLOOR,))
    flat = carried(world, "key").subject
    assert world.rooms[carried(world, "photo").at].name == \
        f"Level {flat.split()[0]}, the ring"
    assert buildworld.door_word(flat) in doors_on(world, FLOOR)
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
    host = AgonHost(stdin=["down", "west", "knock 1200A", "knock 1230A",
                           "east", "quit"], files={})
    out = host.run(game, max_cycles=100_000_000)
    assert host.io_bytes == 0
    # `PRWRAP` decides where the lines break, which is its business and not
    # this assertion's - the same normalisation `tests/test_if.py` uses.
    assert "Juliette Nichols" in " ".join(out.split())


# --- the chain, walked ---------------------------------------------------------
#
# Every test above holds one link: the seed is placed where it belongs, the
# case fills the holes, `CONSULT` copies a subject into `INPBUF`. None of them
# says the links join, and they were verified separately - which is exactly the
# arrangement that passes while the thing it describes does not work.
#
# So one test builds the whole apparatus - compiled world, oracle binary, card
# - and plays it. The card is made here rather than being `data/silo.db`, which
# is 39 MB, gitignored and wants Faker. That is a real limit and worth stating:
# this walks the *mechanism* over a corpus invented for it, not the corpus that
# ships.


def _card(tmp_path, world):
    """A card holding exactly what this world's things point at."""
    import libsearch

    subjects = sorted({t.subject for t in world.things
                       if t.subject is not None})
    leads = [f"{name} is an entry in the archive of Silo 18. "
             f"Filed under {name}." for name in subjects]
    index = libsearch.build(subjects, leads, {})
    libsearch.write_index(index, tmp_path / "S.IDX")
    libsearch.write_text(index, tmp_path / "S.DAT")
    return index.num_docs, {
        "S.IDX": (tmp_path / "S.IDX").read_bytes(),
        "S.DAT": (tmp_path / "S.DAT").read_bytes()}


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Walk the case from the notice to the photograph, asking as we go.

    Judicial is level 5, the Sheriff's Office 4, IT 3 and the ring is on
    `FLOOR`. The route is: take the notice, take the key, carry both up to the
    terminal, ask about each, then walk to the flat the key named and read
    what is in it.
    """
    import buildwikibin

    # Module-scoped, so build the database here rather than reuse `db`.
    conn = _database(tmp_path_factory.mktemp("chain"))
    world = buildworld.build(conn, (FLOOR,))
    conn.close()

    num_docs, files = _card(tmp_path_factory.mktemp("card"), world)
    game = buildwikibin.build(num_docs, index_name="S.IDX",
                              text_name="S.DAT", world=world).build()
    route = [
        "down", "down", "down", "down",          # Level 5
        "east", "take notice", "west",           # Judicial
        "up", "east", "take key", "west",        # Level 4, Sheriff's Office
        "up", "east",                            # Level 3, IT: the terminal
        "consult notice", "consult key",
        "west", "up",                            # back out and up to Level 2
        "west", "knock 100A",                    # onto the ring, the flat's door
        "take photo",
        "east",                                  # back to the stair
        "down", "east", "consult photo",
        "!",
    ]
    host = AgonHost(stdin=route, files=files)
    return host.run(game, max_cycles=2_000_000_000), host, game, files


def test_the_notice_answers_from_the_card(played):
    """Link one: a thing found in a room, asked at a terminal two floors up."""
    out, _host, _game, _files = played
    assert f"{CLEANED} is an entry in the archive" in " ".join(out.split())


def test_the_key_names_a_flat_the_card_knows(played):
    out, _host, _game, _files = played
    flat = schema.address(FLOOR, 60, "A")
    assert f"{flat} is an entry in the archive" in " ".join(out.split())


def test_the_flat_the_key_named_is_a_door_with_the_photograph_outside_it(played):
    """Link two, and the one that makes it a chain rather than two lookups:
    the address the archive just read out is a door the player can knock
    on, and the photograph is lying in the corridor beside it."""
    out, _host, _game, _files = played
    flat = schema.address(FLOOR, 60, "A")
    assert f"Apartment {flat} is a dwelling" in " ".join(out.split())
    assert "Taken." in out


def test_the_photograph_names_the_spouse_and_the_card_has_him(played):
    """Link three. Nothing in the world knows this name - it came out of the
    corpus, into a description, and back to the card."""
    out, _host, _game, _files = played
    assert CLEANED_FACTS["spouse"] in " ".join(out.split())


def test_the_chain_costs_three_questions_and_the_walk_costs_nothing(played):
    """`IF.md`'s claim over the whole apparatus rather than over six rooms.

    Not a bound - an equality. The same three questions asked after twelve
    more moves read the same number of bytes, so the moves cost nothing at
    all rather than nearly nothing. A bound would pass a world that paged.
    """
    _out, host, game, files = played
    wandered = AgonHost(
        stdin=["down", "up", "down", "up", "down", "up",
               "down", "down", "down", "east", "west", "up",
               "down", "down", "east", "take notice", "west",
               "up", "east", "take key", "west", "up", "east",
               "consult notice", "consult key", "west", "up",
               "west", "knock 100A", "knock 1200A", "knock 1230A",
               "take photo", "east", "down", "east", "consult photo",
               "!"],
        files=files)
    wandered.run(game, max_cycles=2_000_000_000)
    assert host.io_bytes == wandered.io_bytes
    assert host.io_bytes > 0                     # and three were still asked

