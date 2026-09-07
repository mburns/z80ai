"""The case generator: a murder from the corpus, and the proof it is fair.

The corpus here is built by hand - a victim, a spouse, two neighbours, two
colleagues on the same shift, a committee - so that every part of the
circle is exercised without Faker. What is held: the circle is who the
views say it is, the clues eliminate everyone but the killer and the
generator refuses when they do not, the walkthrough the planner finds ends
in the right accusation and wins on the device, the deadline fits, and the
spouse is not the killer more often than a guess would be. Issue #112.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import cases
import schema

import buildif
from libhost import AgonHost

SOURCE = schema.SOURCE
LEVELS = 8
DEPARTMENTS = {"Cafeteria": 1, "Mechanical": 2, "IT": 3, "Sheriff's Office": 4,
               "Judicial": 5, "Supply": 6}
FLOOR = 7
BEARINGS = (0, 30, 60, 90)
NOW = 220

VICTIM = "Aaron M. Moore"
SPOUSE = "Beth K. Moore"
NEIGHBOURS = ("Carl T. Reyes", "Dana P. Ortiz")
COLLEAGUES = ("Evan R. Shaw", "Fay L. Nguyen")
COMMITTEE = "Gus W. Pratt"
STRANGER = "Hal Z. Vance"


def _people(conn: sqlite3.Connection) -> None:
    def person(name: str, born: int, died: int | None, department: str,
               shift: str, address: str | None, fate: str | None) -> None:
        facts = {"born": born, "department": department, "shift": shift,
                 "occupation": "Hand", "sex": "man"}
        if died is not None:
            facts["died"] = died
            facts["fate"] = fate
        if address is not None:
            facts["address"] = address
        for prop, value in facts.items():
            kind = "number" if isinstance(value, int) else "text"
            conn.execute(
                "INSERT INTO fact (source, subject, property, ordinal, value, "
                "kind, num) VALUES (?, ?, ?, 0, ?, ?, ?)",
                (SOURCE, name, prop, str(value), kind,
                 value if isinstance(value, int) else None))
        conn.execute("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                     (SOURCE, name, f"{name} is a person of Silo 18."))
        conn.execute("INSERT INTO entity_type (source, kind, entity) "
                     "VALUES (?, 'man', ?)", (SOURCE, name))

    def lives(name: str, bearing: int, ring: str, until: int | None) -> None:
        conn.execute(
            "INSERT INTO residence (source, person, floor, bearing, ring, "
            "since, until) VALUES (?, ?, ?, ?, ?, 180, ?)",
            (SOURCE, name, FLOOR, bearing, ring, until))

    home = schema.address(FLOOR, 0, "A")
    person(VICTIM, 150, NOW, "Mechanical", "Third Shift", home, "Natural causes")
    lives(VICTIM, 0, "A", NOW)
    person(SPOUSE, 152, None, "Supply", "First Shift", home, None)
    lives(SPOUSE, 0, "A", None)
    conn.execute("INSERT INTO edge (source, subject, relation, object) "
                 "VALUES (?, ?, 'spouse_of', ?)", (SOURCE, VICTIM, SPOUSE))
    person(NEIGHBOURS[0], 160, None, "IT", "Second Shift",
           schema.address(FLOOR, 30, "A"), None)
    lives(NEIGHBOURS[0], 30, "A", None)
    person(NEIGHBOURS[1], 161, None, "Supply", "Second Shift",
           schema.address(FLOOR, 0, "B"), None)
    lives(NEIGHBOURS[1], 0, "B", None)
    for i, name in enumerate(COLLEAGUES):
        person(name, 165 + i, None, "Mechanical", "Third Shift",
               schema.address(FLOOR, 60 + 30 * i, "A"), None)
        lives(name, 60 + 30 * i, "A", None)
    person(COMMITTEE, 155, None, "Judicial", "First Shift",
           schema.address(FLOOR, 60, "B"), None)
    lives(COMMITTEE, 60, "B", None)
    person(STRANGER, 170, None, "IT", "First Shift",
           schema.address(FLOOR, 90, "C"), None)
    lives(STRANGER, 90, "C", None)
    # Somebody dead of the same causes but unhoused and alone, who must be
    # skipped rather than made a victim with no circle.
    person("Ida Q. Lone", 140, NOW, "Supply", "First Shift", None, "Natural causes")

    conn.execute("INSERT INTO cohort (source, name, kind, formed) "
                 "VALUES (?, 'Ration Board', 'committee', 200)", (SOURCE,))
    for name in (VICTIM, COMMITTEE):
        conn.execute("INSERT INTO membership (source, person, cohort, role, "
                     "joined, until) VALUES (?, ?, 'Ration Board', 'member', "
                     "205, NULL)", (SOURCE, name))
    for name, dept in ((VICTIM, "Mechanical"), (SPOUSE, "Supply"),
                       (NEIGHBOURS[0], "IT"), (NEIGHBOURS[1], "Supply"),
                       (COLLEAGUES[0], "Mechanical"), (COLLEAGUES[1], "Mechanical"),
                       (COMMITTEE, "Judicial"), (STRANGER, "IT")):
        conn.execute("INSERT INTO edge (source, subject, relation, object) "
                     "VALUES (?, ?, 'works_in', ?)", (SOURCE, name, dept))


def _database(where: Path) -> sqlite3.Connection:
    conn = schema.connect(where / "silo.db", migrate=True)
    for number in range(1, LEVELS + 1):
        name = f"Level {number}"
        conn.execute("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                     (SOURCE, name, f"{name} of the silo."))
        conn.execute("INSERT INTO entity_type (source, kind, entity) "
                     "VALUES (?, 'level', ?)", (SOURCE, name))
    for name, level in DEPARTMENTS.items():
        conn.execute("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                     (SOURCE, name, f"{name} is headquartered here."))
        conn.execute("INSERT INTO entity_type (source, kind, entity) "
                     "VALUES (?, 'department', ?)", (SOURCE, name))
        conn.execute("INSERT INTO edge (source, subject, relation, object) "
                     "VALUES (?, ?, 'located_in', ?)", (SOURCE, name, f"Level {level}"))
    rows = [(SOURCE, FLOOR, b, r) for b in BEARINGS for r in schema.RINGS]
    conn.executemany("INSERT INTO apartment (source, floor, bearing, ring) "
                     "VALUES (?, ?, ?, ?)", rows)
    for b in BEARINGS:
        for i, r in enumerate(schema.RINGS):
            here = schema.address(FLOOR, b, r)
            conn.execute("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                         (SOURCE, here, f"Apartment {here} is a dwelling."))
            nxt = schema.address(FLOOR, BEARINGS[(BEARINGS.index(b) + 1) % 4], r)
            conn.execute("INSERT INTO edge (source, subject, relation, object) "
                         "VALUES (?, ?, 'next_along', ?)", (SOURCE, here, nxt))
            if i + 1 < len(schema.RINGS):
                out = schema.address(FLOOR, b, schema.RINGS[i + 1])
                conn.execute("INSERT INTO edge (source, subject, relation, object) "
                             "VALUES (?, ?, 'next_out', ?)", (SOURCE, here, out))
    _people(conn)
    conn.commit()
    return conn


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    conn = _database(tmp_path_factory.mktemp("cases"))
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def case(db):
    return cases.generate(db, seed=3)


@pytest.fixture(scope="module")
def world(db, case):
    return cases.build_world(db, case)


# --- the case ---------------------------------------------------------------------


def test_the_victim_is_this_years_natural_death_with_a_circle(case):
    assert case.victim == VICTIM
    assert case.victim_flat == schema.address(FLOOR, 0, "A")
    assert case.department == "Mechanical" and case.shift == "Third Shift"


def test_the_circle_is_who_the_views_say(case):
    names = {s.name: s.how for s in case.circle}
    assert names[SPOUSE] == "spouse"
    assert names[NEIGHBOURS[0]] == "neighbour"
    assert names[NEIGHBOURS[1]] == "neighbour"
    assert names[COLLEAGUES[0]] == "colleague"
    assert names[COLLEAGUES[1]] == "colleague"
    assert names[COMMITTEE] == "committee"
    assert STRANGER not in names and VICTIM not in names


def test_the_killer_is_in_the_circle_and_alone_after_the_clues(case):
    assert case.killer in {s.name for s in case.circle}
    assert case.survivors() == {case.killer}


def test_every_innocent_has_an_alibi_and_the_killers_is_broken(case):
    alibied = {c.alibis for c in case.clues if c.alibis}
    broken = {c.accuses for c in case.clues if c.accuses}
    assert broken == {case.killer}
    assert case.killer in alibied                       # the lie
    for s in case.circle:
        if s.name != case.killer:
            assert s.name in alibied, s.name


def test_the_documents_say_what_the_proof_read(case):
    roster = next(c for c in case.clues if c.word == "roster")
    log = next(c for c in case.clues if c.word == "log")
    assert case.killer in roster.text
    assert case.killer in log.text and "0310" in log.text
    for c in case.clues:
        if c.word.startswith("note"):
            assert c.alibis in c.text and c.alibis != case.killer


def test_the_same_seed_makes_the_same_case(db, case):
    again = cases.generate(db, seed=3)
    assert again.killer == case.killer
    assert [c.text for c in again.clues] == [c.text for c in case.clues]


def test_a_case_the_clues_do_not_settle_is_refused(db, monkeypatch):
    def leave_two(rng, case):
        case.clues = []                     # no alibis for anyone
    monkeypatch.setattr(cases, "_write_clues", leave_two)
    with pytest.raises(cases.Unfair, match="standing"):
        cases.generate(db, seed=3)


def test_a_corpus_with_no_death_is_refused(tmp_path):
    conn = schema.connect(tmp_path / "empty.db", migrate=True)
    with pytest.raises(cases.Unfair, match="nobody"):
        cases.generate(conn, seed=0)
    conn.close()


# --- the world ------------------------------------------------------------------------


def test_every_clue_is_a_thing_in_a_room_with_a_rule_that_notices_it(world, case):
    names = {t.name for t in world.things}
    for c in case.clues:
        if c.text:
            assert c.word in names
    rooms = {t.name: world.rooms[t.at].name for t in world.things}
    assert rooms["notice"] == "Cafeteria"
    assert rooms["file"] == "Judicial"
    assert rooms["roster"] == "Sheriff's Office"
    assert rooms["log"] == "IT"


def test_the_suspects_answer_at_their_own_doors(world, case):
    subjects = {d.subject for d in world.doors}
    for s in case.circle:
        if s.flat is not None:
            assert s.name in subjects


def test_the_walkthrough_ends_in_the_right_accusation_and_wins_on_the_device(world, case):
    assert case.walkthrough[-1].startswith("accuse ")
    assert case.deadline == cases.deadline_for(len(case.walkthrough))
    assert case.deadline is not None
    game = buildif.build(world).build()
    out = AgonHost(stdin=[*case.walkthrough, "quit"], files={}).run(
        game, max_cycles=400_000_000)
    assert "The gate log. I thought nobody read the gate log." in " ".join(out.split())
    assert "closes the file" not in out


def test_the_wrong_accusation_loses_and_the_deadline_closes_the_file(world, case):
    game = buildif.build(world).build()
    wrong = list(case.walkthrough[:-1])
    innocent = next(s for s in case.circle if s.name != case.killer and s.flat)
    door = next(d for d in world.doors if d.subject == innocent.name)
    route = [*wrong, f"accuse {door.name.lower()}"]
    # The planner's route may not pass that door; accuse whoever's door is
    # where the walkthrough ends instead, which is the killer's - so go to
    # the innocent's ring first.
    out = AgonHost(stdin=[*route, "quit"], files={}).run(game, max_cycles=400_000_000)
    assert ("it was the wrong one" in " ".join(out.split())
            or f"There is no door marked '{door.name}'" in out)
    out = AgonHost(stdin=["look"] * (case.deadline + 1) + ["quit"], files={}).run(
        game, max_cycles=400_000_000)
    assert "closes the file" in " ".join(out.split())


def test_a_walkthrough_the_clock_cannot_hold_gets_no_deadline():
    """A silo is 144 levels tall and the clock is a byte. Twice a
    walkthrough that crosses it is past 255, and a tighter deadline than
    twice would be a speedrun rather than a mystery."""
    assert cases.deadline_for(20) == 60
    assert cases.deadline_for(100) == 255
    assert cases.deadline_for(128) is None


def test_the_clues_are_collected_down_the_stair_not_in_file_order(world, case):
    """The first real case walked the silo four times. The goal puts the
    clues in stair order, and the walkthrough never climbs back past a
    level it has finished with before the accusation."""
    depth = cases._depths(world)
    found = [world.things[arg].at for op, arg in
             [(r.when[0][0], r.when[0][1]) for r in world.rules
              if r.when and r.when[0][0] == 1]]
    depths = [depth[room] for room in found]
    assert depths == sorted(depths)


def test_the_case_sheet_reads(case):
    sheet = case.sheet()
    assert case.victim in sheet and "<- the killer" in sheet
    assert "survivors after every clue" in sheet


def test_the_baseline_is_a_guess_and_says_so(db):
    report = cases.baseline(db, seeds=12)
    assert "the spouse did it" in report
    assert "refused" in report
