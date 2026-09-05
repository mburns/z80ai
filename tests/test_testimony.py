"""Testimony from records: a household asked about a name, answered from the
graph within two hops, in its department's voice.

`libtestimony.testify` is the reference over the plain edge list; the device
is held to it for every person against one household, and then to the
things the reference cannot show - that a door with nobody behind it says
so, that a name the card does not hold is a name the household never heard,
and that a world with no card says nobody answers and is right. Issue #107.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import buildif
import buildwikibin
import libgraphcard
import libnames
import libsearch
import libtestimony
import libworld
from libhost import AgonHost
from libworld import Door, Room, Rule, Thing, Topic, World

CORPUS = [
    ("Alexander E. Wong", "A person."),         # 0  the household at 600A
    ("Dylan R. Smith", "A person."),            # 1  his father
    ("Amanda M. Wilson", "A person."),          # 2  his wife
    ("Corey W. Wong", "A person."),             # 3  his son
    ("Sarah T. Wong", "A person."),             # 4  his sister
    ("Edward Y. Butler", "A person."),          # 5  a crew-mate
    ("Michelle U. Patterson", "A person."),     # 6  a stranger, in Judicial
    ("Mechanical", "A department."),            # 7
    ("Mechanical First Crew 3", "A crew."),     # 8
    ("Judicial", "A department."),              # 9
    ("Holston Becker", "A person."),            # 10 the household at 630A
    ("Kyle R. Smith", "A person."),             # 11 Dylan's father
    ("2 600 A", "A flat."),                     # 12
    ("2 630 A", "A flat."),                     # 13 clockwise of it
    ("2 600 B", "A flat."),                     # 14 outward of it
    ("Daniel E. Rose", "A person."),            # 15 who lives at 600B
]
PEOPLE = [t for t, lead in CORPUS if lead == "A person."]
EDGES = [
    ("Alexander E. Wong", "father_is", "Dylan R. Smith"),
    ("Alexander E. Wong", "child_of", "Dylan R. Smith"),
    ("Sarah T. Wong", "child_of", "Dylan R. Smith"),
    ("Alexander E. Wong", "spouse_of", "Amanda M. Wilson"),
    ("Corey W. Wong", "child_of", "Alexander E. Wong"),
    ("Alexander E. Wong", "works_in", "Mechanical"),
    ("Edward Y. Butler", "works_in", "Mechanical"),
    ("Michelle U. Patterson", "works_in", "Judicial"),
    ("Alexander E. Wong", "crew_is", "Mechanical First Crew 3"),
    ("Edward Y. Butler", "crew_is", "Mechanical First Crew 3"),
    ("Holston Becker", "works_in", "Judicial"),
    ("Dylan R. Smith", "father_is", "Kyle R. Smith"),
    ("Alexander E. Wong", "lives_at", "2 600 A"),
    ("Holston Becker", "lives_at", "2 630 A"),
    ("Daniel E. Rose", "lives_at", "2 600 B"),
    ("2 600 A", "next_along", "2 630 A"),
    ("2 600 A", "next_out", "2 600 B"),
]
RING, LANDING = 0, 1
DIMS = "The screen on the landing dims for a moment."


def witness_world() -> World:
    """A ring with four doors, a paper naming every person, and a badge on
    the landing so that holding a name up has to be done on purpose."""
    papers = [Thing(f"paper{i}", f"A paper naming {title}.", RING, subject=title)
              for i, title in enumerate(PEOPLE)]
    return World(
        rooms=[Room("Level 2, the ring", "Doors all the way round.",
                    {"EAST": LANDING}),
               Room("Level 2", "The stair.", {"WEST": RING})],
        things=[*papers,
                Thing("ledger", "A ledger naming nobody.", RING),
                Thing("scrap", "A scrap naming nobody the card knows.", RING,
                      subject="Nobody At All"),
                Thing("badge", "A badge.", LANDING, subject="Corey W. Wong")],
        topics=[Topic("wife", ["WIFE", "AMANDA"], titles=["Amanda M. Wilson"]),
                Topic("hatch", ["HATCH"])],
        doors=[Door(RING, "600A", "Wong lives here.", subject="Alexander E. Wong"),
               Door(RING, "630A", "Becker lives here.", subject="Holston Becker"),
               Door(RING, "600B", "Rose lives here.", subject="Daniel E. Rose"),
               Door(RING, "700A", "Nobody has the key.", subject=None),
               Door(RING, "730A", "A name the card lacks.", subject="Nobody Real")],
        messages=[DIMS],
        rules=[Rule(when=[(libworld.C_ASKED, 0)], then=[(libworld.A_PRINT, 0, 0)])],
        terminal=LANDING)


def make_card(out: Path, world: World | None):
    titles = [t for t, _ in CORPUS]
    doc = {t: i for i, t in enumerate(titles)}
    relations = sorted({r for _, r, _ in EDGES})
    rid = {n: i for i, n in enumerate(relations)}
    edges = [(doc[s], rid[r], doc[o]) for s, r, o in EDGES]
    index = libsearch.build(titles, [lead for _, lead in CORPUS], {})
    libsearch.write_index(index, out / "T.IDX")
    libsearch.write_text(index, out / "T.DAT")
    built = libgraphcard.build(titles, edges, relations, {}, paths=[])
    stats = libgraphcard.write(built, out / "T.GRF")
    names = libnames.write(libnames.build(titles), out / "T.NAM")
    spec = buildwikibin.OracleSpec(
        graph_name="T.GRF", forward_at=stats["forward_at"],
        num_edges=stats["edges"],
        types_at=libgraphcard.CardGraph(out / "T.GRF")._types_at,
        num_types=0, num_docs=len(titles), digest=built.digest, paths=[],
        names_name="T.NAM", num_names=names["records"], relations=relations)
    if world is not None:
        libworld.resolve_topics(world, titles)
    game = buildwikibin.build(len(titles), "T.IDX", "T.DAT", oracle=spec,
                              world=world).build()
    files = {n: (out / n).read_bytes() for n in ("T.IDX", "T.DAT", "T.GRF", "T.NAM")}
    return game, files, edges, relations, doc


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    return make_card(tmp_path_factory.mktemp("testimony"), witness_world())


def ask(card, *lines: str) -> tuple[str, AgonHost]:
    game, files, *_ = card
    host = AgonHost(stdin=[*lines, "!"], files=files)
    return host.run(game, max_cycles=2_000_000_000), host


def flat(out: str) -> str:
    return " ".join(out.split())


# --- the reference ------------------------------------------------------------------


def test_the_reference_names_the_closest_path(card):
    _, _, edges, relations, doc = card
    who = doc["Alexander E. Wong"]
    assert libtestimony.testify(edges, relations, who, who) == "self"
    assert libtestimony.testify(edges, relations, who, doc["Dylan R. Smith"]) == "father"
    assert libtestimony.testify(edges, relations, who, doc["Amanda M. Wilson"]) == "spouse"
    assert libtestimony.testify(edges, relations, who, doc["Corey W. Wong"]) == "child"
    assert libtestimony.testify(edges, relations, who, doc["Kyle R. Smith"]) == "grandfather"
    assert libtestimony.testify(edges, relations, who, doc["Sarah T. Wong"]) == "sibling"
    assert libtestimony.testify(edges, relations, who, doc["Edward Y. Butler"]) == "crew"
    assert libtestimony.testify(edges, relations, who, doc["Michelle U. Patterson"]) is None
    assert libtestimony.testify(edges, relations, doc["Holston Becker"],
                                doc["Michelle U. Patterson"]) == "department"
    # Next door both ways round, and across the corridor both ways.
    assert libtestimony.testify(edges, relations, who, doc["Holston Becker"]) == "neighbour"
    assert libtestimony.testify(edges, relations, doc["Holston Becker"], who) == "neighbour"
    assert libtestimony.testify(edges, relations, who, doc["Daniel E. Rose"]) == "neighbour"
    assert libtestimony.testify(edges, relations, doc["Daniel E. Rose"], who) == "neighbour"


def test_a_path_longer_than_the_row_is_refused():
    bad = libtestimony.Path("x", (("child_of", True),) * 4, "x")
    original = libtestimony.PATHS
    libtestimony.PATHS = (bad,)
    try:
        with pytest.raises(ValueError, match="at most 3"):
            libtestimony.resolve(["child_of"])
    finally:
        libtestimony.PATHS = original


def test_a_path_over_a_relation_the_card_lacks_is_left_out():
    names = [p.name for p, _ in libtestimony.resolve(["father_is"])]
    assert names == ["father", "grandfather"]


# --- the device ---------------------------------------------------------------------


def test_a_household_names_its_own_father(card):
    out, _ = ask(card, "take paper1", "ask 600A about paper1")
    assert "'Dylan R. Smith? My father.'" in flat(out)
    assert "somebody from Mechanical" in flat(out)


def test_a_topic_can_be_asked_about_at_a_door_and_counts_as_raised(card):
    out, _ = ask(card, "ask 600A about amanda")
    assert "'Amanda M. Wilson? We are married.'" in flat(out)
    assert DIMS in flat(out)


def test_the_door_next_door_is_a_neighbour_both_ways_round(card):
    """Three steps - my flat, the flat beside it, whoever lives there - with
    the middle one a reverse hop when asked the other way round."""
    out, _ = ask(card, "ask 600A about 630A")
    assert "'Holston Becker? Next door to us.'" in flat(out)
    out, _ = ask(card, "ask 630A about 600A")
    assert "'Alexander E. Wong? Next door to us.'" in flat(out)
    assert "sounds like a citation" in flat(out)


def test_a_name_held_up_has_to_be_held(card):
    out, _ = ask(card, "ask 600A about badge")
    assert "You are not carrying it." in out
    out, _ = ask(card, "east", "take badge", "west", "ask 600A about badge")
    assert "'Corey W. Wong? One of mine.'" in flat(out)


def test_the_device_agrees_with_the_reference_on_every_person(card):
    """Every person against the household at 600A: the path the reference
    names is the sentence the door says, and none where it names none."""
    _, _, edges, relations, doc = card
    who = doc["Alexander E. Wong"]
    for i, title in enumerate(PEOPLE):
        expected = libtestimony.testify(edges, relations, who, doc[title])
        out = flat(ask(card, f"take paper{i}", f"ask 600A about paper{i}")[0])
        assert f"'{title}? " in out
        if expected == "self":
            assert libtestimony.SELF in out
        elif expected is None:
            assert libtestimony.UNKNOWN in out
        else:
            # A name may have several paths and two sentences - next door
            # and across the corridor are both "neighbour".
            assert any(p.said in out for p in libtestimony.PATHS
                       if p.name == expected), (title, expected)


def test_a_flat_nobody_has_the_key_to_does_not_answer(card):
    out, _ = ask(card, "take paper1", "ask 700A about paper1")
    assert "Nobody answers." in out


def test_a_household_the_card_has_never_heard_of_does_not_answer(card):
    out, _ = ask(card, "take paper1", "ask 730A about paper1")
    assert "Nobody answers." in out


def test_a_name_that_is_not_on_the_card_was_never_heard(card):
    out, _ = ask(card, "take scrap", "ask 600A about scrap")
    assert "Never heard the name" in out


def test_a_thing_that_names_nothing_cannot_be_asked_about(card):
    out, _ = ask(card, "take ledger", "ask 600A about ledger")
    assert "The screen has nothing to say about that." in out


def test_asking_a_door_about_nothing_asks_what(card):
    out, _ = ask(card, "ask 600A")
    assert "What do you want to ask about?" in out


def test_a_word_that_is_neither_topic_thing_nor_door_is_named_back(card):
    out, _ = ask(card, "ask 600A about zorkmid")
    assert "I do not know the word 'ZORKMID'." in out


def test_testimony_costs_no_search(card):
    _, asked = ask(card, "take paper1", "ask 600A about paper1")
    _, idle = ask(card, "take paper1")
    assert 0 < asked.io_bytes - idle.io_bytes < 3_000
    assert asked.cpu.instructions - idle.cpu.instructions < 200_000


# --- a world with no card -------------------------------------------------------------


def test_a_door_in_a_world_with_no_card_says_nobody_answers():
    game = buildif.build(witness_world()).build()
    out = AgonHost(stdin=["take paper1", "ask 600A about paper1", "quit"],
                   files={}).run(game, max_cycles=50_000_000)
    assert "Nobody answers." in out


def test_a_door_subject_the_console_could_not_carry_is_refused():
    world = witness_world()
    world.doors[0].subject = "x" * 61
    with pytest.raises(ValueError, match="longer than"):
        world.check()
