"""The graph walk on the machine, against the reference that defines it.

`libgraphcard.CardGraph` says what a walk means; this checks the eZ80 does the
same thing. A binary search that is subtly wrong does not crash - it lands on a
neighbouring record and answers with some other article - so agreement with the
reference is the only signal there is.

The program built here is a harness: it walks a path planted in memory and
prints the document id it reached. The real program will read the path from the
classifier instead, but the walk is the part that can be wrong quietly.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))

import buildgraphwalk
import ingest
import libgraph
import libgraphcard
from libez80 import AGON_LOAD_ADDR, EZ80Builder, agon_header
from libhost import AgonHost

MOS_API = 0x08
MOS_FOPEN = 0x0A
MOS_FCLOSE = 0x0B
MOS_FREAD = 0x1A
MOS_FLSEEK = 0x1C
FA_READ = 0x01

TITLES = ["Jane Austen", "Steventon", "Hampshire", "England",
          "Marie Curie", "Warsaw", "Poland", "Ada Lovelace", "London"]


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    """A small graph, built through libgraph so the edges are the real ones."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(ingest._schema())
    fillers = [(f"{name} filler {i}", name)
               for name in ("England", "Poland") for i in range(3)]
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('w', ?, '')",
                     [(t,) for t in TITLES] + [(t,) for t, _ in fillers])
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('w', ?, ?, 0, ?, 'text', NULL)", [
            ("Jane Austen", "birth_place", "Steventon"),
            ("Steventon", "county", "Hampshire"),
            ("Hampshire", "country", "England"),
            ("Marie Curie", "birth_place", "Warsaw"),
            ("Warsaw", "country", "Poland"),
            ("Ada Lovelace", "birth_place", "London"),
            ("London", "country", "England"),
        ])
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('w', ?, 'country', 0, ?, 'text', NULL)", fillers)
    libgraph.build(conn, "w")

    titles = [t for (t,) in conn.execute(
        "SELECT title FROM article WHERE source = 'w' ORDER BY id")]
    doc = {t: i for i, t in enumerate(titles)}
    relations = sorted({r for (r,) in conn.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = 'w'")})
    rid = {name: i for i, name in enumerate(relations)}

    edges = [(doc[s], rid[r], doc[o]) for s, r, o in conn.execute(
        "SELECT subject, relation, object FROM edge WHERE source = 'w'")]
    types: dict[str, list[int]] = {}
    for kind, entity in conn.execute(
            "SELECT kind, entity FROM entity_type WHERE source = 'w'"):
        types.setdefault(kind, []).append(doc[entity])

    built = libgraphcard.build(titles, edges, relations, types, paths=[])
    path = tmp_path_factory.mktemp("graph") / "WIKI.GRF"
    libgraphcard.write(built, path)
    card = libgraphcard.CardGraph(path)
    return card, path, doc, rid, titles, conn


def harness(card, steps: list[tuple[int, int]], subject: int,
            climb_limit: int = libgraphcard.CLIMB_LIMIT) -> EZ80Builder:
    """A program that walks `steps` from `subject` and prints where it got.

    The path is assembled into the image rather than read from a card, because
    what is under test is the walk and not the plumbing around it.
    """
    b = EZ80Builder(AGON_LOAD_ADDR)
    agon_header(b)

    b.label("START")
    b.ld_hl_label("GRFNAME")
    b.ld_c_n(FA_READ)
    b.ld_a_n(MOS_FOPEN)
    b.rst(MOS_API)
    b.ld_mem_label_a("GRFH")
    b.or_a()
    b.jp_z("DONE")

    b.ld_hl_nn(subject)
    b.ld_mem_label_hl("GW_HERE")
    b.ld_hl_label("PATH")
    b.ld_mem_label_hl("GW_STEPS")
    b.ld_hl_nn(len(steps))
    b.ld_mem_label_hl("GW_LEFT")
    b.ld_hl_nn(card.forward_at)
    b.ld_mem_label_hl("GW_FWD")
    b.ld_hl_nn(card.reverse_at)
    b.ld_mem_label_hl("GW_REV")

    b.call("GW_FOLLOW")
    b.jp_c("NOWHERE")

    # Print the id it reached, as decimal, so the test can read it back.
    b.ld_hl_mem_label("GW_HERE")
    b.call("PRDEC")
    b.jp("DONE")

    b.label("NOWHERE")
    b.ld_hl_label("MSGNONE")
    b.label("PRSTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_z("DONE")
    b.rst(0x10)
    b.inc_hl()
    b.jp("PRSTR")

    b.label("DONE")
    b.ld_a_mem_label("GRFH")
    b.ld_c_a()
    b.ld_a_n(MOS_FCLOSE)
    b.rst(MOS_API)
    b.ret()

    # PRDEC: HL as decimal, by repeated subtraction - there is no divide, and
    # this is a harness rather than something that ships.
    b.label("PRDEC")
    b.ld_ix_label("POWERS")
    b.ld_b_n(7)
    b.ld_a_n(0)
    b.ld_mem_label_a("SEEN")
    b.label("PD_DIGIT")
    b.ld_de_ixd(0)
    b.ld_a_n(0)
    b.label("PD_SUB")
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("PD_BACK")
    b.inc_a()
    b.jp("PD_SUB")
    b.label("PD_BACK")
    b.add_hl_de()
    b.push_af()
    b.or_a()
    b.jp_nz("PD_SHOW")
    b.ld_a_mem_label("SEEN")
    b.or_a()
    b.jp_z("PD_SKIP")
    b.label("PD_SHOW")
    b.pop_af()
    b.push_af()
    b.add_a_n(ord("0"))
    b.rst(0x10)
    b.ld_a_n(1)
    b.ld_mem_label_a("SEEN")
    b.label("PD_SKIP")
    b.pop_af()
    b.inc_ix()
    b.inc_ix()
    b.inc_ix()
    b.djnz("PD_DIGIT")
    b.ld_a_mem_label("SEEN")
    b.or_a()
    b.ret_nz()
    b.ld_a_n(ord("0"))
    b.rst(0x10)
    b.ret()

    buildgraphwalk.emit_walk(
        b, card.num_edges, types_at=card._types_at - 8 * len(card.type_names),
        num_types=len(card.type_names), handle_label="GRFH",
        buffer_label="IOBUF", seekoff_label="SEEKOFF", climb_limit=climb_limit)

    # SEEK and READ, the two the walk borrows from its caller.
    b.label("SEEK")
    b.ld_hl_mem_label("SEEKOFF")
    b.ld_a_mem_label("SEEKOFF", 3)
    b.ld_e_a()
    b.ld_a_n(MOS_FLSEEK)
    b.rst(MOS_API)
    b.ret()

    b.label("READ")
    b.ld_a_n(MOS_FREAD)
    b.rst(MOS_API)
    b.ret()

    b.label("GRFNAME")
    b.emit(*b"WIKI.GRF\x00")
    b.label("MSGNONE")
    b.emit(*b"NOWHERE\x00")
    b.label("POWERS")
    for value in (1000000, 100000, 10000, 1000, 100, 10, 1):
        b.d24(value)
    b.label("PATH")
    for relation, kind in steps:
        b.emit(relation, kind)
    b.label("GRFH")
    b.emit(0)
    b.label("SEEN")
    b.emit(0)
    b.label("SEEKOFF")
    b.emit(0, 0, 0, 0)
    b.label("IOBUF")
    b.emit(*([0] * 16))
    buildgraphwalk.emit_cells(b)
    b.resolve()
    return b


def walk_on_device(graph, steps, subject,
                   climb_limit: int = libgraphcard.CLIMB_LIMIT) -> str:
    card, path, _doc, _rid, _titles, _db = graph
    host = AgonHost(stdin=[], files={"WIKI.GRF": path.read_bytes()})
    return host.run(harness(card, steps, subject, climb_limit).build(),
                    max_cycles=200_000_000).strip()


def both(graph, steps, subject, climb_limit: int = libgraphcard.CLIMB_LIMIT):
    """What the device says, and what the reference says, for one walk."""
    card = graph[0]
    answer, _walked, _missing = card.follow(subject, steps,
                                            climb_limit=climb_limit)
    reference = "NOWHERE" if answer is None else str(answer)
    return walk_on_device(graph, steps, subject, climb_limit), reference


# --- one hop ------------------------------------------------------------------


def test_a_single_hop_agrees(graph):
    _card, _path, doc, rid, _titles, _db = graph
    steps = [(rid["born_in"], libgraphcard.PLAIN)]
    device, reference = both(graph, steps, doc["Marie Curie"])
    assert device == reference == str(doc["Warsaw"])


def test_a_hop_with_no_edge_says_nowhere(graph):
    _card, _path, doc, rid, _titles, _db = graph
    steps = [(rid["born_in"], libgraphcard.PLAIN)]
    device, reference = both(graph, steps, doc["England"])
    assert device == reference == "NOWHERE"


def test_every_subject_hops_the_same_on_both(graph):
    """The whole edge set, one hop each. A binary search off by one record
    still returns *an* article, so this is what catches it."""
    _card, _path, doc, rid, _titles, db = graph
    for subject, relation, obj in db.execute(
            "SELECT subject, relation, object FROM edge WHERE source = 'w'"):
        steps = [(rid[relation], libgraphcard.PLAIN)]
        device, reference = both(graph, steps, doc[subject])
        assert device == reference == str(doc[obj]), subject


# --- the climb ----------------------------------------------------------------


def test_a_climb_of_one_hop_agrees(graph):
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["born_in"], libgraphcard.PLAIN), (rid["located_in"], country)]
    device, reference = both(graph, steps, doc["Marie Curie"])
    assert device == reference == str(doc["Poland"])


def test_a_climb_of_two_hops_agrees(graph):
    """Steventon -> Hampshire -> England. How far it is belongs to the graph."""
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["born_in"], libgraphcard.PLAIN), (rid["located_in"], country)]
    device, reference = both(graph, steps, doc["Jane Austen"])
    assert device == reference == str(doc["England"])


def test_a_climb_already_there_does_not_step(graph):
    """A quarter of real birthplaces are countries already, and a fixed hop
    count walks straight past every one of them."""
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["located_in"], country)]
    device, reference = both(graph, steps, doc["Poland"])
    assert device == reference == str(doc["Poland"])


def test_a_climb_that_runs_out_says_nowhere(graph):
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["born_in"], libgraphcard.PLAIN), (rid["located_in"], country)]
    device, reference = both(graph, steps, doc["Hampshire"])
    assert device == reference == "NOWHERE"


# --- how far a climb may go, which is now a choice ----------------------------
#
# The limit used to be three separate literal 6s - `libgraph`, `libgraphcard`'s
# default argument, and an immediate in `buildgraphwalk` - with a comment in the
# third pointing at a constant in the second that did not exist. One definition
# now, and a card can be built with another.


def climb_to_country(graph, subject: str, limit: int):
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["born_in"], libgraphcard.PLAIN), (rid["located_in"], country)]
    return both(graph, steps, doc[subject], climb_limit=limit)


def test_the_limit_counts_values_examined_and_not_hops(graph):
    """The off-by-one every comment in this repository had backwards.

    Both walkers test the type at the *top* of the loop and give up when the
    count runs out, so the value the last hop reached is never tested: a limit
    of n permits n - 1 hops. Warsaw -> Poland is one hop and needs two.

    The two implementations agree, which is exactly why nothing caught it -
    the eZ80 checks, decrements, gives up on zero, and only then hops.
    """
    _card, _path, doc, _rid, _titles, _db = graph
    assert climb_to_country(graph, "Marie Curie", 1) == ("NOWHERE", "NOWHERE")

    device, reference = climb_to_country(graph, "Marie Curie", 2)
    assert device == reference == str(doc["Poland"])


def test_a_climb_stops_where_the_limit_says(graph):
    """Steventon -> Hampshire -> England is two hops, so it needs three."""
    _card, _path, doc, _rid, _titles, _db = graph
    assert climb_to_country(graph, "Jane Austen", 2) == ("NOWHERE", "NOWHERE")

    device, reference = climb_to_country(graph, "Jane Austen", 3)
    assert device == reference == str(doc["England"])


def test_a_shorter_climb_is_unaffected_by_a_deeper_limit(graph):
    """Raising the limit must not make a walk step past an answer it already
    had - a quarter of real birthplaces are countries already."""
    _card, _path, doc, _rid, _titles, _db = graph
    for limit in (2, 3, 6, 12):
        device, reference = climb_to_country(graph, "Marie Curie", limit)
        assert device == reference == str(doc["Poland"]), limit


def test_a_deeper_limit_costs_no_card_bytes(graph):
    """The limit is an immediate, not an unrolled loop, so it is free in the
    one budget this project has to keep - and `--climb-limit` is therefore a
    choice about answers rather than about size."""
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["born_in"], libgraphcard.PLAIN), (rid["located_in"], country)]
    sizes = {limit: len(harness(card, steps, doc["Jane Austen"], limit).build())
             for limit in (1, 2, 6, 255)}
    assert len(set(sizes.values())) == 1, sizes


def test_a_limit_of_zero_is_refused(graph):
    """It would emit a walk that decrements to zero before its first test and
    answers NOWHERE for everything, including the quarter of subjects that are
    already what was asked for."""
    card, _path, doc, rid, _titles, _db = graph
    country = card.type_names.index("country")
    steps = [(rid["located_in"], country)]
    with pytest.raises(ValueError, match="cannot check anything"):
        harness(card, steps, doc["Poland"], climb_limit=0)


def test_one_definition_reaches_all_three_walkers():
    """`buildgraphwalk` carried a comment saying it matched a constant that
    `libgraphcard` did not have, and `libgraph` held a third copy."""
    assert libgraph.CLIMB_LIMIT is libgraphcard.CLIMB_LIMIT
    assert buildgraphwalk.CLIMB_LIMIT is libgraphcard.CLIMB_LIMIT


# --- inverses on the machine --------------------------------------------------


def test_an_inverse_hop_agrees(graph):
    """The reverse table was written from the first commit and read by nothing.
    One flag on the step's relation byte is all it needed."""
    _card, _path, doc, rid, _titles, _db = graph
    steps = [(rid["born_in"] | libgraphcard.INVERSE, libgraphcard.PLAIN)]
    device, reference = both(graph, steps, doc["Warsaw"])
    assert device == reference == str(doc["Marie Curie"])


def test_an_inverse_with_nothing_pointing_at_it_says_nowhere(graph):
    _card, _path, doc, rid, _titles, _db = graph
    steps = [(rid["born_in"] | libgraphcard.INVERSE, libgraphcard.PLAIN)]
    device, reference = both(graph, steps, doc["Jane Austen"])
    assert device == reference == "NOWHERE"


def test_every_edge_walks_backwards_too(graph):
    """The reverse table indexed wrongly would return a neighbouring subject,
    which is a person who exists and was born somewhere else."""
    _card, _path, doc, rid, _titles, db = graph
    for _subject, relation, obj in db.execute(
            "SELECT subject, relation, object FROM edge WHERE source = 'w'"):
        steps = [(rid[relation] | libgraphcard.INVERSE, libgraphcard.PLAIN)]
        device, reference = both(graph, steps, doc[obj])
        assert device == reference, f"{obj} <- {relation}"


def test_forward_and_inverse_do_not_confuse_their_tables(graph):
    """One flag decides which megabyte is searched; pointing at the wrong one
    finds a well-formed record belonging to a different question."""
    _card, _path, doc, rid, _titles, _db = graph
    forward = [(rid["born_in"], libgraphcard.PLAIN)]
    back = [(rid["born_in"] | libgraphcard.INVERSE, libgraphcard.PLAIN)]

    assert walk_on_device(graph, forward, doc["Marie Curie"]) == str(doc["Warsaw"])
    assert walk_on_device(graph, back, doc["Warsaw"]) == str(doc["Marie Curie"])
    # ...and each is nowhere in the other direction.
    assert walk_on_device(graph, back, doc["Marie Curie"]) == "NOWHERE"
