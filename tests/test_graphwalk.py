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


def harness(card, steps: list[tuple[int, int]], subject: int) -> EZ80Builder:
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
    b.ld_mem_label_hl("GW_BASE")

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
        buffer_label="IOBUF", seekoff_label="SEEKOFF")

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


def walk_on_device(graph, steps, subject) -> str:
    card, path, _doc, _rid, _titles, _db = graph
    host = AgonHost(stdin=[], files={"WIKI.GRF": path.read_bytes()})
    return host.run(harness(card, steps, subject).build(),
                    max_cycles=200_000_000).strip()


def both(graph, steps, subject):
    """What the device says, and what the reference says, for one walk."""
    card = graph[0]
    answer, _walked, _missing = card.follow(subject, steps)
    reference = "NOWHERE" if answer is None else str(answer)
    return walk_on_device(graph, steps, subject), reference


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
