"""The name index, and `LOOKUP` on the device.

`libnames` is the contract: a typed name normalised and hashed the way the
eZ80 does it, sorted records, a lower-bound search. The device half is held
to the Python half the way every card reader here is - two implementations,
one answer - and then to the thing the index is for: a record printed with
no classifier and no search, for a name a door just gave the player.
Issue #105.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import buildwikibin
import libgraphcard
import libnames
import libsearch
import libworld
from libhost import AgonHost
from libworld import Room, Rule, Topic, World

# --- the contract ------------------------------------------------------------------


@pytest.mark.parametrize("text, key", [
    ("Alexander E. Wong", "ALEXANDER E WONG"),
    ("alexander e wong", "ALEXANDER E WONG"),
    ("  ALEXANDER   E.   WONG  ", "ALEXANDER E WONG"),
    ("Sheriff's Office", "SHERIFFS OFFICE"),
    ("Class of 135 (B)", "CLASS OF 135 B"),
    ("Incident Report 214-11: Cistern Pump Failure",
     "INCIDENT REPORT 21411 CISTERN PUMP FAILURE"),
    ("Muñoz", "MUOZ"),
    ("...", ""),
])
def test_normalize_is_what_the_device_computes(text, key):
    assert libnames.normalize(text) == key


def test_the_hashes_are_24_bit_and_independent():
    h1, h2 = libnames.hashes("ALEXANDER E WONG")
    assert 0 <= h1 < 1 << 24 and 0 <= h2 < 1 << 24
    assert h1 != h2
    assert libnames.hashes("A") != libnames.hashes("B")


def test_a_middle_initial_earns_a_second_key():
    assert libnames.keys_for("Alexander E. Wong") == \
        ["ALEXANDER E WONG", "ALEXANDER WONG"]
    assert libnames.keys_for("Level 42") == ["LEVEL 42"]
    assert libnames.keys_for("...") == []


def test_build_sorts_and_refuses_a_collision(monkeypatch):
    records = libnames.build(["B", "A", "A. X. C"])
    assert records == sorted(records)
    monkeypatch.setattr(libnames, "hashes", lambda key: (1, 1))
    with pytest.raises(ValueError, match="hash alike"):
        libnames.build(["A", "B"])


def test_the_reader_finds_every_key_and_lists_shared_ones(tmp_path):
    titles = ["Amanda M. Wilson", "Amanda X. Wilson", "Level 3", "Corey W. Wong"]
    libnames.write(libnames.build(titles), tmp_path / "T.NAM")
    names = libnames.CardNames(tmp_path / "T.NAM")
    assert names.lookup("amanda m wilson") == [0]
    assert names.lookup("Amanda X. Wilson") == [1]
    assert names.lookup("amanda wilson") == [0, 1]
    assert names.lookup("level 3") == [2]
    assert names.lookup("corey wong") == [3]
    assert names.lookup("nobody at all") == []
    assert names.lookup("") == []
    names.close()


def test_a_lookup_is_a_handful_of_probes(tmp_path):
    titles = [f"Person {i} Q. Name" for i in range(10_000)]
    libnames.write(libnames.build(titles), tmp_path / "T.NAM")
    names = libnames.CardNames(tmp_path / "T.NAM")
    names.lookup("person 4242 q name")
    assert names.probes <= 17          # log2(20,000) and the scan


def test_a_title_that_packs_past_what_a_record_reads_is_refused(tmp_path):
    # Not one letter repeated - byte pairs pack that to almost nothing - but
    # enough distinct words that no 49 pair codes bring it under the limit.
    long = " ".join(f"w{i}" for i in range(200))
    index = libsearch.build([long], ["x"], {})
    with pytest.raises(ValueError, match="bytes of title"):
        libsearch.write_text(index, tmp_path / "T.DAT")


# --- the device --------------------------------------------------------------------

CORPUS = [
    ("Alexander E. Wong", "Alexander E. Wong is a person of Silo 18."),
    ("Amanda M. Wilson", "Amanda M. Wilson is a person of Silo 18."),
    ("Amanda X. Wilson", "Amanda X. Wilson is a person of Silo 18."),
    ("Dylan R. Smith", "Dylan R. Smith is a person of Silo 18."),
    ("Mechanical", "Mechanical is a department."),
    ("Year 166", "Year 166 of the silo."),
    ("Standing Order 11: Screen Fitting",
     "A screen is fitted by two people and never by one."),
]
#: The silo's own relations, as `generate.py` names them, and `child_of`
#: because a listing leaves it out - a record that said everyone's father
#: twice would be the failure `liboracle.REDUNDANT` exists for.
EDGES = [
    ("Alexander E. Wong", "father_is", "Dylan R. Smith"),
    ("Alexander E. Wong", "child_of", "Dylan R. Smith"),
    ("Alexander E. Wong", "works_in", "Mechanical"),
    ("Alexander E. Wong", "born_in_year", "Year 166"),
    ("Amanda M. Wilson", "works_in", "Mechanical"),
]


def make_card(out: Path, world: World | None = None):
    """Index, text, graph and names from one article list, and the binary."""
    titles = [t for t, _ in CORPUS]
    doc = {t: i for i, t in enumerate(titles)}
    relations = sorted({r for _, r, _ in EDGES})
    rid = {n: i for i, n in enumerate(relations)}
    edges = [(doc[s], rid[r], doc[o]) for s, r, o in EDGES]

    index = libsearch.build(titles, [lead for _, lead in CORPUS], {})
    libsearch.write_index(index, out / "N.IDX")
    libsearch.write_text(index, out / "N.DAT")
    built = libgraphcard.build(titles, edges, relations, {}, paths=[])
    stats = libgraphcard.write(built, out / "N.GRF")
    names = libnames.write(libnames.build(titles), out / "N.NAM")
    spec = buildwikibin.OracleSpec(
        graph_name="N.GRF", forward_at=stats["forward_at"],
        num_edges=stats["edges"],
        types_at=libgraphcard.CardGraph(out / "N.GRF")._types_at,
        num_types=0, num_docs=len(titles), digest=built.digest, paths=[],
        names_name="N.NAM", num_names=names["records"], relations=relations)
    if world is not None:
        libworld.resolve_topics(world, titles)
    game = buildwikibin.build(len(titles), "N.IDX", "N.DAT", oracle=spec,
                              world=world).build()
    files = {name: (out / name).read_bytes()
             for name in ("N.IDX", "N.DAT", "N.GRF", "N.NAM")}
    return game, files, relations


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    return make_card(tmp_path_factory.mktemp("names"))


def ask(card, *lines: str) -> tuple[str, AgonHost]:
    game, files, _ = card
    host = AgonHost(stdin=[*lines, "!"], files=files)
    return host.run(game, max_cycles=2_000_000_000), host


def flat(out: str) -> str:
    return " ".join(out.split())


def test_a_lookup_prints_the_record_and_nothing_else(card):
    out, _ = ask(card, "lookup alexander e wong")
    text = flat(out)
    assert "Alexander E. Wong" in text
    assert "born: Year 166" in text
    assert "father: Dylan R. Smith" in text
    assert "works in: Mechanical" in text
    assert "child of" not in text                     # said once, as father
    assert "is a person of Silo 18" not in text        # no article, no search


def test_a_lookup_is_case_and_punctuation_blind(card):
    for line in ("LOOKUP Alexander E. Wong", "lookup  ALEXANDER   e   wong",
                 "Lookup alexander wong"):
        out, _ = ask(card, line)
        assert "father: Dylan R. Smith" in flat(out)


def test_a_name_that_is_not_enough_lists_who_shares_it(card):
    out, _ = ask(card, "lookup amanda wilson")
    text = flat(out)
    assert "That name is on more than one record:" in text
    assert "Amanda M. Wilson" in text and "Amanda X. Wilson" in text
    assert "works in" not in text


def test_a_name_the_card_does_not_hold_says_so(card):
    out, _ = ask(card, "lookup nobody at all")
    assert "No record under that name." in out


def test_a_record_with_no_edges_says_the_archive_holds_no_facts(card):
    out, _ = ask(card, "lookup standing order 11 screen fitting")
    text = flat(out)
    assert "Standing Order 11: Screen Fitting" in text
    assert "holds no facts" in text


def test_lookup_alone_asks_whom(card):
    out, _ = ask(card, "lookup")
    assert "Look up whom?" in out
    out, _ = ask(card, "lookup ")
    assert "Look up whom?" in out


def test_a_word_that_starts_with_lookup_is_a_question(card):
    out, _ = ask(card, "lookups")
    assert "Look up whom?" not in out
    assert "No record" not in out


def test_the_device_agrees_with_the_reader_on_every_title(card, tmp_path):
    """Two implementations, one answer, for every key the index holds."""
    _game, files, _ = card
    (tmp_path / "N.NAM").write_bytes(files["N.NAM"])
    names = libnames.CardNames(tmp_path / "N.NAM")
    for title, _ in CORPUS:
        for key in libnames.keys_for(title):
            docs = names.lookup(key)
            out, _ = ask(card, f"lookup {key.lower()}")
            if len(docs) == 1:
                assert flat(out).count(title) >= 1
                assert "more than one" not in out
            else:
                assert "more than one" in out
    names.close()


def test_what_a_lookup_costs(card):
    """No search, so no accumulator scan: a lookup reads the name index,
    the graph, and a title an edge, and nothing else. On seven articles a
    question is cheaper than that - the accumulator is seven bytes - so the
    comparison that matters is on the real card, in `benchcard.py`."""
    _, looked = ask(card, "lookup alexander e wong")
    _, idle = ask(card)
    read = looked.io_bytes - idle.io_bytes
    assert 0 < read < 2_000                    # probes, three edges, four titles
    assert looked.cpu.instructions - idle.cpu.instructions < 150_000


def test_the_lookup_defines_no_label_the_merged_program_already_has(tmp_path):
    """`Z80Builder.label` overwrites silently, and the first version of the
    lookup used `NT_` for its own labels - which the notice routine already
    did, so a `jr` resolved 5,691 bytes away. The same spy `tests/test_if.py`
    keeps, on the build that has every routine in it."""
    import libez80

    seen: dict[str, int] = {}
    original = libez80.EZ80Builder.label

    def spy(self, name: str) -> None:
        seen[name] = seen.get(name, 0) + 1
        original(self, name)

    libez80.EZ80Builder.label = spy
    try:
        make_card(tmp_path, watched_world())
    finally:
        libez80.EZ80Builder.label = original
    assert [n for n, c in seen.items() if c > 1] == []


# --- with a world in front of it ---------------------------------------------------


def watched_world() -> World:
    return World(
        rooms=[Room("IT", "The screen.")], things=[], terminal=0,
        topics=[Topic("wong", ["WONG"], titles=["Alexander E. Wong"], heat=2,
                      censor="THAT RECORD IS SEALED."),
                Topic("wilson", ["WILSON"], titles=["Amanda M. Wilson"])],
        messages=["The screen dims for a moment."],
        rules=[Rule(when=[(libworld.C_ASKED, 1)],
                    then=[(libworld.A_PRINT, 0, 0)])])


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    return make_card(tmp_path_factory.mktemp("watched"), watched_world())


def test_a_lookup_is_noticed_like_a_question(merged):
    """Asked, charged, logged, and sealed if the record is: the Voice hears
    a lookup, or a lookup would be the way round the Voice."""
    out, host = ask(merged, "use", "lookup alexander e wong")
    assert "THAT RECORD IS SEALED." in out
    assert "father" not in out
    assert len(host.files["SILO.LOG"]) == 2

    out, _ = ask(merged, "use", "lookup amanda m wilson")
    assert "works in: Mechanical" in flat(out)
    assert "The screen dims" in out
