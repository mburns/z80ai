"""A tie at the archive is a question back, not a coin.

Two records that earn the same score from the same words are two records
the query does not tell apart, and `BESTID[0]` on a tie is the lower id -
which a plot hook or an answer would take as if the player had meant it.
So the card names the tied records and asks. Nothing is marked asked, the
log records a question about nothing, and the words are kept.
"""

from __future__ import annotations

import pytest

import buildwikibin
import libsearch
import libworld
from libhost import AgonHost
from libworld import Room, Rule, Topic, World

ORDER11 = "Standing Order 11"
ORDER12 = "Standing Order 12"
TITLES = [ORDER11, ORDER12, "Hatch Log"]
#: The two orders get the same lead so their document lengths tie as well as
#: their terms; BM25 would otherwise prefer the shorter.
LEADS = ["A screen is fitted by two people.",
         "A screen is fitted by two people.",
         "The hatch has not been opened in living memory."]
NOTICED = "Somebody upstairs has noticed the eleventh order."


def tied_world() -> World:
    return World(
        rooms=[Room("IT", "Racks of machines, and the screen.")],
        things=[], terminal=0,
        topics=[Topic("eleven", ["ELEVEN"], titles=[ORDER11]),
                Topic("twelve", ["TWELVE"], titles=[ORDER12])],
        messages=[NOTICED],
        rules=[Rule(when=[(libworld.C_ASKED, 0)],
                    then=[(libworld.A_PRINT, 0, 0)])])


@pytest.fixture(scope="module")
def game(tmp_path_factory):
    out = tmp_path_factory.mktemp("specify")
    index = libsearch.build(TITLES, LEADS, {})
    libsearch.write_index(index, out / "W.IDX")
    libsearch.write_text(index, out / "W.DAT")
    world = tied_world()
    libworld.resolve_topics(world, TITLES)
    binary = buildwikibin.build(index.num_docs, index_name="W.IDX",
                                text_name="W.DAT", world=world).build()
    return binary, {"W.IDX": (out / "W.IDX").read_bytes(),
                    "W.DAT": (out / "W.DAT").read_bytes()}


def visit(game, *commands: str) -> tuple[str, AgonHost]:
    binary, files = game
    host = AgonHost(stdin=[*commands, "!"], files=dict(files))
    return host.run(binary, max_cycles=2_000_000_000), host


def test_the_host_reference_agrees_that_the_orders_tie(tmp_path):
    """`libsearch` is the reference the card is held to, so the tie has to
    be real there first, or the card test would be testing the fixture."""
    index = libsearch.build(TITLES, LEADS, {})
    libsearch.write_index(index, tmp_path / "W.IDX")
    libsearch.write_text(index, tmp_path / "W.DAT")
    scores = dict(libsearch.CardSearch(tmp_path / "W.IDX",
                                       tmp_path / "W.DAT").search("standing order"))
    assert scores[0] == scores[1] > scores.get(2, 0)


def test_a_tie_names_both_records_and_asks(game):
    out, _ = visit(game, "use", "standing order")
    flat = " ".join(out.split())
    assert "More than one record matches:" in flat
    assert flat.index(ORDER11) < flat.index(ORDER12) < flat.index("Specify.")
    assert "fitted by two people" not in flat      # no article was chosen


def test_a_tie_marks_nothing_asked(game):
    """The rule on `C_ASKED eleven` fires for the specific query and not
    for the tied one - the plot does not advance on the lower id."""
    out, _ = visit(game, "use", "standing order")
    assert NOTICED not in out
    out, _ = visit(game, "use", "order 11")
    assert NOTICED in out and "fitted by two people" in out


def test_a_tie_is_logged_as_about_nothing_and_its_words_are_kept(game):
    _, host = visit(game, "use", "standing order")
    assert host.files["SILO.LOG"][1] == 0xFF and len(host.files["SILO.LOG"]) == 2
    assert host.files["SILO.ASK"] == b"standing order\r\n"


def test_a_specific_query_is_answered_and_not_kept(game):
    _, host = visit(game, "use", "order 12")
    assert host.files["SILO.LOG"][1] == 1 and len(host.files["SILO.LOG"]) == 2
    assert "SILO.ASK" not in host.files
