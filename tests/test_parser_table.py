"""The word-table parser, which is the fourth row of `examples/parser/`.

`compare.py` needs torch and does not run in CI. The table does not, and the
claim it exists to support is testable without training anything: a table is
told what a model has to infer, so **held-out object pairs cost it nothing**,
and it has no opinion at all about a word it was never given.

That second property is the one an Interactive Fiction needs and a bare argmax
cannot have. Both trained models answer `PUT ZORKMID IN BOX` with a confident
`OK` or `NO`; only something that can decline is able to say which word it did
not know.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "examples" / "parser"))
import gendata
import table

# --- the parse ----------------------------------------------------------------


@pytest.mark.parametrize("command,expected", [
    ("PUT KEY IN BOX", ("KEY", "BOX")),
    ("PUT THE KEY IN THE BOX", ("KEY", "BOX")),
    ("PLACE KEY INTO BOX", ("KEY", "BOX")),
    ("DROP THE KEY IN THE BOX", ("KEY", "BOX")),
    ("put key in box", ("KEY", "BOX")),          # case is not the player's job
])
def test_every_phrasing_reaches_the_same_two_nouns(command, expected):
    assert table.parse(command) == expected


def test_the_order_of_the_nouns_is_the_whole_question():
    """`PUT KEY IN BOX` and `PUT BOX IN KEY` are the same bag of trigrams and
    opposite answers - the case the flat encoder cannot represent at all."""
    assert table.parse("PUT KEY IN BOX") == ("KEY", "BOX")
    assert table.parse("PUT BOX IN KEY") == ("BOX", "KEY")
    assert table.respond("PUT KEY IN BOX") == "OK"
    assert table.respond("PUT BOX IN KEY") == "NO"


def test_the_vocabulary_is_read_out_of_the_templates():
    """Written down twice is written down wrong eventually: a new phrasing in
    `gendata.TEMPLATES` must not leave the parser silently behind."""
    for template in gendata.TEMPLATES:
        command = template.format(a="KEY", b="BOX")
        assert table.parse(command) == ("KEY", "BOX"), command


# --- what it will not do ------------------------------------------------------


@pytest.mark.parametrize("command", [
    "PUT ZORKMID IN BOX",        # a noun it was never given
    "PUT KEY IN GRUE",
    "XYZZY KEY IN BOX",          # a verb it was never given
    "PUT KEY BOX",               # no preposition
    "PUT KEY IN",                # one noun
    "PUT KEY IN BOX IN BAG",     # three
    "",
])
def test_it_declines_rather_than_guessing(command):
    assert table.parse(command) is None
    assert table.respond(command) is None


def test_declining_is_not_an_answer_that_happens_to_be_wrong():
    """The distinction the models cannot draw. `respond` returns None, which is
    neither OK nor NO, so a caller has to handle it and a player can be told
    which word was not understood."""
    assert table.respond("PUT ZORKMID IN BOX") not in ("OK", "NO")


# --- the claim the fourth row makes -------------------------------------------


def test_holding_out_pairs_costs_the_table_nothing():
    """`compare.py` holds out object *pairs*, so the eval set contains
    combinations no model saw in any phrasing - 85.9% flat, 98.4% banded. The
    table is right about all 240 pairs because it was told the sizes rather
    than shown examples."""
    accuracy, declined = table.score(table.corpus(gendata.pairs()))
    assert (accuracy, declined) == (1.0, 0)


def test_it_agrees_with_the_corpus_it_is_scored_against():
    """Two readings of one rule: `gendata.answer` compares size classes and the
    table parses then compares. They have to agree on every pair or the fourth
    row is measuring the parser against itself."""
    for a, b in gendata.pairs():
        for template in gendata.TEMPLATES:
            command = template.format(a=a, b=b)
            assert table.respond(command) == gendata.answer(a, b), command
