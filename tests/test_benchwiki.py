"""The card benchmark's file handling.

Running the card is what the slow oracle tests already do, and doing it here
would mean building a 100MB card to learn what they know. What is worth
checking cheaply is the part that decides *which* bytes the machine is handed:
a card assembled from the wrong pieces produces numbers rather than an error,
and a benchmark that quietly measured a card missing its index would be worse
than no benchmark.
"""

from __future__ import annotations

import pytest

import benchwiki


def card(tmp_path, *suffixes, stem="WIKI"):
    """A stem with the named files beside it, each holding its own name."""
    (tmp_path / f"{stem}.bin").write_bytes(b"\x00" * 16)
    for suffix in suffixes:
        (tmp_path / f"{stem}{suffix}").write_bytes(suffix.encode())
    return tmp_path / stem


def test_a_search_card_needs_no_graph(tmp_path):
    """A card built without `--relations` has no `.GRF`, and finds articles."""
    binary, files = benchwiki.card_files(card(tmp_path, ".IDX", ".DAT"))
    assert len(binary) == 16
    assert sorted(files) == ["WIKI.DAT", "WIKI.IDX"]


def test_a_graph_card_is_served_all_three(tmp_path):
    _binary, files = benchwiki.card_files(card(tmp_path, ".IDX", ".DAT", ".GRF"))
    assert sorted(files) == ["WIKI.DAT", "WIKI.GRF", "WIKI.IDX"]
    assert files["WIKI.GRF"] == b".GRF"


@pytest.mark.parametrize("present,missing", [
    ((".DAT",), ".IDX"),
    ((".IDX",), ".DAT"),
    ((".GRF",), ".DAT"),
])
def test_a_card_missing_a_half_says_which(tmp_path, present, missing):
    """Without this the program runs and reports a cost for finding nothing."""
    with pytest.raises(SystemExit) as caught:
        benchwiki.card_files(card(tmp_path, *present))
    assert missing in str(caught.value)


def test_the_files_are_named_as_the_machine_asks_for_them(tmp_path):
    """`AgonHost` looks them up by name, so a path would never be opened."""
    _binary, files = benchwiki.card_files(card(tmp_path, ".IDX", ".DAT"))
    assert all("/" not in name for name in files)


def test_the_default_queries_cover_both_ends_of_the_tiering():
    """One number for "a query" is meaningless once the accumulator is tiered.

    A query naming a rare subject flags a handful of pages; a common word flags
    most of them and pays the whole-corpus scan plus the table's overhead. A
    default set of only the first kind would report the tiering as free.
    """
    assert len(benchwiki.DEFAULT_QUERIES) >= 4
    assert any(len(q.split()) >= 4 for q in benchwiki.DEFAULT_QUERIES)
