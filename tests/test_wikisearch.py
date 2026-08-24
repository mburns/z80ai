"""The search card, and the eZ80 program that reads it.

Same discipline as the inference tests: build a card, run the generated code
against it in the emulator, and compare with the reference implementation that
reads the identical files. Two independent readers of one format agreeing is
the signal worth having.

The corpus here is a handful of invented articles, so the tests say nothing
about retrieval *quality* - data/wikipedia measures that. These check that the
format round-trips, that the device arithmetic matches libsearch's, and that
the failure modes are the loud kind.
"""

from __future__ import annotations

import pytest

import buildwikibin
import libsearch
from libhost import AgonHost

CORPUS = [
    ("Alexander Graham Bell",
     "Alexander Graham Bell was a scientist who invented the telephone."),
    ("Photosynthesis",
     "Photosynthesis is how green plants make food from sunlight."),
    ("Hamlet",
     "Hamlet is a tragedy play written by William Shakespeare."),
    ("Zilog Z80",
     "The Zilog Z80 is an eight bit microprocessor from 1976."),
    ("Mount Everest",
     "Mount Everest is the highest mountain above sea level on Earth."),
    ("Telephone",
     "A telephone is a machine that carries speech over wires."),
]

ALIASES = {0: ["Graham Bell"], 3: ["Z80", "Zilog Z-80"]}


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    """Build a card once and hand back its paths and the index behind it."""
    out = tmp_path_factory.mktemp("card")
    titles = [t for t, _ in CORPUS]
    leads = [lead for _, lead in CORPUS]
    index = libsearch.build(titles, leads, ALIASES)

    idx = out / "WIKI.IDX"
    dat = out / "WIKI.DAT"
    libsearch.write_index(index, idx)
    libsearch.write_text(index, dat)
    return idx, dat, index


@pytest.fixture(scope="module")
def reference(card):
    idx, dat, _ = card
    searcher = libsearch.CardSearch(idx, dat)
    yield searcher
    searcher.close()


def run_query(card, query: str) -> str:
    """Run the generated binary against the card and return what it printed."""
    idx, dat, index = card
    builder = buildwikibin.build(index.num_docs)
    host = AgonHost(stdin=[query, "!"], files={
        "WIKI.IDX": idx.read_bytes(),
        "WIKI.DAT": dat.read_bytes(),
    })
    return host.run(builder.build(), max_cycles=2_000_000_000)


# --- the format ---------------------------------------------------------------


def test_the_index_round_trips_through_the_reference(reference):
    for term in ("telephone", "photosynthesis", "shakespeare"):
        assert reference._postings(term), term


def test_a_term_that_was_never_indexed_returns_nothing(reference):
    assert reference._postings("aardvark") == []
    assert reference._postings("the") == []      # a stopword, excluded at build


def test_redirects_are_indexed_as_alternate_names(reference):
    """`z80` only finds Zilog Z80 because an alias said so - nothing here does
    fuzzy matching, which is why Wikipedia's redirects are worth carrying."""
    best = reference.search("z80")
    assert best
    assert reference.article(best[0][0])[0] == "Zilog Z80"


def test_weights_fit_the_bits_the_accumulator_assumes(card):
    _, _, index = card
    for postings in index.postings.values():
        for posting in postings:
            assert 1 <= posting.weight <= libsearch.MAX_WEIGHT


def test_eight_terms_cannot_overflow_one_byte():
    """The whole reason the accumulator is a byte per article."""
    assert libsearch.MAX_WEIGHT * libsearch.MAX_QUERY_TERMS <= 255


def test_the_text_file_round_trips(reference):
    for doc, (title, lead) in enumerate(CORPUS):
        assert reference.article(doc) == (title, lead)


# --- the device against the reference -----------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("telephone", "Telephone"),
    ("photosynthesis", "Photosynthesis"),
    ("shakespeare", "Hamlet"),
    ("highest mountain", "Mount Everest"),
    ("microprocessor", "Zilog Z80"),
])
def test_the_binary_finds_what_the_reference_finds(card, reference, query, expected):
    best = reference.search(query)
    assert best, f"the reference found nothing for {query!r}"
    assert reference.article(best[0][0])[0] == expected

    printed = run_query(card, query)
    assert expected in printed


def test_the_binary_prints_the_lead_not_just_the_title(card):
    printed = run_query(card, "photosynthesis")
    assert "green plants make food" in printed


def test_a_query_matching_nothing_says_so(card):
    printed = run_query(card, "aardvark")
    assert "Nothing on the card matches" in printed


def test_scores_agree_between_the_device_and_the_reference(card, reference):
    """The device adds bytes; libsearch adds the same bytes. They must land on
    the same document, or one of the two readers has the format wrong."""
    for query in ("telephone", "plants sunlight", "tragedy play"):
        best = reference.search(query)
        top_title = reference.article(best[0][0])[0]
        assert top_title in run_query(card, query)


# --- the loud failures ---------------------------------------------------------


def test_a_card_in_the_wrong_format_is_refused(card, tmp_path):
    """A stale card must fail visibly, not score against misread bytes."""
    idx, dat, index = card
    corrupt = bytearray(idx.read_bytes())
    corrupt[:6] = b"ZWIKI0"
    builder = buildwikibin.build(index.num_docs)
    host = AgonHost(stdin=["telephone", "!"], files={
        "WIKI.IDX": bytes(corrupt),
        "WIKI.DAT": dat.read_bytes(),
    })
    printed = host.run(builder.build(), max_cycles=100_000_000)
    assert "different format" in printed


def test_a_missing_card_says_so(card):
    _, _, index = card
    builder = buildwikibin.build(index.num_docs)
    host = AgonHost(stdin=["telephone", "!"], files={})
    printed = host.run(builder.build(), max_cycles=100_000_000)
    assert "Cannot open" in printed


def test_the_reference_refuses_a_foreign_index(tmp_path):
    bad = tmp_path / "bad.IDX"
    bad.write_bytes(b"NOTZWIK" + bytes(64))
    with pytest.raises(ValueError, match="not a ZWIKI1 index"):
        libsearch.CardSearch(bad, bad)


# --- memory -------------------------------------------------------------------


def test_the_accumulator_clears_the_image():
    """284,000 articles is 277KB of accumulator; the build must prove the two
    do not overlap rather than discover it on hardware."""
    builder = buildwikibin.build(283_997)
    assert builder.org + len(builder.code) <= builder.accumulator


def test_a_corpus_too_large_to_score_fails_at_build_time():
    with pytest.raises(AssertionError, match="too large to score"):
        buildwikibin.build(600_000)
