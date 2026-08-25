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
    return run_session(card, [query])


def run_session(card, queries: list[str]) -> str:
    idx, dat, index = card
    builder = buildwikibin.build(index.num_docs)
    host = AgonHost(stdin=[*queries, "!"], files={
        "WIKI.IDX": idx.read_bytes(),
        "WIKI.DAT": dat.read_bytes(),
    })
    return host.run(builder.build(), max_cycles=2_000_000_000)


# --- the format ---------------------------------------------------------------


def test_the_index_round_trips_through_the_reference(reference):
    for term in ("telephone", "photosynthesis", "shakespeare"):
        assert reference._postings(term), term


# --- gap-encoded postings -------------------------------------------------------
#
# Doc ids ascend within a term and cluster, so the gaps are small: 65.8% fit in
# a byte over the full card. Storing the gap rather than the id took the index
# from 33.1 MB to 23.1. What makes it safe is that the width rides in the two
# spare bits of a byte that only ever needed five for the weight - so nothing
# about the accumulator changes, and neither does the count of bytes read per
# posting being knowable before it is read.


@pytest.mark.parametrize("docs", [
    [0],                                  # the first gap is measured from zero
    [0, 1, 2],                            # every gap one byte
    [5, 300],                             # a two-byte gap
    [0, 200_000],                         # a three-byte gap
    [1, 2, 300, 400, 100_000, 300_000],   # all three widths in one list
])
def test_postings_survive_the_gap_encoding(docs):
    entries = [libsearch.Posting(doc, 1 + (doc % libsearch.MAX_WEIGHT))
               for doc in docs]
    decoded = libsearch.decode_postings(libsearch.encode_postings(entries))
    assert [(p.doc, p.weight) for p in decoded] == \
           [(p.doc, p.weight) for p in entries]


def test_the_encoding_is_smaller_than_the_flat_one_it_replaced():
    """Four bytes a posting was the old format; the gain is the whole point."""
    entries = [libsearch.Posting(doc, 7) for doc in range(0, 2000, 3)]
    assert len(libsearch.encode_postings(entries)) < 4 * len(entries)


def test_a_posting_out_of_order_is_still_read_back_correctly():
    """The encoder sorts, because the gaps are meaningless otherwise.

    Nothing upstream produces an unsorted list today - the builder walks
    documents in order - but the encoding now depends on that, and a dependency
    worth having is one that does not rely on a dict's iteration order.
    """
    entries = [libsearch.Posting(9, 1), libsearch.Posting(2, 2),
               libsearch.Posting(700, 3)]
    decoded = libsearch.decode_postings(libsearch.encode_postings(entries))
    assert [(p.doc, p.weight) for p in decoded] == [(2, 2), (9, 1), (700, 3)]


def test_the_width_rides_in_the_bits_the_weight_never_uses():
    """A weight is five bits, so bits 5 and 6 were spare and are now the width.

    If a weight ever needed six bits this would corrupt it silently, so the
    two are pinned against each other here rather than in separate places.
    """
    assert libsearch.MAX_WEIGHT == 0x1F
    tagged = libsearch.encode_postings([libsearch.Posting(1, libsearch.MAX_WEIGHT)])
    assert tagged[0] & libsearch.MAX_WEIGHT == libsearch.MAX_WEIGHT
    assert tagged[0] >> libsearch.WEIGHT_BITS == 0          # a one-byte gap


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
    """Including a card in the format this one replaced.

    The magic is read from `libsearch` rather than spelled out, because the
    version in it moves whenever the layout does and a test that pins the old
    spelling fails for the one reason that is not a bug.
    """
    bad = tmp_path / "bad.IDX"
    bad.write_bytes(b"NOTZWIK" + bytes(64))
    with pytest.raises(ValueError, match=f"not a {libsearch.MAGIC.decode()} index"):
        libsearch.CardSearch(bad, bad)

    stale = tmp_path / "stale.IDX"
    stale.write_bytes(b"ZWIKI1" + bytes(64))
    with pytest.raises(ValueError, match="not a "):
        libsearch.CardSearch(stale, stale)


# --- the page-tiered accumulator ------------------------------------------------
#
# Six articles fit in one 256-article page, so the corpus above never exercises
# the page table: skipping unflagged pages, the short final page, or clearing
# between queries. Three pages here, the last one partial.

BIG_DOCS = 600


@pytest.fixture(scope="module")
def big_card(tmp_path_factory):
    out = tmp_path_factory.mktemp("bigcard")
    titles = [f"Article {i}" for i in range(BIG_DOCS)]
    leads = [f"filler text {i}" for i in range(BIG_DOCS)]
    leads[0] = "zxqfirst appears only here"
    leads[300] = "zxqmid appears only here"
    leads[599] = "zxqlast appears only here"
    index = libsearch.build(titles, leads, {})
    idx = out / "WIKI.IDX"
    dat = out / "WIKI.DAT"
    libsearch.write_index(index, idx)
    libsearch.write_text(index, dat)
    return idx, dat, index


@pytest.mark.parametrize("query,title", [
    ("zxqfirst", "Article 0"),     # first page flagged, later pages skipped
    ("zxqmid", "Article 300"),     # a middle page
    ("zxqlast", "Article 599"),    # the short final page
])
def test_a_flagged_page_is_found_wherever_it_sits(big_card, query, title):
    assert title in run_session(big_card, [query])


def test_scores_do_not_leak_between_queries(big_card):
    """Clearing only flagged pages must still leave every byte zero, or the
    next query would inherit scores it never earned."""
    printed = run_session(big_card, ["zxqmid", "aardvark"])
    assert "Article 300" in printed
    assert "Nothing on the card matches" in printed


def test_a_later_query_still_scores_after_a_clear(big_card):
    printed = run_session(big_card, ["zxqfirst", "zxqlast"])
    assert "Article 0" in printed
    assert "Article 599" in printed


def test_a_term_spanning_every_page_still_agrees_with_the_reference(big_card):
    """The page tier's worst case: a term in every article flags every page,
    so nothing is skipped - and the answer must be the reference's answer."""
    idx, dat, _ = big_card
    reference = libsearch.CardSearch(idx, dat)
    try:
        best = reference.search("filler")
        assert best
        expected = reference.article(best[0][0])[0]
    finally:
        reference.close()
    assert expected in run_session(big_card, ["filler"])


# --- memory -------------------------------------------------------------------


def test_the_accumulator_clears_the_image():
    """284,000 articles is 277KB of accumulator; the build must prove the two
    do not overlap rather than discover it on hardware."""
    builder = buildwikibin.build(283_997)
    assert builder.org + len(builder.code) <= builder.accumulator


def test_a_corpus_too_large_to_score_fails_at_build_time():
    with pytest.raises(AssertionError, match="too large to score"):
        buildwikibin.build(600_000)


# --- notability, which decides which article someone meant --------------------


def test_alternate_names_outrank_a_shorter_page_that_repeats_the_word():
    """The bug this exists for, in miniature.

    BM25 rewards a short document that repeats a term, and neither is evidence
    of being the article someone meant. Over twenty question probes on the full
    corpus every miss had this shape - "Albert Einstein Square" over Albert
    Einstein, "East Berlin" over Berlin, "Napoleon II" over Napoleon - because
    the derived page is a stub that says the name twice.

    Wikipedia's editors have already voted, by writing redirects.
    """
    titles = ["Napoleon", "Napoleon II"]
    leads = [
        "Napoleon was a French military leader who became emperor. "
        "He fought many wars across Europe and changed its borders and laws.",
        "Napoleon II was the son of Napoleon.",
    ]
    aliases = {0: ["Napoleon Bonaparte", "Bonaparte", "Emperor Napoleon"],
               1: ["Napoleon the Second"]}

    unfamed = libsearch.build(titles, leads, {})
    famed = libsearch.build(titles, leads, aliases)

    def best(index):
        scores = {p.doc: p.weight for p in index.postings["napoleon"]}
        return max(scores, key=lambda d: scores[d])

    assert best(unfamed) == 1, "the stub should win without notability"
    assert best(famed) == 0, "and lose once its alternate names are counted"


def test_notability_does_not_invent_a_match():
    """A famous article that never mentions the term must stay unmatched -
    the boost scales a score, it does not create one."""
    index = libsearch.build(
        ["Famous", "Obscure"], ["a b c", "telephone"],
        {0: [f"alias {i}" for i in range(20)]})
    docs = {p.doc for p in index.postings["telephone"]}
    assert docs == {1}
