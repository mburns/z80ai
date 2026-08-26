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

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "data" / "wikipedia"))
import buildwikibin
import ingest
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
    # Two people who differ by one character. Without joined initials these
    # are not similar queries, they are the *same* query - a single-character
    # token is dropped at both ends - and no ranking can separate them.
    ("Amanda M. Wilson",
     "Amanda M. Wilson is a cook who works the first shift."),
    ("Amanda X. Wilson",
     "Amanda X. Wilson is a welder who works the third shift."),
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


# --- byte-pair packed text ------------------------------------------------------
#
# 49 byte values never occur in the corpus, so a code needs no escape and no
# shift state: a byte is either itself or it stands for a string. The table
# stores expansions flattened, so a merge of merges is still one lookup.


def packable(times: int = 40) -> str:
    """Text with digraphs common enough to clear MIN_PAIR_USES."""
    return " ".join("the theory of the thermometer is there" for _ in range(times))


def test_packing_learns_pairs_and_gives_the_text_back():
    text = packable().encode()
    merges = libsearch.learn_pairs(text, libsearch.free_codes(text))
    assert merges, "text this repetitive must yield merges"
    packed = libsearch.pack_text(text, merges)
    assert len(packed) < len(text)
    assert libsearch.unpack_text(packed, libsearch.pair_table(merges)) == text


def test_a_code_never_collides_with_a_byte_the_text_uses():
    text = packable().encode()
    merges = libsearch.learn_pairs(text, libsearch.free_codes(text))
    used = set(text)
    assert not used & {code for _pair, code in merges}


def test_no_code_may_hide_a_terminator():
    """The device counts the two NULs to know where the lead ends, and copies a
    code's expansion with a block move it does not inspect. A NUL inside an
    expansion is one the device never sees, so it reads on into the next
    article - which the real corpus did do, via `.\\x00`, most leads ending in a
    full stop."""
    text = (b"the article ends here.\x00Next article.\x00" * 200)
    merges = libsearch.learn_pairs(text, libsearch.free_codes(text))
    assert merges
    for code, expansion in libsearch.pair_table(merges).items():
        assert b"\x00" not in expansion, (code, expansion)


def test_text_with_no_free_bytes_is_stored_as_it_is():
    """Every byte value occurring leaves nothing to encode with. The packer has
    to notice rather than reuse a byte that means something."""
    text = bytes(range(256)) * 4
    merges = libsearch.learn_pairs(text, libsearch.free_codes(text))
    assert merges == []
    assert libsearch.pack_text(text, merges) == text


def test_the_device_unpacks_a_corpus_that_actually_has_pairs(tmp_path):
    """The bug this exists for: `UNPACK` held its slot pointer in BC and wrote
    the byte through `ld_c_a`, which is C. It printed `leArArtA300` for
    `Article 300`. Every small-corpus test passed, because a corpus that small
    learns no pairs and so never expands anything."""
    titles = [f"Article {n}" for n in range(300)]
    leads = [f"This is the article about the theory of the thing numbered {n}, "
             f"and there is theoretically nothing else in it." for n in range(300)]
    index = libsearch.build(titles, leads, {})

    idx, dat = tmp_path / "WIKI.IDX", tmp_path / "WIKI.DAT"
    libsearch.write_index(index, idx)
    libsearch.write_text(index, dat)

    with dat.open("rb") as fh:
        fh.seek(len(libsearch.TEXT_MAGIC))
        assert int.from_bytes(fh.read(2), "little") > 0, "no pairs, so no test"

    builder = buildwikibin.build(index.num_docs)
    host = AgonHost(stdin=["theory", "!"], files={
        "WIKI.IDX": idx.read_bytes(), "WIKI.DAT": dat.read_bytes()})
    printed = host.run(builder.build(), max_cycles=2_000_000_000)

    reference = libsearch.CardSearch(idx, dat)
    try:
        assert reference.article(0) == (titles[0], leads[0])
    finally:
        reference.close()
    assert "Article " in printed
    assert "theoretically nothing else in it." in printed


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


# --- fame ---------------------------------------------------------------------


def test_fame_is_the_value_that_was_swept():
    """A toy corpus cannot pin this, and pretending otherwise would be worse
    than not trying.

    `FAME` boosts an article by how many alternate names point at it. Its
    damage - a famous page outranking the article somebody asked for by name -
    needs a corpus with enough famous pages to swamp a ranking, and it scales
    with corpus size: 77.7% of articles found by their own title at 40,000
    articles, 53.3% at 120,000, 47.8% at 283,997. On the eight documents in
    this file the difference between 0.25 and 1.0 is under a point.

    So this pins the number and points at the measurement, which lives in
    `tools/probe_entities.py --sample N` and needs the real database. What it
    catches is somebody changing the constant without re-running that.
    """
    assert libsearch.FAME == 0.25


def test_the_corpus_probe_sets_are_built_from_what_the_card_indexes(tmp_path):
    """The twenty hand-written probes could not choose between four values of
    FAME - 5% of resolution each, and assembled from the symptoms of the one
    bug FAME repairs. These come out of the corpus instead."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import probe_entities

    db_path = tmp_path / "corpus.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(ingest._schema())
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('simplewiki', ?, '')",
                     [(t,) for t, _ in CORPUS])
    conn.executemany("INSERT INTO redirect (source, title, target) "
                     "VALUES ('simplewiki', ?, ?)",
                     [("Graham Bell", "Alexander Graham Bell"),
                      ("Nowhere", "An Article That Does Not Exist")])
    conn.commit()
    conn.close()

    by_title, by_redirect = probe_entities.corpus_probes(
        db_path, "simplewiki", sample=100, seed=0)
    assert len(by_title) == len(CORPUS)
    assert all(query == wanted for query, wanted in by_title)
    # A redirect pointing outside the corpus is not a probe: it has no right
    # answer, and counting it would make every card look worse than it is.
    assert by_redirect == [("Graham Bell", "Alexander Graham Bell")]


# --- initials -----------------------------------------------------------------


def test_an_initial_is_glued_to_the_name_after_it():
    assert libsearch.tokenize("Amanda M. Wilson", join_initials=True) == [
        "amanda", "mwilson"]
    assert libsearch.tokenize("Amanda M. Wilson") == ["amanda", "wilson"]


def test_a_single_character_stopword_is_left_alone():
    """Gluing `a` eats the word after it, which cost a probe that had always
    passed: `what is a black hole` became `ablack hole`."""
    assert libsearch.tokenize("what is a black hole", join_initials=True) == [
        "black", "hole"]


def test_only_two_stopwords_are_a_single_character():
    """`NEXT_TOKEN` on the eZ80 tests for exactly `a` and `i` rather than
    carrying the whole list. A third would make the machine disagree with
    `libsearch.tokenize` in silence, which is why this is pinned here."""
    assert {w for w in libsearch.STOPWORDS if len(w) == 1} == {"a", "i"}


def test_the_device_tells_apart_two_names_that_differ_by_an_initial(card):
    """The whole point, end to end: the reference separates them and so does
    the binary. Before the initials were joined both queries scored the two
    documents identically and the tie-break decided who you meant."""
    for query, expected in (("amanda m wilson", "Amanda M. Wilson"),
                            ("amanda x wilson", "Amanda X. Wilson")):
        assert expected in run_query(card, query)


def test_an_initial_at_the_end_of_a_query_terminates(card):
    """The glue is a one-shot. Without `NTGLUED` a query ending in a lone
    initial comes back from `NT_SKIP` with the token still one character long
    and goes round again forever - the emulator would burn its cycle budget
    rather than fail an assertion, so what this really pins is that the run
    finishes at all.

    The trailing initial glues to nothing and is dropped, which leaves
    `wilson`, which both of them have in their leads.
    """
    printed = run_query(card, "wilson q")
    assert "Amanda M. Wilson" in printed
    assert "Amanda X. Wilson" in printed
    assert libsearch.tokenize("wilson q", join_initials=True) == ["wilson"]


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


def test_the_unpacking_buffers_cost_the_image_nothing():
    """They were declared with `ds` first, which put 11,040 zeros in the .bin
    and grew it from 7.4KB to 16.4KB. Nothing failed - the card simply got
    bigger - so the property is pinned rather than left to the next reader."""
    builder = buildwikibin.build(283_997)
    scratch = buildwikibin.scratch_base(283_997)
    assert builder.org + len(builder.code) <= scratch
    for name in ("PACKBUF", "TEXTBUF", "PAIRTAB", "BLOBBUF"):
        assert builder.labels[name] >= scratch
    assert scratch + buildwikibin.SCRATCH_BYTES <= builder.accumulator


# --- how long an article may be -----------------------------------------------
#
# `READ_ARTICLE` reads exactly one CHUNK and `UNPACK` walks it until it has seen
# the two NULs ending the title and the lead. An article packing to more than
# that does not truncate - the second NUL is not in what was read, so the
# decoder carries on into whatever the last query left in SRAM. It looked clean
# when it was found only because emulated SRAM starts zeroed.
#
# Nothing checked this while every lead was 300 characters, which is the
# condition under which a limit is easiest to be wrong about.

PROSE = ("the pump on level forty two failed at oh six hundred and the shift "
         "lead reported water in the lower corridor before the seal was "
         "replaced by the maintenance crew who logged it as routine ")


def one_article_card(out, lead: str):
    index = libsearch.build(["Incident Report", "Filler"],
                            [lead, "nothing here at all"], {})
    idx, dat = out / "WIKI.IDX", out / "WIKI.DAT"
    libsearch.write_index(index, idx)
    libsearch.write_text(index, dat)
    return idx, dat, index


def test_an_article_the_device_cannot_finish_is_refused_at_build_time(tmp_path):
    """And names the article, because a corpus is a lot of places to look."""
    with pytest.raises(ValueError, match="Incident Report"):
        one_article_card(tmp_path, (PROSE * 40)[:6000])


def test_the_refusal_says_what_the_device_can_actually_hold(tmp_path):
    with pytest.raises(ValueError, match=str(libsearch.MAX_PACKED_ARTICLE)):
        one_article_card(tmp_path, (PROSE * 40)[:6000])


def test_an_article_at_the_limit_still_reaches_the_screen(tmp_path):
    """The boundary is worth exercising from both sides: one byte over raises,
    and the largest thing that does not raise must still come back whole."""
    lead = (PROSE * 40)[:libsearch.MAX_PACKED_ARTICLE - 200]
    card = one_article_card(tmp_path, lead)
    idx, dat, _index = card

    reference = libsearch.CardSearch(idx, dat)
    try:
        assert reference.article(0) == ("Incident Report", lead)
    finally:
        reference.close()

    printed = "".join(run_query(card, "incident").split())
    assert "".join(lead.split()) in printed


def test_the_reference_reads_what_the_device_reads(tmp_path):
    """`article` used to read 4096 packed bytes against the device's 2048, so a
    card the machine could not finish was one the reference finished fine - and
    every test that compares the two would have agreed with the wrong one."""
    assert libsearch.MAX_PACKED_ARTICLE == buildwikibin.CHUNK
    assert libsearch.MAX_ARTICLE == 2 * buildwikibin.CHUNK


# --- the ceiling, which used to be a guess ------------------------------------
#
# `buildwikisearch` warned above "380KB of accumulator" on the grounds that the
# program wanted the other 130KB. The program is 4.7KB, so the guess was low by
# 113,000 articles, and it had never been tested because nothing had built a
# card anywhere near it. `build` takes a count and no corpus, so the real
# boundary costs milliseconds to find.


def _bisect_the_ceiling() -> int:
    """The largest corpus `build` will emit for, found by asking it."""
    low, high = 1, 1_000_000
    while low < high:
        mid = (low + high + 1) // 2
        try:
            buildwikibin.build(mid)
            low = mid
        except AssertionError:
            high = mid - 1
    return low


def test_the_search_card_tops_out_where_it_tops_out():
    """The number itself, so that a change to the program's size shows up here
    as a diff rather than as a card that quietly stopped fitting."""
    assert _bisect_the_ceiling() == 502_016


def test_max_docs_agrees_with_building_until_it_breaks():
    """`max_docs` solves the inequality and the bisection runs the emitter, so
    the two are independent. Dropping the page table's byte per 256 articles
    from the solve moves the answer by about 2,000 and nothing else notices."""
    ceiling = _bisect_the_ceiling()
    image = len(buildwikibin.build(ceiling).code)
    assert buildwikibin.max_docs(
        buildwikibin.fixed_bytes(ceiling, image)) == ceiling


def test_one_more_article_than_fits_is_refused():
    ceiling = _bisect_the_ceiling()
    buildwikibin.build(ceiling)
    with pytest.raises(AssertionError, match="too large to score"):
        buildwikibin.build(ceiling + 1)


def test_the_refusal_says_what_would_have_fitted():
    """A build that fails at 600,000 is not actionable; one that names the
    limit is, and the limit depends on the image rather than being a constant
    somebody has to look up."""
    with pytest.raises(AssertionError, match="limit of 502,016"):
        buildwikibin.build(600_000)


def test_the_headroom_runs_out_at_the_ceiling_and_not_before():
    ceiling = _bisect_the_ceiling()
    assert buildwikibin.headroom(
        ceiling, len(buildwikibin.build(ceiling).code)) >= 0
    assert buildwikibin.headroom(ceiling + 1, len(
        buildwikibin.build(ceiling).code) + 1) < 0


def test_a_page_of_articles_costs_the_image_a_byte_and_the_gap_257():
    """Both bases round down to 256, so the accumulator and the buffers below
    it move a whole page at a time while the image grows by one byte. That is
    why an article costs 257/256 bytes and not one."""
    page = buildwikibin.num_pages(1) * 256
    assert buildwikibin.scratch_base(0) - buildwikibin.scratch_base(page) == 256
    assert (len(buildwikibin.build(100_000 + page).code)
            - len(buildwikibin.build(100_000).code)) == 1
    assert buildwikibin.max_docs(0) - buildwikibin.max_docs(257) == 256


def test_the_classifier_is_paid_for_in_articles():
    """An oracle card carries its model in the image, and every byte of it is
    an article the accumulator cannot have. The silo's two classifier widths
    were 94.4KB and 38.9KB, which is 55,000 articles between them."""
    search = buildwikibin.fixed_bytes(1, len(buildwikibin.build(1).code))
    plain = buildwikibin.max_docs(search)
    with_model = buildwikibin.max_docs(search + 55_000)
    assert plain - with_model == pytest.approx(55_000, abs=256)


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
