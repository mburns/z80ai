"""Walking the fact graph.

The measurement behind this module: `birth_place -> country` completes for
1.7% of subjects, and the same chain written as `born_in -> located_in`
completes for 40.7%. The graph was not sparse - a city records its country as
`subdivision_name`, so a chain asking for `country` failed on data that plainly
contained the answer.

These pin the collapsing, the traversal, and the part an oracle needs most:
that a broken chain says *where* it broke rather than only that it did.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import libgraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))
import ingest


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    # The real schema, not a hand-written stand-in. A copy is how these tests
    # kept passing against a `fact` table the ingest had already changed.
    conn.executescript(ingest._schema())
    return conn


def load(db, articles, facts, redirects=(), categories=()):
    """`facts` are (subject, property, value) or (subject, property, ordinal,
    value) - the ordinal defaults to 0, the way an unindexed field does.
    `categories` are (title, name), the way a page files itself."""
    db.executemany("INSERT INTO article (source, title, lead) "
                   "VALUES ('w', ?, '')", [(a,) for a in articles])
    db.executemany("INSERT INTO redirect VALUES ('w', ?, ?)", redirects)
    db.executemany(
        "INSERT OR REPLACE INTO fact (source, subject, property, ordinal, "
        "value, kind, num) VALUES ('w', ?, ?, ?, ?, 'text', NULL)",
        [(f[0], f[1], 0, f[2]) if len(f) == 3 else f for f in facts])
    db.executemany("INSERT OR REPLACE INTO category VALUES ('w', ?, ?)",
                   categories)
    libgraph.build(db, "w")


# --- the collapsing -----------------------------------------------------------


def test_synonymous_properties_become_one_relation(db):
    """The finding this module exists for: a city says `subdivision_name`
    where a country says `country`, and a chain asking for either fails."""
    load(db,
         ["Alan Turing", "London", "Paris", "England", "France"],
         [("Alan Turing", "birth_place", "London"),
          ("London", "subdivision_name", "England"),   # a city's vocabulary
          ("Paris", "country", "France")])             # a different one

    assert libgraph.follow(db, "w", "London", ["located_in"]).value == "England"
    assert libgraph.follow(db, "w", "Paris", ["located_in"]).value == "France"


def test_the_most_specific_field_wins(db):
    """A page carrying both must contribute one edge, not two that disagree."""
    load(db, ["Berlin", "Germany", "Brandenburg"],
         [("Berlin", "country", "Germany"),
          ("Berlin", "subdivision_name", "Brandenburg")])
    edges = db.execute("SELECT object FROM edge WHERE subject='Berlin' "
                       "AND relation='located_in'").fetchall()
    assert edges == [("Germany",)]      # `country` outranks `subdivision_name`


def test_a_lower_ranked_field_wins_when_the_better_one_names_no_article(db):
    """Ranking decides between usable values, not on behalf of an unusable one.

    `writer` outranks `artist` for `created_by`, and a song's writer is often a
    list of three people that names no article while its artist names one. The
    rank alone would drop a `created_by` answer the page plainly holds; this
    cost 199 edges when a change to the ingest started populating fields that
    had previously been empty.
    """
    load(db, ["A Broken Wing", "Martina McBride"],
         [("A Broken Wing", "writer", "Phil Barnhart, Sam Hogin, James House"),
          ("A Broken Wing", "artist", "Martina McBride")])
    edges = db.execute("SELECT object FROM edge WHERE subject='A Broken Wing' "
                       "AND relation='created_by'").fetchall()
    assert edges == [("Martina McBride",)]


def test_the_better_field_still_wins_when_both_resolve(db):
    """The fallback must not become a preference for whatever resolves last."""
    load(db, ["A Song", "A Writer", "A Singer"],
         [("A Song", "writer", "A Writer"), ("A Song", "artist", "A Singer")])
    edges = db.execute("SELECT object FROM edge WHERE subject='A Song' "
                       "AND relation='created_by'").fetchall()
    assert edges == [("A Writer",)]


def test_a_value_naming_no_article_is_dropped(db):
    """A hop can only continue to something the corpus actually holds."""
    load(db, ["Alan Turing"], [("Alan Turing", "birth_place", "Nowhere-at-all")])
    assert db.execute("SELECT COUNT(*) FROM edge").fetchone()[0] == 0


# --- containment a page files rather than tabulates ---------------------------
#
# `Infobox U.S. state` has no country field, so Michigan is in the United
# States only by being filed under `1837 establishments in the United States`.
# Reading those rescued 2,869 of the 4,686 distinct places a birthplace climb
# died on, and took `born_in in_country` from 56.2% to 72.1%. The guards below
# are each here because the rule without them produced nonsense.


def located(db, subject):
    row = db.execute("SELECT object FROM edge WHERE subject = ? "
                     "AND relation = 'located_in'", (subject,)).fetchone()
    return row[0] if row else None


def test_a_category_supplies_containment_the_infobox_never_had(db):
    """The finding: a place whose template has no country field at all.

    Also the leading-article case, which is most of them. English writes "in
    the United States" and Wikipedia titles the article "United States", so a
    tail taken literally resolves to nothing and Michigan stays unplaced.
    """
    load(db, ["Michigan", "United States", "Detroit"],
         [("Detroit", "birth_place", "Michigan"),      # makes Michigan a place
          ("Detroit", "country", "United States")],
         categories=[("Michigan", "1837 establishments in the United States")])
    assert located(db, "Michigan") == "United States"


def test_an_infobox_outranks_a_category(db):
    """A page that said where it was keeps what it said. The category is the
    weaker evidence, so this fills gaps and never replaces - which is what
    makes it monotonic: no chain that completed before stops completing."""
    load(db, ["Berlin", "Germany", "France"],
         [("Berlin", "country", "Germany")],
         categories=[("Berlin", "Cities in France")])
    assert located(db, "Berlin") == "Germany"


def test_a_person_is_not_contained_by_their_category(db):
    """`Presidents of France` parses exactly like `Cities in France`."""
    load(db, ["Charles de Gaulle", "France", "Paris"],
         [("Charles de Gaulle", "birth_place", "Paris"),
          ("Paris", "country", "France")],
         categories=[("Charles de Gaulle", "Presidents of France")])
    assert located(db, "Charles de Gaulle") is None


def test_a_target_that_is_not_a_place_is_refused(db):
    """"Bands established in 2022" parses, and this corpus has an article on
    2022. Without this guard the naive rule filed things inside years."""
    load(db, ["Some Band", "2022", "Somewhere", "A Country"],
         [("Some Band", "birth_place", "Somewhere"),
          ("Somewhere", "country", "A Country")],
         categories=[("Some Band", "Bands established in 2022")])
    assert located(db, "Some Band") is None


def test_a_subject_that_is_not_a_place_is_refused(db):
    """Otherwise every band formed in California is filed inside it - 70,844
    edges, of which 3,763 were about places."""
    load(db, ["A Band", "California", "United States"],
         [("California", "country", "United States")],
         categories=[("A Band", "Musical groups in California")])
    assert located(db, "A Band") is None


def test_the_returned_count_is_what_the_table_holds(db):
    """The caller writes it into `meta` as the size of the graph.

    Counting only the infobox edges recorded 163,977 against a table holding
    167,922 - a provenance number that disagrees with the thing it describes,
    which is worse than not having one.
    """
    load(db, ["Aarhus", "Denmark", "Copenhagen"],
         [("Copenhagen", "birth_place", "Aarhus")],
         categories=[("Aarhus", "Cities in Denmark")])
    reported, _dropped = libgraph.build(db, "w")
    held = db.execute("SELECT COUNT(*) FROM edge WHERE source = 'w'").fetchone()[0]
    assert reported == held


def test_a_country_is_preferred_to_a_continent(db):
    """A climb carries on from a country and stops dead at a continent, and
    rows arrive in primary-key order, which is alphabetical and means nothing.

    Denmark is deliberately given no `located_in` of its own, so the only
    thing that can prefer it here is that three infoboxes call it a country -
    which means this fails if `entity_type` is not built before the categories
    are read. Written the other way round, it passes on the alphabet.

    That ordering is worth 365 edges and 821 completed chains on the real
    corpus, and it is invisible to any measurement taken by re-running `build`
    against a database that already has an `entity_type`: the stale rows are
    right, so the wrong order gets the right answer. Only a fresh ingest shows
    it, which is why this is a test and not a number in a README.
    """
    load(db, ["Aarhus", "Denmark", "Europe", "Copenhagen", "Berlin", "Oslo",
              "Iceland"],
         [("Copenhagen", "birth_place", "Aarhus"),   # makes Aarhus a place
          ("Copenhagen", "country", "Denmark"),
          ("Berlin", "country", "Denmark"),
          ("Oslo", "country", "Denmark"),            # three votes: a country
          ("Iceland", "country", "Europe")],         # one: a place, not a country
         categories=[("Aarhus", "Cities in Denmark"),
                     ("Aarhus", "Cities in Europe")])
    assert libgraph.is_a(db, "w", "Denmark", "country")
    assert not libgraph.is_a(db, "w", "Europe", "country")
    assert located(db, "Aarhus") == "Denmark"


def test_a_value_resolves_through_a_redirect(db):
    load(db, ["Alan Turing", "United Kingdom"],
         [("Alan Turing", "birth_place", "Britain")],
         [("Britain", "United Kingdom")])
    assert libgraph.follow(db, "w", "Alan Turing", ["born_in"]).value == \
        "United Kingdom"


def test_a_trailing_qualifier_is_dropped_to_find_the_article(db):
    """"Edinburgh, Scotland" is a sentence about a place, not its title."""
    load(db, ["Bell", "Edinburgh"],
         [("Bell", "birth_place", "Edinburgh, Scotland")])
    assert libgraph.follow(db, "w", "Bell", ["born_in"]).value == "Edinburgh"


def test_a_middle_segment_is_taken_when_the_narrowest_is_not_an_article(db):
    """Jane Austen's birthplace, and 11% of all edges, hang on this.

    The corpus has no article on Steventon Rectory, so an edge that insists on
    the most specific segment is no edge at all. Falling through to Hampshire
    trades precision for an answer that is still true.
    """
    load(db, ["Jane Austen", "Hampshire"],
         [("Jane Austen", "birth_place",
           "Steventon Rectory, Hampshire, England")])
    assert libgraph.follow(db, "w", "Jane Austen", ["born_in"]).value == "Hampshire"


def test_the_narrowest_segment_still_wins_when_the_corpus_has_it(db):
    """Falling through must be a fallback, not the rule - precision first."""
    load(db, ["Jane Austen", "Steventon", "Hampshire"],
         [("Jane Austen", "birth_place", "Steventon, Hampshire")])
    assert libgraph.follow(db, "w", "Jane Austen", ["born_in"]).value == "Steventon"


# --- traversal ----------------------------------------------------------------


def chain_fixture(db):
    load(db,
         ["Alan Turing", "London", "England", "United Kingdom"],
         [("Alan Turing", "birth_place", "London"),
          ("London", "subdivision_name", "England"),
          ("England", "country", "United Kingdom")])


def test_a_two_hop_chain_completes(db):
    chain_fixture(db)
    answer = libgraph.follow(db, "w", "Alan Turing", ["born_in", "located_in"])
    assert answer.value == "England"
    assert answer.complete


def test_a_three_hop_chain_completes(db):
    chain_fixture(db)
    answer = libgraph.follow(
        db, "w", "Alan Turing", ["born_in", "located_in", "located_in"])
    assert answer.value == "United Kingdom"
    assert answer.path == ["Alan Turing", "London", "England", "United Kingdom"]


def test_a_broken_chain_reports_where_it_broke(db):
    """The difference between "I don't know" and "I know where he was born,
    but not what country that is in" - which is the whole voice of an oracle
    that is wrong half the time."""
    load(db, ["Alan Turing", "London"],
         [("Alan Turing", "birth_place", "London")])

    answer = libgraph.follow(db, "w", "Alan Turing", ["born_in", "located_in"])
    assert not answer.complete
    assert answer.path == ["Alan Turing", "London"]     # got one hop in
    assert answer.missing == "located_in"


def test_an_unknown_subject_stops_at_the_first_hop(db):
    chain_fixture(db)
    answer = libgraph.follow(db, "w", "Nobody", ["born_in"])
    assert answer.path == ["Nobody"]
    assert answer.missing == "born_in"


# --- inverses and counting ----------------------------------------------------


def test_the_graph_walks_backwards(db):
    """"Who was born in London" from the rows saying where people were born."""
    load(db, ["Alan Turing", "Ada Lovelace", "London"],
         [("Alan Turing", "birth_place", "London"),
          ("Ada Lovelace", "birth_place", "London")])
    assert sorted(libgraph.inverse(db, "w", "London", "born_in")) == \
        ["Ada Lovelace", "Alan Turing"]


def test_counting_is_free_once_the_inverse_index_exists(db):
    load(db, ["Alan Turing", "Ada Lovelace", "Charles Babbage", "London"],
         [("Alan Turing", "birth_place", "London"),
          ("Ada Lovelace", "birth_place", "London"),
          ("Charles Babbage", "birth_place", "London")])
    assert libgraph.count(db, "w", "London", "born_in") == 3
    assert libgraph.count(db, "w", "Paris", "born_in") == 0


def test_rebuilding_replaces_rather_than_accumulates(db):
    chain_fixture(db)
    before = db.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    libgraph.build(db, "w")
    assert db.execute("SELECT COUNT(*) FROM edge").fetchone()[0] == before


def test_sources_do_not_leak_into_each_other(db):
    """MetaQA and Wikipedia share the table; a walk must not cross between."""
    chain_fixture(db)
    db.execute("INSERT INTO edge VALUES ('metaqa', 'Alan Turing', 'born_in', "
               "'Somewhere Else')")
    assert libgraph.follow(db, "w", "Alan Turing", ["born_in"]).value == "London"
    assert libgraph.follow(db, "metaqa", "Alan Turing", ["born_in"]).value == \
        "Somewhere Else"


# --- asking for a type rather than a hop count --------------------------------
#
# The corpus records birthplaces at whatever granularity the author felt like,
# so the distance from a person to their country is 0 hops for 26.2% of people,
# 1 for 35.7% and 2 for 1.6%. Every one of these is a case a fixed-length path
# gets wrong in a different direction.


def countries(db, *names, times=libgraph.TYPE_FLOOR):
    """Make the corpus call something a country, TYPE_FLOOR authors deep."""
    db.executemany(
        "INSERT OR REPLACE INTO fact (source, subject, property, ordinal, "
        "value, kind, num) VALUES ('w', ?, 'country', 0, ?, 'text', NULL)",
        [(f"Article {i}", name) for name in names for i in range(times)])
    db.executemany("INSERT OR REPLACE INTO article (source, title, lead) "
                   "VALUES ('w', ?, '')",
                   [(f"Article {i}",) for i in range(times)])


def test_a_type_is_what_enough_infoboxes_say_it_is(db):
    load(db, ["France", "Paris"], [("Paris", "country", "France")])
    assert not libgraph.is_a(db, "w", "France", "country")   # one author only

    countries(db, "France")
    libgraph.build(db, "w")
    assert libgraph.is_a(db, "w", "France", "country")


def test_a_climb_stops_the_moment_it_is_already_there(db):
    """26.2% of birthplaces. A fixed hop would step past the answer."""
    countries(db, "France")
    load(db, ["Rene", "France"], [("Rene", "birth_place", "France")])
    answer = libgraph.follow(db, "w", "Rene", ["born_in", "in_country"])
    assert answer.value == "France"
    assert answer.path == ["Rene", "France"]        # no wasted step


def test_a_climb_keeps_going_until_the_type_matches(db):
    """1.6% of birthplaces need two. A fixed hop would stop short."""
    countries(db, "England")
    load(db, ["Jane", "Steventon", "Hampshire", "England"],
         [("Jane", "birth_place", "Steventon"),
          ("Steventon", "county", "Hampshire"),
          ("Hampshire", "country", "England")])
    answer = libgraph.follow(db, "w", "Jane", ["born_in", "in_country"])
    assert answer.value == "England"
    assert answer.path == ["Jane", "Steventon", "Hampshire", "England"]


def test_a_climb_that_never_reaches_the_type_declines(db):
    """Better than Einstein's old answer, which was that Ulm is in a country
    called Baden-Wurttemberg."""
    load(db, ["Albert", "Ulm", "Baden-Wurttemberg"],
         [("Albert", "birth_place", "Ulm"),
          ("Ulm", "subdivision_name", "Baden-Wurttemberg")])
    answer = libgraph.follow(db, "w", "Albert", ["born_in", "in_country"])
    assert not answer.complete
    assert answer.missing == "in_country"
    # It still reports how far it got, so the oracle can say so.
    assert answer.path == ["Albert", "Ulm", "Baden-Wurttemberg"]


def test_a_cycle_in_the_data_terminates(db):
    """Two places each inside the other is a thing real infoboxes do."""
    load(db, ["A", "B"],
         [("A", "country", "B"), ("B", "country", "A")])
    assert not libgraph.follow(db, "w", "A", ["in_country"]).complete


def test_rebuilding_replaces_types_rather_than_accumulating(db):
    countries(db, "France")
    load(db, ["France"], [])
    libgraph.build(db, "w")
    before = db.execute("SELECT COUNT(*) FROM entity_type").fetchone()[0]
    libgraph.build(db, "w")
    assert db.execute("SELECT COUNT(*) FROM entity_type").fetchone()[0] == before
