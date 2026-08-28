"""The synthetic silo corpus: is it coherent, and does the graph agree with it?

Two different jobs, and keeping them apart is the point of this file.

**Is the corpus coherent** - nobody is their own ancestor, no parent died
before their child was born, no tenancy begins after the archive is dated.
These are properties of the simulation, and they are checked here because a
generator is the one place a bug produces data that looks perfectly fine.

**Do the three readings agree** - the `edge` graph that `libgraph` walks, the
SQL views in `schema.py`, and the `fact`/`residence`/`membership` tables.
Neighbours are the sharpest of these: the graph reaches them by following
`next_along` and `next_out`, and the view finds them by arithmetic on the
bearings. Two implementations of one idea, written to disagree if either is
wrong.

The corpus is built once per session at 600 people rather than 10,000. Same
seven generations, same shape, about a fifth of a second.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from random import Random

import pytest

pytest.importorskip("faker", reason="pip install -r data/silo/requirements.txt")

REPO = Path(__file__).resolve().parent.parent
PEOPLE = 600
SEED = 5


@pytest.fixture(scope="session")
def silo(tmp_path_factory, request):
    """Generate a small corpus, and hand back (module, module, connection)."""
    import sys

    for path in (str(REPO), str(REPO / "data" / "silo")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import generate
    import schema

    db_path = tmp_path_factory.mktemp("silo") / "silo.db"
    world = generate.populate(Random(SEED), SEED, PEOPLE)
    db = schema.connect(db_path, migrate=True)
    generate.write(db, world, SEED)
    db.row_factory = sqlite3.Row
    return generate, schema, db


@pytest.fixture(scope="session")
def planted(tmp_path_factory):
    """The same corpus with contradictions in it, and the key that says which.

    A separate fixture rather than a flag on the one above, because every other
    test in this file asserts the corpus is coherent and they all have to keep
    passing - planting is off by default for exactly that reason.
    """
    import sys

    for path in (str(REPO), str(REPO / "data" / "silo")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import generate
    import plant
    import schema

    rng = Random(SEED)
    world = generate.populate(rng, SEED, PEOPLE)
    anomalies = plant.plant(rng, world, 3)
    db_path = tmp_path_factory.mktemp("planted") / "silo.db"
    db = schema.connect(db_path, migrate=True)
    generate.write(db, world, SEED, planted=len(anomalies))
    db.row_factory = sqlite3.Row
    return anomalies, db


def test_a_count_label_names_the_end_the_subjects_come_from(silo):
    """`resolve` decides who a question is worth asking about, and a count runs
    backwards: "how many children does X have" counts `child_of` records
    pointing *at* X, so X has to be an object of that relation rather than a
    subject of it.

    `count_child_of` and `born_on count_born_on` now ship, so this pins the
    reading rather than reserving it. The walkability test below is what checks
    they reach the card; this checks they point the right way, which nothing
    downstream would notice - a count drawn from subjects rather than objects
    trains the classifier on people who have no children and teaches it the
    phrasing against an answer of nought.
    """
    import sys

    if str(REPO / "data" / "silo") not in sys.path:  # pragma: no cover
        sys.path.insert(0, str(REPO / "data" / "silo"))
    import relationpaths

    import libgraph

    have = {"child_of", "born_on"}
    assert relationpaths.resolve("count_child_of", have) == ("child_of", True)
    assert relationpaths.resolve("child_of", have) == ("child_of", False)
    assert relationpaths.resolve("count_born_on", have) == ("born_on", True)
    assert libgraph.COUNT == "count_"


def test_planting_is_off_unless_asked_for(silo):
    """Every measurement in `data/silo/README.md` was taken on a corpus with
    none of this in it."""
    _, _, db = silo
    assert db.execute("SELECT value FROM meta WHERE key = 'silo.planted'"
                      ).fetchone()[0] == "0"


def test_an_impossible_father_is_planted_exactly_once_each(planted):
    """The invariant `test_no_parent_died_before_their_child_was_born` asserts
    is the detector, and the count has to match the key exactly.

    It did not at first, twice. Moving a death earlier makes every child born
    after it impossible, so aiming at a random child planted three and created
    ten; and the purge moved deaths earlier too, which made four more that were
    indistinguishable from the planted ones.
    """
    anomalies, db = planted
    want = {a.subject for a in anomalies if a.kind == "impossible_father"}
    found = {r["name"] for r in db.execute(
        "SELECT c.name FROM edge e "
        "JOIN person c ON c.source = e.source AND c.name = e.subject "
        "JOIN person p ON p.source = e.source AND p.name = e.object "
        "WHERE e.relation = 'father_is' AND p.died IS NOT NULL "
        "AND p.died < c.born")}
    assert found == want


def test_an_altered_record_disagrees_with_the_graph_and_nothing_else_does(planted):
    """The `fact` table and the `edge` table are written from one pass, so they
    cannot disagree unless somebody made them - which is what a falsified
    record looks like from the inside."""
    anomalies, db = planted
    want = {a.subject for a in anomalies if a.kind == "altered_parentage"}
    found = {r["name"] for r in db.execute(
        "SELECT p.name FROM person p JOIN edge e "
        "ON e.source = 'silo' AND e.subject = p.name AND e.relation = 'father_is' "
        "WHERE p.father <> e.object")}
    assert found == want


def test_the_key_is_beside_the_database_and_not_in_it(planted, tmp_path):
    """Answers in the corpus would be answers a player could query for."""
    import plant

    anomalies, db = planted
    key = tmp_path / "silo.key.json"
    plant.write_key(key, anomalies, SEED)
    assert json.loads(key.read_text())["planted"] == len(anomalies)

    tables = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not any("anomal" in t or "planted" in t or "key" in t for t in tables)


def test_the_database_is_internally_consistent(silo):
    _, _, db = silo
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_every_address_matches_its_generated_column(silo):
    """The writer formats an address in Python; the table formats it in SQL.

    Two implementations exist because the writer needs the string before the
    row does. Nothing but this test says they agree, and an address that
    disagrees is invisible - both spellings look like addresses.
    """
    _, schema, db = silo
    rows = db.execute("SELECT floor, bearing, ring, address FROM apartment "
                      "WHERE source = ?", (schema.SOURCE,)).fetchall()
    assert rows
    wrong = [r["address"] for r in rows
             if schema.address(r["floor"], r["bearing"], r["ring"]) != r["address"]]
    assert wrong == []


def test_the_address_format_is_the_one_that_was_asked_for(silo):
    """`FLOOR TIME RING` - "42 600 A" is floor 42, six o'clock, inner ring."""
    _, schema, _ = silo
    assert schema.address(42, 360, "A") == "42 600 A"
    assert schema.address(1, 0, "C") == "1 1200 C"
    assert schema.address(144, 690, "B") == "144 1130 B"


def test_nobody_is_their_own_ancestor(silo):
    """A cycle in the pedigree would make the recursive view run forever."""
    _, _, db = silo
    assert db.execute("SELECT COUNT(*) FROM ancestor "
                      "WHERE person = ancestor").fetchone()[0] == 0


def test_no_parent_died_before_their_child_was_born(silo):
    _, _, db = silo
    assert db.execute(
        "SELECT COUNT(*) FROM edge e "
        "JOIN person c ON c.source = e.source AND c.name = e.subject "
        "JOIN person p ON p.source = e.source AND p.name = e.object "
        "WHERE e.relation = 'child_of' AND p.died IS NOT NULL "
        "AND p.died < c.born").fetchone()[0] == 0


def test_the_archive_records_nothing_from_the_future(silo):
    """Births and tenancies both stop at the present year.

    This one has bitten: seven generations at 22-34 years apart run past year
    200, and with the archive dated 175 the youngest generation was born in the
    future, alive, and living in flats they had not moved into - which made
    them neighbours of people they had never met.
    """
    generate, _, db = silo
    assert db.execute("SELECT COUNT(*) FROM person WHERE born >= ?",
                      (generate.NOW,)).fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM residence WHERE since > ?",
                      (generate.NOW,)).fetchone()[0] == 0


def test_the_graph_carries_the_present_and_the_table_carries_the_history(silo):
    """`lives_at` is for the living; `residence` is for everyone.

    A deliberate asymmetry rather than an oversight, so it is worth a test that
    fails if either half drifts: the graph has no notion of time, and "who
    lives next door" answered across two centuries of occupants is wrong rather
    than merely broad.
    """
    _, schema, db = silo
    living_with_edge = db.execute(
        "SELECT COUNT(*) FROM person p WHERE p.died IS NULL AND NOT EXISTS ("
        " SELECT 1 FROM edge e WHERE e.source = p.source AND e.subject = p.name"
        " AND e.relation = 'lives_at')").fetchone()[0]
    dead_with_edge = db.execute(
        "SELECT COUNT(*) FROM person p JOIN edge e ON e.source = p.source "
        "AND e.subject = p.name AND e.relation = 'lives_at' "
        "WHERE p.died IS NOT NULL").fetchone()[0]
    assert (living_with_edge, dead_with_edge) == (0, 0)
    assert db.execute("SELECT COUNT(*) FROM residence WHERE source = ?",
                      (schema.SOURCE,)).fetchone()[0] > 0


def test_the_ring_edges_and_the_bearing_arithmetic_find_the_same_neighbours(silo):
    """The graph walks `next_along`; the view subtracts bearings.

    The whole reason adjacency is stored as edges is that an eZ80 has no
    modulo, so the arithmetic happens once at build time. If the two ever
    disagree, the device is wrong and nothing else would say so.
    """
    _, schema, db = silo
    walked = {(r["subject"], r["object"]) for r in db.execute(
        "SELECT subject, object FROM edge WHERE source = ? "
        "AND relation IN ('next_along', 'next_out')", (schema.SOURCE,))}
    assert walked

    computed: set[tuple[str, str]] = set()
    dwellings = {(r["floor"], r["bearing"], r["ring"]) for r in db.execute(
        "SELECT floor, bearing, ring FROM apartment WHERE source = ?",
        (schema.SOURCE,))}
    for floor, bearing, ring in dwellings:
        along = (floor, (bearing + schema.BEARING_STEP) % schema.CLOCK, ring)
        if along in dwellings:
            computed.add((schema.address(floor, bearing, ring),
                          schema.address(*along)))
        index = schema.RINGS.index(ring)
        if index + 1 < len(schema.RINGS):
            out = (floor, bearing, schema.RINGS[index + 1])
            if out in dwellings:
                computed.add((schema.address(floor, bearing, ring),
                              schema.address(*out)))
    assert walked == computed


def test_the_neighbour_view_never_pairs_people_who_missed_each_other(silo):
    """Overlapping tenancies, or they are not neighbours.

    2,088 flats hold 10,000 people across the corpus's whole history, so a flat
    has several successive households and "same floor, next door along" without
    a date is a claim about the building.
    """
    _, _, db = silo
    assert db.execute(
        "SELECT COUNT(*) FROM neighbour n "
        "JOIN residence a ON a.person = n.person "
        "JOIN residence b ON b.person = n.neighbour "
        "WHERE a.since > coalesce(b.until, 1000000) "
        "   OR b.since > coalesce(a.until, 1000000)").fetchone()[0] == 0


def test_the_sibling_view_agrees_with_the_parent_edges(silo):
    """Half-siblings share one parent and full siblings share two.

    Checked against a second count taken straight off `child_of`, because the
    `shared_parents` column is what tells the two apart and an aggregate that
    is quietly wrong looks exactly like an aggregate that is right.
    """
    _, schema, db = silo
    parents: dict[str, set[str]] = {}
    for subject, obj in db.execute(
            "SELECT subject, object FROM edge WHERE source = ? "
            "AND relation = 'child_of'", (schema.SOURCE,)):
        parents.setdefault(subject, set()).add(obj)
    for row in db.execute("SELECT person, sibling, shared_parents FROM sibling "
                          "WHERE source = ? LIMIT 400", (schema.SOURCE,)):
        shared = parents[row["person"]] & parents[row["sibling"]]
        assert len(shared) == row["shared_parents"]
        assert 1 <= row["shared_parents"] <= 2


def test_a_class_list_and_a_class_edge_say_the_same_thing(silo):
    """`membership` and `class_is` are written from the same enrolment.

    They came apart once: a child who died before six was given a class edge,
    and the matching membership row failed its own `until >= joined` check and
    was dropped by an `INSERT OR IGNORE` without a word.
    """
    _, schema, db = silo
    assert db.execute(
        "SELECT COUNT(*) FROM edge e WHERE e.source = ? AND e.relation = 'class_is'"
        " AND NOT EXISTS (SELECT 1 FROM membership m WHERE m.source = e.source"
        " AND m.person = e.subject AND m.cohort = e.object)",
        (schema.SOURCE,)).fetchone()[0] == 0


def founders_reached(libgraph, schema, db) -> dict[int, set[bool]]:
    """Which generations reach their founder, at whatever limit is in force."""
    reached: dict[int, set[bool]] = {}
    for row in db.execute(
            "SELECT name, generation FROM person WHERE source = ? "
            "AND generation > 0 ORDER BY name", (schema.SOURCE,)):
        answer = libgraph.follow(db, schema.SOURCE, row["name"],
                                 ["founding_father"])
        reached.setdefault(row["generation"], set()).add(answer.complete)
    return reached


def test_every_generation_now_reaches_its_founder(silo):
    """Seven cohorts, and a limit of eight examines enough to answer them all.

    This used to stop at generation 5 and assert that 6 fell one short. The
    limit went to 8 because importing Wikidata's containment made *Wikipedia's*
    chains longer, and this corpus came along for the ride: generation g is
    exactly g hops from its founder, so eight covers every generation there is.
    """
    import libgraph

    _, schema, db = silo
    assert libgraph.CLIMB["founding_father"] == ("father_is", "founder")
    reached = founders_reached(libgraph, schema, db)
    assert reached, "no generations to check"
    for generation, complete in sorted(reached.items()):
        assert complete == {True}, generation
    # A limit of n examines n values and permits n - 1 hops, so this corpus no
    # longer reaches it. That is worth asserting rather than assuming: if the
    # generator ever grows an eighth cohort, this is what says so.
    assert max(reached) <= libgraph.CLIMB_LIMIT - 1


def test_the_hop_limit_is_still_what_stops_a_climb(silo, monkeypatch):
    """The limit is a real cost of running where a loop must terminate.

    It stopped being visible in this corpus when it was raised, so it is
    demonstrated at a limit rather than at *the* limit - otherwise the only
    test of it would be one that passes because nothing exercises it.
    """
    import libgraph

    _, schema, db = silo
    deepest = max(founders_reached(libgraph, schema, db))
    monkeypatch.setattr(libgraph, "CLIMB_LIMIT", deepest)
    reached = founders_reached(libgraph, schema, db)
    for generation in range(1, deepest):
        assert reached[generation] == {True}, generation
    assert reached[deepest] == {False}


def test_the_generator_stops_long_before_the_card_does(silo):
    """Issue #62 assumed the card was what limited this corpus. It is not.

    A search card scores 502,016 articles (`buildwikibin.max_docs`), and the
    generator refuses somewhere near 37,558 people - about 55,000 articles, a
    ninth of that. Two separate walls stand in front of the card, and neither
    of them is in the card:

    the calendar, at 37,559, because the seventh cohort is born in year 220
    and `NOW` is 220; and the dwellings, at around 57,500, because 144 levels
    of 24 bearings by 3 rings is 10,368 homes and no more.

    Pinned as an inequality rather than as the two numbers, which move with the
    seed. What must not change quietly is which side of the card they are on -
    if a generator change ever put them past it, `buildwikibin`'s assertion
    would start firing from `data/silo/buildcard.py` with no warning here.
    """
    generate, _schema, _db = silo
    import buildwikibin

    ceiling = buildwikibin.max_docs(
        buildwikibin.fixed_bytes(1, len(buildwikibin.build(1).code)))
    dwellings = generate.LEVELS * _schema.BEARINGS * len(_schema.RINGS)
    assert dwellings == 10_368
    assert dwellings < ceiling

    with pytest.raises(SystemExit, match="archive is dated"):
        generate.populate(Random(18), 18, 37_559)


def test_the_same_seed_builds_the_same_silo(silo):
    """Faker is seeded, so a corpus is reproducible from its number alone.

    Which is what lets the database stay out of git: a 25MB file that three
    seconds rebuilds exactly is not something to commit.
    """
    generate, _, _ = silo
    again = generate.populate(Random(SEED), SEED, PEOPLE)
    once = generate.populate(Random(SEED), SEED, PEOPLE)
    assert [p.name for p in again.people] == [p.name for p in once.people]
    assert [p.home for p in again.people] == [p.home for p in once.people]
    assert [p.crew for p in again.people] == [p.crew for p in once.people]


def test_every_phrase_the_classifier_knows_is_a_path_the_card_can_walk(silo):
    """The one card-build failure with no symptom.

    `buildwikigraph.paths_for` turns each of the classifier's labels into a
    list of steps, and a label it cannot read becomes an **empty** row in the
    card's path table rather than an error. The machine then classifies the
    question correctly and answers it with silence, which is indistinguishable
    from a corpus that has no answer.

    Three of these labels are climbs, and climbs live in `libgraph.CLIMB`,
    which `data/silo/generate.py` populates on import. A card built without
    that import is exactly the failure above - so this is really a test that
    importing the corpus is enough.
    """
    import sys

    if str(REPO / "data" / "silo") not in sys.path:  # pragma: no cover
        sys.path.insert(0, str(REPO / "data" / "silo"))
    import relationpaths

    import buildwikigraph
    from buildwikibin import PATH_STRIDE

    _, schema, db = silo
    have = sorted({r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (schema.SOURCE,))})
    kinds = sorted({k for (k,) in db.execute(
        "SELECT DISTINCT kind FROM entity_type WHERE source = ?",
        (schema.SOURCE,))})

    import libgraphcard

    labels = list(relationpaths.PATHS)
    steps = buildwikigraph.paths_for(labels, have, kinds)

    # `[]` is the failure - a label whose path this corpus has no edges for.
    # `None` is the refusal class, which is deliberate, and testing `not path`
    # cannot tell them apart.
    inert = [label for label, path in zip(labels, steps, strict=True)
             if path == []]
    assert inert == [], f"the card would answer these with silence: {inert}"

    refusals = [label for label, path in zip(labels, steps, strict=True)
                if path is None]
    assert refusals == [libgraphcard.REFUSE_PATH], refusals

    # One byte of length plus two per step, and the table is fixed-stride. A
    # refusal is one byte and no steps.
    assert max(1 + 2 * len(path) for path in steps
               if path is not None) <= PATH_STRIDE


def test_the_climbs_are_registered_by_importing_the_corpus(silo):
    import libgraph

    generate, _, _ = silo
    for name, step in generate.CLIMBS.items():
        assert libgraph.CLIMB[name] == step


def test_the_card_builder_takes_its_arguments_as_arguments(silo):
    """`buildcard.py` calls `buildwikisearch.main(argv)` rather than setting
    `sys.argv` around it, which only works because `main` accepts a list."""
    import inspect

    import buildwikisearch

    assert "argv" in inspect.signature(buildwikisearch.main).parameters


def test_the_corpus_tables_are_the_ones_ingest_defines(silo):
    """Shared, not copied.

    `libgraph`, `liboracle` and `buildwikisearch` read this corpus with no idea
    it is not Wikipedia, and that only holds while the definitions are the same
    text rather than two texts that agree today.
    """
    _, schema, _ = silo
    shared = schema.load_ingest()._schema()
    assert shared in schema.script()
    for table in ("article", "fact", "edge", "entity_type"):
        assert f"CREATE TABLE IF NOT EXISTS {table} " in shared


def test_the_answers_are_worth_more_than_a_guess(silo):
    """The point of the corpus, reduced to one assertion.

    A synthetic dataset can be made to report any number, so the harness prints
    a trivial baseline beside every result. This checks that the baselines are
    real competitors and the walk still beats them - "who is X's father" is
    guessable from the surname roughly half the time, because children take
    their father's name.
    """
    import sys

    if str(REPO / "data" / "silo") not in sys.path:  # pragma: no cover
        sys.path.insert(0, str(REPO / "data" / "silo"))
    import questions

    _, _, db = silo
    archive = questions.load(db)
    subjects = Random(0).sample(archive.people, 200)
    scored = {q.label: (q, s) for q, s in questions.run(db, archive, subjects, 0)}

    _, score = scored["who is X's father"]
    assert score.asked > 100
    assert score.walked == score.asked
    assert 0.25 < score.guessed / score.asked < 0.75, "the baseline is a straw man"

    _, department = scored["which department does X's father work in"]
    assert department.walked == department.asked
    assert department.guessed / department.asked < 0.6
