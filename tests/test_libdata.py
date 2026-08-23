"""Training-data handling: parsing, the validation split, and the data checks.

These are deliberately torch-free so they run in CI, where training is not
installed.  The split in particular is worth pinning: measuring accuracy on the
training set is what hid an 18-point generalization gap in the tinychat data,
and a split that leaks queries would hide it just as well.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import pytest

import libdata

# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("hello|HI", ("HELLO", "HI")),
        ("  spaced  |  out  ", ("SPACED", "OUT")),
        ("lower|case", ("LOWER", "CASE")),
        ("a|b", None),                      # query shorter than 2 characters
        ("query|", None),                   # empty response
        ("no pipe here", None),
        ("", None),
        ("   ", None),
        ("# a comment", None),
        ("take the lamp|TAKE|EXTRA", ("TAKE THE LAMP", "TAKE|EXTRA")),
    ],
)
def test_parse_pair(line, expected):
    assert libdata.parse_pair(line) == expected


def test_comments_are_skipped_even_when_they_contain_a_pipe():
    """The old parser only ignored comments that happened to have no pipe."""
    assert libdata.parse_pair("# vocabulary: YES|NO|MAYBE") is None


def test_long_queries_are_truncated_not_dropped():
    query = "WORD " * 20
    parsed = libdata.parse_pair(f"{query}|OK")
    assert parsed is not None
    assert len(parsed[0]) <= libdata.MAX_QUERY_LEN


def test_load_pairs_honours_a_limit():
    lines = [f"a{i}|X" for i in range(10)]
    assert len(libdata.load_pairs(lines)) == 10
    assert len(libdata.load_pairs(lines, 3)) == 3


# --- charset -----------------------------------------------------------------


def test_charset_comes_from_responses_only():
    """Queries are hashed into buckets, never spelled out, so they add nothing."""
    pairs = [("ZZZ QUERY", "OK"), ("ANOTHER", "NO")]
    assert libdata.build_charset(pairs) == "KNO"


def test_a_single_rare_character_still_enters_the_charset():
    """One line with a slash costs an output neuron for the whole model."""
    pairs = [("QUERY ONE", "OK")] * 100 + [("QUERY TWO", "K W/E")]
    assert "/" in libdata.build_charset(pairs)


# --- the split ---------------------------------------------------------------


def _pairs(n: int) -> list[tuple[str, str]]:
    return [(f"QUERY {i}", "YES" if i % 2 else "NO") for i in range(n)]


def test_split_holds_out_roughly_the_requested_fraction():
    train, val = libdata.split_pairs(_pairs(100), val_frac=0.2)
    assert len(val) == 20
    assert len(train) == 80


def test_split_never_leaks_a_query_across_the_boundary():
    """The whole point: a query in both halves makes validation meaningless."""
    pairs = _pairs(50) * 3  # every query appears three times
    train, val = libdata.split_pairs(pairs, val_frac=0.2)
    assert {q for q, _ in train}.isdisjoint({q for q, _ in val})
    assert len(train) + len(val) == len(pairs)


def test_split_is_deterministic_for_a_given_seed():
    pairs = _pairs(200)
    assert libdata.split_pairs(pairs, 0.1, 7) == libdata.split_pairs(pairs, 0.1, 7)


def test_split_seed_actually_changes_the_split():
    pairs = _pairs(200)
    assert libdata.split_pairs(pairs, 0.1, 0) != libdata.split_pairs(pairs, 0.1, 1)


def test_split_does_not_depend_on_input_order():
    """Otherwise a shuffled data file would silently change the held-out set."""
    pairs = _pairs(200)
    _, val_a = libdata.split_pairs(pairs, 0.1, 3)
    _, val_b = libdata.split_pairs(list(reversed(pairs)), 0.1, 3)
    assert sorted(val_a) == sorted(val_b)


@pytest.mark.parametrize("off", [0, -0.5])
def test_a_non_positive_fraction_disables_the_split(off):
    pairs = _pairs(10)
    train, val = libdata.split_pairs(pairs, val_frac=off)
    assert val == []
    assert train == pairs


@pytest.mark.parametrize("bad", [1.0, 1.5])
def test_fractions_of_one_or_more_are_rejected(bad):
    """Holding out everything would leave nothing to train on."""
    with pytest.raises(ValueError, match="val_frac"):
        libdata.split_pairs(_pairs(10), val_frac=bad)


# --- contradictions ----------------------------------------------------------


def test_accuracy_ceiling_is_one_when_labels_are_consistent():
    assert libdata.accuracy_ceiling([("A B", "YES"), ("C D", "NO")]) == 1.0


def test_accuracy_ceiling_reflects_a_contradiction():
    """Two of three pairs agree, so the best any model can do is 2/3."""
    pairs = [("A B", "YES"), ("A B", "YES"), ("A B", "NO")]
    assert libdata.accuracy_ceiling(pairs) == pytest.approx(2 / 3)


def test_accuracy_ceiling_of_an_empty_set_is_one():
    assert libdata.accuracy_ceiling([]) == 1.0


def test_contradictions_reports_the_conflicting_labels():
    pairs = [("A B", "YES"), ("A B", "NO"), ("C D", "OK")]
    assert libdata.contradictions(pairs) == {"A B": {"YES", "NO"}}


# --- scoring -----------------------------------------------------------------


def test_score_predictions_counts_both_ways():
    pairs = [("A B", "YES")] * 3 + [("C D", "NO")]
    overall, macro = libdata.score_predictions(pairs, lambda q: "YES")
    assert overall == pytest.approx(0.75)   # 3 of 4 pairs
    assert macro == pytest.approx(0.5)      # 1 of 2 answers


def test_macro_exposes_a_majority_class_guesser():
    """The whole reason macro is reported: overall flatters always-say-NO.

    On guess this is not hypothetical - 58% of the data is NO, so a constant
    guesser scores 58% overall, which is most of what the keyword baseline
    gets.  Macro puts it at 25%.
    """
    pairs = [("Q ONE", "NO")] * 58 + [("Q TWO", "YES")] * 21 + [("Q SIX", "MAYBE")] * 21
    overall, macro = libdata.score_predictions(pairs, lambda q: "NO")
    assert overall == pytest.approx(0.58)
    assert macro == pytest.approx(1 / 3)


def test_score_predictions_is_one_when_everything_is_right():
    pairs = [("A B", "YES"), ("C D", "NO")]
    assert libdata.score_predictions(pairs, dict(pairs).__getitem__) == (1.0, 1.0)


def test_score_predictions_of_an_empty_set():
    assert libdata.score_predictions([], lambda q: "X") == (1.0, 1.0)


def test_a_class_never_predicted_drags_macro_down_hard():
    """guess's keyword table never says WIN; macro is where that shows up."""
    pairs = [("Q A", "NO")] * 99 + [("Q B", "WIN")]
    overall, macro = libdata.score_predictions(pairs, lambda q: "NO")
    assert overall == pytest.approx(0.99)
    assert macro == pytest.approx(0.5)


# --- the linter --------------------------------------------------------------


@pytest.fixture(scope="session")
def lint(repo_root):
    sys.path.insert(0, os.path.join(repo_root, "data"))
    import lint as module

    return module


def _clean(n: int = 200) -> list[tuple[str, str]]:
    """A dataset with nothing wrong with it."""
    return [(f"QUERY NUMBER {i}", "YES" if i % 2 else "NO") for i in range(n)]


def test_clean_data_reports_no_problems(lint, capsys):
    assert lint.report(_clean()) == []
    capsys.readouterr()


def test_a_dominant_label_is_flagged(lint, capsys):
    pairs = _clean(100) + [(f"EXTRA QUERY {i}", "NO") for i in range(300)]
    problems = lint.report(pairs)
    capsys.readouterr()
    assert any("even split" in p for p in problems)


def test_an_even_two_class_split_is_not_flagged_as_dominant(lint, capsys):
    """With two responses an even split is 50%, which is not imbalance."""
    problems = lint.report(_clean(200))
    capsys.readouterr()
    assert not any("even split" in p for p in problems)


def test_too_many_distinct_responses_is_flagged(lint, capsys):
    pairs = [(f"QUERY {i}", f"R{i}") for i in range(60)]
    problems = lint.report(pairs)
    capsys.readouterr()
    assert any("distinct responses" in p for p in problems)


def test_contradictions_are_flagged_with_the_ceiling(lint, capsys):
    pairs = _clean(100) + [("AMBIGUOUS ONE", "YES"), ("AMBIGUOUS ONE", "NO")] * 10
    problems = lint.report(pairs)
    capsys.readouterr()
    assert any("caps accuracy" in p for p in problems)


def test_a_character_used_once_is_flagged(lint, capsys):
    """It costs 128 output weights whether it appears once or a thousand times."""
    problems = lint.report([*_clean(200), ("ONE ODD QUERY", "K W/E")])
    capsys.readouterr()
    assert any("128 output weights" in p for p in problems)


def test_colliding_queries_are_only_flagged_when_they_disagree(lint, capsys):
    """Two phrasings of the same command may collide harmlessly, and often do."""
    agree = [("PLEASE PAUSE", "WAIT"), ("PAUSE PLEASE", "WAIT")]
    assert not any("hash to the same" in p for p in lint.report(_clean() + agree))
    capsys.readouterr()

    disagree = [("PLEASE PAUSE", "WAIT"), ("PAUSE PLEASE", "HELP")]
    problems = lint.report(_clean() + disagree)
    capsys.readouterr()
    assert any("hash to the same" in p for p in problems)


# --- the vendored CLINC150 data ----------------------------------------------


@pytest.fixture(scope="session")
def subset(repo_root):
    sys.path.insert(0, os.path.join(repo_root, "data", "clinc150"))
    import subset as module

    return module


def test_smalltalk_data_matches_its_recipe(subset, examples_dir):
    """The shipped training file must be what subset.py produces.

    Vendoring third-party data is only worth anything if the derivation from it
    is reproducible; otherwise the checked-in file is just more invented data
    with a citation attached.
    """
    # Both sides through parse_pair: the file on disk holds full queries, and
    # the 60-character truncation happens when they are read back.
    generated = [
        libdata.parse_pair(f"{q}|{r}")
        for q, r in subset.build(subset.RECIPES["smalltalk"], seed=0)
    ]
    shipped = libdata.read_files(
        [os.path.join(examples_dir, "smalltalk", "training-data.txt.gz")]
    )
    assert generated == shipped, "training-data.txt.gz is stale; regenerate it"


def test_smalltalk_recipe_is_balanced_and_clean(subset):
    pairs = subset.build(subset.RECIPES["smalltalk"], seed=0)
    counts = Counter(reply for _, reply in pairs)

    assert len(counts) == len(subset.RECIPES["smalltalk"])
    assert len(set(counts.values())) == 1, "classes are not balanced"
    assert libdata.accuracy_ceiling(pairs) > 0.99
    assert max(len(r) for r in counts) <= 12
    assert len(libdata.build_charset(pairs)) < 30, "charset too wide"


def test_out_of_scope_is_capped_with_the_rest(subset):
    """CLINC ships 1,200 oos against 150 per intent; uncapped it would dominate."""
    pairs = subset.build(subset.RECIPES["smalltalk"], seed=0)
    counts = Counter(reply for _, reply in pairs)
    assert counts["IDK"] / len(pairs) < 0.1


def test_subset_is_deterministic(subset):
    a = subset.build(subset.RECIPES["smalltalk"], seed=0)
    assert a == subset.build(subset.RECIPES["smalltalk"], seed=0)
    assert a != subset.build(subset.RECIPES["smalltalk"], seed=1)


def test_an_unknown_intent_is_rejected(subset):
    with pytest.raises(SystemExit, match="no such intent"):
        subset.build({"not_a_real_intent": "NOPE"})


def test_the_shipped_datasets_parse(examples_dir):
    """Whatever else is wrong with them, they must still load."""
    for example in ("guess", "tinychat"):
        path = os.path.join(examples_dir, example, "training-data.txt.gz")
        if not os.path.exists(path):
            pytest.skip(f"{example} data not present")
        pairs = libdata.read_files([path])
        assert len(pairs) > 100
        assert all(len(q) >= 2 and r for q, r in pairs)
