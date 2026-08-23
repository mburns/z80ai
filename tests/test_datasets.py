"""The CLINC150 recipes, and the lint thresholds that judge them.

The invariants worth testing here are the ones whose failure mode is a model
that trains fine and answers wrong: two intents sharing a reply (they silently
become one class), a domain list that has drifted from the reply table (--domain
quietly emits less than it says), and a rarity threshold that calls a perfectly
balanced dataset starved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import lint


def _load_subset(repo_root):
    """data/clinc150/subset.py is not importable by name - it is in a subdir."""
    path = Path(repo_root) / "data" / "clinc150" / "subset.py"
    spec = importlib.util.spec_from_file_location("clinc_subset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def subset(repo_root):
    return _load_subset(repo_root)


def test_every_intent_has_its_own_reply(subset):
    """Two intents sharing a reply merge into one class with no warning."""
    replies = list(subset.REPLIES.values())
    assert len(set(replies)) == len(replies)


def test_the_domains_cover_exactly_the_reply_table(subset):
    assert set(subset.DOMAIN_OF) | {"oos"} == set(subset.REPLIES)
    assert len(subset.DOMAINS) == 10
    assert all(len(intents) == 15 for intents in subset.DOMAINS.values())


def test_no_reply_is_long_enough_for_parse_pair_to_truncate(subset):
    """libdata.parse_pair truncates at a word boundary rather than erroring, so
    an over-long reply would become a *different* reply and split its class."""
    import libdata
    for intent, reply in subset.REPLIES.items():
        assert len(reply) <= libdata.MAX_RESPONSE_LEN, intent
        assert "|" not in reply, intent


def test_the_shipped_smalltalk_recipe_is_untouched(subset):
    """A shipped example depends on its recipe reproducing byte for byte."""
    assert len(subset.RECIPES["smalltalk"]) == 19
    assert subset.RECIPES["smalltalk"]["greeting"] == "HI"
    assert subset.RECIPES["smalltalk"]["oos"] == "IDK"


def test_the_router_recipe_maps_fifteen_intents_onto_each_domain(subset):
    recipe, oos_cap = subset.resolve("clinc-router", None)
    assert len(set(recipe.values())) == 11          # ten domains plus IDK
    assert recipe["greeting"] == "SMALL TALK"
    assert recipe["balance"] == "BANKING"
    assert recipe["oos"] == "IDK"
    # Capped per-intent alongside 150 others, the catch-all lands at 0.7% of the
    # data - under what lint accepts. Uncapped it is about 5%.
    assert oos_cap == float("inf")


def test_a_domain_ships_its_intents_plus_the_catch_all(subset):
    recipe, _ = subset.resolve(None, "banking")
    assert len(recipe) == 16
    assert "oos" in recipe
    assert recipe["balance"] == subset.REPLIES["balance"]


def test_a_balanced_dataset_is_not_reported_as_starved():
    """151 classes at 0.66% each is the best case, not the worst.

    A flat 1% floor flagged every class of a perfectly balanced 151-way set.
    The dominance check already judges against an even split; so does this now.
    """
    pairs = [(f"QUERY NUMBER {i} OF MANY", f"REPLY {i % 151}") for i in range(1510)]
    problems = lint.report(pairs, phrasebook=True)
    assert not [p for p in problems if "Too rare to learn" in p]


def test_a_genuinely_rare_class_is_still_reported():
    pairs = [(f"COMMON QUERY {i}", "USUAL") for i in range(1000)]
    pairs.append(("A SINGLE ODD ONE OUT", "RARE"))
    problems = lint.report(pairs, phrasebook=True)
    assert any("Too rare to learn" in p for p in problems)


def test_phrasebook_mode_drops_the_character_decoder_thresholds():
    """Reply text costs a phrasebook nothing, so length and count stop binding."""
    pairs = [(f"QUERY {i} WORDS HERE", f"A MUCH LONGER REPLY THAN TWELVE {i}")
             for i in range(600)]
    assert any("longer than" in p for p in lint.report(pairs))
    assert not any("longer than" in p for p in lint.report(pairs, phrasebook=True))


def test_contradictions_are_reported_either_way():
    """Two answers for one query caps accuracy no matter how it is decoded."""
    pairs = [("SAME QUERY EVERY TIME", "YES")] * 50
    pairs += [("SAME QUERY EVERY TIME", "NO")] * 50
    for phrasebook in (False, True):
        problems = lint.report(pairs, phrasebook=phrasebook)
        assert any("more than one response" in p for p in problems)
