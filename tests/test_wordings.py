"""The wordings pipeline, run against the fake backend.

What is worth holding: a candidate without exactly one `{s}` never reaches
the file, a duplicate of a shipped wording never does either, and the block
is sorted with the most novel first. None of that needs a model, which is why
`fake_backend` exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import wordings

SHIPPED = {
    "father_is": ("who is {s}'s father", "name {s}'s father", "{s}'s dad"),
    "mother_is": ("who is {s}'s mother", "name {s}'s mother", "{s}'s mum"),
}


def test_parse_strips_numbering_bullets_and_quotes():
    text = "1. Who is {s}'s father\n- \"name {s}'s dad\"\n\n  • {s} dad?  "
    assert wordings.parse(text) == ["who is {s}'s father", "name {s}'s dad",
                                    "{s} dad?"]


def test_a_wording_needs_exactly_one_subject():
    seen: set[str] = set()
    assert wordings.keep("who fathered {s}", seen)
    assert not wordings.keep("who fathered nobody", seen)
    assert not wordings.keep("{s} and {s}", seen)


def test_a_shipped_wording_is_refused_and_so_is_a_repeat():
    found = wordings.generate(SHIPPED, wordings.fake_backend, per_persona=3,
                              personas=wordings.PERSONAS[:2])
    shipped = {w for ws in SHIPPED.values() for w in ws}
    for path, candidates in found.items():
        seen = [c.wording for c in candidates]
        assert len(seen) == len(set(seen))
        assert not shipped & set(seen)
        assert all(c.wording.count("{s}") == 1 for c in candidates)
        assert all(c.path == path for c in candidates)


def test_candidates_come_most_novel_first_and_name_their_twin():
    found = wordings.generate(SHIPPED, wordings.fake_backend, per_persona=3,
                              personas=wordings.PERSONAS[:3])
    for candidates in found.values():
        novelties = [c.novelty for c in candidates]
        assert novelties == sorted(novelties, reverse=True)
        assert all(0.0 <= n <= 1.0 for n in novelties)
        # The fake rewrites shipped wordings, so every twin is a shipped one.
        assert all(c.nearest.split(": ", 1)[1] in
                   {w for ws in SHIPPED.values() for w in ws}
                   for c in candidates)


def test_novelty_is_measured_against_the_whole_phrasebook():
    """A candidate for one path may be nearest to another path's wording,
    and the reviewer should be shown which."""
    only_father = {"father_is": SHIPPED["father_is"]}
    found = wordings.generate(only_father, wordings.fake_backend, per_persona=3,
                              personas=wordings.PERSONAS[:1], shipped=SHIPPED)
    paths_seen = {c.nearest.split(":", 1)[0] for c in found["father_is"]}
    assert paths_seen <= set(SHIPPED)


def test_the_review_file_keeps_the_placeholder(tmp_path: Path):
    found = wordings.generate(SHIPPED, wordings.fake_backend, per_persona=2,
                              personas=wordings.PERSONAS[:1])
    out = tmp_path / "review.txt"
    wordings.write_review(found, out)
    text = out.read_text()
    assert "# father_is" in text and "# mother_is" in text
    assert "{s}" in text
    assert wordings.SUBJECT not in text
