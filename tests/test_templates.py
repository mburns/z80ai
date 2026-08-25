"""What the template scan counts, and why its `survives` has to be borrowed.

The scan exists to aim a fix, so its numbers decide which templates get an
expansion rule and which do not. Two things could make those numbers wrong in
the flattering direction:

    counting a value as lost that the ingest actually keeps
    missing a template because it was nested, or spelled differently

`survives` guards the first by running the ingest's own three tests rather than
a paraphrase of them - if `infobox_fields` changes its mind about what a value
must look like, the scan changes with it or these fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))
import ingest
import templates

# --- finding the templates ----------------------------------------------------


def test_a_nested_template_is_counted_too():
    """`{{convert}}` inside `{{nowrap}}` is two rules missing, not one."""
    assert templates.template_names("{{nowrap|{{convert|5|km}}}}") == [
        "nowrap", "convert"]


def test_spelling_is_folded_the_way_a_property_key_is():
    """Authors write `Birth_date`, `birth date` and `Birth Date` for one thing,
    and three rows for one template would hide how much it costs."""
    assert templates.template_names("{{Birth_date|1847}}") == ["birth date"]
    assert templates.template_names("{{ Birth Date |1847}}") == ["birth date"]


def test_a_value_with_no_template_names_none():
    assert templates.template_names("Edinburgh, Scotland") == []


def test_a_bare_brace_is_not_a_template():
    """A value may hold prose with a brace in it; that is not a rule to write."""
    assert templates.template_names("see {sic} the note") == []


# --- deciding what was lost ---------------------------------------------------


def test_a_templated_value_is_lost_because_cleaning_empties_it():
    """The finding the scan exists to size: this is a date, and it is dropped.

    `clean` strips every `{{...}}` so the lead survives an unclosed infobox;
    applied to a value the same rule deletes the fact.
    """
    assert ingest.clean("{{birth date|1847|3|3}}").strip() == ""
    assert not templates.survives("{{birth date|1847|3|3}}")


def test_a_template_around_plain_text_keeps_the_text():
    """Not every template is fatal, which is why the scan reports a rate.

    `{{nowrap|Paris, France}}` loses its wrapper and keeps its value, so
    counting every templated value as lost would overstate the problem.
    """
    assert templates.survives("Paris, {{nowrap|France}}")


def test_survives_agrees_with_the_ingest_on_junk():
    """A value the ingest throws out as junk was never a template's to lose.

    `yes` is dropped whether or not anyone wrapped it, so the scan has to apply
    the junk test as well as the empty one - otherwise every `{{yesno}}` reads
    as a fact the expansion table could win back, and none of them is.
    """
    assert ingest.JUNK_VALUE.match("yes")
    assert not templates.survives("yes")
    assert not templates.survives("n/a")


def test_survives_agrees_with_the_ingest_on_length():
    """The third of the three tests, and the one a long citation trips."""
    assert templates.survives("x" * ingest.MAX_VALUE_LEN)
    assert not templates.survives("x" * (ingest.MAX_VALUE_LEN + 1))


# --- the scan itself ----------------------------------------------------------


def test_a_field_is_counted_once_however_often_a_template_repeats(tmp_path):
    """`{{flag}}` three times in one value is one lost value, not three.

    Counting appearances rather than values would rank a template that repeats
    inside lists above one that costs a whole property, and the ranking is what
    the fix is aimed by.
    """
    dump = tmp_path / "d.xml"
    dump.write_text(
        "<page><title>A</title><ns>0</ns><text>"
        "{{Infobox person|spouse={{marriage|X|1990}}{{marriage|Y|2000}}"
        "{{marriage|Z|2010}}}}"
        "</text></page>\n", encoding="utf-8")

    tally = templates.scan(dump)
    assert tally.lost["marriage"] == 1
    assert tally.totals["templated_lost"] == 1


def test_kept_means_the_value_survived_not_that_the_template_did(tmp_path):
    """The reported loss rate is a lower bound, and this is why.

    `_strip_braced` removes a template's *contents* along with its braces, so
    `{{nowrap|Paris}}` standing alone is a lost value while
    `Paris, {{nowrap|France}}` is a kept one that quietly lost "France".
    Both count as one appearance of `nowrap`; only the first counts as lost.
    """
    dump = tmp_path / "d.xml"
    dump.write_text(
        "<page><title>A</title><ns>0</ns><text>"
        "{{Infobox person|birth_place=Paris, {{nowrap|France}}}}"
        "</text></page>\n", encoding="utf-8")

    tally = templates.scan(dump)
    assert tally.kept["nowrap"] == 1
    assert tally.lost["nowrap"] == 0
    assert tally.totals["templated_lost"] == 0


def test_a_template_standing_alone_takes_the_whole_value_with_it(tmp_path):
    """The other half of the pair above, and the common case in an infobox."""
    dump = tmp_path / "d.xml"
    dump.write_text(
        "<page><title>A</title><ns>0</ns><text>"
        "{{Infobox person|birth_place={{nowrap|Paris}}}}"
        "</text></page>\n", encoding="utf-8")

    tally = templates.scan(dump)
    assert tally.lost["nowrap"] == 1
    assert tally.totals["templated_lost"] == 1


def test_a_redirect_contributes_nothing(tmp_path):
    """It has no infobox, and counting it would dilute every rate reported."""
    dump = tmp_path / "d.xml"
    dump.write_text(
        '<page><title>A</title><ns>0</ns><redirect title="B" />'
        "<text>#REDIRECT [[B]]</text></page>\n", encoding="utf-8")

    assert templates.scan(dump).totals["infoboxes"] == 0
