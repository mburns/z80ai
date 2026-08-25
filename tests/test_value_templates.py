"""Reading an infobox value that was written as a template.

`clean()` strips every `{{...}}` because a lead has to survive an infobox that
ran past the window we captured. Applied to a value the same rule is a
deletion, and `templates.py` measured what it cost: 232,947 values, 8.0% of
every named field, including 6,199 `spouse` values and 4,660
`subdivision_name` values - both fields `libgraph.CANONICAL` maps, so both
edges the graph never saw.

The risk in repairing that is inventing facts, so these pin the boundary as
hard as the behaviour: a template with no rule, or with arguments that do not
fit its rule, has to come out exactly as it did before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))
import ingest


def value(raw: str) -> str:
    """A raw infobox value, through the path `infobox_fields` puts it through."""
    return ingest.normalize_value(
        ingest.clean(ingest.unexpanded(ingest.expand_templates(raw))))


# --- dates: the largest single loss -------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("{{birth date|1847|3|3}}", "1847-03-03"),
    ("{{birth date and age|1847|3|3}}", "1847-03-03"),
    ("{{Birth_date|1847|3|3}}", "1847-03-03"),      # authors vary the spelling
    ("{{start date|1602|12|26}}", "1602-12-26"),
    ("{{end date|1990|1|1}}", "1990-01-01"),
])
def test_a_date_template_becomes_the_one_shape_value_kind_reads(raw, expected):
    """ISO, because `DATES[2]` is the only shape typed without a month name.

    Expanding to "3 March 1847" would read as well and type as a date only if
    the month name matched `MONTHS`; ISO needs no lookup and no guess.
    """
    assert value(raw) == expected
    assert ingest.value_kind(expected) == ("date", float(expected[:4]))


def test_death_date_and_age_leads_with_the_death():
    """`{{death date and age|1990|5|1|1920|3|2}}` is a death then a birth.

    Taking the last three would record everyone's death as their birthday.
    """
    assert value("{{death date and age|1990|5|1|1920|3|2}}") == "1990-05-01"


def test_a_year_on_its_own_stays_a_year():
    """`{{start date|1602}}` is a fact; padding it to January the first is not.

    Nobody wrote a month or a day, so inventing them would be inventing
    precision - the same objection `DATES` records against parsing "9th
    century".
    """
    assert value("{{start date|1602}}") == "1602"


def test_a_display_flag_is_not_a_date_part():
    """`df=y` says how to print the date, and is not one of its numbers."""
    assert value("{{birth date|1847|3|3|df=y}}") == "1847-03-03"


@pytest.mark.parametrize("raw", [
    "{{birth date|not|a|date}}",
    "{{birth date}}",
    "{{birth date|12}}",              # too short to be a year
    "{{birth date|1847|13|40}}",      # no such month or day
])
def test_a_date_that_does_not_fit_is_left_alone_rather_than_guessed(raw):
    """A rule that does not fit falls back to the old behaviour, not a guess.

    `1847|13|40` is the interesting one: it has three numbers and still is not
    a date, so the check has to be on the values and not on their count.
    """
    assert value(raw) in ("", "1847")


# --- the two that cost edges --------------------------------------------------


def test_a_marriage_names_a_spouse():
    """6,199 `spouse` values, and `spouse` is a `CANONICAL` field.

    Every one of these was a `spouse_of` edge the graph never got.
    """
    assert value("{{marriage|Jane Doe|1990}}") == "Jane Doe"
    assert "spouse" in ingest.libgraph.CANONICAL["spouse_of"]


def test_a_flag_standing_alone_names_a_country():
    """4,660 of these are `subdivision_name` - the field that made chaining
    work, taking `birth_place -> country` from 1.7% to 40.7%."""
    assert value("{{flag|France}}") == "France"
    assert value("{{flagcountry|Japan}}") == "Japan"
    assert "subdivision_name" in ingest.libgraph.CANONICAL["located_in"]


@pytest.mark.parametrize("raw,expected", [
    ("{{flagicon|IRI}} Urmia", "Urmia"),
    ("{{flag|Great Britain}} London", "London"),
    ("{{flag|Iran}} Nough, Iran", "Nough, Iran"),
])
def test_a_flag_beside_a_value_is_decoration(raw, expected):
    """The same template, the other way round, and it cost 155 edges.

    `birth_place = {{flagicon|IRI}} Urmia` puts an icon next to the place;
    reading it gives "IRI Urmia", which names no article, so a birthplace that
    resolved stops resolving. Standing alone the template is the value, beside
    something it is furniture, and position is the only thing that says which.
    """
    assert value(raw) == expected


def test_flagicon_is_never_read_even_alone():
    """Its argument is an IOC code, so it yields "IRI" rather than "Iran"."""
    assert value("{{flagicon|IRI}}") == ""


# --- wrappers, lists and measurements -----------------------------------------


def test_a_nowrap_contributes_its_contents():
    """It exists to keep a value on one line, so its contents are the value."""
    assert value("{{nowrap|Paris}}") == "Paris"


@pytest.mark.parametrize("wrapper", ["small", "big", "nobold", "noitalic"])
def test_an_annotation_wrapper_is_still_deleted(wrapper):
    """Reading these cost 686 edges, which is how they came to be left out.

    `| successor = Osman Hussein {{small|(Acting)}}` is the idiom. Expanding it
    gives "Osman Hussein (Acting)", which names no article, so a title that
    resolved before stops resolving - the value gains text and loses the edge.
    Deleting an annotation is what the old cleaner did well.
    """
    assert value(f"Osman Hussein {{{{{wrapper}|(Acting)}}}}") == "Osman Hussein"


def test_a_language_template_holds_the_text_second():
    """`{{lang|fr|Paris}}` is the code then the text; the code is not a fact."""
    assert value("{{lang|fr|Paris}}") == "Paris"


@pytest.mark.parametrize("raw", [
    "{{hlist|Actor|Singer}}",
    "{{ubl|Actor|Singer}}",
    "{{plainlist|\n* Actor\n* Singer\n}}",
    "{{flatlist|\n*Actor\n*Singer}}",
])
def test_a_list_reads_the_same_whichever_way_it_was_spelled(raw):
    """One item per argument, or a bullet list inside one argument. Both are
    common, and reading only the first would drop the other silently."""
    assert value(raw) == "Actor, Singer"


def test_a_measurement_keeps_its_unit():
    """`{{convert|5|km|mi}}` is five kilometres; the third argument is display."""
    assert value("{{convert|5|km|mi}}") == "5 km"


# --- nesting and the boundary -------------------------------------------------


def test_a_nested_template_is_expanded_innermost_first():
    """`{{nowrap|{{convert|5|km}}}}` needs the inner one read before the outer
    one can contribute anything at all."""
    assert value("{{nowrap|{{convert|5|km|mi}}}}") == "5 km"


def test_an_unmapped_template_is_dropped_exactly_as_before():
    """The boundary. Anything not understood keeps the old behaviour."""
    assert value("{{some template nobody mapped|1990}}") == ""


def test_an_unmapped_template_does_not_take_its_neighbours_with_it():
    """The old cleaner removed the braces and their contents; text outside them
    still survives, and expansion must not change that either."""
    assert value("Paris, {{unmapped|x}} France") == "Paris, France"


def test_a_wikidata_fetching_template_has_nothing_to_expand():
    """The single biggest loss in the corpus, and not repairable here.

    15,471 `{{france metadata wikidata}}` values hold no text - the template
    fetches a population at render time. There is nothing inside to read, which
    makes it an argument for ingesting Wikidata rather than a rule to write.
    """
    assert value("{{france metadata wikidata}}") == ""


def test_nesting_within_the_bound_is_read_all_the_way_down():
    depth = ingest.MAX_EXPANSIONS - 1
    assert value("{{nowrap|" * depth + "x" + "}}" * depth) == "x"


def test_nesting_past_the_bound_degrades_to_the_old_behaviour():
    """A value is not a page, so expansion is bounded rather than exhaustive.

    What matters is which way it fails: past the bound the leftover braces go
    back to `clean()` and the value is dropped, exactly as it would have been
    before any of this existed. Slower is not an option; wrong is not either.
    """
    depth = ingest.MAX_EXPANSIONS + 10
    assert value("{{nowrap|" * depth + "x" + "}}" * depth) == ""


def test_unbalanced_braces_terminate():
    assert value("{{" * 200) == ""
    assert value("}}" * 200) == ""


def test_braces_never_reach_the_value():
    """The reason the old behaviour was chosen: braces on screen are a bug."""
    for raw in ("{{birth date|1847|3|3}}", "{{unmapped|x}}",
                "{{nowrap|{{convert|5|km}}}}", "{{" * 10):
        assert "{{" not in value(raw) and "}}" not in value(raw)
