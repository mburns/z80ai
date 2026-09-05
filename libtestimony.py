"""
Testimony from records: what a household says about a name, from the graph.

    > ask 600A about notice
    'Alexandra H. Anderson? My mother.' says a voice on the other side of
    the door, in the flat tone of somebody from Mechanical.

Ten thousand people are on the card and four have lines written for them.
This is how the other 9,996 answer: a door's household is a person on the
card, the name the player carries is another, and what the first says about
the second is *which path on the graph joins them*, if one does. A father
is one hop, a sibling is a hop up and a scan down, a crew-mate is a hop to
the crew and a scan of its members. Nothing is written per person; the
sentence is written per path, and the graph fills in who.

## Two hops, and why that is fair

A witness knows what a witness would know. Parents, children, a spouse, the
people under the same roof, the crew, the class, the committee - every one
of those is within two hops of anybody, and beyond two hops a household
says it does not know the name, which is the refuse class in a voice. A
player who wants to know more about somebody has to find somebody closer,
which is what an investigation is.

## The register

Who is speaking is a department. `REGISTERS` says how each one sounds, and
the device reads the speaker's `works_in` and picks the line - so the same
fact comes back in fourteen voices, which is what makes a corpus of ten
thousand feel populated rather than templated.

`PATHS` and `REGISTERS` are the whole of what an author edits. Both are
resolved against the card at build time, and a path over a relation the
card does not carry is left out rather than emitted inert.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Set on a step's relation to walk the reverse table: "everyone whose
#: `child_of` points at me" is the step that finds children and siblings.
INVERSE = 0x80


#: The most steps a path may take. Three is what "next door" needs - my
#: flat, the flat beside it, whoever lives there - and it is the row width
#: on the device.
MAX_STEPS = 3


@dataclass(frozen=True)
class Path:
    """One thing a household can be to a name, and what it says about it."""

    name: str
    #: (relation, inverse), up to `MAX_STEPS` of them. Every step but the
    #: last is a *hop* - the first edge, on the forward table or, inverted,
    #: the reverse one - and the last is a hop that must land on the name or,
    #: inverted, a scan of the run that must pass it. An inverse hop in the
    #: middle takes the first record and no other, which is exact where the
    #: reverse is unique (the flat beside mine) and a choice where it is not
    #: (my first child): write the path the way round that makes it unique.
    steps: tuple[tuple[str, bool], ...]
    #: What is said, after `'<name>? ` and before the closing quote.
    said: str


#: In the order they are tried, which is the order they are close: the first
#: path that joins the two is the answer, so a sister is a sister before she
#: is a classmate.
PATHS: tuple[Path, ...] = (
    Path("father", (("father_is", False),), "My father."),
    Path("mother", (("mother_is", False),), "My mother."),
    Path("spouse", (("spouse_of", False),), "We are married."),
    Path("child", (("child_of", True),), "One of mine."),
    Path("grandfather", (("father_is", False), ("father_is", False)),
         "My father's father."),
    Path("grandmother", (("mother_is", False), ("mother_is", False)),
         "My mother's mother."),
    Path("sibling", (("child_of", False), ("child_of", True)),
         "We had the same parents."),
    Path("parent-in-law", (("spouse_of", False), ("father_is", False)),
         "My wife's father, or my husband's. Family, either way."),
    Path("housemate", (("lives_at", False), ("lives_at", True)),
         "Under this roof."),
    # Next door, four ways: the flat beside mine clockwise and the one it is
    # clockwise of, the one outward and the one inward. `next_along` and
    # `next_out` are stored edges rather than arithmetic for exactly this -
    # the schema says the machine that walks them has no modulo - and each
    # inverse is unique, so the middle hop is exact.
    Path("neighbour", (("lives_at", False), ("next_along", False),
                       ("lives_at", True)), "Next door to us."),
    Path("neighbour", (("lives_at", False), ("next_along", True),
                       ("lives_at", True)), "Next door to us."),
    Path("neighbour", (("lives_at", False), ("next_out", False),
                       ("lives_at", True)), "Across the corridor from us."),
    Path("neighbour", (("lives_at", False), ("next_out", True),
                       ("lives_at", True)), "Across the corridor from us."),
    Path("crew", (("crew_is", False), ("crew_is", True)),
         "On my crew. I see them every shift."),
    Path("class", (("class_is", False), ("class_is", True)),
         "We were schooled together."),
    Path("committee", (("sits_on", False), ("sits_on", True)),
         "We sat on a committee together, once."),
    Path("department", (("works_in", False), ("works_in", True)),
         "Somebody from my department. That is all I could tell you."),
)

#: What a household says about itself when asked about itself.
SELF = "That is me you are asking about."

#: What a household says about a name no path reaches.
UNKNOWN = "Not a name I know."

#: How a department sounds, keyed by the title of its article. Anything not
#: here gets `DEFAULT_REGISTER`, so a department added to the corpus speaks
#: rather than staying silent.
REGISTERS: dict[str, str] = {
    "Mechanical": "says a voice on the other side of the door, in the flat "
                  "tone of somebody from Mechanical.",
    "IT": "says a voice through the door, carefully, the way IT says things.",
    "Judicial": "says a voice through the door, and it sounds like a citation.",
    "Supply": "says a voice through the door, as if reading it off a ledger.",
    "Sheriff's Office": "says a voice through the door that is used to asking "
                        "and not to being asked.",
    "Cafeteria": "says a voice through the door, over the sound of something "
                 "being stirred.",
    "Nursery": "says a voice through the door, quietly, as if somebody were "
               "asleep behind it.",
    "Farms": "says a voice through the door that smells faintly of soil.",
    "Water Treatment": "says a voice through the door, over a tap left running.",
    "Electrical": "says a voice through the door, and something behind it hums.",
}
DEFAULT_REGISTER = "says a voice on the other side of the door."


def resolve(relations: list[str]) -> list[tuple[Path, tuple[int, ...]]]:
    """The paths this card can walk, with relation ids in place of names.

    A path over a relation the card does not carry is left out. One longer
    than `MAX_STEPS` is refused, because that is the row on the device.
    """
    rid = {name: i for i, name in enumerate(relations)}
    out = []
    for path in PATHS:
        if not 1 <= len(path.steps) <= MAX_STEPS:
            raise ValueError(f"path {path.name!r} has {len(path.steps)} steps, "
                             f"and the device walks at most {MAX_STEPS}")
        if not all(name in rid for name, _ in path.steps):
            continue
        out.append((path, tuple(rid[name] | (INVERSE if inverse else 0)
                                for name, inverse in path.steps)))
    return out


def testify(edges: list[tuple[int, int, int]], relations: list[str],
            who: int, whom: int) -> str | None:
    """The reference: the name of the first path from `who` that reaches
    `whom`, `"self"` for the same document, or `None`.

    Written over the plain edge list the way the device walks the card, so
    that the two can disagree.
    """
    if who == whom:
        return "self"
    forward: dict[tuple[int, int], list[int]] = {}
    reverse: dict[tuple[int, int], list[int]] = {}
    for s, r, o in edges:
        forward.setdefault((s, r), []).append(o)
        reverse.setdefault((o, r), []).append(s)

    for path, steps in resolve(relations):
        here = who
        reached: list[int] = []
        for index, step in enumerate(steps):
            relation, inverse = step & ~INVERSE, bool(step & INVERSE)
            table = reverse if inverse else forward
            found = table.get((here, relation), [])
            if not found:
                break
            if index < len(steps) - 1:
                here = found[0]              # a hop takes the first edge
            elif inverse:
                reached = found              # a scan sees the whole run
            else:
                reached = found[:1]          # a hop lands on one
        if whom in reached:
            return path.name
    return None
