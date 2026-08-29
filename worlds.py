"""
A few floors of Silo 18, as somewhere to stand.

Small on purpose. This exists so `buildif.py` has a world to emit and
`tests/test_if.py` has one to walk, not because six rooms is an Interactive
Fiction - #62's second scope is about whether the machine can hold one, and
what it costs per turn, and neither question needs three hundred rooms to
answer.

The geography agrees with `data/silo/generate.py`: the levels run from 1 at the
top to 144 at the bottom, `Up Top` is 1-20, `The Mids` 21-120 and `Down Deep`
121-144, and the departments sit where that file puts them.
"""

from __future__ import annotations

import libworld
from libworld import Room, Rule, Thing, World

#: Room indices, named so the exits below read as something other than digits.
LANDING, CAFETERIA, STAIR_MID, IT_OFFICE, STAIR_DEEP, GENERATOR = range(6)


def silo() -> World:
    """Six rooms down the stair, and four things to carry between them."""
    rooms = [
        Room("Level 1 Landing",
             "The top of the stair. A sealed hatch above you has not been "
             "opened in living memory, and the light through the screen is "
             "the colour of dust.",
             {"DOWN": CAFETERIA}),
        Room("The Cafeteria",
             "Long tables, and the great screen along the far wall showing "
             "the hills outside. Nobody sits at the tables nearest it.",
             {"UP": LANDING, "DOWN": STAIR_MID}),
        Room("The Mids Stair",
             "A landing halfway down, where the stair widens enough for two "
             "people to pass. Somebody has chalked a number on the rail and "
             "somebody else has half rubbed it out.",
             {"UP": CAFETERIA, "EAST": IT_OFFICE, "DOWN": STAIR_DEEP}),
        Room("IT, Level 34",
             "Racks of machines behind glass, and a bench where a screen "
             "sits with its back off. A standing order on the wall says a "
             "screen is fitted by two people and never by one.",
             {"WEST": STAIR_MID}),
        Room("Down Deep Stair",
             "The air is warmer here and the rail is worn smooth. The hum "
             "from below is the sort you stop hearing after a week.",
             {"UP": STAIR_MID, "DOWN": GENERATOR}),
        Room("Generator Three",
             "Level 140, and the machine that carries the base load. The "
             "outboard bearing runs hot in the summer months and has done "
             "since year 211.",
             {"UP": STAIR_DEEP}),
    ]

    things = [
        # Two of the four are references to something on the card and two are
        # not, which is the distinction `CONSULT` exists to make. A wrench is a
        # tool; a ledger is a piece of paper with a name on it.
        Thing("wrench", "A generator technician's wrench, worn to the shape "
                        "of a hand that is not yours.", GENERATOR),
        Thing("ledger", "A ration ledger from year 188. The back of it has "
                        "been drawn on.", CAFETERIA,
              subject="Supply"),
        Thing("screen", "A screen with its back off, half fitted. The "
                        "standing order is clear about who may open one.",
              IT_OFFICE, portable=False),
        Thing("badge", "A deputy's badge, tarnished. You have no idea whose.",
              STAIR_MID, subject="Sheriff's Office"),
    ]

    # The four shapes `data/silo/README.md` says a path cannot express, as far
    # as a flat condition list gets: conjunction, a bounded count, and a state
    # that persists. Ranking is not here because this cannot do it - see IF.md.
    messages = [
        "Your hands are full. Whatever else you find down here is going to "
        "have to wait, or something you already have is going down the stair "
        "without you.",
        "The badge and the wrench together look like a story you would rather "
        "not have to tell a deputy.",
        "You have been to the bottom and back, and the hum is different now "
        "you know what makes it.",
    ]

    rules = [
        # A count over a set: two things at once, which a path cannot ask.
        Rule(when=[(libworld.C_CARRYING, 2)],
             then=[(libworld.A_PRINT, 0, 0)]),
        # Conjunction: two particular things, not merely two things.
        Rule(when=[(libworld.C_HAVE, 3), (libworld.C_HAVE, 0)],
             then=[(libworld.A_PRINT, 1, 0), (libworld.A_SET, 0, 0)]),
        # A flag remembering somewhere you have been, tested somewhere else.
        Rule(when=[(libworld.C_AT, GENERATOR)],
             then=[(libworld.A_SET, 1, 0)]),
        Rule(when=[(libworld.C_AT, LANDING), (libworld.C_FLAG, 1)],
             then=[(libworld.A_PRINT, 2, 0)]),
    ]

    return World(rooms=rooms, things=things, start=LANDING,
                 rules=rules, messages=messages)
