"""
The same six floors, with people in them and something to work out.

`worlds.silo()` is a world you can walk. This is a world you can be *wrong*
about, which is a different thing and needs four mechanisms `silo()` has none
of: people who answer, topics that both a person and the archive know about,
a record of what has been asked, and an attention counter that makes asking
cost something.

## Fair play, and what enforces it

A fair-play mystery promises that everything needed to reach the answer can be
found before the answer is asked for. Prose cannot keep that promise and
neither can an author reading their own source - the state space is where the
promise lives, so it is checked there:

    python -c "import worlds_mystery, json; \\
               w = worlds_mystery.mystery(); w.check(); \\
               s = w.explore(); print(s.solve())"

`goal` is what winning means, `Search.solve` finds the shortest way to it, and
`Search.unseen` names any line, room or thing no reachable state can show. A
clue behind a door that never opens reads to a player as the author cheating,
and it is the failure this catches.

## The chain

Marnes will say Allison worked in IT. Knowing that, Walk will say she was
fitting a screen alone, which the standing order on his own wall forbids. That
is the deduction; the machine does not make it and does not need to. What the
machine does is serve the two clues in an order, refuse the second until the
first has landed, and check one accusation at the end - which is the whole of
what `data/silo/README.md` says an eZ80 can do, used as a librarian rather
than as a detective.

## What asking costs

The archive is logged. `ALLISON` is sealed, so consulting it returns the seal
rather than the record and puts three on the attention counter, and asking the
mayor about her while she is standing there puts on two more. At five the
deputy comes up the stair looking for whoever has been asking.

That counter is the reason the oracle is an antagonist rather than a hint
button. A question is slow - ~370,000 instructions against a move's ~3,400 -
and it is *noticed*, so the wall clock and the fiction agree about what it
costs to want to know something.

## And it acts

Two of the rules move the archive rather than the world. At five on the
counter the deputy comes up the stair and the pump report - which was never
about Allison - is sealed, because the Voice reacts to the asking and not
the subject. With both clues in hand the standing order is rewritten to
say one person may fit a screen, which the player can catch: the order on
Walk's wall still says two. `data/silo/plant.py` set the rule for the
corpus and it holds at the terminal - a record that is wrong in a fixed,
discoverable way is a clue, and one that is unreliable at random is noise.
"""

from __future__ import annotations

import libworld
from libworld import Line, Person, Room, Rule, Thing, Topic, World

#: Room indices, as in `worlds.py` and for the same reason.
LANDING, CAFETERIA, STAIR_MID, IT_OFFICE, STAIR_DEEP, GENERATOR = range(6)

#: Thing indices.
WRENCH, LEDGER, SCREEN, BADGE = range(4)

#: Person indices.
JAHNS, MARNES, WALK, KNOX = range(4)

#: Topic indices.
T_ALLISON, T_PUMP, T_SCREEN, T_BADGE, T_HATCH = range(5)

#: Flags. The first two are `worlds.silo()`'s and mean the same things; the
#: rest are what the conversation teaches, which is why they are set by lines
#: rather than by rules.
F_HANDS_FULL = 0
F_BEEN_DEEP = 1
F_ALLISON_WAS_IT = 2       # Marnes said where she worked
F_SCREEN_ALONE = 3         # Walk said she fitted one by herself
F_DEPUTY_CAME = 4          # attention reached five and something happened
F_WON = 5                  # the accusation named the right person
F_LOST = 6                 # it did not, and there was only the one


def mystery() -> World:
    """Six rooms, four people, five topics and one thing to work out."""
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
        # `subject` is `CONSULT`'s half of the join, and it meets `NOTICE` on
        # the way through: holding a piece of paper up to the screen reaches
        # the card by the same path a typed question does, so it is marked
        # asked and charged its attention without either feature knowing
        # about the other.
        Thing("wrench", "A generator technician's wrench, worn to the shape "
                        "of a hand that is not yours.", GENERATOR,
              subject="cistern pump"),
        Thing("ledger", "A ration ledger from year 188. The back of it has "
                        "been drawn on.", CAFETERIA),
        Thing("screen", "A screen with its back off, half fitted. The "
                        "standing order is clear about who may open one.",
              IT_OFFICE, portable=False),
        Thing("badge", "A deputy's badge, tarnished. You have no idea whose.",
              STAIR_MID, subject="cleaning record 218-04"),
    ]

    # A person's description is the sentence that puts them in the room, and
    # the default is what they say about anything nobody wrote a line for -
    # the refuse class, in a voice. A deflection in character is worth more
    # than a correct answer to a question the author never considered.
    people = [
        Person("jahns", "Mayor Jahns is here, not sitting down, with the "
                        "look of somebody who came up to see the screen and "
                        "has been standing a while.",
               LANDING,
               "Jahns considers that for longer than it deserves. 'That is "
               "not really mine to talk about.'"),
        Person("marnes", "Deputy Marnes leans on the end of a table with a "
                         "cup he is not drinking.",
               CAFETERIA,
               "'Could be,' says Marnes. 'I would not want to say so where "
               "it could be written down.'"),
        Person("walk", "Walk is under the bench with only her boots showing, "
                       "and does not come out.",
               IT_OFFICE,
               "'Ask me something I can answer with a part number,' says "
               "Walk, from under the bench."),
        Person("knox", "Knox is wiping down the outboard bearing housing "
                       "with a rag that has not been clean in a year.",
               GENERATOR,
               "'Down here we fix what is in front of us,' says Knox. 'You "
               "want somebody who reads.'"),
    ]

    # A topic is one thing whether it is asked of a person or of the card.
    # `titles` are resolved against the index by `libworld.resolve_topics`,
    # which is what the merged build calls; the standalone binary has no card
    # and simply never uses them.
    topics = [
        Topic("allison", ["ALLISON", "ALLIE"],
              titles=["Cleaning Record 218-04"], heat=3,
              censor="RECORD SEALED BY ORDER OF JUDICIAL. THIS ACCESS HAS "
                     "BEEN LOGGED."),
        Topic("pump", ["PUMP", "CISTERN"],
              titles=["Incident Report 214-11: Cistern Pump Failure"]),
        # The record the Voice rewrites once the player has put the two
        # clues together. The order on Walk's wall still says two; the
        # archive now says otherwise, and the player has read both.
        Topic("screenfit", ["FITTING", "ORDER"],
              titles=["Standing Order 11: Screen Fitting"],
              alter="Standing Order 11: Screen Fitting (as amended, year "
                    "218). A screen may be fitted by one person where a "
                    "second is not available. Judicial."),
        Topic("badgetopic", ["DEPUTY", "BADGE"]),
        Topic("hatch", ["HATCH", "OUTSIDE"], heat=1),
    ]

    # The chain, most specific first - which is the whole of the conditional
    # mechanism, and why `check` refuses an ungated line written above a
    # gated one for the same pair.
    lines = [
        Line(MARNES, T_ALLISON,
             "'She was IT,' says Marnes, and puts the cup down. 'Thirty-four. "
             "Everybody says the hills got into her, but she was IT before "
             "she was anything else.'",
             sets=F_ALLISON_WAS_IT),
        Line(WALK, T_ALLISON,
             "The boots go still. 'She fitted the landing screen the week "
             "before. On her own. You have read the order on that wall - I "
             "did not write it and I did not sign off on her breaking it.'",
             gate=F_ALLISON_WAS_IT, sets=F_SCREEN_ALONE),
        Line(WALK, T_ALLISON,
             "'Allison who,' says Walk, in the voice of somebody who knows "
             "exactly which Allison."),
        Line(JAHNS, T_ALLISON,
             "Jahns looks at the screen rather than at you. 'She went out. "
             "That is the whole of the record and it is enough.'"),
        Line(WALK, T_SCREEN,
             "'Two people,' says Walk. 'One holds the frame and one does the "
             "seals. One person cannot do both, which is why it says two.'"),
        Line(MARNES, T_BADGE,
             "'Mine is on my chest,' says Marnes. 'If you have found one that "
             "is not, I would like to know which stair it was on.'"),
        Line(KNOX, T_PUMP,
             "'One-forty-two went in the spring,' says Knox. 'Took the down "
             "deep cistern with it. There is a report, if you like reports.'"),
        Line(JAHNS, T_HATCH,
             "'Nobody opens it,' says Jahns. 'That is not a rule. It is the "
             "shape of the world.'"),
    ]

    messages = [
        # 0 and 1 are `worlds.silo()`'s, kept so the two worlds print the same
        # sentences for the same reasons.
        "Your hands are full. Whatever else you find down here is going to "
        "have to wait, or something you already have is going down the stair "
        "without you.",
        "The badge and the wrench together look like a story you would rather "
        "not have to tell a deputy.",
        # 2: the mayor notices the subject rather than the question.
        "Jahns does not answer that one. She looks at you for a moment "
        "longer than she needs to, and then back at the hills.",
        # 3: what attention buys.
        "There are boots on the stair below you, coming up at the pace of "
        "somebody who has been told a name. Marnes is not in the cafeteria "
        "any more.",
        # 4: the two clues together, which is the accusation the world checks.
        "A screen fitted by one person, on the landing, the week before she "
        "went out. The order on Walk's wall says two. Somebody signed that "
        "off, and it was not Walk.",
    ]

    rules = [
        Rule(when=[(libworld.C_CARRYING, 2)],
             then=[(libworld.A_PRINT, 0, 0)]),
        Rule(when=[(libworld.C_HAVE, BADGE), (libworld.C_HAVE, WRENCH)],
             then=[(libworld.A_PRINT, 1, 0), (libworld.A_SET, F_HANDS_FULL, 0)]),

        # Asking about her where the mayor can hear costs more than asking
        # elsewhere. `C_ASKED` and `C_WITH` together are the shape a graph
        # path cannot reach at any length: a question, and who was present.
        Rule(when=[(libworld.C_ASKED, T_ALLISON), (libworld.C_WITH, JAHNS)],
             then=[(libworld.A_PRINT, 2, 0), (libworld.A_HEAT, 2, 0)]),

        # What attention buys, once. The deputy leaves the cafeteria and comes
        # to the landing, which is `A_SEND` - a person moving is not a thing
        # moving and does not go through `WHERE`. And the archive closes a
        # record that was open: the pump report was never about Allison, and
        # sealing it is the Voice reacting to the *asking*, not the subject.
        Rule(when=[(libworld.C_HEAT, 5)],
             then=[(libworld.A_PRINT, 3, 0), (libworld.A_SEND, MARNES, LANDING),
                   (libworld.A_SET, F_DEPUTY_CAME, 0),
                   (libworld.A_SEAL, T_PUMP, 0)]),

        # Both clues in hand. The machine prints the reading; the player did
        # the deduction, which is the division of labour the hardware forces
        # and the genre happens to want. And the archive quietly rewrites
        # the standing order, which is the Voice doing the one thing a
        # reliable narrator cannot: the player has read the order on the
        # wall, so the amendment is a lie they can catch.
        Rule(when=[(libworld.C_FLAG, F_ALLISON_WAS_IT),
                   (libworld.C_FLAG, F_SCREEN_ALONE)],
             then=[(libworld.A_PRINT, 4, 0), (libworld.A_ALTER, T_SCREEN, 0)]),

        Rule(when=[(libworld.C_AT, GENERATOR)],
             then=[(libworld.A_SET, F_BEEN_DEEP, 0)]),
    ]

    return World(rooms=rooms, things=things, people=people, topics=topics,
                 lines=lines, start=LANDING, terminal=IT_OFFICE,
                 rules=rules, messages=messages,
                 # Somebody signed off a screen fitted by one person, and it
                 # was not Walk. The mayor has one accusation coming, and the
                 # player has one to make.
                 culprit="jahns", won=F_WON, lost=F_LOST,
                 win_text="Jahns looks at the hills for a long time. 'She "
                          "asked to do it alone,' she says. 'And I let her. "
                          "You can write that down.' Marnes already is.",
                 lose_text="Marnes looks at you the way he looked at the "
                           "badge. 'That is a serious thing to say about "
                           "somebody,' he says, 'and it is the only one you "
                           "get to say.' The case is closed with the wrong "
                           "name on it.",
                 # Winning is the accusation, made *after* both clues are in
                 # hand: an accusation on turn one is legal and wins the
                 # game, and a goal that did not name the clues would let
                 # `solve` report a walkthrough that never found them.
                 # `explore` is what proves this is reachable, and
                 # `test_mystery.py` is where it is asserted rather than hoped.
                 goal=[(libworld.C_FLAG, F_SCREEN_ALONE),
                       (libworld.C_HAVE, BADGE),
                       (libworld.C_FLAG, F_WON)])
