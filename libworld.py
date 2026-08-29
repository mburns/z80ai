"""
A world small enough to hold in SRAM, and the tables an eZ80 walks it with.

The oracle answers questions *about* a corpus on a card. This is the other
half of issue #62: somewhere you are, which changes when you say so. The two
are deliberately different machines - see `data/silo/README.md` - and the
difference starts here, with a world that is resident rather than read.

## Why resident

`buildwikibin`'s card costs about 4,600 bytes of I/O and 370,000 instructions
to answer one question, which is fine for a question and hopeless for a step.
An Interactive Fiction takes a turn every few seconds and most turns are
`NORTH`. A move has to be free, and the only way it is free is if nothing about
it touches the card.

So the world is tables in the image and one small mutable overlay in RAM:

    rooms      name, description, six exits          fixed stride, in the image
    things     name, description, starting place     fixed stride, in the image
    where[]    thing -> where it is now              one byte each, in RAM
    flags[]    one bit a proposition                 in RAM

Only `where` and `flags` change, which is what makes a saved game small: the
image is the same on every card, so a save is the overlay and nothing else.

## Sizes, which are not the problem

A room is 8 bytes of table plus its text, a thing 5 plus its text. Three
hundred rooms and two hundred things is 4,400 bytes of table - against 388 KB
of free SRAM the accumulator does not want. Prose dominates, and #67 already
measured that a card holds far more of it than anybody will write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

#: The directions a room can lead in, in table order. Six is what a silo needs
#: - four around, two along the stair - and each is one byte in a room's row,
#: so adding a seventh costs every room a byte whether it leads anywhere.
DIRECTIONS: tuple[str, ...] = ("NORTH", "SOUTH", "EAST", "WEST", "UP", "DOWN")

#: Short forms a player actually types. Mapped here rather than in the parser
#: so that a world with different geography can rename them.
ALIASES: dict[str, str] = {
    "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
    "U": "UP", "D": "DOWN",
}

#: The longest word the parser will hold, and therefore the longest a thing can
#: be named. Here rather than in `buildif` for the same reason `ALIASES` is: it
#: is a fact about the vocabulary a world may use, which an author has to obey
#: while writing one and `World.check` is what tells them so.
MAX_WORD_LEN = 12

#: The longest line the console reads, and therefore the longest a thing's
#: `subject` may be: consulting one copies it into the same `INPBUF` a player
#: types into, so anything longer would be a question truncated mid-word.
MAX_INPUT_LEN = 60

#: No exit that way. 255 rooms is the limit a one-byte room id implies, and a
#: world that wants more wants a different overlay rather than a wider table.
NOWHERE = 0xFF

#: Where a thing is when the player has it, rather than in a room.
CARRIED = 0xFE

#: No gate on a line of dialogue, and no flag set by speaking it. `NOWHERE`
#: bounds rooms and this bounds the two optional fields of a line, which are
#: the only bytes in any table that are allowed to be absent.
NONE = 0xFF


@dataclass
class Room:
    name: str
    description: str
    #: direction -> room index. Missing means no exit that way.
    exits: dict[str, int] = field(default_factory=dict)


@dataclass
class Thing:
    name: str
    description: str
    #: Room index it starts in, or CARRIED.
    at: int
    #: Whether the player can pick it up. A door is scenery; a key is not.
    portable: bool = True
    #: What this thing is a reference to, in the archive's own words - the
    #: line `CONSULT` types at the terminal on the player's behalf.
    #:
    #: This is the whole of the join between the two programs. The card holds
    #: ten thousand people and the world can carry none of them; what it can
    #: carry is a *name*, on a ledger or a work order or a death notice, and a
    #: thing with a subject is that piece of paper. Without one, `CONSULT`
    #: says the screen has nothing to say about it - which is also the answer
    #: for a wrench.
    subject: str | None = None


@dataclass
class Topic:
    """Something that can be asked about, of the archive or of a person.

    One table for both, which is the point rather than a saving. A player who
    reads the incident report about the cistern pump and a player who asks the
    deputy about it have learned the same thing, and a world that recorded
    those separately would need every rule written twice.

    `docs` are article ids on the card. They are ids rather than titles because
    `buildwikibin` is handed a document count and never sees the index - see
    `resolve_topics`, which is where a title becomes a number.
    """

    #: What the author calls it. Never printed; it names the index in errors.
    name: str
    #: What a player may type. Uppercased into the topic word table.
    words: list[str]
    #: Article titles on the card that are about this. `resolve_topics`
    #: turns them into `docs`, and refuses one the card does not hold.
    titles: list[str] = field(default_factory=list)
    #: Article ids on the card that are about this, for the merged build.
    docs: list[int] = field(default_factory=list)
    #: What asking costs in attention. Most topics cost nothing.
    heat: int = 0
    #: What the archive says instead of the article, for a record that has
    #: been sealed. `None` prints the article. A censored topic still marks
    #: itself asked and still costs its heat: the sealing is what was learned.
    censor: str | None = None


@dataclass
class Person:
    """Somebody standing in a room, who can be asked about a topic.

    Not a `Thing` with `portable=False`. A thing that cannot be carried is
    scenery and is listed as "You can see screen."; a person is listed by
    standing there, is asked rather than taken, and moves under `A_SEND`
    rather than `A_MOVE`. Sharing the table would have saved eight bytes and
    cost every message that mentions one.
    """

    name: str
    description: str
    #: Room index they start in.
    at: int
    #: What they say about a topic no line covers. The refuse class again:
    #: `IF.md` sets out why a confident answer to an unwritten question is
    #: worse than a deflection, and a person has the same failure a parser
    #: does. Every person needs one, so it is not optional.
    default: str


@dataclass
class Line:
    """One thing one person says about one topic.

    Rows are scanned in order and the first whose gate is satisfied wins, so
    the author writes the most specific line first. That ordering is the whole
    of the conditional-dialogue mechanism: there is no condition list here,
    because a line that needed one is a rule.
    """

    person: int
    topic: int
    text: str
    #: A flag that must be set for this line to be the one spoken, or `NONE`.
    gate: int = NONE
    #: A flag speaking it sets, or `NONE`. This is how a conversation teaches
    #: the world something - `C_ASKED` records that a topic came up, and this
    #: records that a particular person answered it.
    sets: int = NONE


#: Condition opcodes. Every condition in a rule must hold, which is the point:
#: `data/silo/README.md` sets out that a graph path composes and then stops at
#: conjunction, and a list of conditions ANDed together is the smallest thing
#: that does not.
C_AT = 0        # the player is in room `arg`
C_HAVE = 1      # the player is carrying thing `arg`
C_HERE = 2      # thing `arg` is in the room the player is in
C_FLAG = 3      # flag `arg` is set
C_NFLAG = 4     # flag `arg` is clear
C_CARRYING = 5  # the player is carrying at least `arg` things
C_ASKED = 6     # topic `arg` has been asked about, of the archive or a person
C_HEAT = 7      # attention stands at `arg` or above
C_WITH = 8      # person `arg` is in the room the player is in

#: Action opcodes. `A_MOVE` and `A_SEND` take two bytes because each names
#: something and a destination; everything else takes one.
A_SET = 0       # set flag `arg`
A_CLEAR = 1     # clear flag `arg`
A_PRINT = 2     # print message `arg`
A_GOTO = 3      # move the player to room `arg`
A_MOVE = 4      # move thing `arg` to `arg2`
A_HEAT = 5      # add `arg` to attention, saturating at 255
A_COOL = 6      # take `arg` off attention, floored at 0
A_SEND = 7      # move person `arg` to room `arg2`

CONDITION_NAMES = {C_AT: "AT", C_HAVE: "HAVE", C_HERE: "HERE", C_FLAG: "FLAG",
                   C_NFLAG: "NFLAG", C_CARRYING: "CARRYING", C_ASKED: "ASKED",
                   C_HEAT: "HEAT", C_WITH: "WITH"}
ACTION_NAMES = {A_SET: "SET", A_CLEAR: "CLEAR", A_PRINT: "PRINT",
                A_GOTO: "GOTO", A_MOVE: "MOVE", A_HEAT: "HEAT",
                A_COOL: "COOL", A_SEND: "SEND"}


@dataclass
class Rule:
    """`when all of these, do all of those`, checked after every turn.

    Flat on purpose. There is no `or`, no nesting and no arithmetic beyond a
    count, because the question this answers is what the *smallest* step past
    a path buys - and the answer turns out to be three of the four things a
    path cannot express rather than all four. See `IF.md`.
    """

    #: (opcode, argument) pairs, all of which must hold.
    when: list[tuple[int, int]]
    #: (opcode, argument, argument) triples. The second argument is 0 unless
    #: the opcode is `A_MOVE`.
    then: list[tuple[int, int, int]]
    #: Fire at most once. Most rules are events rather than standing facts,
    #: and a rule that printed its message every turn would be a bug that
    #: looks like a design decision.
    once: bool = True


@dataclass(frozen=True)
class Reach:
    """What `World.reach` found, which is a ceiling rather than a description.

    Every field is a superset of what a player can really bring about - see
    `World.reach` for the three ways it errs upward and why that direction is
    the useful one.
    """

    #: Rooms the player can stand in.
    rooms: frozenset[int]
    #: Things that can be in the player's hands.
    held: frozenset[int]
    #: Things that can be in a room the player can stand in.
    present: frozenset[int]
    #: Flags that can be set.
    flags: frozenset[int]
    #: Rules that can fire. Anything outside this is dead code with a story
    #: attached, which is the bug this whole analysis exists to find.
    rules: frozenset[int]


@dataclass
class World:
    rooms: list[Room]
    things: list[Thing]
    #: Room the player starts in.
    start: int = 0
    #: Which room the archive terminal stands in, or None for a world with no
    #: card behind it. `buildwikibin` is what puts one there; the standalone
    #: `buildif` binary has no card to consult and says so.
    terminal: int | None = None
    #: Rules, checked in order after every turn that did something.
    rules: list[Rule] = field(default_factory=list)
    #: Messages rules print, by index.
    messages: list[str] = field(default_factory=list)
    #: How many one-bit propositions the world reserves. Costs one byte per
    #: eight and nothing else, so the number is a guess that can be generous.
    flags: int = 64
    #: What can be asked about, of the archive or of a person.
    topics: list[Topic] = field(default_factory=list)
    #: Who is standing about to be asked.
    people: list[Person] = field(default_factory=list)
    #: What they say, most specific first.
    lines: list[Line] = field(default_factory=list)
    #: What the world is won by: a condition list in the same shape as
    #: `Rule.when`. `solve` is what makes it more than documentation - a goal
    #: no reachable state satisfies is a game that cannot be finished, and
    #: that is a build-time question rather than a playtesting one.
    goal: list[tuple[int, int]] = field(default_factory=list)

    def check(self) -> None:
        """Refuse a world that cannot be walked, before anything is emitted.

        Every one of these is a mistake that produces a playable-looking game
        with a room nobody can leave or a thing nobody can find, which is the
        kind of bug an author discovers ten minutes in rather than at build
        time.
        """
        self._check_shape()
        self._check_rules()
        self._check_impossible()

    def _check_shape(self) -> None:
        if not self.rooms:
            raise ValueError("a world needs at least one room")
        if len(self.rooms) >= NOWHERE:
            raise ValueError(
                f"{len(self.rooms)} rooms, and a room id is one byte with "
                f"{NOWHERE:#x} reserved for 'no exit'")
        if not 0 <= self.start < len(self.rooms):
            raise ValueError(f"the game starts in room {self.start}, which "
                             f"is not one of {len(self.rooms)}")

        for index, room in enumerate(self.rooms):
            for direction, target in room.exits.items():
                if direction not in DIRECTIONS:
                    raise ValueError(
                        f"{room.name!r} leads {direction!r}, which is not one "
                        f"of {', '.join(DIRECTIONS)}")
                if not 0 <= target < len(self.rooms):
                    raise ValueError(
                        f"{room.name!r} leads {direction} to room {target}, "
                        f"which does not exist")
                if target == index:
                    raise ValueError(f"{room.name!r} leads {direction} to "
                                     f"itself")

        for thing in self.things:
            if thing.at != CARRIED and not 0 <= thing.at < len(self.rooms):
                raise ValueError(f"{thing.name!r} starts in room {thing.at}, "
                                 f"which does not exist")

        self._check_names()

    def _check_names(self) -> None:
        """A thing's name against what the parser can actually match.

        `SPLIT` takes two words and no more and stops copying either at
        `MAX_WORD_LEN`, so a thing whose name is two words or thirteen
        characters is one the player cannot name. Nothing said so until this
        existed, and the failure is a quiet one twice over: the build succeeds,
        and `DESCRIBE` then lists the thing in the room every turn - so the
        player is told they can see something, types its name, and is told the
        game does not know the word.

            > take identification
            I do not know the word 'IDENTIFICATI'.

        The truncation in that reply is `SP_ONE` stopping at twelve, which is
        the only evidence on the screen that the world was built wrong rather
        than the player spelling it wrong.
        """
        names = [t.name.upper() for t in self.things]
        if len(set(names)) != len(names):
            raise ValueError("two things share a name, and the parser resolves "
                             "a noun to exactly one")

        for thing in self.things:
            if len(thing.name.split()) != 1:
                raise ValueError(
                    f"{thing.name!r} is not one word, and a command is a verb "
                    f"and at most one noun")
            if len(thing.name) > MAX_WORD_LEN:
                raise ValueError(
                    f"{thing.name!r} is {len(thing.name)} characters and the "
                    f"parser holds {MAX_WORD_LEN}, so it would be matched "
                    f"against {thing.name.upper()[:MAX_WORD_LEN]!r} and never "
                    f"found")
            if thing.subject is not None and not thing.subject.strip():
                raise ValueError(
                    f"{thing.name!r} has an empty subject, which the terminal "
                    f"cannot tell from having none - leave it None")
            if thing.subject is not None and len(thing.subject) > MAX_INPUT_LEN:
                raise ValueError(
                    f"{thing.name!r} names a subject of {len(thing.subject)} "
                    f"characters and the console reads {MAX_INPUT_LEN}, so it "
                    f"would reach the archive cut off mid-word")
            # Deliberately *not* refused when `terminal is None`. One `World`
            # is built twice - once standalone, once carried by the oracle -
            # and only the second has a card, so a subject unreadable in the
            # first is not a mistake. The standalone binary says there is no
            # terminal here, which is true of every room in it.

        self._check_people(set(names))
        self._check_topics()
        self._check_lines()
        self._check_rules()

        for op, arg in self.goal:
            if op not in CONDITION_NAMES:
                raise ValueError(f"the goal has no condition {op}")
            self._check_arg(-1, CONDITION_NAMES[op], op, arg)

    def _check_people(self, thing_names: set[str]) -> None:
        """Rooms that exist, names that are one person's and nobody else's."""
        if len(self.people) >= NOWHERE:
            raise ValueError(f"{len(self.people)} people, and a person id is "
                             f"one byte with {NONE:#x} reserved for 'none'")
        seen: set[str] = set()
        for person in self.people:
            if not 0 <= person.at < len(self.rooms):
                raise ValueError(f"{person.name!r} stands in room {person.at},"
                                 f" which does not exist")
            name = person.name.upper()
            if name in seen:
                raise ValueError(f"two people are called {person.name!r}, and "
                                 f"ASK resolves a name to exactly one")
            # A word that is both is not ambiguous to the machine - TAKE reads
            # the noun table and ASK reads this one - but it is ambiguous to
            # the player, who has no way to know which table they are in.
            if name in thing_names:
                raise ValueError(f"{person.name!r} is both a person and a "
                                 f"thing, and a player cannot tell which")
            if len(person.name.split()) != 1 or len(person.name) > MAX_WORD_LEN:
                raise ValueError(
                    f"{person.name!r} is not one word of at most "
                    f"{MAX_WORD_LEN} characters, so `ASK` can never resolve "
                    f"it - the same limit a thing's name has")
            seen.add(name)
            if not person.default:
                raise ValueError(f"{person.name!r} has no default line, so a "
                                 f"topic nobody wrote gets silence")

    def _check_topics(self) -> None:
        """One word, one topic. The word table resolves it to exactly one."""
        if len(self.topics) >= NOWHERE:
            raise ValueError(f"{len(self.topics)} topics, and a topic id is "
                             f"one byte with {NONE:#x} reserved")
        words: dict[str, str] = {}
        for topic in self.topics:
            if not topic.words:
                raise ValueError(f"topic {topic.name!r} has no words, so "
                                 f"nothing a player types can reach it")
            if not 0 <= topic.heat <= 255:
                raise ValueError(f"topic {topic.name!r} costs {topic.heat} "
                                 f"attention, and that is one byte")
            for word in topic.words:
                upper = word.upper()
                if len(word.split()) != 1 or len(word) > MAX_WORD_LEN:
                    raise ValueError(
                        f"topic {topic.name!r} can be reached by {word!r}, "
                        f"which is not one word of at most {MAX_WORD_LEN} "
                        f"characters and so can never be typed")
                if upper in words:
                    raise ValueError(
                        f"{word!r} names both {words[upper]!r} and "
                        f"{topic.name!r}, and a word resolves to one topic")
                words[upper] = topic.name

    def _check_lines(self) -> None:
        """Every index, and every line that can never be the one spoken."""
        seen: set[tuple[int, int, int]] = set()
        for number, line in enumerate(self.lines):
            if not 0 <= line.person < len(self.people):
                raise ValueError(f"line {number} is spoken by person "
                                 f"{line.person}, who does not exist")
            if not 0 <= line.topic < len(self.topics):
                raise ValueError(f"line {number} is about topic {line.topic}, "
                                 f"which does not exist")
            for name, flag in (("gate", line.gate), ("sets", line.sets)):
                if flag != NONE and not 0 <= flag < self.flags:
                    raise ValueError(f"line {number} {name}s flag {flag}, and "
                                     f"the world reserves {self.flags}")
            key = (line.person, line.topic, line.gate)
            if key in seen:
                # The scan takes the first match, so the second is dead text.
                raise ValueError(
                    f"line {number} repeats person {line.person} on topic "
                    f"{line.topic} behind the same gate, and the scan takes "
                    f"the first - the second can never be spoken")
            seen.add(key)
            if line.gate == NONE:
                continue
            # An ungated line before a gated one shadows it for the same
            # reason, and is the mistake that actually gets made.
            if (line.person, line.topic, NONE) in seen:
                raise ValueError(
                    f"line {number} is gated behind flag {line.gate} but an "
                    f"ungated line for person {line.person} on topic "
                    f"{line.topic} comes first and always wins")

    def _check_rules(self) -> None:
        """Every argument, against what it indexes.

        A rule with a bad argument is the worst kind of bug this can have: it
        fires on the wrong thing, or never, and the game merely feels wrong.
        """
        for number, rule in enumerate(self.rules):
            if not rule.when:
                raise ValueError(f"rule {number} has no conditions, so it "
                                 f"fires on the first turn and every turn")
            for op, arg in rule.when:
                if op not in CONDITION_NAMES:
                    raise ValueError(f"rule {number}: no condition {op}")
                self._check_arg(number, CONDITION_NAMES[op], op, arg)
            for op, arg, arg2 in rule.then:
                if op not in ACTION_NAMES:
                    raise ValueError(f"rule {number}: no action {op}")
                self._check_arg(number, ACTION_NAMES[op], op, arg, arg2)

    def _check_impossible(self) -> None:
        """Rules that can never fire, exactly first and then approximately.

        The two passes are different kinds of claim and are kept apart for
        that reason. A contradiction is arithmetic - `AT 3` and `AT 5` are one
        byte compared against two values and the comparison cannot pass twice
        - and is worth naming precisely. `dead_rules` is a search over an
        over-approximation, so it says less and says it about more.
        """
        portable = sum(1 for t in self.things if t.portable)
        for number, rule in enumerate(self.rules):
            rooms = {arg for op, arg in rule.when if op == C_AT}
            if len(rooms) > 1:
                raise ValueError(
                    f"rule {number} needs the player in rooms "
                    f"{sorted(rooms)} at once")
            set_flags = {arg for op, arg in rule.when if op == C_FLAG}
            clear_flags = {arg for op, arg in rule.when if op == C_NFLAG}
            if both := set_flags & clear_flags:
                raise ValueError(f"rule {number} needs flag {min(both)} both "
                                 f"set and clear")
            carried = {arg for op, arg in rule.when if op == C_HAVE}
            here = {arg for op, arg in rule.when if op == C_HERE}
            if both := carried & here:
                thing = self.things[min(both)]
                raise ValueError(
                    f"rule {number} needs {thing.name!r} carried and in the "
                    f"room, and a thing is in one place")
            for op, arg in rule.when:
                if op == C_CARRYING and arg > portable:
                    raise ValueError(
                        f"rule {number}: CARRYING {arg}, and only {portable} "
                        f"of {len(self.things)} things can be picked up")

        for number, why in self.dead_rules():
            raise ValueError(f"rule {number} can never fire: {why}")

    def _check_arg(self, number: int, name: str, op: int, arg: int,
                   arg2: int = 0) -> None:
        rooms, things = len(self.rooms), len(self.things)
        where = "the goal" if number < 0 else f"rule {number}"
        if (name in ("AT", "GOTO") and not 0 <= arg < rooms):
            raise ValueError(f"{where}: {name} {arg}, and there are "
                             f"{rooms} rooms")
        if name in ("HAVE", "HERE", "MOVE") and not 0 <= arg < things:
            raise ValueError(f"{where}: {name} {arg}, and there are "
                             f"{things} things")
        if name == "MOVE" and not (0 <= arg2 < rooms or arg2 == CARRIED):
            raise ValueError(f"{where}: MOVE to {arg2}, which is neither "
                             f"a room nor CARRIED")
        if name in ("FLAG", "NFLAG", "SET", "CLEAR") and not 0 <= arg < self.flags:
            raise ValueError(f"{where}: {name} {arg}, and the world "
                             f"reserves {self.flags} flags")
        if name == "PRINT" and not 0 <= arg < len(self.messages):
            raise ValueError(f"{where}: PRINT {arg}, and there are "
                             f"{len(self.messages)} messages")
        if name == "CARRYING" and not 0 <= arg <= things:
            raise ValueError(f"{where}: CARRYING {arg}, and there are "
                             f"only {things} things to carry")
        if name == "ASKED" and not 0 <= arg < len(self.topics):
            raise ValueError(f"{where}: ASKED {arg}, and there are "
                             f"{len(self.topics)} topics")
        if name in ("WITH", "SEND") and not 0 <= arg < len(self.people):
            raise ValueError(f"{where}: {name} {arg}, and there are "
                             f"{len(self.people)} people")
        if name == "SEND" and not 0 <= arg2 < rooms:
            raise ValueError(f"{where}: SEND to room {arg2}, which does "
                             f"not exist")
        if name in ("HEAT", "COOL") and not 0 <= arg <= 255:
            raise ValueError(f"{where}: {name} {arg}, and attention is "
                             f"one byte")
        if name == "HEAT" and op == C_HEAT and arg == 0:
            # Heat is never negative, so `HEAT 0` holds on the first turn and
            # every turn - the same mistake as a rule with no conditions.
            raise ValueError(f"{where}: HEAT 0 always holds, so the rule "
                             f"fires before the player has done anything")

    @property
    def overlay_bytes(self) -> int:
        """RAM the world needs to be *mutable*, which is the whole save file.

        One byte a thing for where it is, one a flag, one a one-shot rule that
        has already fired, and one for the room the player is in.

        A byte a flag rather than a bit. Bits would be eight times smaller and
        need a shift and a mask at four call sites; a world binary has half a
        megabyte of SRAM spare, so the trade is not close - and a restore that
        replayed every event the player had already seen would be the bug that
        packing them saved sixty bytes to earn.
        """
        return (1 + max(1, len(self.things)) + self.flags
                + max(1, len(self.rules))
                + max(1, len(self.topics))     # ASKED, one byte a topic
                + 1                            # HEAT
                + max(1, len(self.people)))    # PWHERE

    @property
    def asked_bytes(self) -> int:
        """`ASKED`, which is the one part of the overlay that only grows.

        Named separately because it is the piece a save file must not lose and
        a rule must not touch. `A_CLEAR` can put a flag back; nothing clears a
        topic. A mystery is fair only if what the player has been told stays
        told, and monotone state is how that is enforced rather than promised.
        """
        return max(1, len(self.topics))

    def reachable(self, start: int | None = None) -> set[int]:
        """Rooms reachable from the start, for the check nobody runs by hand."""
        first = self.start if start is None else start
        seen, queue = {first}, [first]
        while queue:
            room = self.rooms[queue.pop()]
            for target in room.exits.values():
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen

    # --- what a player can get to, which is not the same as what exists ------

    def reach(self) -> Reach:
        """Everything a player could ever bring about, over-approximated.

        The one property that matters is the direction of the error: this set
        is a **superset** of what is really attainable, so a rule outside it is
        certainly dead and a rule inside it is only probably live. That is the
        useful direction. An analysis that under-approximated would report
        locked doors that are not locked, which an author would learn to
        ignore, and an analysis that claimed to be exact would be lying - the
        turn loop's state space is every arrangement of every thing.

        Three deliberate over-approximations, each of which drops a way the
        world can go *backwards*:

        - `A_CLEAR` is ignored, so a flag once settable stays settable.
        - `C_NFLAG` always holds, because every flag is clear on turn one and
          a rule may fire then. This is why an `NFLAG` condition can never be
          what makes a rule dead.
        - a thing is treated as being everywhere it could ever be at once,
          rather than in one place at a time.
        - a flag a line of dialogue sets counts as settable, without asking
          whether that person can be found or that gate opened.

        The fixpoint is monotone under all four, so it terminates.

        That fourth one was not optional. `A_SET` inside a rule is not the
        only way a flag goes up any more - `Line.sets` is how a conversation
        teaches the world something - and without it this analysis condemned
        every rule in `worlds_mystery` that reads what somebody said. An
        over-approximation that has not been told about a mechanism does not
        degrade gracefully; it reports the mechanism as dead.
        """
        rooms = self.reachable()
        # Where each thing might be. `CARRIED` is a place like any other here,
        # which is what lets `A_MOVE thing CARRIED` feed `C_HAVE`.
        at: list[set[int]] = [{thing.at} for thing in self.things]
        flags: set[int] = {line.sets for line in self.lines
                           if line.sets != NONE}
        fired: set[int] = set()

        while True:
            held = {t for t, places in enumerate(at)
                    if CARRIED in places
                    or (self.things[t].portable and places & rooms)}
            present = {t for t, places in enumerate(at) if places & rooms}
            before = (len(rooms), sum(len(p) for p in at), len(flags),
                      len(fired))

            for number, rule in enumerate(self.rules):
                if number in fired:
                    continue
                if not all(self._holds(op, arg, rooms, held, present, flags)
                           for op, arg in rule.when):
                    continue
                fired.add(number)
                for op, arg, arg2 in rule.then:
                    if op == A_SET:
                        flags.add(arg)
                    elif op == A_GOTO:
                        rooms |= self.reachable(arg)
                    elif op == A_MOVE:
                        at[arg].add(arg2)

            after = (len(rooms), sum(len(p) for p in at), len(flags),
                     len(fired))
            if after == before:
                return Reach(frozenset(rooms), frozenset(held),
                             frozenset(present), frozenset(flags),
                             frozenset(fired))

    def _holds(self, op: int, arg: int, rooms: set[int], held: set[int],
               present: set[int], flags: set[int]) -> bool:
        if op == C_AT:
            return arg in rooms
        if op == C_HAVE:
            return arg in held
        if op == C_HERE:
            return arg in present
        if op == C_FLAG:
            return arg in flags
        if op == C_CARRYING:
            return len(held) >= arg
        # `C_NFLAG`, and `C_ASKED`, `C_HEAT` and `C_WITH` with it. The
        # fall-through is not laziness about the three new ones: this analysis
        # is a ceiling and every one of them is something a player can bring
        # about by typing - any topic can be raised, attention only climbs
        # when it is, and a person stands in a room. Answering `True` keeps
        # the error pointing upward, which is the property `reach` rests on.
        #
        # `explore` is where these are decided exactly, at the cost of an
        # actual state search. The two are not rivals: this one is total and
        # cheap and can only be wrong in the safe direction, and that one is
        # exact and can produce a walkthrough. See `explore`.
        return True

    def dead_rules(self) -> list[tuple[int, str]]:
        """Rules no play of this world can ever fire, and the reason.

        This is the locked-key bug in the only form it can be seen in: a key
        behind the door it opens is not a wrong table entry, it is a rule whose
        conditions never all hold, and nothing about the emitted binary says
        so. The game runs, and the ending is simply never reached.
        """
        got = self.reach()
        dead = []
        for number, rule in enumerate(self.rules):
            if number in got.rules:
                continue
            why = next((self._why_not(op, arg, got) for op, arg in rule.when
                        if not self._holds(op, arg, set(got.rooms),
                                           set(got.held), set(got.present),
                                           set(got.flags))),
                       "its conditions cannot hold together")
            dead.append((number, why))
        return dead

    def _why_not(self, op: int, arg: int, got: Reach) -> str:
        if op == C_AT:
            return f"room {arg} ({self.rooms[arg].name!r}) cannot be reached"
        if op == C_HAVE:
            thing = self.things[arg]
            if not thing.portable:
                return f"thing {arg} ({thing.name!r}) is not portable"
            return f"thing {arg} ({thing.name!r}) cannot be picked up"
        if op == C_HERE:
            return (f"thing {arg} ({self.things[arg].name!r}) is never in a "
                    f"room that can be reached")
        if op == C_FLAG:
            return f"flag {arg} is never set"
        return (f"{arg} things cannot be carried at once; only "
                f"{len(got.held)} can ever be picked up")
    # --- fair play, as a build-time question ----------------------------------

    def explore(self, max_states: int = 200_000) -> Search:
        """Every state a player can reach, and the shortest way to each.

        `reachable` walks the map. This walks the *game*: a state is where the
        player is, what everything is holding, which flags are set, which
        rules have fired, what has been asked and how much attention that has
        cost. Every command is an edge.

        A fair-play mystery makes a promise the author cannot keep by reading
        their own source - that the ending can be reached, and that every clue
        it rests on can be found first. This is the machine that checks it.
        Both `solve` and `unseen` are readings of what it returns.

        ## And `reach`, which is the other half

        `reach` answers a neighbouring question and the two are not rivals.
        It is a monotone fixpoint - total, cheap, and wrong only ever in the
        safe direction, so a rule it calls dead is certainly dead. This is
        exact, and pays an exponential search for it.

        The division that follows is worth stating plainly. `dead_rules` is
        the check to run on every build, because it always terminates and
        never cries wolf. This is the one to run when a world has a *goal*,
        because an over-approximation cannot prove a game winnable and cannot
        produce the sequence of commands that wins it - and a walkthrough is
        the only artefact here that can be replayed through the emulator and
        checked against the binary, which is what `test_the_mystery_can_be_won`
        does.

        **It models the device rather than an idealisation of it.** Rules are
        one pass a turn, not a fixpoint, because `RULES_RUN` walks the table
        once and a rule made true by a later rule does not fire until the next
        turn. `LOOK` is therefore a move: it is the turn that costs nothing and
        lets a cascade finish, and a walkthrough that needs one will contain
        one.

        Attention is clamped at the largest threshold any condition tests,
        which is exact rather than approximate - nothing in the world can tell
        one value above that from another - and is what keeps the space finite.
        """
        cap = self._heat_cap()
        start = _State(
            here=self.start, at_terminal=False,
            where=tuple(t.at for t in self.things),
            flags=(0,) * self.flags,
            fired=(0,) * len(self.rules),
            asked=(0,) * len(self.topics),
            heat=0,
            pwhere=tuple(p.at for p in self.people))

        # The start is a turn: the program describes the room and then runs
        # the rules before it reads the first line.
        seen_msgs: set[int] = set()
        spoken: set[int] = set()
        start = self._settle(start, cap, seen_msgs)

        parents: dict[_State, tuple[_State, str] | None] = {start: None}
        order: list[_State] = [start]
        queue = [start]
        while queue:
            state = queue.pop(0)
            for command, successor in self._moves(state, cap, seen_msgs,
                                                  spoken):
                if successor in parents:
                    continue
                if len(parents) >= max_states:
                    raise RuntimeError(
                        f"more than {max_states:,} states, so this is not "
                        f"the instrument for this world - raise max_states "
                        f"if you mean it, and know it grows with 2^flags")
                parents[successor] = (state, command)
                order.append(successor)
                queue.append(successor)

        return Search(world=self, parents=parents, states=order,
                      printed=seen_msgs, spoken=spoken)

    def droppable(self) -> set[int]:
        """Things whose being *put down somewhere* any rule can notice.

        The reduction that makes the search finish, and it is sound rather
        than a sample. Dropping a thing changes exactly three conditions:
        `C_HAVE` goes false, `C_CARRYING` falls, and `C_HERE` goes true. The
        first two can only stop a rule firing, never start one - so the only
        way putting something down can *open* anything is through a `C_HERE`
        that names it.

        For every other thing, which floor it is lying on is a distinction the
        rule language cannot make, and modelling it multiplies the state space
        by the number of rooms per object for nothing. `worlds_mystery` has no
        `C_HERE` at all, which takes it from over 200,000 states to a few
        thousand.

        The one thing this gives up is a world where *not* holding something
        matters - a rule with `HAVE` in it that the player would rather did
        not fire. That is exotic, it is stated here rather than buried, and
        `explore` can be given a world with a `C_HERE` on the thing to model
        it.
        """
        observed = {arg for rule in self.rules for op, arg in rule.when
                    if op == C_HERE}
        observed |= {arg for op, arg in self.goal if op == C_HERE}
        return observed

    def _heat_cap(self) -> int:
        """The largest attention any condition asks about.

        Above it the world is blind, so the search need not count higher - and
        must not, or a rule that adds one every turn makes the space infinite.
        """
        thresholds = [arg for rule in self.rules for op, arg in rule.when
                      if op == C_HEAT]
        thresholds += [arg for op, arg in self.goal if op == C_HEAT]
        return max(thresholds, default=0)

    def _moves(self, state: _State, cap: int, printed: set[int],
               spoken: set[int]) -> list[tuple[str, _State]]:
        """Every command that is legal here, and where it leads."""
        out: list[tuple[str, _State]] = []

        def turn(name: str, changed: _State) -> None:
            out.append((name, self._settle(changed, cap, printed)))

        if state.at_terminal:
            # The classifier is listening, so the word table is not. This is
            # `ATTERM` and it is the whole of the switch.
            turn("leave", state._replace(at_terminal=False))
            for index, topic in enumerate(self.topics):
                asked = _set(state.asked, index, 1)
                turn(f"archive {topic.words[0].lower()}",
                     state._replace(asked=asked,
                                    heat=min(cap, state.heat + topic.heat)))
            return out

        # A turn that does nothing but let the rules run. Not padding: rules
        # are one pass, so a cascade of two needs two turns and this is the
        # cheaper one.
        turn("look", state)

        for direction, target in self.rooms[state.here].exits.items():
            turn(direction.lower(), state._replace(here=target))

        for index, thing in enumerate(self.things):
            if thing.portable and state.where[index] == state.here:
                turn(f"take {thing.name}",
                     state._replace(where=_set(state.where, index, CARRIED)))
            elif state.where[index] == CARRIED and index in self.droppable():
                turn(f"drop {thing.name}",
                     state._replace(where=_set(state.where, index,
                                               state.here)))

        for pid, person in enumerate(self.people):
            if state.pwhere[pid] != state.here:
                continue
            for tid, topic in enumerate(self.topics):
                line = self._line_for(pid, tid, state.flags)
                if line is not None:
                    spoken.add(line)
                flags = state.flags
                if line is not None and self.lines[line].sets != NONE:
                    flags = _set(flags, self.lines[line].sets, 1)
                turn(f"ask {person.name} about {topic.words[0].lower()}",
                     state._replace(asked=_set(state.asked, tid, 1),
                                    flags=flags))

        if self.terminal is not None and state.here == self.terminal:
            turn("use", state._replace(at_terminal=True))

        return out

    def _line_for(self, person: int, topic: int,
                  flags: tuple[int, ...]) -> int | None:
        """The row that would be spoken, or None for the person's default."""
        for index, line in enumerate(self.lines):
            if line.person != person or line.topic != topic:
                continue
            if line.gate == NONE or flags[line.gate]:
                return index
        return None

    def _settle(self, state: _State, cap: int,
                printed: set[int]) -> _State:
        """One pass of the rule table, which is what a turn actually costs."""
        for number, rule in enumerate(self.rules):
            if rule.once and state.fired[number]:
                continue
            if not all(self._holds_in(state, op, arg) for op, arg in rule.when):
                continue
            for op, arg, arg2 in rule.then:
                state = self._apply(state, op, arg, arg2, cap, printed)
            state = state._replace(fired=_set(state.fired, number, 1))
        return state

    def _holds_in(self, state: _State, op: int, arg: int) -> bool:
        if op == C_AT:
            return state.here == arg
        if op == C_HAVE:
            return state.where[arg] == CARRIED
        if op == C_HERE:
            return state.where[arg] == state.here
        if op == C_FLAG:
            return bool(state.flags[arg])
        if op == C_NFLAG:
            return not state.flags[arg]
        if op == C_CARRYING:
            return sum(1 for w in state.where if w == CARRIED) >= arg
        if op == C_ASKED:
            return bool(state.asked[arg])
        if op == C_HEAT:
            return state.heat >= arg
        if op == C_WITH:
            return state.pwhere[arg] == state.here
        raise ValueError(f"no condition {op}")

    def _apply(self, state: _State, op: int, arg: int, arg2: int, cap: int,
               printed: set[int]) -> _State:
        if op == A_SET:
            return state._replace(flags=_set(state.flags, arg, 1))
        if op == A_CLEAR:
            return state._replace(flags=_set(state.flags, arg, 0))
        if op == A_PRINT:
            printed.add(arg)
            return state
        if op == A_GOTO:
            return state._replace(here=arg)
        if op == A_MOVE:
            return state._replace(where=_set(state.where, arg, arg2))
        if op == A_HEAT:
            return state._replace(heat=min(cap, state.heat + arg))
        if op == A_COOL:
            return state._replace(heat=max(0, state.heat - arg))
        if op == A_SEND:
            return state._replace(pwhere=_set(state.pwhere, arg, arg2))
        raise ValueError(f"no action {op}")


class _State(NamedTuple):
    """Everything a turn can change, hashable so the search can visit it once.

    This is the overlay and nothing else, with `at_terminal` for `ATTERM` -
    which is not saved on the device, because standing up is what a restore
    does anyway.
    """

    here: int
    at_terminal: bool
    where: tuple[int, ...]
    flags: tuple[int, ...]
    fired: tuple[int, ...]
    asked: tuple[int, ...]
    heat: int
    pwhere: tuple[int, ...]


def _set(values: tuple[int, ...], index: int, value: int) -> tuple[int, ...]:
    """One element changed, since a state has to stay hashable."""
    if values[index] == value:
        return values
    return (*values[:index], value, *values[index + 1:])


@dataclass
class Search:
    """What `World.explore` found, and the three questions asked of it."""

    world: World
    parents: dict[_State, tuple[_State, str] | None]
    states: list[_State]
    #: Message indices some reachable state prints.
    printed: set[int]
    #: Line indices some reachable state speaks.
    spoken: set[int]

    def solve(self) -> list[str] | None:
        """The shortest sequence of commands that satisfies the goal.

        `None` means the goal is unreachable, which is the bug this exists to
        find: a game that looks finished, plays for an hour and cannot be won.
        A world with no goal is not solvable-or-not, so it returns `[]`.
        """
        if not self.world.goal:
            return []
        for state in self.states:
            if all(self.world._holds_in(state, op, arg)
                   for op, arg in self.world.goal):
                return self._path(state)
        return None

    def _path(self, state: _State) -> list[str]:
        commands: list[str] = []
        while True:
            step = self.parents[state]
            if step is None:
                return list(reversed(commands))
            state, command = step
            commands.append(command)

    def unseen(self) -> dict[str, list[str]]:
        """Authored content no reachable state can show.

        The practical fair-play bug is not an unwinnable game - that one gets
        noticed. It is a clue behind a door that never opens, which reads to
        the player as the author having been unfair when they were only wrong.
        """
        world = self.world
        visited = {s.here for s in self.states}
        held = {i for s in self.states for i, w in enumerate(s.where)
                if w == CARRIED}
        stood = {w for s in self.states for w in s.pwhere}
        return {
            "rooms": [r.name for i, r in enumerate(world.rooms)
                      if i not in visited],
            "things": [t.name for i, t in enumerate(world.things)
                       if t.portable and i not in held],
            "messages": [world.messages[i][:40]
                         for i in range(len(world.messages))
                         if i not in self.printed],
            "lines": [f"{world.people[ln.person].name} on "
                      f"{world.topics[ln.topic].name}"
                      for i, ln in enumerate(world.lines)
                      if i not in self.spoken],
            "people": [p.name for i, p in enumerate(world.people)
                       if p.at not in visited and p.at not in stood],
            # A rule no reachable state fires is the quietest authoring bug
            # there is: it has no error, no output and nothing to notice. The
            # rule that was supposed to charge for reading a sealed record was
            # one of these, and this is what found it.
            "rules": [f"rule {i}" for i in range(len(world.rules))
                      if not any(s.fired[i] for s in self.states)],
        }


def resolve_topics(world: World, titles: list[str]) -> None:
    """Turn every topic's article titles into the ids the card knows.

    `Topic.docs` are numbers because `buildwikibin.build` is handed a document
    count and never sees the index. Somebody has to bridge that, and doing it
    here means the world is the only thing that knows both - a build that
    resolved titles itself would need the index passed through three call
    sites that have no other use for it.

    Titles are matched case-insensitively and an unknown one is an error
    rather than a topic that quietly marks nothing: a mistyped title is
    exactly the bug that makes a clue unfindable, and it is invisible at play
    time because the archive still answers.
    """
    index = {title.upper(): doc for doc, title in enumerate(titles)}
    for topic in world.topics:
        docs = []
        for title in topic.titles:
            doc = index.get(title.upper())
            if doc is None:
                raise ValueError(
                    f"topic {topic.name!r} names the article {title!r}, "
                    f"which is not on this card")
            docs.append(doc)
        topic.docs = docs
