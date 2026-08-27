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

#: No exit that way. 255 rooms is the limit a one-byte room id implies, and a
#: world that wants more wants a different overlay rather than a wider table.
NOWHERE = 0xFF

#: Where a thing is when the player has it, rather than in a room.
CARRIED = 0xFE


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

#: Action opcodes. `A_MOVE` takes two bytes because it names a thing and a
#: destination; everything else takes one.
A_SET = 0       # set flag `arg`
A_CLEAR = 1     # clear flag `arg`
A_PRINT = 2     # print message `arg`
A_GOTO = 3      # move the player to room `arg`
A_MOVE = 4      # move thing `arg` to `arg2`

CONDITION_NAMES = {C_AT: "AT", C_HAVE: "HAVE", C_HERE: "HERE", C_FLAG: "FLAG",
                   C_NFLAG: "NFLAG", C_CARRYING: "CARRYING"}
ACTION_NAMES = {A_SET: "SET", A_CLEAR: "CLEAR", A_PRINT: "PRINT",
                A_GOTO: "GOTO", A_MOVE: "MOVE"}


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


@dataclass
class World:
    rooms: list[Room]
    things: list[Thing]
    #: Room the player starts in.
    start: int = 0
    #: Rules, checked in order after every turn that did something.
    rules: list[Rule] = field(default_factory=list)
    #: Messages rules print, by index.
    messages: list[str] = field(default_factory=list)
    #: How many one-bit propositions the world reserves. Costs one byte per
    #: eight and nothing else, so the number is a guess that can be generous.
    flags: int = 64

    def check(self) -> None:
        """Refuse a world that cannot be walked, before anything is emitted.

        Every one of these is a mistake that produces a playable-looking game
        with a room nobody can leave or a thing nobody can find, which is the
        kind of bug an author discovers ten minutes in rather than at build
        time.
        """
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

        names = [t.name.upper() for t in self.things]
        if len(set(names)) != len(names):
            raise ValueError("two things share a name, and the parser resolves "
                             "a noun to exactly one")

        self._check_rules()

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

    def _check_arg(self, number: int, name: str, op: int, arg: int,
                   arg2: int = 0) -> None:
        rooms, things = len(self.rooms), len(self.things)
        if (name in ("AT", "GOTO") and not 0 <= arg < rooms):
            raise ValueError(f"rule {number}: {name} {arg}, and there are "
                             f"{rooms} rooms")
        if name in ("HAVE", "HERE", "MOVE") and not 0 <= arg < things:
            raise ValueError(f"rule {number}: {name} {arg}, and there are "
                             f"{things} things")
        if name == "MOVE" and not (0 <= arg2 < rooms or arg2 == CARRIED):
            raise ValueError(f"rule {number}: MOVE to {arg2}, which is neither "
                             f"a room nor CARRIED")
        if name in ("FLAG", "NFLAG", "SET", "CLEAR") and not 0 <= arg < self.flags:
            raise ValueError(f"rule {number}: {name} {arg}, and the world "
                             f"reserves {self.flags} flags")
        if name == "PRINT" and not 0 <= arg < len(self.messages):
            raise ValueError(f"rule {number}: PRINT {arg}, and there are "
                             f"{len(self.messages)} messages")
        if name == "CARRYING" and not 0 <= arg <= things:
            raise ValueError(f"rule {number}: CARRYING {arg}, and there are "
                             f"only {things} things to carry")

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
                + max(1, len(self.rules)))

    def reachable(self) -> set[int]:
        """Rooms reachable from the start, for the check nobody runs by hand."""
        seen, queue = {self.start}, [self.start]
        while queue:
            room = self.rooms[queue.pop()]
            for target in room.exits.values():
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen
