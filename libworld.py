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


@dataclass
class World:
    rooms: list[Room]
    things: list[Thing]
    #: Room the player starts in.
    start: int = 0
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

    @property
    def overlay_bytes(self) -> int:
        """RAM the world needs to be *mutable*, which is the whole save file.

        One byte a thing for where it is, one bit a flag, and one byte for the
        room the player is in.
        """
        return len(self.things) + (self.flags + 7) // 8 + 1

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
