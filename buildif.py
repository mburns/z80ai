#!/usr/bin/env python3
"""
An Agon binary that is somewhere rather than one that answers questions.

    python buildif.py --output SILO.bin

The oracle reads a card to answer. This reads nothing: the world is tables in
the image and a handful of mutable bytes in RAM, so a turn costs no I/O at all.
That is the claim `tests/test_if.py` measures, because it is the whole reason
the two are separate programs - `data/silo/README.md` sets out the argument and
issue #62 is where it came from.

## The parser is a word table

Not a model. `examples/parser/` measured both on the same commands: a table is
100% on held-out object pairs against a position-aware model's 98.4% and a flat
one's 85.9%, and - the part that decides it - a table *declines* a word it was
never given while both models answer confidently. A player types an unknown
noun every few turns, and "I don't know the word 'zorkmid'" is the only useful
reply.

So the model stays off the critical path of moving and taking, which is what
issue #62 argued for and this is the first program that acts on.

## What a turn does

Read a line, split it into at most two words, look the first up in the verb
table and the second in the noun table, dispatch. Six directions, LOOK, TAKE,
DROP, INVENTORY, QUIT. Everything else is refused by name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import libagonio
import libworld
from libagonio import MOS_OUTCHAR
from libez80 import AGON_LOAD_ADDR, AGON_SRAM_TOP, EZ80Builder, agon_header
from libworld import CARRIED, DIRECTIONS, NOWHERE, World

#: An input line, and the most words a command may hold. Two is verb plus noun,
#: which is every command this understands; a third word is a phrasing it will
#: say it cannot read rather than one it silently ignores.
MAX_INPUT_LEN = 60
#: The longest word either table holds, plus room to notice an over-long one.
#: `INVENTORY` is nine; a player who types more gets it truncated and then
#: named back at them, which is a legible failure rather than a wrong verb.
MAX_WORD_LEN = 12

#: Room row: name pointer, description pointer, then one byte an exit.
ROOM_STRIDE = 3 + 3 + len(DIRECTIONS)
#: Thing row: name pointer, description pointer, starting place, portable.
THING_STRIDE = 3 + 3 + 1 + 1

#: Verb ids, in the order the verb table lists them. Directions come first so
#: that a verb id below `len(DIRECTIONS)` *is* the direction index.
V_LOOK = len(DIRECTIONS)
V_TAKE = V_LOOK + 1
V_DROP = V_TAKE + 1
V_INVENTORY = V_DROP + 1
V_QUIT = V_INVENTORY + 1

#: Stack margin below the top of SRAM, matching every other Agon build here.
STACK_MARGIN = 0x1000


def _words(world: World) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """(verb table, noun table) as (word, id), longest match not required.

    Directions and their short forms share the verb table because `NORTH` is a
    command on its own - a player types `N`, not `GO N`, and a table that made
    `GO` mandatory would be wrong about the commonest thing anybody types.
    """
    verbs: list[tuple[str, int]] = []
    for index, direction in enumerate(DIRECTIONS):
        verbs.append((direction, index))
    for short, full in libworld.ALIASES.items():
        verbs.append((short, DIRECTIONS.index(full)))
    verbs += [("LOOK", V_LOOK), ("L", V_LOOK),
              ("TAKE", V_TAKE), ("GET", V_TAKE),
              ("DROP", V_DROP), ("PUT", V_DROP),
              ("INVENTORY", V_INVENTORY), ("I", V_INVENTORY),
              ("QUIT", V_QUIT), ("Q", V_QUIT)]

    nouns = [(thing.name.upper(), index)
             for index, thing in enumerate(world.things)]
    return verbs, nouns


def _emit_word_table(b: EZ80Builder, label: str,
                     words: list[tuple[str, int]]) -> None:
    """A word table: length, bytes, id - terminated by a zero length.

    Fixed stride would waste more than it saved; a word is short and the scan
    is linear over a few dozen entries, which is nothing beside a forward pass
    the alternative would have cost.
    """
    b.label(label)
    for word, ident in words:
        b.db(len(word))
        b.ascii(word)
        b.db(ident)
    b.db(0)


def build(world: World, org: int = AGON_LOAD_ADDR) -> EZ80Builder:
    """Emit the whole game: tables, turn loop, and the text it prints."""
    world.check()

    b = EZ80Builder(org=org)
    agon_header(b, "START")
    verbs, nouns = _words(world)

    b.label("START")
    b.ld_a_n(world.start)
    b.ld_mem_label_a("HERE")
    _emit_reset_things(b, world)
    b.ld_hl_label("BANNER")
    b.call("PRWRAP")
    b.call("PRNL")
    b.call("DESCRIBE")

    # --- the turn loop --------------------------------------------------------
    b.label("TURN")
    b.call("PRNL")
    b.ld_hl_label("PROMPT")
    b.call("PRSTR")
    b.call("READ_INPUT")
    b.call("SPLIT")                  # INPBUF -> W1LEN/W1, W2LEN/W2
    b.ld_a_mem_label("W1LEN")
    b.or_a()
    b.jr_z("TURN")                   # an empty line is not a turn

    b.ld_hl_label("VERBS")
    b.ld_de_label("W1")
    b.ld_a_mem_label("W1LEN")
    b.call("LOOKUP")
    b.jr_c("BADVERB")
    b.ld_mem_label_a("VERB")

    b.cp_n(V_QUIT)
    b.jp_z("BYE")
    b.cp_n(V_LOOK)
    b.jr_z("DO_LOOK")
    b.cp_n(V_INVENTORY)
    b.jp_z("DO_INV")
    b.cp_n(V_TAKE)
    b.jp_z("DO_TAKE")
    b.cp_n(V_DROP)
    b.jp_z("DO_DROP")
    b.jp("DO_GO")                    # below LOOK: the id is a direction

    b.label("DO_LOOK")
    b.call("DESCRIBE")
    b.jr("TURN")

    b.label("BADVERB")
    b.ld_hl_label("MSGVERB")
    b.call("PRSTR")
    b.ld_hl_label("W1")
    b.ld_a_mem_label("W1LEN")
    b.call("PRWORD")
    b.ld_hl_label("MSGQUOTE")
    b.call("PRSTR")
    b.jr("TURN")

    _emit_go(b, world)
    _emit_take_drop(b, world)
    _emit_describe(b, world)
    _emit_split(b)
    _emit_upper(b)
    _emit_lookup(b)
    _emit_prword(b)
    _emit_noun(b)
    _emit_room_row(b)
    _emit_thing_row(b)
    _emit_where_ptr(b)
    _ldptr(b)

    b.label("BYE")
    b.ld_hl_label("MSGBYE")
    b.call("PRWRAP")
    b.call("PRNL")
    b.ld_hl_nn(0)
    b.ret()

    libagonio.emit_console(b, MAX_INPUT_LEN)
    _emit_tables(b, world, verbs, nouns)
    _emit_ram(b, world, org)
    return b


def _ldptr(b: EZ80Builder) -> None:
    """HL points at a 3-byte pointer in a table; leave that pointer in HL.

    The eZ80 has no `LD HL,(HL)`, so it goes through a cell. Called from every
    routine that prints something out of a table, which is all of them.
    """
    b.label("LDPTR")
    for offset in range(3):
        b.ld_a_hl()
        b.ld_mem_label_a("PTMP", offset)
        b.inc_hl()
    b.ld_hl_mem_label("PTMP")
    b.ret()


def _emit_reset_things(b: EZ80Builder, world: World) -> None:
    """Copy the starting places into the overlay.

    The image holds where a thing *starts* and RAM holds where it is, which is
    the whole distinction that makes a saved game the overlay and nothing else.
    """
    if not world.things:
        return
    b.ld_hl_label("INITWHERE")
    b.ld_de_label("WHERE")
    b.ld_b_n(len(world.things))
    b.label("RESET_LP")
    b.ld_a_hl()
    b.ld_de_a()
    b.inc_hl()
    b.inc_de()
    b.djnz("RESET_LP")


def _emit_room_row(b: EZ80Builder) -> None:
    """HL = the row for the room in A."""
    b.label("ROOMROW")
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.add_hl_hl()
    b.add_hl_hl()                    # x4
    b.push_hl()
    b.pop_de()
    b.add_hl_hl()                    # x8
    b.add_hl_de()                    # x12
    b.ld_de_label("ROOMS")
    b.add_hl_de()
    b.ret()


def _emit_thing_row(b: EZ80Builder) -> None:
    """HL = the row for the thing in A. Eight bytes, so three doublings."""
    b.label("THINGROW")
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.ld_de_label("THINGS")
    b.add_hl_de()
    b.ret()


def _emit_where_ptr(b: EZ80Builder) -> None:
    """HL = &WHERE[A], the one byte that says where a thing is now."""
    b.label("WHEREPTR")
    b.ld_hl_label("WHERE")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ret()


def _emit_upper(b: EZ80Builder) -> None:
    """A lowercase letter in A becomes uppercase; anything else is left alone.

    The same `cp / add` the tokenizer does elsewhere in this repository - a
    player does not capitalise and a word table does not want to hold both.
    """
    b.label("UPPER")
    b.cp_n(ord("a"))
    b.ret_c()
    b.cp_n(ord("z") + 1)
    b.ret_nc()
    b.sub_n(32)
    b.ret()


def _emit_split(b: EZ80Builder) -> None:
    """INPBUF -> W1/W1LEN and W2/W2LEN, uppercased.

    Two words and no more. Everything this understands is a verb and at most
    one noun, and a third word is a phrasing it will decline by name rather
    than one it quietly ignores - `PUT KEY IN BOX` is not a command here and
    saying so is better than doing half of it.
    """
    b.label("SPLIT")
    b.xor_a()
    b.ld_mem_label_a("W1LEN")
    b.ld_mem_label_a("W2LEN")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.ret_z()
    b.ld_b_a()
    b.ld_hl_label("INPBUF")

    b.ld_de_label("W1")
    b.call("SP_ONE")
    b.ld_a_c()
    b.ld_mem_label_a("W1LEN")
    b.ld_de_label("W2")
    b.call("SP_ONE")
    b.ld_a_c()
    b.ld_mem_label_a("W2LEN")
    b.ret()

    # SP_ONE: HL source, B bytes left, DE destination -> C length.
    b.label("SP_ONE")
    b.ld_c_n(0)
    b.label("SP_SKIP")
    b.ld_a_b()
    b.or_a()
    b.ret_z()
    b.ld_a_hl()
    b.cp_n(32)
    b.jr_nz("SP_COPY")
    b.inc_hl()
    b.dec_b()
    b.jr("SP_SKIP")

    b.label("SP_COPY")
    b.ld_a_b()
    b.or_a()
    b.ret_z()
    b.ld_a_hl()
    b.cp_n(32)
    b.ret_z()
    b.call("UPPER")
    b.ld_de_a()
    b.inc_de()
    b.inc_hl()
    b.dec_b()
    b.inc_c()
    b.ld_a_c()
    b.cp_n(MAX_WORD_LEN)
    b.ret_nc()                       # a word longer than the table can hold
    b.jr("SP_COPY")


def _emit_lookup(b: EZ80Builder) -> None:
    """HL a word table, DE the word, A its length -> A the id, carry on miss.

    A linear scan over a few dozen short entries, which is the entire cost of
    the decision a classifier would have spent a forward pass on. Carry is the
    answer that matters: it is how the program knows to name the word back
    rather than act on a guess.
    """
    b.label("LOOKUP")
    b.ld_mem_label_a("LKLEN")
    b.ld_mem_label_de("LKPTR")

    b.label("LK_ENTRY")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("LK_FAIL")
    b.ld_b_a()
    b.ld_a_mem_label("LKLEN")
    b.cp_b()
    b.jr_nz("LK_SKIP")

    b.push_hl()
    b.inc_hl()
    b.ld_de_mem_label("LKPTR")
    b.label("LK_CMP")
    b.ld_a_de()
    b.cp_hl()
    b.jr_nz("LK_NO")
    b.inc_hl()
    b.inc_de()
    b.djnz("LK_CMP")
    b.ld_a_hl()                      # the id byte follows the characters
    b.pop_de()                       # discard the saved entry pointer
    b.or_a()                         # and clear carry: this is a hit
    b.ret()

    b.label("LK_NO")
    b.pop_hl()

    b.label("LK_SKIP")
    b.ld_a_hl()
    b.inc_a()
    b.inc_a()                        # one length byte, the word, one id byte
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.jr("LK_ENTRY")

    b.label("LK_FAIL")
    b.scf()
    b.ret()


def _emit_prword(b: EZ80Builder) -> None:
    """Print A characters from HL, for naming a word back to the player."""
    b.label("PRWORD")
    b.or_a()
    b.ret_z()
    b.ld_b_a()
    b.label("PRW_LP")
    b.ld_a_hl()
    b.rst(MOS_OUTCHAR)
    b.inc_hl()
    b.djnz("PRW_LP")
    b.ret()


def _emit_noun(b: EZ80Builder) -> None:
    """The second word as a thing id, carry set if there was not one."""
    b.label("NOUNID")
    b.ld_a_mem_label("W2LEN")
    b.or_a()
    b.jr_z("NN_FAIL")
    b.ld_hl_label("NOUNS")
    b.ld_de_label("W2")
    b.ld_a_mem_label("W2LEN")
    b.jp("LOOKUP")                   # carry and A come straight back

    b.label("NN_FAIL")
    b.scf()
    b.ret()


def _emit_describe(b: EZ80Builder, world: World) -> None:
    """Room name, room description, then whatever is lying about in it."""
    b.label("DESCRIBE")
    b.ld_a_mem_label("HERE")
    b.call("ROOMROW")
    b.push_hl()
    b.call("LDPTR")
    b.call("PRWRAP")
    b.call("PRNL")
    b.pop_hl()
    b.ld_de_nn(3)
    b.add_hl_de()
    b.call("LDPTR")
    b.call("PRWRAP")
    b.call("PRNL")

    if not world.things:
        b.ret()

    b.ld_b_n(len(world.things))
    b.ld_c_n(0)
    b.label("LH_LP")
    b.push_bc()
    b.ld_a_c()
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jr_nz("LH_NEXT")
    b.ld_hl_label("MSGSEE")
    b.call("PRSTR")
    b.ld_a_c()
    b.call("THINGROW")
    b.call("LDPTR")
    b.call("PRWRAP")
    b.ld_hl_label("MSGDOT")
    b.call("PRSTR")
    b.call("PRNL")
    b.label("LH_NEXT")
    b.pop_bc()
    b.inc_c()
    b.djnz("LH_LP")
    b.ret()


def _emit_go(b: EZ80Builder, world: World) -> None:
    """A direction verb: the id is the index into the room's exit bytes."""
    b.label("DO_GO")
    b.ld_a_mem_label("HERE")
    b.call("ROOMROW")
    b.ld_de_nn(6)
    b.add_hl_de()
    b.ld_a_mem_label("VERB")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()
    b.cp_n(NOWHERE)
    b.jr_z("GO_NO")
    b.ld_mem_label_a("HERE")
    b.call("DESCRIBE")
    b.jp("TURN")

    b.label("GO_NO")
    b.ld_hl_label("MSGNOWAY")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")


def _emit_take_drop(b: EZ80Builder, world: World) -> None:
    """TAKE, DROP and INVENTORY, which are the whole of this world's physics.

    Every one of them is a byte in the overlay changing value. That is the
    property that makes a saved game the overlay and nothing else, and the
    reason a turn touches no card: there is nowhere else for the state to be.
    """
    b.label("DO_TAKE")
    b.call("NOUNID")
    b.jp_c("BADNOUN")
    b.ld_c_a()
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jp_z("TK_HAVE")
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jp_nz("TK_ABSENT")

    b.ld_a_c()
    b.call("THINGROW")
    b.ld_de_nn(7)
    b.add_hl_de()
    b.ld_a_hl()
    b.or_a()
    b.jp_z("TK_FIXED")

    b.ld_a_c()
    b.call("WHEREPTR")
    b.ld_a_n(CARRIED)
    b.ld_hl_a()
    b.ld_hl_label("MSGTAKEN")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")

    for label, message in (("TK_HAVE", "MSGHAVE"), ("TK_ABSENT", "MSGABSENT"),
                           ("TK_FIXED", "MSGFIXED")):
        b.label(label)
        b.ld_hl_label(message)
        b.call("PRWRAP")
        b.call("PRNL")
        b.jp("TURN")

    b.label("DO_DROP")
    b.call("NOUNID")
    b.jp_c("BADNOUN")
    b.ld_c_a()
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jp_nz("TK_HAVENOT")
    b.ld_a_c()
    b.call("WHEREPTR")
    b.ld_a_mem_label("HERE")
    b.ld_hl_a()
    b.ld_hl_label("MSGDROPPED")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")

    b.label("TK_HAVENOT")
    b.ld_hl_label("MSGHAVENOT")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")

    b.label("BADNOUN")
    b.ld_a_mem_label("W2LEN")
    b.or_a()
    b.jr_z("BN_NONE")
    b.ld_hl_label("MSGNOUN")
    b.call("PRSTR")
    b.ld_hl_label("W2")
    b.ld_a_mem_label("W2LEN")
    b.call("PRWORD")
    b.ld_hl_label("MSGQUOTE")
    b.call("PRSTR")
    b.jp("TURN")

    b.label("BN_NONE")
    b.ld_hl_label("MSGWHAT")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")

    b.label("DO_INV")
    if not world.things:
        b.ld_hl_label("MSGEMPTY")
        b.call("PRWRAP")
        b.call("PRNL")
        b.jp("TURN")
        return

    b.xor_a()
    b.ld_mem_label_a("NCARRIED")
    b.ld_b_n(len(world.things))
    b.ld_c_n(0)
    b.label("IV_LP")
    b.push_bc()
    b.ld_a_c()
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jr_nz("IV_NEXT")
    b.ld_a_mem_label("NCARRIED")
    b.inc_a()
    b.ld_mem_label_a("NCARRIED")
    b.ld_hl_label("MSGCARRY")
    b.call("PRSTR")
    b.ld_a_c()
    b.call("THINGROW")
    b.call("LDPTR")
    b.call("PRWRAP")
    b.ld_hl_label("MSGDOT")
    b.call("PRSTR")
    b.call("PRNL")
    b.label("IV_NEXT")
    b.pop_bc()
    b.inc_c()
    b.djnz("IV_LP")

    b.ld_a_mem_label("NCARRIED")
    b.or_a()
    b.jp_nz("TURN")
    b.ld_hl_label("MSGEMPTY")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")


#: The messages, which are the whole of this program's manners. Kept together
#: so that a world in another language changes one table.
MESSAGES: dict[str, str] = {
    "BANNER": "Silo 18. You are somewhere, and it is dark outside.",
    "PROMPT": "> ",
    "MSGSEE": "You can see ",
    "MSGCARRY": "You are carrying ",
    "MSGDOT": ".",
    "MSGQUOTE": "'.\r\n",
    "MSGVERB": "I do not know the word '",
    "MSGNOUN": "I do not know the word '",
    "MSGWHAT": "What do you want to do that to?",
    "MSGNOWAY": "You cannot go that way.",
    "MSGTAKEN": "Taken.",
    "MSGDROPPED": "Dropped.",
    "MSGHAVE": "You already have it.",
    "MSGHAVENOT": "You are not carrying it.",
    "MSGABSENT": "That is not here.",
    "MSGFIXED": "That is not something you can carry.",
    "MSGEMPTY": "You are empty-handed.",
    "MSGBYE": "Goodbye.",
}


def _emit_tables(b: EZ80Builder, world: World,
                 verbs: list[tuple[str, int]],
                 nouns: list[tuple[str, int]]) -> None:
    """Everything that does not change: rooms, things, words and text."""
    b.label("ROOMS")
    for index, room in enumerate(world.rooms):
        b.fixup_word(f"RNAME{index}")
        b.fixup_word(f"RDESC{index}")
        for direction in DIRECTIONS:
            b.db(room.exits.get(direction, NOWHERE))

    b.label("THINGS")
    for index, thing in enumerate(world.things):
        b.fixup_word(f"TNAME{index}")
        b.fixup_word(f"TDESC{index}")
        b.db(thing.at)
        b.db(1 if thing.portable else 0)

    if world.things:
        b.label("INITWHERE")
        for thing in world.things:
            b.db(thing.at)

    _emit_word_table(b, "VERBS", verbs)
    _emit_word_table(b, "NOUNS", nouns)

    for index, room in enumerate(world.rooms):
        b.label(f"RNAME{index}")
        b.ascii(room.name)
        b.db(0)
        b.label(f"RDESC{index}")
        b.ascii(room.description)
        b.db(0)
    for index, thing in enumerate(world.things):
        b.label(f"TNAME{index}")
        b.ascii(thing.name)
        b.db(0)
        b.label(f"TDESC{index}")
        b.ascii(thing.description)
        b.db(0)

    for label, text in MESSAGES.items():
        b.label(label)
        b.ascii(text)
        b.db(0)


def _emit_ram(b: EZ80Builder, world: World, org: int) -> None:
    """The mutable half, and it is small.

    `HERE`, `WHERE` and `FLAGS` are the saved game. Everything else here is
    scratch the turn loop needs and nothing outlives a turn.
    """
    # The overlay first and contiguous, because it is the save file: where the
    # player is, where everything else is, and the flags. Putting the scratch
    # buffers in the middle would make saving three writes and a format, and
    # this way it is one `mos_fwrite` over one run of bytes.
    b.label("HERE")
    b.db(0)
    b.label("WHERE")
    b.ds(max(1, len(world.things)))
    b.label("FLAGS")
    b.ds((world.flags + 7) // 8)

    # Everything below is scratch that does not outlive a turn.
    for name in ("VERB", "W1LEN", "W2LEN", "LKLEN", "WRAPCOL",
                 "INPLEN", "NCARRIED"):
        b.label(name)
        b.db(0)
    for name in ("PTMP", "LKPTR"):
        b.label(name)
        b.d24(0)

    b.label("W1")
    b.ds(MAX_WORD_LEN + 1)
    b.label("W2")
    b.ds(MAX_WORD_LEN + 1)
    b.label("INPBUF")
    b.ds(MAX_INPUT_LEN + 1)


def overlay_at(builder: EZ80Builder, world: World) -> tuple[int, int]:
    """(address, length) of the bytes a save file would hold.

    `HERE` is deliberately the first of them and `FLAGS` the last, so the
    saved game is one contiguous run and `mos_fwrite` can put it down in a
    single call - see `tests/test_fwrite.py` for the call itself, which is
    emitted with the rest of save and restore rather than here.
    """
    start = builder.labels["HERE"]
    end = builder.labels["FLAGS"] + (world.flags + 7) // 8
    return start, end - start


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", "-o", default="SILO.bin", type=Path)
    args = ap.parse_args()

    import worlds
    world = worlds.silo()
    builder = build(world)
    builder.save(str(args.output))

    top = builder.org + len(builder.code)
    print(f"{args.output}  {len(builder.code):,} bytes   "
          f"{len(world.rooms)} rooms, {len(world.things)} things")
    print(f"  overlay {world.overlay_bytes} bytes - the whole saved game")
    print(f"  image ends at {top:06X}h, "
          f"{AGON_SRAM_TOP - STACK_MARGIN - top:,} bytes of SRAM unused")


if __name__ == "__main__":
    main()
