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
#: Sit down at the archive terminal, where a different parser is listening.
#: Only the merged build can act on it; the standalone world says there is no
#: terminal here, which is true of every room in it.
V_USE = V_QUIT + 1

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
              ("QUIT", V_QUIT), ("Q", V_QUIT),
              ("USE", V_USE), ("CONSULT", V_USE), ("ASK", V_USE)]

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

    b.label("START")
    b.ld_a_n(world.start)
    b.ld_mem_label_a("HERE")
    _emit_reset_things(b, world)
    b.ld_hl_label("WBANNER")
    b.call("PRWRAP")
    b.call("PRNL")
    b.call("DESCRIBE")

    # --- the turn loop --------------------------------------------------------
    b.label("TURN")
    b.call("RULES_RUN")              # the world reacts before it asks again
    b.call("PRNL")
    b.ld_hl_label("WPROMPT")
    b.call("PRSTR")
    b.call("READ_INPUT")
    emit_dispatch(b, quit_label="BYE")
    emit_world_routines(b, world)

    b.label("BYE")
    b.ld_hl_label("MSGBYE")
    b.call("PRWRAP")
    b.call("PRNL")
    b.ld_hl_nn(0)
    b.ret()

    libagonio.emit_console(b, MAX_INPUT_LEN)
    emit_world_tables(b, world)
    emit_world_ram(b, world)
    return b


def emit_dispatch(b: EZ80Builder, quit_label: str) -> None:
    """A line already in INPBUF -> the verb it means, acted on.

    Split from the loop around it so that the oracle binary can call the same
    dispatch when its mode byte says the player is walking rather than at the
    screen. Everything here jumps to `TURN` when it is done, and what `TURN`
    is depends on which program emitted it.
    """
    b.call("SPLIT")                  # INPBUF -> W1LEN/W1, W2LEN/W2
    b.ld_a_mem_label("W1LEN")
    b.or_a()
    b.jp_z("TURN")                   # an empty line is not a turn

    b.ld_hl_label("VERBS")
    b.ld_de_label("W1")
    b.ld_a_mem_label("W1LEN")
    b.call("LOOKUP")
    b.jr_c("BADVERB")
    b.ld_mem_label_a("VERB")

    b.cp_n(V_QUIT)
    b.jp_z(quit_label)
    b.cp_n(V_LOOK)
    b.jr_z("DO_LOOK")
    b.cp_n(V_INVENTORY)
    b.jp_z("DO_INV")
    b.cp_n(V_TAKE)
    b.jp_z("DO_TAKE")
    b.cp_n(V_DROP)
    b.jp_z("DO_DROP")
    b.cp_n(V_USE)
    b.jp_z("DO_USE")
    b.jp("DO_GO")                    # below LOOK: the id is a direction

    b.label("DO_LOOK")
    b.call("DESCRIBE")
    b.jp("TURN")

    b.label("BADVERB")
    b.ld_hl_label("MSGVERB")
    b.call("PRSTR")
    b.ld_hl_label("W1")
    b.ld_a_mem_label("W1LEN")
    b.call("PRWORD")
    b.ld_hl_label("MSGQUOTE")
    b.call("PRSTR")
    b.jp("TURN")


def emit_reset(b: EZ80Builder, world: World) -> None:
    """Put every thing back where it starts. Exposed for the merged build."""
    _emit_reset_things(b, world)


def emit_world_routines(b: EZ80Builder, world: World) -> None:
    """Everything a turn needs, and nothing about how the program starts.

    Split out so the oracle binary can hold a world as well. That direction
    round is the one that fits: the world is 4,050 bytes and the oracle 38,912,
    so a terminal standing in a room is the small thing inside the large one
    rather than the other way about - which is not how issue #62 pictured it.
    """
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
    _emit_rules(b, world)


def emit_world_tables(b: EZ80Builder, world: World) -> None:
    """Rooms, things, words and text - none of which change."""
    verbs, nouns = _words(world)
    _emit_tables(b, world, verbs, nouns)


def emit_world_ram(b: EZ80Builder, world: World,
                   shared_console: bool = False) -> None:
    """The overlay and the turn's scratch.

    `shared_console` leaves `INPBUF`, `INPLEN` and `WRAPCOL` to the host
    program. Sharing the input buffer is not a compromise: it is the point.
    Two parsers reading one line is what "the two input paths can coexist"
    means, and they can because both only ever read it.
    """
    _emit_ram(b, world, shared_console)


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

    # DO_USE: sit down at the terminal, if there is one here.
    #
    # `TERMROOM` is 0xFF in the standalone world, which has no card to consult,
    # and the room the archive terminal stands in when the oracle binary
    # carries a world. The whole of the switch is one byte: from here the
    # classifier reads the same INPBUF the word table just did.
    b.label("DO_USE")
    b.ld_a_mem_label("HERE")
    b.ld_hl_label("TERMROOM")
    b.cp_hl()
    b.jr_z("USE_YES")
    b.ld_hl_label("MSGNOTERM")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")

    b.label("USE_YES")
    b.ld_a_n(1)
    b.ld_mem_label_a("ATTERM")
    b.ld_hl_label("MSGSITDOWN")
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


def _emit_rules(b: EZ80Builder, world: World) -> None:
    """Check every rule; fire the ones whose conditions all hold.

    The step past a path, and a small one. A graph walk composes - follow this,
    then that - and stops at conjunction. A flat list of conditions ANDed
    together is the least that does not, and `IF.md` reports which of the four
    shapes `data/silo/README.md` names it actually closes. It is three.

    A rule is length-prefixed so that skipping one is an addition rather than a
    walk over its parts, which is what the first version of this did and got
    wrong twice.
    """
    b.label("RULES_RUN")
    if not world.rules:
        b.ret()
        return

    b.ld_hl_label("RULETAB")
    b.ld_mem_label_hl("RULEPTR")
    b.ld_c_n(0)

    b.label("RU_NEXT")
    b.ld_hl_mem_label("RULEPTR")
    b.ld_a_hl()
    b.or_a()
    b.ret_z()                        # a zero length ends the table

    b.push_bc()
    b.call("RU_ONE")
    b.pop_bc()

    b.ld_hl_mem_label("RULEPTR")
    b.ld_a_hl()
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_mem_label_hl("RULEPTR")
    b.inc_c()
    b.jr("RU_NEXT")

    # RU_ONE: the rule at RULEPTR, with C its number. Leaves RULEPTR alone -
    # the caller steps over it by its length whatever happens here.
    b.label("RU_ONE")
    b.ld_hl_mem_label("RULEPTR")
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a("RU_ONCE")
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a("RU_NC")
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a("RU_NA")
    b.inc_hl()
    b.ld_mem_label_hl("RU_CUR")

    b.ld_a_mem_label("RU_ONCE")
    b.or_a()
    b.jr_z("RU_COND")
    b.call("FIREDP")
    b.or_a()
    b.ret_nz()                       # already fired, and fires once

    b.label("RU_COND")
    b.ld_a_mem_label("RU_NC")
    b.or_a()
    b.jr_z("RU_ACT")
    b.ld_b_a()

    b.label("RU_CLP")
    b.push_bc()
    b.ld_hl_mem_label("RU_CUR")
    b.ld_a_hl()
    b.ld_mem_label_a("RU_OP")
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a("RU_ARG")
    b.inc_hl()
    b.ld_mem_label_hl("RU_CUR")
    b.call("RU_TEST")
    b.pop_bc()
    b.ret_nc()                       # one condition short is the whole rule
    b.djnz("RU_CLP")

    b.label("RU_ACT")
    b.ld_a_mem_label("RU_NA")
    b.or_a()
    b.jr_z("RU_FIRED")
    b.ld_b_a()
    b.label("RU_ALP")
    b.push_bc()
    b.call("RU_DO")
    b.pop_bc()
    b.djnz("RU_ALP")

    b.label("RU_FIRED")
    b.call("MARKFIRED")
    b.ret()

    _emit_rule_test(b, world)
    _emit_rule_do(b, world)
    _emit_rule_state(b, world)


def _emit_rule_test(b: EZ80Builder, world: World) -> None:
    """One condition, from RU_OP and RU_ARG -> carry set when it holds."""
    b.label("RU_TEST")
    b.ld_a_mem_label("RU_OP")

    b.cp_n(libworld.C_AT)
    b.jr_nz("RT_HAVE")
    b.ld_a_mem_label("HERE")
    b.ld_hl_label("RU_ARG")
    b.cp_hl()
    b.jr_z("RT_YES")
    b.jr("RT_NO")

    b.label("RT_HAVE")
    b.cp_n(libworld.C_HAVE)
    b.jr_nz("RT_HERE")
    b.ld_a_mem_label("RU_ARG")
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jr_z("RT_YES")
    b.jr("RT_NO")

    b.label("RT_HERE")
    b.cp_n(libworld.C_HERE)
    b.jr_nz("RT_FLAG")
    b.ld_a_mem_label("RU_ARG")
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jr_z("RT_YES")
    b.jr("RT_NO")

    b.label("RT_FLAG")
    b.cp_n(libworld.C_FLAG)
    b.jr_nz("RT_NFLAG")
    b.ld_a_mem_label("RU_ARG")
    b.call("FLAGPTR")
    b.ld_a_hl()
    b.or_a()
    b.jr_nz("RT_YES")
    b.jr("RT_NO")

    b.label("RT_NFLAG")
    b.cp_n(libworld.C_NFLAG)
    b.jr_nz("RT_COUNT")
    b.ld_a_mem_label("RU_ARG")
    b.call("FLAGPTR")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("RT_YES")
    b.jr("RT_NO")

    # The count a path cannot do, and the reason this opcode exists at all.
    b.label("RT_COUNT")
    b.cp_n(libworld.C_CARRYING)
    b.jr_nz("RT_NO")
    b.call("COUNTHELD")
    b.ld_hl_label("RU_ARG")
    b.cp_hl()
    b.jr_c("RT_NO")                  # carrying fewer than the rule asked for
    b.jr("RT_YES")

    b.label("RT_NO")
    b.or_a()
    b.ret()
    b.label("RT_YES")
    b.scf()
    b.ret()


def _emit_rule_do(b: EZ80Builder, world: World) -> None:
    """One action, three bytes at RU_CUR, which it steps past."""
    b.label("RU_DO")
    b.ld_hl_mem_label("RU_CUR")
    b.ld_a_hl()
    b.ld_mem_label_a("RU_OP")
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a("RU_ARG")
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a("RU_ARG2")
    b.inc_hl()
    b.ld_mem_label_hl("RU_CUR")

    b.ld_a_mem_label("RU_OP")
    b.cp_n(libworld.A_SET)
    b.jr_nz("RD_CLEAR")
    b.ld_a_mem_label("RU_ARG")
    b.call("FLAGPTR")
    b.ld_a_n(1)
    b.ld_hl_a()
    b.ret()

    b.label("RD_CLEAR")
    b.cp_n(libworld.A_CLEAR)
    b.jr_nz("RD_PRINT")
    b.ld_a_mem_label("RU_ARG")
    b.call("FLAGPTR")
    b.xor_a()
    b.ld_hl_a()
    b.ret()

    b.label("RD_PRINT")
    b.cp_n(libworld.A_PRINT)
    b.jr_nz("RD_GOTO")
    b.ld_a_mem_label("RU_ARG")
    b.call("MSGROW")
    b.call("PRWRAP")
    b.jp("PRNL")

    b.label("RD_GOTO")
    b.cp_n(libworld.A_GOTO)
    b.jr_nz("RD_MOVE")
    b.ld_a_mem_label("RU_ARG")
    b.ld_mem_label_a("HERE")
    b.jp("DESCRIBE")

    b.label("RD_MOVE")
    b.cp_n(libworld.A_MOVE)
    b.ret_nz()
    b.ld_a_mem_label("RU_ARG")
    b.call("WHEREPTR")
    b.ld_a_mem_label("RU_ARG2")
    b.ld_hl_a()
    b.ret()


def _emit_rule_state(b: EZ80Builder, world: World) -> None:
    """Flags, the fired markers, the carried count, and message lookup.

    A byte a flag rather than a bit. Bits would be eight times smaller and
    need a shift and a mask at four call sites; there are 517,068 bytes of
    SRAM unused, so the trade is not close.
    """
    b.label("FLAGPTR")
    b.ld_hl_label("FLAGS")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ret()

    b.label("FIREDP")
    b.ld_hl_label("FIRED")
    b.ld_de_nn(0)
    b.ld_a_c()
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()
    b.ret()

    b.label("MARKFIRED")
    b.ld_hl_label("FIRED")
    b.ld_de_nn(0)
    b.ld_a_c()
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_n(1)
    b.ld_hl_a()
    b.ret()

    b.label("COUNTHELD")
    b.xor_a()
    b.ld_mem_label_a("RU_CNT")
    if world.things:
        b.ld_hl_label("WHERE")
        b.ld_b_n(len(world.things))
        b.label("CH_LP")
        b.ld_a_hl()
        b.cp_n(CARRIED)
        b.jr_nz("CH_NX")
        b.ld_a_mem_label("RU_CNT")
        b.inc_a()
        b.ld_mem_label_a("RU_CNT")
        b.label("CH_NX")
        b.inc_hl()
        b.djnz("CH_LP")
    b.ld_a_mem_label("RU_CNT")
    b.ret()

    b.label("MSGROW")
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.push_hl()
    b.pop_de()
    b.add_hl_hl()
    b.add_hl_de()                    # x3, a pointer apiece
    b.ld_de_label("MSGTAB")
    b.add_hl_de()
    b.jp("LDPTR")


MESSAGES: dict[str, str] = {
    "WBANNER": "Silo 18. You are somewhere, and it is dark outside.",
    "WPROMPT": "> ",
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
    "MSGNOTERM": "There is no terminal here.",
    "MSGSITDOWN": "The screen wakes. Type a name to look it up, or LEAVE to "
                  "stand up again.",
    "TERMPROMPT": "archive> ",
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

    if world.rules:
        b.label("RULETAB")
        for rule in world.rules:
            length = 4 + 2 * len(rule.when) + 3 * len(rule.then)
            if length > 0xFF:
                raise ValueError(f"a rule of {length} bytes does not fit its "
                                 f"one-byte length; split it")
            b.db(length)
            b.db(1 if rule.once else 0)
            b.db(len(rule.when))
            b.db(len(rule.then))
            for op, arg in rule.when:
                b.db(op)
                b.db(arg)
            for op, arg, arg2 in rule.then:
                b.db(op)
                b.db(arg)
                b.db(arg2)
        b.db(0)                      # the table ends with a zero length

    if world.messages:
        b.label("MSGTAB")
        for index in range(len(world.messages)):
            b.fixup_word(f"RMSG{index}")
        for index, text in enumerate(world.messages):
            b.label(f"RMSG{index}")
            b.ascii(text)
            b.db(0)

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

    # Which room the archive terminal stands in, or 0xFF for a world with no
    # card behind it. One byte, and it is the whole of the wiring.
    b.label("TERMROOM")
    b.db(NOWHERE if world.terminal is None else world.terminal)


def _emit_ram(b: EZ80Builder, world: World, shared_console: bool = False) -> None:
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
    b.ds(world.flags)
    b.label("FIRED")
    b.ds(max(1, len(world.rules)))

    # Everything below is scratch that does not outlive a turn.
    scratch = ["VERB", "W1LEN", "W2LEN", "LKLEN", "NCARRIED", "RU_ONCE",
               "RU_NC", "RU_NA", "RU_OP", "RU_ARG", "RU_ARG2", "RU_CNT",
               "ATTERM"]
    if not shared_console:
        scratch += ["WRAPCOL", "INPLEN"]
    for name in scratch:
        b.label(name)
        b.db(0)
    for name in ("PTMP", "LKPTR", "RULEPTR", "RU_CUR"):
        b.label(name)
        b.d24(0)

    b.label("W1")
    b.ds(MAX_WORD_LEN + 1)
    b.label("W2")
    b.ds(MAX_WORD_LEN + 1)
    if not shared_console:
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
    end = builder.labels["FIRED"] + max(1, len(world.rules))
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
