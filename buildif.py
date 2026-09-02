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
from libagon import (
    FA_CREATE_ALWAYS,
    FA_OPEN_APPEND,
    FA_READ,
    FA_WRITE,
    MOS_API,
    MOS_FCLOSE,
    MOS_FOPEN,
    MOS_FREAD,
    MOS_FWRITE,
)
from libagonio import MOS_OUTCHAR
from libez80 import AGON_LOAD_ADDR, AGON_SRAM_TOP, EZ80Builder, agon_header
from libworld import CARRIED, DIRECTIONS, NOWHERE, World

#: An input line, and the most words a command may hold. Two is verb plus noun,
#: which is every command this understands; a third word is a phrasing it will
#: say it cannot read rather than one it silently ignores.
#:
#: In `libworld` because it also bounds a thing's `subject`, which `CONSULT`
#: copies into this same buffer.
MAX_INPUT_LEN = libworld.MAX_INPUT_LEN
#: The longest word either table holds, plus room to notice an over-long one.
#: `INVENTORY` is nine; a player who types more gets it truncated and then
#: named back at them, which is a legible failure rather than a wrong verb.
#:
#: Defined in `libworld` because it also bounds what an author may *name* a
#: thing, and `World.check` is where they find that out.
MAX_WORD_LEN = libworld.MAX_WORD_LEN

#: Room row: name pointer, description pointer, then one byte an exit.
ROOM_STRIDE = 3 + 3 + len(DIRECTIONS)
#: Thing row: name pointer, description pointer, starting place, portable.
THING_STRIDE = 3 + 3 + 1 + 1
#: Person row: description pointer, default-line pointer, starting room.
#: There is no name pointer - a person's name is only ever typed, never
#: printed, because the description is the sentence that puts them in a room.
PERSON_STRIDE = 3 + 3 + 1
#: Dialogue row: person, topic, gate flag, flag to set, text pointer.
LINE_STRIDE = 1 + 1 + 1 + 1 + 3

#: Words dropped between the ones that mean something, so that `ASK MARNES
#: ABOUT ALLISON` reaches the same three slots as `ASK MARNES ALLISON`.
#:
#: Dropping them in the splitter rather than the handlers is what keeps the
#: slot count at three. `ASK` needs a verb, a person and a topic, and every
#: natural phrasing of that has a preposition in the middle - so either the
#: splitter loses it or every command becomes four words wide, and four words
#: costs `TAKE` and `DROP` a slot neither has ever used.
NOISE_WORDS: tuple[str, ...] = ("ABOUT", "THE", "A", "AN", "TO", "AT", "FOR")

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
#: The verb that reads a thing's description. Every thing row has carried a
#: pointer to one since this file was written and nothing ever printed it - the
#: text was emitted into the image, indexed, and unreachable. This is the
#: cheaper of the two ways to resolve that; the other was to stop emitting it,
#: and `worlds.py` had already written four descriptions worth reading.
V_EXAMINE = V_USE + 1
#: Ask a *person* about a topic. `ASK` was an alias for `USE` and is not any
#: more, and `CONSULT <thing>` is why it could stop being one: holding a piece
#: of paper up to the archive already has a verb, so `ASK` is free to mean the
#: other way of finding something out. A machine and a person are asked in the
#: same words and answer out of different tables, and one verb would have had
#: to guess which was meant.
V_ASK = V_EXAMINE + 1
#: The overlay to a slot on the card, and back. Neither is a turn: the rules
#: do not run and the clock does not tick, so a game saved and restored plays
#: on exactly as one that was not - which `tests/test_save.py` holds it to.
V_SAVE = V_ASK + 1
V_RESTORE = V_SAVE + 1

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
              ("USE", V_USE), ("CONSULT", V_USE),
              ("EXAMINE", V_EXAMINE), ("X", V_EXAMINE),
              ("READ", V_EXAMINE),
              ("ASK", V_ASK), ("TALK", V_ASK),
              ("SAVE", V_SAVE), ("RESTORE", V_RESTORE), ("LOAD", V_RESTORE)]

    nouns = [(thing.name.upper(), index)
             for index, thing in enumerate(world.things)]
    return verbs, nouns


def _people_words(world: World) -> list[tuple[str, int]]:
    """Who can be named. A separate table from the nouns on purpose.

    `libworld.World.check` refuses a word that is both, so nothing here is
    ambiguous - but the tables stay apart because `TAKE` and `ASK` want
    different failures. Naming a person to `TAKE` should say the machine does
    not know the word, not that a deputy is too heavy.
    """
    return [(person.name.upper(), index)
            for index, person in enumerate(world.people)]


def _topic_words(world: World) -> list[tuple[str, int]]:
    """Every word that reaches a topic, flattened out of the topic list.

    One topic has several words because a player types `PUMP` for what the
    archive files as `Cistern Pump Failure, Level 142`. `check` refuses two
    topics claiming one word, so this table resolves to exactly one id.
    """
    return [(word.upper(), index)
            for index, topic in enumerate(world.topics)
            for word in topic.words]


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
    emit_reset(b, world)
    b.ld_hl_label("WBANNER")
    b.call("PRWRAP")
    b.call("PRNL")
    b.call("DESCRIBE")

    # --- the turn loop --------------------------------------------------------
    b.label("TURN")
    b.call("RULES_RUN")              # the world reacts before it asks again
    # Where a command that is not a turn comes back to: an empty line, a
    # save, a restore. Nothing the world can notice happened, so the rules do
    # not run and the clock does not tick.
    b.label("NOTURN")
    b.call("PRNL")
    b.ld_hl_label("WPROMPT")
    b.call("PRSTR")
    b.call("READ_INPUT")
    emit_dispatch(b, quit_label="BYE")
    emit_world_routines(b, world, ask_label="USE_NOCARD")

    # Where `CONSULT <thing>` goes in a binary with no card behind it. The
    # subject is in `INPBUF` by now and there is nothing to read it, which is
    # true of every room in this world rather than of this one.
    b.label("USE_NOCARD")
    b.ld_hl_label("MSGNOTERM")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("TURN")

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
    b.jp_z("NOTURN")                 # an empty line is not a turn

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
    b.cp_n(V_EXAMINE)
    b.jp_z("DO_EXAM")
    b.cp_n(V_ASK)
    b.jp_z("DO_ASK")
    b.cp_n(V_SAVE)
    b.jp_z("DO_SAVE")
    b.cp_n(V_RESTORE)
    b.jp_z("DO_RESTORE")
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
    """Put every thing back where it starts, and read what the card remembers.

    Exposed for the merged build. `LOGCOUNT` is here rather than in `START`
    because both programs start a game the same way, and the archive's log
    is the one thing on the card that outlives one: a fresh game on a card
    that already holds a log starts with `LOGGED` at its length, which is
    how the Voice knows it has met this player before.
    """
    _emit_reset_things(b, world)
    b.call("LOGCOUNT")


def emit_world_routines(b: EZ80Builder, world: World,
                        ask_label: str) -> None:
    """Everything a turn needs, and nothing about how the program starts.

    Split out so the oracle binary can hold a world as well. That direction
    round is the one that fits: the world is 4,050 bytes and the oracle 38,912,
    so a terminal standing in a room is the small thing inside the large one
    rather than the other way about - which is not how issue #62 pictured it.

    `ask_label` is where `CONSULT <thing>` goes once the subject is in
    `INPBUF`: the card's ask path in the merged binary, and a stub that
    says there is no terminal in the standalone one. Passed in rather than
    defined twice, because `Z80Builder.label` overwrites silently and
    `test_the_two_programs_define_no_label_twice` exists to say so.
    """
    _emit_go(b, world)
    _emit_take_drop(b, world, ask_label)
    _emit_describe(b, world)
    _emit_split(b)
    _emit_upper(b)
    _emit_lookup(b)
    _emit_prword(b)
    _emit_noun(b)
    _emit_room_row(b)
    _emit_thing_row(b)
    _emit_subj_row(b)
    _emit_where_ptr(b)
    _ldptr(b)
    _emit_ask(b, world)
    _emit_attention(b, world)
    _emit_rules(b, world)
    _emit_save_restore(b, world)


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
    if world.things:
        b.ld_hl_label("INITWHERE")
        b.ld_de_label("WHERE")
        b.ld_b_n(len(world.things))
        b.label("RESET_LP")
        b.ld_a_hl()
        b.ld_de_a()
        b.inc_hl()
        b.inc_de()
        b.djnz("RESET_LP")

    if world.people:
        b.ld_hl_label("INITPWHERE")
        b.ld_de_label("PWHERE")
        b.ld_b_n(len(world.people))
        b.label("RESETP_LP")
        b.ld_a_hl()
        b.ld_de_a()
        b.inc_hl()
        b.inc_de()
        b.djnz("RESETP_LP")

    if world.topics:
        # Which records start sealed is authored; which are altered is not,
        # so the first is copied from a table and the second is cleared.
        b.ld_hl_label("INITSEALED")
        b.ld_de_label("SEALED")
        b.ld_b_n(len(world.topics))
        b.label("RESETS_LP")
        b.ld_a_hl()
        b.ld_de_a()
        b.inc_hl()
        b.inc_de()
        b.djnz("RESETS_LP")
        b.ld_hl_label("ALTERED")
        b.ld_b_n(len(world.topics))
        b.xor_a()
        b.label("RESETA_LP")
        b.ld_hl_a()
        b.inc_hl()
        b.djnz("RESETA_LP")


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


def _emit_subj_row(b: EZ80Builder) -> None:
    """HL = &SUBJECTS[A], a three-byte pointer per thing.

    Its own table rather than a fourth field on the thing row, which would
    take the stride from eight to eleven and cost `THINGROW` its three
    `ADD HL,HL`. Three is `2A + A`, and the push is cheaper than a multiply.
    """
    b.label("SUBJROW")
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.push_hl()
    b.add_hl_hl()
    b.pop_de()
    b.add_hl_de()
    b.ld_de_label("SUBJECTS")
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
    """INPBUF -> W1, W2 and W3, uppercased, with the noise words dropped.

    Three words, because `ASK MARNES ABOUT ALLISON` is the shortest natural
    phrasing of the one command that names two things. `TAKE` and `DROP` still
    read two and a third word to them is a phrasing this declines by name
    rather than one it quietly does half of.

    The noise words come out in the splitter rather than in `DO_ASK`, which is
    what keeps the count at three: a preposition sits between the person and
    the topic in every wording anybody types, so either it is dropped here or
    every slot in the program widens by one to carry it.
    """
    b.label("SPLIT")
    b.xor_a()
    b.ld_mem_label_a("W1LEN")
    b.ld_mem_label_a("W2LEN")
    b.ld_mem_label_a("W3LEN")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.ret_z()
    b.ld_b_a()
    b.ld_hl_label("INPBUF")

    for slot in ("W1", "W2", "W3"):
        b.ld_de_label(slot)
        b.call("SP_WORD")
        b.ld_a_c()
        b.ld_mem_label_a(f"{slot}LEN")
    b.ret()

    # SP_WORD: one word that is not noise, into DE. HL/B advance over the
    # input as `SP_ONE` leaves them; a noise word is copied and then written
    # over by the next, which is why the destination is reloaded each time.
    b.label("SP_WORD")
    b.ld_mem_label_de("SPDST")

    b.label("SPW_TRY")
    b.ld_de_mem_label("SPDST")
    b.call("SP_ONE")
    b.ld_a_c()
    b.or_a()
    b.ret_z()                        # the line ran out
    b.push_hl()
    b.push_bc()
    b.ld_hl_label("NOISE")
    b.ld_de_mem_label("SPDST")
    b.ld_a_c()
    b.call("LOOKUP")
    b.pop_bc()                       # C is the length again
    b.pop_hl()                       # and HL the place in the line
    b.ret_c()                        # a miss: this word means something
    b.jr("SPW_TRY")

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

    if world.things:
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

    if not world.people:
        b.ret()
        return

    # Whoever is standing here, by their description rather than their name.
    # "You can see Marnes." is what a thing gets; a person gets the sentence
    # that puts them in the room, which is why `Person` has no name pointer.
    b.ld_b_n(len(world.people))
    b.ld_c_n(0)
    b.label("LP_LP")
    b.push_bc()
    b.ld_a_c()
    b.call("PWHEREPTR")
    b.ld_a_hl()
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jr_nz("LP_NEXT")
    b.ld_a_c()
    b.call("PERSONROW")
    b.call("LDPTR")
    b.call("PRWRAP")
    b.call("PRNL")
    b.label("LP_NEXT")
    b.pop_bc()
    b.inc_c()
    b.djnz("LP_LP")
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


def _emit_take_drop(b: EZ80Builder, world: World,
                    ask_label: str) -> None:
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

    # DO_EXAM: the thing's description, which is offset 3 of its row.
    #
    # Carried or in the room, both - a player who has picked something up has
    # not stopped being able to look at it, and refusing there would be a
    # distinction the game made and nothing else did.
    b.label("DO_EXAM")
    b.call("NOUNID")
    b.jp_c("BADNOUN")
    b.ld_c_a()
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jp_z("EX_SHOW")
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jp_nz("TK_ABSENT")

    b.label("EX_SHOW")
    b.ld_a_c()
    b.call("THINGROW")
    b.ld_de_nn(3)
    b.add_hl_de()
    b.call("LDPTR")
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
    # `CONSULT LEDGER` rather than `CONSULT`: the thing is the question.
    #
    # Ten thousand people are on the card and the world can carry none of
    # them. What it can carry is a *name* - on a ledger, a work order, a death
    # notice - and a thing's `subject` is that piece of paper. Consulting one
    # types it at the archive on the player's behalf, so the only entries
    # reachable are the ones somebody has physically found a reference to.
    b.ld_a_mem_label("W2LEN")
    b.or_a()
    b.jp_z("DO_USE_BARE")

    b.call("NOUNID")
    b.jp_c("BADNOUN")
    b.ld_c_a()
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jp_nz("TK_HAVENOT")
    b.ld_a_mem_label("HERE")
    b.ld_hl_label("TERMROOM")
    b.cp_hl()
    b.jp_nz("US_NOTERM")

    b.ld_a_c()
    b.call("SUBJROW")
    b.call("LDPTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_z("US_NOSUBJ")              # an empty string is "no subject"

    # The subject into INPBUF, which is the line the archive is about to read.
    # `World.check` refuses a subject longer than the console, so the cap here
    # is the second of two rather than the only one.
    b.ld_de_label("INPBUF")
    b.ld_c_n(0)
    b.label("US_COPY")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("US_ASK")
    b.ld_de_a()
    b.inc_hl()
    b.inc_de()
    b.inc_c()
    b.ld_a_c()
    b.cp_n(MAX_INPUT_LEN)
    b.jr_nc("US_ASK")
    b.jr("US_COPY")

    b.label("US_ASK")
    b.ld_a_c()
    b.ld_mem_label_a("INPLEN")
    b.jp(ask_label)

    for label, message in (("US_NOTERM", "MSGNOTERM"),
                           ("US_NOSUBJ", "MSGNOSUBJ")):
        b.label(label)
        b.ld_hl_label(message)
        b.call("PRWRAP")
        b.call("PRNL")
        b.jp("TURN")

    b.label("DO_USE_BARE")
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


def _emit_ask(b: EZ80Builder, world: World) -> None:
    """`ASK <person> ABOUT <topic>` - the oracle's shape, in the image.

    Three lookups and a linear scan, and every byte of it is resident. That is
    the same asymmetry the turn loop is built on: the card costs ~4,600 bytes
    of I/O and ~370,000 instructions to answer a question, and a person
    answers out of a table for nothing. A world where talking cost what
    consulting the archive costs would be a world nobody talked in.

    The scan takes the first row whose gate is satisfied, so the author writes
    the most specific line first and `libworld.World.check` refuses a row that
    an earlier one has already shadowed - which is the mistake that produces
    dialogue nobody can ever hear.
    """
    b.label("DO_ASK")
    if not world.people or not world.topics:
        # Nobody to ask, or nothing to ask about. Said plainly rather than
        # left as an unknown verb, because `ASK` is in the table either way.
        b.ld_hl_label("MSGNOASK")
        b.call("PRWRAP")
        b.call("PRNL")
        b.jp("TURN")
        return

    b.ld_a_mem_label("W2LEN")
    b.or_a()
    b.jp_z("ASK_WHO")
    b.ld_hl_label("PEOPLEW")
    b.ld_de_label("W2")
    b.ld_a_mem_label("W2LEN")
    b.call("LOOKUP")
    b.jp_c("ASK_NOBODY")
    b.ld_mem_label_a("ASKWHO")

    b.call("PWHEREPTR")
    b.ld_a_hl()
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jp_nz("ASK_ABSENT")

    b.ld_a_mem_label("W3LEN")
    b.or_a()
    b.jp_z("ASK_WHAT")
    b.ld_hl_label("TOPICW")
    b.ld_de_label("W3")
    b.ld_a_mem_label("W3LEN")
    b.call("LOOKUP")
    b.jp_c("ASK_NOTOPIC")
    b.ld_mem_label_a("ASKTOP")

    # Marked asked before the answer is chosen, and marked whatever the answer
    # turns out to be. `C_ASKED` records that the subject came up, not that it
    # was productively answered - a deflection is a thing the player learned.
    b.call("MARK_ASKED")
    b.call("SAY")
    b.jp("TURN")

    for label, message in (("ASK_WHO", "MSGASKWHO"),
                           ("ASK_ABSENT", "MSGASKGONE"),
                           ("ASK_WHAT", "MSGASKWHAT")):
        b.label(label)
        b.ld_hl_label(message)
        b.call("PRWRAP")
        b.call("PRNL")
        b.jp("TURN")

    for label, message, word, length in (
            ("ASK_NOBODY", "MSGNOONE", "W2", "W2LEN"),
            ("ASK_NOTOPIC", "MSGNOUN", "W3", "W3LEN")):
        b.label(label)
        b.ld_hl_label(message)
        b.call("PRSTR")
        b.ld_hl_label(word)
        b.ld_a_mem_label(length)
        b.call("PRWORD")
        b.ld_hl_label("MSGQUOTE")
        b.call("PRSTR")
        b.jp("TURN")

    # SAY: the line for (ASKWHO, ASKTOP), or the person's default.
    #
    # IX walks the rows because the gate and the flag it sets both want
    # `FLAGPTR`, which returns in HL - so the row pointer cannot live there.
    b.label("SAY")
    b.ld_ix_label("DIALOG")

    b.label("SAY_LP")
    b.ld_a_ixd(0)
    b.cp_n(libworld.NONE)
    b.jr_z("SAY_DEF")
    b.ld_hl_label("ASKWHO")
    b.cp_hl()
    b.jr_nz("SAY_NEXT")
    b.ld_a_ixd(1)
    b.ld_hl_label("ASKTOP")
    b.cp_hl()
    b.jr_nz("SAY_NEXT")

    b.ld_a_ixd(2)                    # the gate
    b.cp_n(libworld.NONE)
    b.jr_z("SAY_HIT")
    b.call("FLAGPTR")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("SAY_NEXT")

    b.label("SAY_HIT")
    b.ld_a_ixd(3)                    # the flag speaking it sets
    b.cp_n(libworld.NONE)
    b.jr_z("SAY_PRINT")
    b.call("FLAGPTR")
    b.ld_a_n(1)
    b.ld_hl_a()

    b.label("SAY_PRINT")
    b.ld_hl_ixd(4)
    b.call("PRWRAP")
    b.jp("PRNL")

    b.label("SAY_NEXT")
    b.ld_de_nn(LINE_STRIDE)
    b.add_ix_de()
    b.jr("SAY_LP")

    b.label("SAY_DEF")
    b.ld_a_mem_label("ASKWHO")
    b.call("PERSONROW")
    b.ld_de_nn(3)                    # past the description, to the default
    b.add_hl_de()
    b.call("LDPTR")
    b.call("PRWRAP")
    b.jp("PRNL")


def _emit_attention(b: EZ80Builder, world: World) -> None:
    """`ASKED`, `HEAT` and `PWHERE`: the three arrays a question moves.

    `ASKED` only ever grows. No action clears it and nothing in the rule set
    can, which is deliberate rather than an omission - `A_CLEAR` can put a
    flag back, and a mystery whose record of what the player had been told
    could be rewound would not be fair. It is the one monotone thing in the
    overlay.

    `HEAT` saturates at both ends rather than wrapping. A counter that rolled
    over from 255 to 0 would hand the player an escape from every consequence
    by asking enough questions, which is exactly backwards.
    """
    if world.people:
        # x7, as one doubling too many and a subtraction. `THINGROW` gets
        # three doublings and `ROOMROW` an add; seven is the stride that has
        # neither, and the shift-and-subtract is still cheaper than a multiply
        # the eZ80 would have to set up registers for.
        b.label("PERSONROW")
        b.ld_hl_nn(0)
        b.ld_l_a()
        b.push_hl()
        b.pop_de()
        b.add_hl_hl()
        b.add_hl_hl()
        b.add_hl_hl()                    # x8
        b.or_a()                         # clear the carry the doublings left
        b.sbc_hl_de()                    # x7, which is PERSON_STRIDE
        b.ld_de_label("PEOPLE")
        b.add_hl_de()
        b.ret()

    b.label("ASKEDPTR")
    b.ld_hl_label("ASKED")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ret()

    # The Voice's two arrays, a byte a topic like `ASKED` and read the same
    # way: `SEALED` says the archive declines, `ALTERED` says it lies.
    b.label("SEALEDPTR")
    b.ld_hl_label("SEALED")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ret()

    b.label("ALTEREDPTR")
    b.ld_hl_label("ALTERED")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ret()

    b.label("PWHEREPTR")
    b.ld_hl_label("PWHERE")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ret()

    # MARK_ASKED: the topic in ASKTOP, counted rather than flagged. Only the
    # zero test is exposed as `C_ASKED`; the count is kept because the byte is
    # spent either way and a threshold opcode would want it already there.
    b.label("MARK_ASKED")
    b.ld_a_mem_label("ASKTOP")
    b.call("ASKEDPTR")
    b.ld_a_hl()
    b.inc_a()
    b.ret_z()                        # 255 was already as asked as it gets
    b.ld_hl_a()
    b.ret()

    b.label("ADDHEAT")
    b.ld_c_a()
    b.ld_a_mem_label("HEAT")
    b.add_a_c()
    b.jr_nc("AH_OK")
    b.ld_a_n(0xFF)
    b.label("AH_OK")
    b.ld_mem_label_a("HEAT")
    b.ret()

    b.label("SUBHEAT")
    b.ld_c_a()
    b.ld_a_mem_label("HEAT")
    b.sub_c()
    b.jr_nc("SH_OK")
    b.xor_a()
    b.label("SH_OK")
    b.ld_mem_label_a("HEAT")
    b.ret()


def _emit_rules(b: EZ80Builder, world: World) -> None:
    """Check every rule; fire the ones whose conditions all hold.

    The step past a path, and a small one. A graph walk composes - follow this,
    then that - and stops at conjunction. A flat list of conditions ANDed
    together is the least that does not, and `IF.md` reports which of the four
    shapes `data/silo/README.md` names it actually closes. It is three.

    A rule is length-prefixed so that skipping one is an addition rather than a
    walk over its parts, which is what the first version of this did and got
    wrong twice.

    `RULES_RUN` is one pass and then a tick. The clock counts turns already
    taken, so a rule on turn N reads N after the N-th command and the opening
    pass reads zero - the order `World._settle` models, and the one thing the
    two have to agree on for a deadline to mean the same thing on both. It
    saturates at 255 rather than wrapping, for the reason `HEAT` does: a
    clock that rolled over would hand back every deadline that had passed.
    """
    b.label("RULES_RUN")
    if world.rules:
        b.call("RULES_PASS")
    b.ld_a_mem_label("CLOCK")
    b.cp_n(255)
    b.ret_z()
    b.inc_a()
    b.ld_mem_label_a("CLOCK")
    b.ret()
    if not world.rules:
        return

    b.label("RULES_PASS")
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
    b.jp_z("RT_YES")
    b.jp("RT_NO")

    b.label("RT_HAVE")
    b.cp_n(libworld.C_HAVE)
    b.jr_nz("RT_HERE")
    b.ld_a_mem_label("RU_ARG")
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.cp_n(CARRIED)
    b.jp_z("RT_YES")
    b.jp("RT_NO")

    b.label("RT_HERE")
    b.cp_n(libworld.C_HERE)
    b.jr_nz("RT_FLAG")
    b.ld_a_mem_label("RU_ARG")
    b.call("WHEREPTR")
    b.ld_a_hl()
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jp_z("RT_YES")
    b.jp("RT_NO")

    b.label("RT_FLAG")
    b.cp_n(libworld.C_FLAG)
    b.jr_nz("RT_NFLAG")
    b.ld_a_mem_label("RU_ARG")
    b.call("FLAGPTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_nz("RT_YES")
    b.jp("RT_NO")

    b.label("RT_NFLAG")
    b.cp_n(libworld.C_NFLAG)
    b.jr_nz("RT_COUNT")
    b.ld_a_mem_label("RU_ARG")
    b.call("FLAGPTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_z("RT_YES")
    b.jp("RT_NO")

    # The count a path cannot do, and the reason this opcode exists at all.
    b.label("RT_COUNT")
    b.cp_n(libworld.C_CARRYING)
    b.jr_nz("RT_ASKED")
    b.call("COUNTHELD")
    b.ld_hl_label("RU_ARG")
    b.cp_hl()
    b.jp_c("RT_NO")                  # carrying fewer than the rule asked for
    b.jp("RT_YES")

    # The three the archive and the people put there. `C_ASKED` is what makes
    # a question part of the plot rather than a lookup beside it: the world
    # can react to what the player wanted to know, which is the one thing a
    # stateless card could never be asked.
    b.label("RT_ASKED")
    b.cp_n(libworld.C_ASKED)
    b.jr_nz("RT_HEAT")
    b.ld_a_mem_label("RU_ARG")
    b.call("ASKEDPTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_nz("RT_YES")
    b.jp("RT_NO")

    b.label("RT_HEAT")
    b.cp_n(libworld.C_HEAT)
    b.jr_nz("RT_TURN")
    b.ld_a_mem_label("HEAT")
    b.ld_hl_label("RU_ARG")
    b.cp_hl()
    b.jp_c("RT_NO")                  # quieter than the rule was watching for
    b.jp("RT_YES")

    # The clock, read the same way as attention: at or past the deadline.
    # `CLOCK` holds the turns already taken, because `RULES_RUN` ticks it
    # after the pass and not before - see `_emit_rules`.
    b.label("RT_TURN")
    b.cp_n(libworld.C_TURN)
    b.jr_nz("RT_LOGGED")
    b.ld_a_mem_label("CLOCK")
    b.ld_hl_label("RU_ARG")
    b.cp_hl()
    b.jp_c("RT_NO")                  # not yet
    b.jp("RT_YES")

    # The archive's log, which is the one counter here that a previous game
    # can have left behind. `LOGGED` is read off the card when a game starts
    # and `LOGAPPEND` keeps it in step from then on.
    b.label("RT_LOGGED")
    b.cp_n(libworld.C_LOGGED)
    b.jr_nz("RT_SEALED")
    b.ld_a_mem_label("LOGGED")
    b.ld_hl_label("RU_ARG")
    b.cp_hl()
    b.jp_c("RT_NO")
    b.jp("RT_YES")

    # What the archive is doing to a record, which a rule may read back so
    # that a person can react to a seal the player has not yet run into.
    b.label("RT_SEALED")
    b.cp_n(libworld.C_SEALED)
    b.jr_nz("RT_ALTERED")
    b.ld_a_mem_label("RU_ARG")
    b.call("SEALEDPTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_nz("RT_YES")
    b.jp("RT_NO")

    b.label("RT_ALTERED")
    b.cp_n(libworld.C_ALTERED)
    b.jr_nz("RT_WITH")
    b.ld_a_mem_label("RU_ARG")
    b.call("ALTEREDPTR")
    b.ld_a_hl()
    b.or_a()
    b.jp_nz("RT_YES")
    b.jp("RT_NO")

    b.label("RT_WITH")
    b.cp_n(libworld.C_WITH)
    b.jp_nz("RT_NO")
    b.ld_a_mem_label("RU_ARG")
    b.call("PWHEREPTR")
    b.ld_a_hl()
    b.ld_hl_label("HERE")
    b.cp_hl()
    b.jp_z("RT_YES")

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
    b.jr_nz("RD_HEAT")
    b.ld_a_mem_label("RU_ARG")
    b.call("WHEREPTR")
    b.ld_a_mem_label("RU_ARG2")
    b.ld_hl_a()
    b.ret()

    # Attention, up and down. Both exist because a world with only `A_HEAT`
    # is one the player can only lose: there has to be somewhere to lie low,
    # or the counter is a countdown wearing a different name.
    b.label("RD_HEAT")
    b.cp_n(libworld.A_HEAT)
    b.jr_nz("RD_COOL")
    b.ld_a_mem_label("RU_ARG")
    b.jp("ADDHEAT")

    b.label("RD_COOL")
    b.cp_n(libworld.A_COOL)
    b.jr_nz("RD_SEND")
    b.ld_a_mem_label("RU_ARG")
    b.jp("SUBHEAT")

    b.label("RD_SEND")
    b.cp_n(libworld.A_SEND)
    b.jr_nz("RD_SEAL")
    b.ld_a_mem_label("RU_ARG")
    b.call("PWHEREPTR")
    b.ld_a_mem_label("RU_ARG2")
    b.ld_hl_a()
    b.ret()

    # The Voice acting on the record. Four opcodes and two arrays: a byte
    # goes to 1 or 0, and the archive reads it the next time it is asked.
    b.label("RD_SEAL")
    b.cp_n(libworld.A_SEAL)
    b.jr_nz("RD_UNSEAL")
    b.ld_a_mem_label("RU_ARG")
    b.call("SEALEDPTR")
    b.ld_a_n(1)
    b.ld_hl_a()
    b.ret()

    b.label("RD_UNSEAL")
    b.cp_n(libworld.A_UNSEAL)
    b.jr_nz("RD_ALTER")
    b.ld_a_mem_label("RU_ARG")
    b.call("SEALEDPTR")
    b.xor_a()
    b.ld_hl_a()
    b.ret()

    b.label("RD_ALTER")
    b.cp_n(libworld.A_ALTER)
    b.jr_nz("RD_TRUTH")
    b.ld_a_mem_label("RU_ARG")
    b.call("ALTEREDPTR")
    b.ld_a_n(1)
    b.ld_hl_a()
    b.ret()

    b.label("RD_TRUTH")
    b.cp_n(libworld.A_TRUTH)
    b.ret_nz()
    b.ld_a_mem_label("RU_ARG")
    b.call("ALTEREDPTR")
    b.xor_a()
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
    "MSGNOSUBJ": "The screen has nothing to say about that.",
    "MSGSITDOWN": "The screen wakes. Type a name to look it up, or LEAVE to "
                  "stand up again.",
    "TERMPROMPT": "archive> ",
    "MSGNOASK": "There is nobody here to ask.",
    "MSGNOONE": "You do not know anybody called '",
    "MSGASKWHO": "Who do you want to ask?",
    "MSGASKWHAT": "What do you want to ask about?",
    "MSGASKGONE": "They are not here.",
    "MSGSAVED": "Saved.",
    "MSGRESTORED": "Restored.",
    "MSGSLOT": "Which slot? 1 to 9.",
    "MSGNOSAVE": "There is no saved game in that slot.",
    "MSGBADSAVE": "That is not a saved game for this silo.",
    "MSGNOWRITE": "The card would not take it.",
}


def _emit_save_restore(b: EZ80Builder, world: World) -> None:
    """`SAVE [n]`, `RESTORE [n]`, and the archive's log.

    A saved game is a four-byte header and the overlay, copied into `SAVEBUF`
    so that it goes to the card in one `mos_fwrite` and comes back in one
    `mos_fread`. The header is `SV` and `World.stamp`, and a restore that
    finds any other header - or fewer bytes than the file should hold -
    touches nothing, because the overlay is a run of bytes with no names in
    it and a save from another world would load without complaint.

    Neither verb is a turn. Both come back through `NOTURN`, so the rules do
    not run and the clock does not tick, and a game saved and restored plays
    on as one that was not. `ATTERM` is cleared on a restore because standing
    up is what a restore does.

    The log is different: `LOGAPPEND` puts `(CLOCK, topic)` on the end of
    `SILO.LOG` for every question the archive sees, `LOGCOUNT` reads the
    length back when a game starts, and `C_LOGGED` lets a rule read it.
    `LOGGED` is outside the overlay on purpose - the file is the truth and
    the byte is its length - so a restore leaves it alone: the archive does
    not forget what it was asked because the player wound the clock back.
    """
    size = 4 + world.overlay_bytes
    prefix = len(world.save_name)

    b.label("DO_SAVE")
    b.call("SLOT")
    b.jp_c("NOTURN")
    b.ld_hl_label("SAVEHDR")
    b.ld_de_label("SAVEBUF")
    b.ld_bc_nn(4)
    b.ldir()
    b.ld_hl_label("HERE")
    b.ld_de_label("SAVEOVL")
    b.ld_bc_nn(world.overlay_bytes)
    b.ldir()
    b.ld_hl_label("SAVNAME")
    b.ld_c_n(FA_WRITE | FA_CREATE_ALWAYS)
    b.ld_a_n(MOS_FOPEN)
    b.rst(MOS_API)
    b.or_a()
    b.jr_z("SV_FAIL")
    b.ld_mem_label_a("SAVEH")
    b.ld_c_a()
    b.ld_hl_label("SAVEBUF")
    b.ld_de_nn(size)
    b.ld_a_n(MOS_FWRITE)
    b.rst(MOS_API)
    b.push_de()                      # DE = bytes written
    b.call("SAVECLOSE")
    b.pop_de()
    b.ld_hl_nn(size)
    b.or_a()
    b.sbc_hl_de()
    b.jr_nz("SV_FAIL")               # a short write is a failed save
    b.ld_hl_label("MSGSAVED")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("NOTURN")

    b.label("SV_FAIL")
    b.ld_hl_label("MSGNOWRITE")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("NOTURN")

    b.label("SAVECLOSE")
    b.ld_a_mem_label("SAVEH")
    b.ld_c_a()
    b.ld_a_n(MOS_FCLOSE)
    b.rst(MOS_API)
    b.ret()

    b.label("DO_RESTORE")
    b.call("SLOT")
    b.jp_c("NOTURN")
    b.ld_hl_label("SAVNAME")
    b.ld_c_n(FA_READ)
    b.ld_a_n(MOS_FOPEN)
    b.rst(MOS_API)
    b.or_a()
    b.jr_z("RS_NONE")
    b.ld_mem_label_a("SAVEH")
    b.ld_c_a()
    b.ld_hl_label("SAVEBUF")
    b.ld_de_nn(size)
    b.ld_a_n(MOS_FREAD)
    b.rst(MOS_API)
    b.push_de()                      # DE = bytes read
    b.call("SAVECLOSE")
    b.pop_de()
    b.ld_hl_nn(size)
    b.or_a()
    b.sbc_hl_de()
    b.jr_nz("RS_BAD")                # short, so not one of ours
    b.ld_hl_label("SAVEHDR")
    b.ld_de_label("SAVEBUF")
    b.ld_b_n(4)
    b.label("RS_CMP")
    b.ld_a_de()
    b.cp_hl()
    b.jr_nz("RS_BAD")                # another world's, or not a save at all
    b.inc_hl()
    b.inc_de()
    b.djnz("RS_CMP")
    b.ld_hl_label("SAVEOVL")
    b.ld_de_label("HERE")
    b.ld_bc_nn(world.overlay_bytes)
    b.ldir()
    b.xor_a()
    b.ld_mem_label_a("ATTERM")       # standing up is what a restore does
    b.ld_hl_label("MSGRESTORED")
    b.call("PRWRAP")
    b.call("PRNL")
    b.call("DESCRIBE")
    b.jp("NOTURN")

    b.label("RS_NONE")
    b.ld_hl_label("MSGNOSAVE")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("NOTURN")

    b.label("RS_BAD")
    b.ld_hl_label("MSGBADSAVE")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp("NOTURN")

    # SLOT: W2 -> the digit in SAVNAME. No word means slot 1. Carry set,
    # and the complaint already printed, for anything that is not 1 to 9.
    b.label("SLOT")
    b.ld_a_mem_label("W2LEN")
    b.or_a()
    b.jr_z("SL_ONE")
    b.cp_n(1)
    b.jr_nz("SL_BAD")
    b.ld_a_mem_label("W2")
    b.cp_n(ord("1"))
    b.jr_c("SL_BAD")
    b.cp_n(ord("9") + 1)
    b.jr_nc("SL_BAD")
    b.jr("SL_SET")
    b.label("SL_ONE")
    b.ld_a_n(ord("1"))
    b.label("SL_SET")
    b.ld_mem_label_a("SAVNAME", prefix)
    b.or_a()
    b.ret()
    b.label("SL_BAD")
    b.ld_hl_label("MSGSLOT")
    b.call("PRWRAP")
    b.call("PRNL")
    b.scf()
    b.ret()

    # LOGCOUNT: LOGGED = min(255, records in the log), or 0 with no log. Two
    # bytes a read rather than the file into a buffer, because the count is
    # all that is wanted and nothing here has to know how long a log can get.
    b.label("LOGCOUNT")
    b.xor_a()
    b.ld_mem_label_a("LOGGED")
    b.ld_hl_label("LOGNAME")
    b.ld_c_n(FA_READ)
    b.ld_a_n(MOS_FOPEN)
    b.rst(MOS_API)
    b.or_a()
    b.ret_z()
    b.ld_mem_label_a("SAVEH")
    b.label("LC_LP")
    b.ld_a_mem_label("SAVEH")
    b.ld_c_a()
    b.ld_hl_label("LOGREC")
    b.ld_de_nn(2)
    b.ld_a_n(MOS_FREAD)
    b.rst(MOS_API)
    b.ld_a_e()
    b.cp_n(2)
    b.jr_nz("LC_END")                # short read: the end of the log
    b.ld_a_mem_label("LOGGED")
    b.cp_n(255)
    b.jr_z("LC_LP")
    b.inc_a()
    b.ld_mem_label_a("LOGGED")
    b.jr("LC_LP")
    b.label("LC_END")
    b.jp("SAVECLOSE")

    # LOGAPPEND: (CLOCK, LOGTOP) onto the end of the log. Only the merged
    # build calls it - the standalone world has no archive to be asked - but
    # it is emitted with the rest so the two programs read one definition.
    b.label("LOGAPPEND")
    b.ld_a_mem_label("CLOCK")
    b.ld_mem_label_a("LOGREC")
    b.ld_a_mem_label("LOGTOP")
    b.ld_mem_label_a("LOGREC", 1)
    b.ld_hl_label("LOGNAME")
    b.ld_c_n(FA_WRITE | FA_OPEN_APPEND)
    b.ld_a_n(MOS_FOPEN)
    b.rst(MOS_API)
    b.or_a()
    b.ret_z()                        # a card that will not take it: unlogged
    b.ld_mem_label_a("SAVEH")
    b.ld_c_a()
    b.ld_hl_label("LOGREC")
    b.ld_de_nn(2)
    b.ld_a_n(MOS_FWRITE)
    b.rst(MOS_API)
    b.call("SAVECLOSE")
    b.ld_a_mem_label("LOGGED")
    b.cp_n(255)
    b.ret_z()
    b.inc_a()
    b.ld_mem_label_a("LOGGED")
    b.ret()

    # The header a save file starts with. `SV`, then the world's stamp.
    b.label("SAVEHDR")
    b.db(ord("S"))
    b.db(ord("V"))
    b.db(world.stamp & 0xFF)
    b.db(world.stamp >> 8)


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

    # One pointer a thing, to the line `CONSULT` types at the archive on its
    # behalf. A thing that names nothing points at an empty string rather than
    # at zero, so the test for "no subject" is the first byte and needs no
    # 24-bit compare.
    b.label("SUBJECTS")
    for index, _thing in enumerate(world.things):
        b.fixup_word(f"TSUBJ{index}")

    if world.things:
        b.label("INITWHERE")
        for thing in world.things:
            b.db(thing.at)

    if world.people:
        b.label("PEOPLE")
        for index, person in enumerate(world.people):
            b.fixup_word(f"PDESC{index}")
            b.fixup_word(f"PDEF{index}")
            b.db(person.at)
        b.label("INITPWHERE")
        for person in world.people:
            b.db(person.at)

    if world.topics:
        b.label("INITSEALED")
        for topic in world.topics:
            b.db(int(topic.starts_sealed))

    if world.people and world.topics:
        # The dialogue, in the order the author wrote it. The scan is linear
        # and first-match-wins, which is what makes ordering the whole of the
        # conditional mechanism - see `_emit_ask`.
        b.label("DIALOG")
        for index, line in enumerate(world.lines):
            b.db(line.person)
            b.db(line.topic)
            b.db(line.gate)
            b.db(line.sets)
            b.fixup_word(f"DLINE{index}")
        b.db(libworld.NONE)          # a person of 255 ends the table

    _emit_word_table(b, "VERBS", verbs)
    _emit_word_table(b, "NOUNS", nouns)
    _emit_word_table(b, "NOISE", [(w, 0) for w in NOISE_WORDS])
    if world.people:
        _emit_word_table(b, "PEOPLEW", _people_words(world))
    if world.topics:
        _emit_word_table(b, "TOPICW", _topic_words(world))

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
        b.label(f"TSUBJ{index}")
        b.ascii(thing.subject or "")
        b.db(0)
    for index, person in enumerate(world.people):
        b.label(f"PDESC{index}")
        b.ascii(person.description)
        b.db(0)
        b.label(f"PDEF{index}")
        b.ascii(person.default)
        b.db(0)
    if world.people and world.topics:
        for index, line in enumerate(world.lines):
            b.label(f"DLINE{index}")
            b.ascii(line.text)
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

    `HERE` through `PWHERE` is the saved game. Everything else here is scratch
    the turn loop needs and nothing outlives a turn.
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
    # `ASKED` and `HEAT` are inside the run rather than beside it, because a
    # restore that put the player back somewhere but forgot what they had
    # already been told would be a worse bug than losing the save: the world
    # would re-explain everything and the rules keyed on `C_ASKED` would fire
    # a second time.
    b.label("ASKED")
    b.ds(max(1, len(world.topics)))
    b.label("HEAT")
    b.db(0)
    # The clock is overlay for the same reason `ASKED` is: a restore that put
    # it back to zero would give the player every deadline a second time.
    b.label("CLOCK")
    b.db(0)
    # What the archive is doing to each record. Overlay, because a restore
    # that unsealed everything the Voice had closed would be the Voice
    # forgetting it had been threatened.
    b.label("SEALED")
    b.ds(max(1, len(world.topics)))
    b.label("ALTERED")
    b.ds(max(1, len(world.topics)))
    b.label("PWHERE")
    b.ds(max(1, len(world.people)))

    # After the overlay and not in it: the length of the archive's log, which
    # the card holds. Saving it would let a restore forget questions the
    # archive has on record.
    b.label("LOGGED")
    b.db(0)

    # Everything below is scratch that does not outlive a turn.
    scratch = ["VERB", "W1LEN", "W2LEN", "W3LEN", "LKLEN", "NCARRIED",
               "RU_ONCE", "RU_NC", "RU_NA", "RU_OP", "RU_ARG", "RU_ARG2",
               "RU_CNT", "ATTERM", "ASKWHO", "ASKTOP", "LOGTOP", "SAVEH"]
    if not shared_console:
        scratch += ["WRAPCOL", "INPLEN"]
    for name in scratch:
        b.label(name)
        b.db(0)
    for name in ("PTMP", "LKPTR", "RULEPTR", "RU_CUR", "SPDST"):
        b.label(name)
        b.d24(0)

    b.label("W1")
    b.ds(MAX_WORD_LEN + 1)
    b.label("W2")
    b.ds(MAX_WORD_LEN + 1)
    b.label("W3")
    b.ds(MAX_WORD_LEN + 1)
    if not shared_console:
        b.label("INPBUF")
        b.ds(MAX_INPUT_LEN + 1)

    # The card's side of a saved game. `SAVNAME` carries its slot digit at
    # `len(save_name)`, which `SLOT` overwrites; the buffers are what one
    # `mos_fwrite` or `mos_fread` moves in a piece.
    b.label("SAVNAME")
    b.ascii(f"{world.save_name}1.SAV")
    b.db(0)
    b.label("LOGNAME")
    b.ascii(f"{world.save_name}.LOG")
    b.db(0)
    b.label("SAVEBUF")
    b.ds(4)
    b.label("SAVEOVL")
    b.ds(world.overlay_bytes)
    b.label("LOGREC")
    b.ds(2)


def overlay_at(builder: EZ80Builder, world: World) -> tuple[int, int]:
    """(address, length) of the bytes a save file would hold.

    `HERE` is deliberately the first of them and `FLAGS` the last, so the
    saved game is one contiguous run and `mos_fwrite` can put it down in a
    single call - see `tests/test_fwrite.py` for the call itself, which is
    emitted with the rest of save and restore rather than here.
    """
    start = builder.labels["HERE"]
    end = builder.labels["PWHERE"] + max(1, len(world.people))
    return start, end - start


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", "-o", default="SILO.bin", type=Path)
    ap.add_argument("--world", default="silo", choices=("silo", "mystery"),
                    help="silo is six rooms and four things; mystery adds "
                         "the people, topics and attention counter")
    args = ap.parse_args()

    if args.world == "mystery":
        import worlds_mystery
        world = worlds_mystery.mystery()
    else:
        import worlds
        world = worlds.silo()
    builder = build(world)
    builder.save(str(args.output))

    top = builder.org + len(builder.code)
    print(f"{args.output}  {len(builder.code):,} bytes   "
          f"{len(world.rooms)} rooms, {len(world.things)} things, "
          f"{len(world.people)} people, {len(world.lines)} lines")
    print(f"  overlay {world.overlay_bytes} bytes - the whole saved game")
    print(f"  image ends at {top:06X}h, "
          f"{AGON_SRAM_TOP - STACK_MARGIN - top:,} bytes of SRAM unused")

    # The check nobody runs by hand, and the one that makes "fair play" a
    # property of the build rather than a promise in the prose.
    search = world.explore()
    walkthrough = search.solve()
    print(f"  {len(search.states):,} reachable states")
    if walkthrough is None:
        raise SystemExit("  the goal is not reachable, so this cannot be won")
    if world.goal:
        print(f"  solved in {len(walkthrough)}: {', '.join(walkthrough)}")
    for kind, missing in search.unseen().items():
        if missing:
            print(f"  unreachable {kind}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
