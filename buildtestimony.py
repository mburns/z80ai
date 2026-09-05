"""
Testimony on the eZ80: a door's household asked about a name.

    TS_ENTRY    two titles in TS_WHO / TS_WHOM -> two documents, then TESTIFY
    TESTIFY     the first `libtestimony.PATHS` row joining them, said aloud

The world side (`buildif._emit_ask`) has copied the household's title and the
subject's title into two buffers and jumped here. Both go through the name
index - `NM_HASHAT` and `NM_FIND`, the same routines `LOOKUP` uses - so
nothing about a door has to be resolved when the world is built, and a door
in a world with no card behind it simply says nobody answers.

Then, path by path: a hop is `GW_HOP` over the forward table, a scan is the
reverse table's run for the key with each record's far end compared to the
subject. The first path that reaches the subject is the answer, and the
sentence is the path's, so the fact a household states is exactly an edge
or two on the graph and never a guess.
"""

from __future__ import annotations

import libtestimony
from libagon import MOS_OUTCHAR
from libez80 import EZ80Builder
from libgraphcard import EDGE_SIZE, RELATION

#: A path row in the image: step count, `MAX_STEPS` step bytes, a sentence
#: pointer.
ROW = 1 + libtestimony.MAX_STEPS + 3

CELLS = ("TS_WHO", "TS_WHOM", "TS_HERE")


def emit_cells(b: EZ80Builder) -> None:
    """The two documents and the walk's position. Call once, in the data."""
    for name in CELLS:
        b.label(name)
        b.ds24(1)
    b.label("TS_LEFT")
    b.db(0)


def emit_testimony(b: EZ80Builder, relations: list[str], forward_at: int,
                   num_edges: int, buffer_label: str = "IOBUF",
                   who_buffer: str = "TS_WHOBUF", whom_buffer: str = "TS_WHOMBUF",
                   turn_label: str = "TURN") -> None:
    """Emit `TS_ENTRY`, `TESTIFY`, and their tables.

    `relations` is the card's relation list, which picks the paths this card
    can walk and gives them their ids. `turn_label` is where the world's turn
    resumes once something has been said.
    """
    paths = libtestimony.resolve(relations)
    reverse_at = forward_at + num_edges * EDGE_SIZE
    works_in = relations.index("works_in") if "works_in" in relations else None

    # --- TS_ENTRY: titles to documents, or the reasons that failed ------------------
    b.label("TS_ENTRY")
    b.ld_hl_label(who_buffer)
    b.call("TS_RESOLVE")
    b.jp_c("TS_NOBODY")              # a household the card has never heard of
    b.ld_hl_mem_label(buffer_label, 6)
    b.ld_mem_label_hl("TS_WHO")
    b.ld_hl_label(whom_buffer)
    b.call("TS_RESOLVE")
    b.jp_c("TS_NONAME")
    b.ld_hl_mem_label(buffer_label, 6)
    b.ld_mem_label_hl("TS_WHOM")
    b.call("TESTIFY")
    b.jp(turn_label)

    b.label("TS_NOBODY")
    b.call("PRNL")
    b.ld_hl_label("MSGTSNOBODY")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp(turn_label)

    b.label("TS_NONAME")
    b.call("PRNL")
    b.ld_hl_label("MSGTSNONAME")
    b.call("PRWRAP")
    b.call("PRNL")
    b.jp(turn_label)

    # TS_RESOLVE: HL a NUL-terminated title -> its record in IOBUF, carry on
    # no such name. Length first, because the hash wants a count.
    b.label("TS_RESOLVE")
    b.push_hl()
    b.pop_ix()
    b.ld_b_n(0)
    b.label("TSR_LEN")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("TSR_GO")
    b.inc_hl()
    b.inc_b()
    b.jr("TSR_LEN")
    b.label("TSR_GO")
    b.call("NM_HASHAT")
    b.jp("NM_FIND")

    # --- TESTIFY: TS_WHO about TS_WHOM ----------------------------------------------
    b.label("TESTIFY")
    b.ld_hl_mem_label("TS_WHO")
    b.ld_de_mem_label("TS_WHOM")
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("TS_SELF")

    b.ld_ix_label("TSPATHS")
    b.label("TS_NEXT")
    b.ld_a_ixd(0)
    b.or_a()
    b.jp_z("TS_UNKNOWN")             # the table ends: no path joins them
    b.ld_hl_mem_label("TS_WHO")
    b.ld_mem_label_hl("GW_HERE")
    b.ld_mem_label_a("TS_LEFT")      # steps still to take
    b.push_ix()
    b.pop_iy()
    b.inc_iy()                       # IY walks the step bytes

    # Every step but the last is a hop: the first edge on the forward table,
    # or on the reverse one for an inverse step. The last is a hop that must
    # land on the subject, or a scan that must pass it.
    b.label("TS_STEP")
    b.ld_a_iyd(0)
    b.inc_iy()
    b.ld_c_a()                       # the step byte, kept for the flag
    b.and_n(RELATION)
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.ld_mem_label_hl("GW_REL")
    b.ld_a_mem_label("TS_LEFT")
    b.dec_a()
    b.ld_mem_label_a("TS_LEFT")
    b.jr_z("TS_LAST")
    b.ld_a_c()
    b.and_n(libtestimony.INVERSE)
    b.jr_nz("TS_REV_HOP")
    b.call("TS_FWD_HOP")
    b.jp_c("TS_MISS")
    b.jr("TS_STEP")
    b.label("TS_REV_HOP")
    b.call("TS_REV_HOP_GO")
    b.jp_c("TS_MISS")
    b.jr("TS_STEP")

    b.label("TS_LAST")
    b.ld_a_c()
    b.and_n(libtestimony.INVERSE)
    b.jp_nz("TS_SCAN")
    b.call("TS_FWD_HOP")
    b.jp_c("TS_MISS")
    b.ld_hl_mem_label("GW_HERE")     # where the hops ended
    b.ld_de_mem_label("TS_WHOM")
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("TS_HIT")
    b.jp("TS_MISS")

    # A scan: every reverse record for (GW_HERE, GW_REL), its far end
    # against the subject. The run is contiguous, and `GW_FIND` leaves its
    # lower bound in GW_LOW.
    b.label("TS_SCAN")
    b.ld_hl_nn(reverse_at)
    b.ld_mem_label_hl("GW_BASE")
    b.ld_hl_mem_label("GW_HERE")
    b.ld_mem_label_hl("GW_KEY")
    b.call("GW_FIND")
    b.jp_c("TS_MISS")
    b.ld_hl_mem_label("GW_LOW")
    b.ld_mem_label_hl("GW_MID")
    b.label("TS_SCAN_LP")
    b.ld_hl_mem_label("GW_MID")
    b.ld_de_nn(num_edges)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("TS_MISS")
    b.call("GW_FETCH")
    b.call("GW_SAME")
    b.jp_nz("TS_MISS")               # the run has ended
    b.ld_hl_mem_label(buffer_label, 4)
    b.ld_de_mem_label("TS_WHOM")
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("TS_HIT")
    b.ld_hl_mem_label("GW_MID")
    b.inc_hl()
    b.ld_mem_label_hl("GW_MID")
    b.jr("TS_SCAN_LP")

    b.label("TS_MISS")
    b.ld_de_nn(ROW)
    b.add_ix_de()
    b.jp("TS_NEXT")

    # TS_FWD_HOP: GW_HERE := object of the first (GW_HERE, GW_REL) forward
    # edge; carry when there is none.
    b.label("TS_FWD_HOP")
    b.ld_hl_nn(forward_at)
    b.ld_mem_label_hl("GW_BASE")
    b.jp("GW_HOP")

    # TS_REV_HOP_GO: the same over the reverse table - the first subject
    # whose (GW_REL) edge points at GW_HERE. GW_HOP reads the far end at
    # the same offset on either table, so only the base differs.
    b.label("TS_REV_HOP_GO")
    b.ld_hl_nn(reverse_at)
    b.ld_mem_label_hl("GW_BASE")
    b.jp("GW_HOP")

    # --- what is said -----------------------------------------------------------
    b.label("TS_HIT")
    b.ld_hl_ixd(1 + libtestimony.MAX_STEPS)   # the path's sentence
    b.push_hl()
    b.call("TS_OPEN")                # '<name>? '
    b.pop_hl()
    b.call("PRSTR")
    b.jp("TS_CLOSE")

    b.label("TS_SELF")
    b.call("TS_OPEN")
    b.ld_hl_label("MSGTSSELF")
    b.call("PRSTR")
    b.jp("TS_CLOSE")

    b.label("TS_UNKNOWN")
    b.call("TS_OPEN")
    b.ld_hl_label("MSGTSUNKNOWN")
    b.call("PRSTR")
    b.jp("TS_CLOSE")

    # TS_OPEN: a newline, a quote, the subject's title and a question mark.
    b.label("TS_OPEN")
    b.call("PRNL")
    b.ld_a_n(ord("'"))
    b.rst(MOS_OUTCHAR)
    b.ld_hl_mem_label("TS_WHOM")
    b.call("READ_TITLE")
    b.ld_hl_label("TEXTBUF")
    b.call("PRSTR")
    b.ld_hl_label("MSGTSQ")
    b.jp("PRSTR")

    # TS_CLOSE: the closing quote, then the register of the household's
    # department - one hop to `works_in`, its title, and a table of how each
    # department sounds. A household with no department, or one the table
    # does not name, gets the default.
    b.label("TS_CLOSE")
    b.ld_a_n(ord("'"))
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(ord(" "))
    b.rst(MOS_OUTCHAR)
    if works_in is None:
        b.ld_hl_label("MSGTSDEFAULT")
        b.call("PRWRAP")
        b.jp("PRNL")
    else:
        b.ld_hl_mem_label("TS_WHO")
        b.ld_mem_label_hl("GW_HERE")
        b.ld_hl_nn(works_in)
        b.ld_mem_label_hl("GW_REL")
        b.call("TS_FWD_HOP")
        b.jr_c("TS_DEFAULT")
        b.ld_hl_mem_label("GW_HERE")
        b.call("READ_TITLE")
        b.ld_ix_label("TSREGS")
        b.label("TS_REG_LP")
        b.ld_hl_ixd(0)
        b.ld_de_nn(0)
        b.or_a()
        b.sbc_hl_de()
        b.jr_z("TS_DEFAULT")         # the table ends
        b.ld_hl_ixd(0)
        b.ld_de_label("TEXTBUF")
        b.call("TS_STREQ")
        b.jr_z("TS_REG_HIT")
        b.ld_de_nn(6)
        b.add_ix_de()
        b.jr("TS_REG_LP")
        b.label("TS_REG_HIT")
        b.ld_hl_ixd(3)
        b.call("PRWRAP")
        b.jp("PRNL")
        b.label("TS_DEFAULT")
        b.ld_hl_label("MSGTSDEFAULT")
        b.call("PRWRAP")
        b.jp("PRNL")

        # TS_STREQ: zero flag set when the NUL-terminated strings at HL and
        # DE are the same.
        b.label("TS_STREQ")
        b.ld_a_de()
        b.cp_hl()
        b.ret_nz()
        b.or_a()
        b.ret_z()                    # both ended together
        b.inc_hl()
        b.inc_de()
        b.jr("TS_STREQ")

    # --- data -------------------------------------------------------------------
    b.label("TSPATHS")
    for index, (_path, steps) in enumerate(paths):
        b.db(len(steps))
        for position in range(libtestimony.MAX_STEPS):
            b.db(steps[position] if position < len(steps) else 0)
        b.fixup_word(f"TSSAID{index}")
    b.db(0)
    for index, (path, _steps) in enumerate(paths):
        b.label(f"TSSAID{index}")
        b.ascii(path.said)
        b.db(0)

    b.label("TSREGS")
    registers = list(libtestimony.REGISTERS.items())
    for index, _ in enumerate(registers):
        b.fixup_word(f"TSREGT{index}")
        b.fixup_word(f"TSREGS{index}")
    b.d24(0)
    for index, (title, says) in enumerate(registers):
        b.label(f"TSREGT{index}")
        b.ascii(title)
        b.db(0)
        b.label(f"TSREGS{index}")
        b.ascii(says)
        b.db(0)

    for label, text in (("MSGTSSELF", libtestimony.SELF),
                        ("MSGTSUNKNOWN", libtestimony.UNKNOWN),
                        ("MSGTSDEFAULT", libtestimony.DEFAULT_REGISTER),
                        ("MSGTSQ", "? "),
                        ("MSGTSNOBODY", "Nobody answers."),
                        ("MSGTSNONAME", "'Never heard the name,' says a voice "
                                        "through the door.")):
        b.label(label)
        b.ascii(text)
        b.db(0)
