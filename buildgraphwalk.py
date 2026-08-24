#!/usr/bin/env python3
"""
The graph walk, in eZ80: binary search over a card file, and the climb.

`libgraphcard.CardGraph` is the reference; this is the same walk on the machine.
Emitted as routines rather than a program, so a caller supplies the open handle,
the scratch buffer and the seek cell, and there is one path to the card rather
than two.

## What a hop costs

A hop is a binary search over fixed-width records: about eighteen iterations
for 150,335 edges, each one a seek, a seven-byte read and two 24-bit compares.
No multiply - the record offset is `n * 7`, which is `n * 8 - n`: three
`ADD HL,HL` and a subtract.

That is the argument for putting the oracle on this machine at all. Reading an
answer out of an article is comprehension and out of reach. Comparing two
24-bit numbers is four instructions.

## Why the state is in memory

Everything here is 24-bit and there are three index registers, so the search
state lives in named cells rather than registers. Slower and legible; the
alternative is a register allocation nobody can check against the reference.

Two eZ80 details worth naming, because both would be silent bugs:

**Halving is done in memory.** There is no 24-bit shift of HL, so the midpoint
is shifted where it sits - `SRA` the top byte, then `RR` down through the other
two. `SRA` rather than `SRL` because the eZ80 has no `SRL (HL)`; it is only
equal to a logical shift while the top byte's sign bit is clear, which holds
because `low + high` cannot exceed twice the edge count.

**Zero is tested with a subtract.** `LD A,L : OR H` tests sixteen bits and
would call 0x010000 zero, so the test is `SBC HL,DE` against zero instead.
"""

from __future__ import annotations

from libez80 import EZ80Builder
from libgraphcard import EDGE_SIZE, PLAIN

#: Every walk cell is 24-bit, which is a word on this machine.
CELLS = (
    "GW_BASE",      # byte offset of the table being searched
    "GW_LOW", "GW_HIGH", "GW_MID",
    "GW_KEY",       # the subject (or object) being looked up
    "GW_REL",       # the relation, in the low byte
    "GW_HERE",      # where the walk has got to
    "GW_STEPS",     # address of the next (relation, kind) pair
    "GW_LEFT",      # how many steps remain
    "GW_CLIMB",     # climbs left before giving up
    "GW_TYPEAT",    # byte offset of the current type's id list
    "GW_TYPEN",     # how many ids are in it
)

#: Matches libgraphcard.CLIMB_LIMIT. A containment hierarchy is shallow, and a
#: cycle in the data - two places each inside the other - must still terminate.
CLIMB_LIMIT = 6


def emit_cells(b: EZ80Builder) -> None:
    """Reserve the walk's state. Call once, in the data section."""
    for name in CELLS:
        b.label(name)
        b.ds24(1)


def _halve(b: EZ80Builder, cell: str) -> None:
    """Shift a 24-bit cell right by one, in place."""
    b.ld_hl_label(cell, 2)
    b.sra_hl_ind()                     # top byte; its sign bit is always clear
    b.ld_hl_label(cell, 1)
    b.rr_hl_ind()
    b.ld_hl_label(cell)
    b.rr_hl_ind()


def _is_zero(b: EZ80Builder, cell: str, target: str) -> None:
    """Jump to `target` when a 24-bit cell is zero."""
    b.ld_hl_mem_label(cell)
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()
    b.jp_z(target)


def _seek_read(b: EZ80Builder, handle_label: str, buffer_label: str,
               seekoff_label: str, count: int) -> None:
    """HL holds a byte offset; leave `count` bytes at the buffer."""
    b.ld_mem_label_hl(seekoff_label)
    b.xor_a()
    b.ld_mem_label_a(seekoff_label, 3)
    b.ld_a_mem_label(handle_label)
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label(handle_label)
    b.ld_c_a()
    b.ld_hl_label(buffer_label)
    b.ld_de_nn(count)
    b.call("READ")


def emit_walk(b: EZ80Builder, num_edges: int, types_at: int, num_types: int,
              handle_label: str, buffer_label: str, seekoff_label: str) -> None:
    """Emit the walk routines.

    ``types_at`` is the card offset of the type table, whose entries are an
    offset and a count; the id lists follow it. Reading a type's span off the
    card costs one extra seek per climb and keeps this self-contained.
    """
    types_body_at = types_at + 8 * num_types

    # --- GW_MUL7: HL = HL * EDGE_SIZE -----------------------------------------
    b.label("GW_MUL7")
    b.push_de()
    # PUSH HL / POP DE, not EX DE,HL: the exchange puts n in DE but leaves
    # whatever was in DE in HL, so the doublings below multiplied the caller's
    # leftovers. The symptom was a seek to a plausible offset and a record full
    # of some other edge, which is exactly the kind of wrong this cannot see.
    b.push_hl()
    b.pop_de()                         # DE = n, HL still n
    b.add_hl_hl()                      # 2n
    b.add_hl_hl()                      # 4n
    b.add_hl_hl()                      # 8n
    b.or_a()
    b.sbc_hl_de()                      # 7n
    b.pop_de()
    b.ret()

    # --- GW_FETCH: read record (GW_MID) of the table at GW_BASE ---------------
    b.label("GW_FETCH")
    b.ld_hl_mem_label("GW_MID")
    b.call("GW_MUL7")
    b.ld_de_mem_label("GW_BASE")
    b.add_hl_de()
    _seek_read(b, handle_label, buffer_label, seekoff_label, EDGE_SIZE)
    b.ret()

    # --- GW_CMP: carry set when the fetched record sorts *before* the wanted
    #     (GW_KEY, GW_REL). That is the test a lower-bound search needs.
    b.label("GW_CMP")
    b.ld_hl_mem_label(buffer_label)    # the record's 24-bit key
    b.ld_de_mem_label("GW_KEY")
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("GW_CMP_LT")
    b.jp_nz("GW_CMP_GE")
    b.ld_a_mem_label(buffer_label, 3)  # keys equal: the relation decides
    b.ld_hl_label("GW_REL")
    b.cp_hl()
    b.jp_c("GW_CMP_LT")
    b.label("GW_CMP_GE")
    b.or_a()                           # carry clear
    b.ret()
    b.label("GW_CMP_LT")
    b.scf()
    b.ret()

    # --- GW_SAME: zero flag set when the fetched record *is* the wanted one.
    #     GW_CMP can only say "before or not", which cannot distinguish equal
    #     from after, and a lower bound lands on the first not-before record.
    b.label("GW_SAME")
    b.ld_hl_mem_label(buffer_label)
    b.ld_de_mem_label("GW_KEY")
    b.or_a()
    b.sbc_hl_de()
    b.ret_nz()
    b.ld_a_mem_label(buffer_label, 3)
    b.ld_hl_label("GW_REL")
    b.cp_hl()
    b.ret()

    # --- GW_FIND: lower bound for (GW_KEY, GW_REL); index left in GW_LOW.
    #     Carry set when there is no such edge.
    b.label("GW_FIND")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("GW_LOW")
    b.ld_hl_nn(num_edges)
    b.ld_mem_label_hl("GW_HIGH")

    b.label("GW_FIND_LP")
    b.ld_hl_mem_label("GW_LOW")
    b.ld_de_mem_label("GW_HIGH")
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("GW_FIND_DONE")            # low >= high
    b.ld_hl_mem_label("GW_LOW")
    b.ld_de_mem_label("GW_HIGH")
    b.add_hl_de()
    b.ld_mem_label_hl("GW_MID")
    _halve(b, "GW_MID")
    b.call("GW_FETCH")
    b.call("GW_CMP")
    b.jp_nc("GW_FIND_HIGH")
    b.ld_hl_mem_label("GW_MID")        # low = mid + 1
    b.inc_hl()
    b.ld_mem_label_hl("GW_LOW")
    b.jp("GW_FIND_LP")
    b.label("GW_FIND_HIGH")
    b.ld_hl_mem_label("GW_MID")        # high = mid
    b.ld_mem_label_hl("GW_HIGH")
    b.jp("GW_FIND_LP")

    b.label("GW_FIND_DONE")
    b.ld_hl_mem_label("GW_LOW")        # off the end is a miss
    b.ld_de_nn(num_edges)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("GW_FIND_MISS")
    b.ld_hl_mem_label("GW_LOW")
    b.ld_mem_label_hl("GW_MID")
    b.call("GW_FETCH")
    b.call("GW_SAME")
    b.jp_nz("GW_FIND_MISS")
    b.or_a()                           # carry clear: found
    b.ret()
    b.label("GW_FIND_MISS")
    b.scf()
    b.ret()

    # --- GW_HOP: GW_HERE := object of the first (GW_HERE, GW_REL) edge.
    b.label("GW_HOP")
    b.ld_hl_mem_label("GW_HERE")
    b.ld_mem_label_hl("GW_KEY")
    b.call("GW_FIND")
    b.ret_c()
    b.ld_hl_mem_label(buffer_label, 4)  # the object, 24-bit
    b.ld_mem_label_hl("GW_HERE")
    b.or_a()
    b.ret()

    # --- GW_TYPESET: A = type id -> GW_TYPEAT, GW_TYPEN from the card's table.
    b.label("GW_TYPESET")
    # Widen the kind byte to 24 bits through a zeroed cell: there is no
    # "LD HL,A", and leaving the upper bytes as they were would index the type
    # table at a wild offset that still reads as a plausible span.
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("GW_MID")
    b.ld_mem_label_a("GW_MID")
    b.ld_hl_mem_label("GW_MID")
    b.add_hl_hl()                      # 2
    b.add_hl_hl()                      # 4
    b.add_hl_hl()                      # 8 bytes per entry
    b.ld_de_nn(types_at)
    b.add_hl_de()
    _seek_read(b, handle_label, buffer_label, seekoff_label, 8)
    b.ld_hl_mem_label(buffer_label)    # offset within the id body
    b.ld_de_nn(types_body_at)
    b.add_hl_de()
    b.ld_mem_label_hl("GW_TYPEAT")
    b.ld_hl_mem_label(buffer_label, 4)
    b.ld_mem_label_hl("GW_TYPEN")
    b.ret()

    # --- GW_ISA: is GW_HERE in the sorted id list at (GW_TYPEAT, GW_TYPEN)?
    b.label("GW_ISA")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("GW_LOW")
    b.ld_hl_mem_label("GW_TYPEN")
    b.ld_mem_label_hl("GW_HIGH")

    b.label("GW_ISA_LP")
    b.ld_hl_mem_label("GW_LOW")
    b.ld_de_mem_label("GW_HIGH")
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("GW_ISA_NO")
    b.ld_hl_mem_label("GW_LOW")
    b.ld_de_mem_label("GW_HIGH")
    b.add_hl_de()
    b.ld_mem_label_hl("GW_MID")
    _halve(b, "GW_MID")
    b.ld_hl_mem_label("GW_MID")        # offset = TYPEAT + mid * 3
    b.push_hl()
    b.add_hl_hl()
    b.pop_de()
    b.add_hl_de()
    b.ld_de_mem_label("GW_TYPEAT")
    b.add_hl_de()
    _seek_read(b, handle_label, buffer_label, seekoff_label, 3)

    b.ld_hl_mem_label(buffer_label)
    b.ld_de_mem_label("GW_HERE")
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("GW_ISA_YES")
    b.jp_nc("GW_ISA_HIGH")
    b.ld_hl_mem_label("GW_MID")        # entry < here: low = mid + 1
    b.inc_hl()
    b.ld_mem_label_hl("GW_LOW")
    b.jp("GW_ISA_LP")
    b.label("GW_ISA_HIGH")
    b.ld_hl_mem_label("GW_MID")
    b.ld_mem_label_hl("GW_HIGH")
    b.jp("GW_ISA_LP")

    b.label("GW_ISA_YES")
    b.or_a()                           # carry clear: it is one
    b.ret()
    b.label("GW_ISA_NO")
    b.scf()
    b.ret()

    # --- GW_CLIMBTO: repeat GW_REL until GW_HERE is of the current type.
    #
    # The type check comes first, so an entity that is already what was asked
    # for is returned rather than stepped past - which is the point, and a
    # quarter of the birthplaces in this corpus.
    b.label("GW_CLIMBTO")
    b.ld_hl_nn(CLIMB_LIMIT)
    b.ld_mem_label_hl("GW_CLIMB")
    b.label("GW_CLIMB_LP")
    b.call("GW_ISA")
    b.jp_nc("GW_CLIMB_OK")
    b.ld_hl_mem_label("GW_CLIMB")
    b.dec_hl()
    b.ld_mem_label_hl("GW_CLIMB")
    _is_zero(b, "GW_CLIMB", "GW_CLIMB_NO")
    b.call("GW_HOP")
    b.ret_c()                          # ran out of graph
    b.jp("GW_CLIMB_LP")
    b.label("GW_CLIMB_OK")
    b.or_a()
    b.ret()
    b.label("GW_CLIMB_NO")
    b.scf()
    b.ret()

    # --- GW_FOLLOW: walk GW_LEFT steps from GW_HERE, reading (relation, kind)
    #     pairs at GW_STEPS. Carry set when a step had nowhere to go.
    b.label("GW_FOLLOW")
    b.label("GW_FOLLOW_LP")
    _is_zero(b, "GW_LEFT", "GW_FOLLOW_OK")
    b.ld_hl_mem_label("GW_LEFT")
    b.dec_hl()
    b.ld_mem_label_hl("GW_LEFT")

    b.ld_hl_mem_label("GW_STEPS")
    b.ld_a_hl()                        # relation
    b.ld_mem_label_a("GW_REL")
    b.inc_hl()
    b.ld_a_hl()                        # kind
    b.inc_hl()
    b.ld_mem_label_hl("GW_STEPS")
    b.cp_n(PLAIN)
    b.jp_z("GW_FOLLOW_PLAIN")

    b.call("GW_TYPESET")               # a climb: point at that type's ids
    b.call("GW_CLIMBTO")
    b.ret_c()
    b.jp("GW_FOLLOW_LP")

    b.label("GW_FOLLOW_PLAIN")
    b.call("GW_HOP")
    b.ret_c()
    b.jp("GW_FOLLOW_LP")

    b.label("GW_FOLLOW_OK")
    b.or_a()
    b.ret()
