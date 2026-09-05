"""
`LOOKUP <name>` on the eZ80: the name index searched, the record printed.

    archive> lookup alexander e wong
    Alexander E. Wong
      born: Year 166
      father: Dylan R. Smith
      ...

Three routines and no classifier. `NM_HASH` computes `libnames.hashes` of
`libnames.normalize` of what was typed, byte for byte the way that module
does. `NM_FIND` is a lower-bound binary search over the nine-byte records of
`SILO.NAM`, fourteen probes for ten thousand names. `NM_RECORD` is the
forward edge table's lower bound for the document and a scan while the
subject holds, one title read per edge - `libgraph.record` on the device.

That is the deterministic surface `data/silo/README.md` argued the archive
was missing: a bare name gets everything the graph holds about that person,
which is always coherent, always true, and needs no guess about which
question was meant. The classifier stays where it is measured, on
questions. `IF.md` calls this the archive as records.

## A name that is not enough

`First Last` keys land on everyone with that first and last name, and the
device lists them under a heading rather than picking one - the honest
answer to the middle-initial problem #56 measured, where two people were
one document and the graph was right about the wrong person.
"""

from __future__ import annotations

from libez80 import EZ80Builder
from libnames import HEADER, RECORD

#: The word at the front of the line. Compared case-insensitively.
WORD = b"LOOKUP"

#: 24-bit state the search and the scan keep.
CELLS = ("NM_H1", "NM_H2", "NM_LOW", "NM_HIGH", "NM_MID", "NM_DOC", "NM_EDGE")


def emit_cells(b: EZ80Builder) -> None:
    """Reserve the lookup's state. Call once, in the data section."""
    for name in CELLS:
        b.label(name)
        b.ds24(1)
    for name in ("NM_PEND", "NM_SEEN", "NM_REL", "NM_ANY"):
        b.label(name)
        b.db(0)


def _seek_read(b: EZ80Builder, handle_label: str, buffer_label: str,
               seekoff_label: str, count: int) -> None:
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


def _halve(b: EZ80Builder, cell: str) -> None:
    b.ld_hl_label(cell, 2)
    b.sra_hl_ind()
    b.ld_hl_label(cell, 1)
    b.rr_hl_ind()
    b.ld_hl_label(cell)
    b.rr_hl_ind()


def emit_lookup(b: EZ80Builder, num_names: int, num_edges: int,
                forward_at: int, labels: list[str],
                handle_label: str = "NAMH", buffer_label: str = "IOBUF",
                seekoff_label: str = "SEEKOFF",
                notice_label: str | None = None) -> None:
    """Emit `NM_TRY` and everything under it.

    `NM_TRY` reads the line in `INPBUF`: carry set means it was a `LOOKUP`
    and has been answered, carry clear means it was not and the caller
    should treat the line as it would have. `labels` is one printable label
    per relation id, and an empty one means the relation is left out of a
    record - `child_of` says what `father_is` and `mother_is` already said.

    `notice_label`, when given, is called with the resolved document in
    `BESTID` before the record is printed, so a world's seals, alterations,
    attention and log apply to a lookup exactly as to a question. A nonzero
    return means the archive said something instead, and the record is not
    printed.
    """
    # --- NM_TRY --------------------------------------------------------------
    b.label("NM_TRY")
    b.ld_a_mem_label("INPLEN")
    b.cp_n(len(WORD))
    b.jp_c("NMT_NO")
    b.ld_hl_label("INPBUF")
    b.ld_de_label("NM_WORD")
    b.ld_b_n(len(WORD))
    b.label("NMT_CMP")
    b.ld_a_hl()
    b.cp_n(ord("a"))
    b.jr_c("NMT_UP")
    b.cp_n(ord("z") + 1)
    b.jr_nc("NMT_UP")
    b.sub_n(0x20)
    b.label("NMT_UP")
    b.ld_c_a()
    b.ld_a_de()
    b.cp_c()
    b.jp_nz("NMT_NO")
    b.inc_hl()
    b.inc_de()
    b.djnz("NMT_CMP")
    b.ld_a_mem_label("INPLEN")
    b.cp_n(len(WORD) + 1)
    b.jr_c("NMT_WHOM")                # LOOKUP and nothing after it
    b.ld_a_mem_label("INPBUF", len(WORD))
    b.cp_n(ord(" "))
    b.jp_nz("NMT_NO")                 # LOOKUPS is some other word
    b.ld_a_mem_label("INPLEN")
    b.cp_n(len(WORD) + 2)
    b.jr_c("NMT_WHOM")                # LOOKUP and a space
    b.call("NM_HASH")
    b.call("NM_FIND")
    b.jr_c("NMT_NONE")
    b.call("NM_ANSWER")
    b.scf()
    b.ret()

    b.label("NMT_WHOM")
    b.call("PRNL")
    b.ld_hl_label("MSGWHOM")
    b.call("PRSTR")
    b.call("PRNL")
    b.scf()
    b.ret()

    b.label("NMT_NONE")
    b.call("PRNL")
    b.ld_hl_label("MSGNONAME")
    b.call("PRSTR")
    b.call("PRNL")
    b.scf()
    b.ret()

    b.label("NMT_NO")
    b.or_a()
    b.ret()

    # --- NM_HASH: INPBUF past the word -> NM_H1, NM_H2 ---------------------------
    #
    # `libnames.normalize` and `libnames.hashes` in one pass: upper-case the
    # letters, keep letters and digits, feed one space between kept runs,
    # drop the rest. A pending space is only fed once something has been,
    # so leading and trailing spaces vanish the way the Python does it.
    b.label("NM_HASH")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("NM_H1")
    b.ld_mem_label_hl("NM_H2")
    b.xor_a()
    b.ld_mem_label_a("NM_PEND")
    b.ld_mem_label_a("NM_SEEN")
    b.ld_a_mem_label("INPLEN")
    b.sub_n(len(WORD) + 1)
    b.ret_z()
    b.ld_b_a()
    b.ld_ix_label("INPBUF")
    b.ld_de_nn(len(WORD) + 1)
    b.add_ix_de()

    b.label("NMH_LP")
    b.ld_a_ixd(0)
    b.inc_ix()
    b.cp_n(ord("a"))
    b.jr_c("NMH_CASED")
    b.cp_n(ord("z") + 1)
    b.jr_nc("NMH_CASED")
    b.sub_n(0x20)
    b.label("NMH_CASED")
    b.cp_n(ord(" "))
    b.jr_nz("NMH_KEEP")
    b.ld_a_n(1)
    b.ld_mem_label_a("NM_PEND")
    b.jr("NMH_NEXT")
    b.label("NMH_KEEP")
    b.cp_n(ord("0"))
    b.jr_c("NMH_NEXT")
    b.cp_n(ord("9") + 1)
    b.jr_c("NMH_EMIT")
    b.cp_n(ord("A"))
    b.jr_c("NMH_NEXT")
    b.cp_n(ord("Z") + 1)
    b.jr_nc("NMH_NEXT")
    b.label("NMH_EMIT")
    b.ld_c_a()
    b.ld_a_mem_label("NM_PEND")
    b.or_a()
    b.jr_z("NMH_FEED")
    b.ld_a_mem_label("NM_SEEN")
    b.or_a()
    b.jr_z("NMH_FEED")
    b.ld_a_n(ord(" "))
    b.call("NM_FEED")
    b.label("NMH_FEED")
    b.xor_a()
    b.ld_mem_label_a("NM_PEND")
    b.ld_a_n(1)
    b.ld_mem_label_a("NM_SEEN")
    b.ld_a_c()
    b.call("NM_FEED")
    b.label("NMH_NEXT")
    b.djnz("NMH_LP")
    b.ret()

    # NM_FEED: one byte in A into both hashes. h1 = h1*31 + c, h2 = h2*33 + c,
    # each mod 2^24 - which is what a 24-bit ADD HL,HL gives for free.
    b.label("NM_FEED")
    b.push_bc()
    b.ld_c_a()
    b.ld_hl_mem_label("NM_H1")
    b.push_hl()
    b.pop_de()
    for _ in range(5):
        b.add_hl_hl()
    b.or_a()
    b.sbc_hl_de()
    b.ld_de_nn(0)
    b.ld_e_c()
    b.add_hl_de()
    b.ld_mem_label_hl("NM_H1")
    b.ld_hl_mem_label("NM_H2")
    b.push_hl()
    b.pop_de()
    for _ in range(5):
        b.add_hl_hl()
    b.add_hl_de()
    b.ld_de_nn(0)
    b.ld_e_c()
    b.add_hl_de()
    b.ld_mem_label_hl("NM_H2")
    b.pop_bc()
    b.ret()

    # --- NM_FIND: lower bound for (NM_H1, NM_H2); index in NM_LOW -----------------
    #     Carry set when no record has that key.
    b.label("NM_FETCH")              # record NM_MID -> IOBUF
    b.ld_hl_mem_label("NM_MID")
    b.push_hl()
    b.pop_de()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_de()                    # x9
    b.ld_de_nn(HEADER.size)
    b.add_hl_de()
    _seek_read(b, handle_label, buffer_label, seekoff_label, RECORD)
    b.ret()

    b.label("NM_CMP")                # carry set when the record sorts before
    b.ld_hl_mem_label(buffer_label)
    b.ld_de_mem_label("NM_H1")
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("NM_LT")
    b.jp_nz("NM_GE")
    b.ld_hl_mem_label(buffer_label, 3)
    b.ld_de_mem_label("NM_H2")
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("NM_LT")
    b.label("NM_GE")
    b.or_a()
    b.ret()
    b.label("NM_LT")
    b.scf()
    b.ret()

    b.label("NM_SAME")               # zero flag set when the record is the key
    b.ld_hl_mem_label(buffer_label)
    b.ld_de_mem_label("NM_H1")
    b.or_a()
    b.sbc_hl_de()
    b.ret_nz()
    b.ld_hl_mem_label(buffer_label, 3)
    b.ld_de_mem_label("NM_H2")
    b.or_a()
    b.sbc_hl_de()
    b.ret()

    b.label("NM_FIND")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("NM_LOW")
    b.ld_hl_nn(num_names)
    b.ld_mem_label_hl("NM_HIGH")
    b.label("NMF_LP")
    b.ld_hl_mem_label("NM_LOW")
    b.ld_de_mem_label("NM_HIGH")
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("NMF_DONE")
    b.ld_hl_mem_label("NM_LOW")
    b.ld_de_mem_label("NM_HIGH")
    b.add_hl_de()
    b.ld_mem_label_hl("NM_MID")
    _halve(b, "NM_MID")
    b.call("NM_FETCH")
    b.call("NM_CMP")
    b.jp_nc("NMF_HIGH")
    b.ld_hl_mem_label("NM_MID")
    b.inc_hl()
    b.ld_mem_label_hl("NM_LOW")
    b.jp("NMF_LP")
    b.label("NMF_HIGH")
    b.ld_hl_mem_label("NM_MID")
    b.ld_mem_label_hl("NM_HIGH")
    b.jp("NMF_LP")
    b.label("NMF_DONE")
    b.ld_hl_mem_label("NM_LOW")
    b.ld_de_nn(num_names)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("NMF_MISS")
    b.ld_hl_mem_label("NM_LOW")
    b.ld_mem_label_hl("NM_MID")
    b.call("NM_FETCH")
    b.call("NM_SAME")
    b.jp_nz("NMF_MISS")
    b.or_a()
    b.ret()
    b.label("NMF_MISS")
    b.scf()
    b.ret()

    # --- NM_ANSWER: one record, or the list of who shares the name -------------------
    b.label("NM_ANSWER")
    b.ld_hl_mem_label(buffer_label, 6)   # the first match's document
    b.ld_mem_label_hl("NM_DOC")
    b.ld_hl_mem_label("NM_MID")
    b.inc_hl()
    b.ld_mem_label_hl("NM_MID")
    b.ld_de_nn(num_names)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("NM_ONE")                # the last record: nobody shares it
    b.call("NM_FETCH")
    b.call("NM_SAME")
    b.jp_nz("NM_ONE")

    # More than one. The heading, then every title while the key holds.
    b.call("PRNL")
    b.ld_hl_label("MSGWHICH")
    b.call("PRSTR")
    b.call("PRNL")
    b.ld_hl_mem_label("NM_LOW")
    b.ld_mem_label_hl("NM_MID")
    b.label("NML_LP")
    b.ld_hl_mem_label("NM_MID")
    b.ld_de_nn(num_names)
    b.or_a()
    b.sbc_hl_de()
    b.ret_nc()
    b.call("NM_FETCH")
    b.call("NM_SAME")
    b.ret_nz()
    b.ld_hl_label("MSGINDENT")
    b.call("PRSTR")
    b.ld_hl_mem_label(buffer_label, 6)
    b.call("READ_TITLE")
    b.ld_hl_label("TEXTBUF")
    b.call("PRSTR")
    b.call("PRNL")
    b.ld_hl_mem_label("NM_MID")
    b.inc_hl()
    b.ld_mem_label_hl("NM_MID")
    b.jp("NML_LP")

    b.label("NM_ONE")
    if notice_label is not None:
        # The world hears a lookup the way it hears a question: the subject
        # is noticed, charged and logged, and a sealed or altered record is
        # what the archive says instead.
        b.ld_hl_mem_label("NM_DOC")
        b.ld_mem_label_hl("BESTID")
        b.ld_a_n(1)
        b.ld_mem_label_a("BESTSC")
        b.call(notice_label)
        b.or_a()
        b.ret_nz()
    # Fall through to the record.

    # --- NM_RECORD: the title, then a line an edge ----------------------------------
    b.label("NM_RECORD")
    b.call("PRNL")
    b.ld_hl_mem_label("NM_DOC")
    b.call("READ_TITLE")
    b.ld_hl_label("TEXTBUF")
    b.call("PRSTR")
    b.call("PRNL")
    b.xor_a()
    b.ld_mem_label_a("NM_ANY")

    # The forward table's lower bound for (doc, relation 0), which is the
    # first edge of the subject whether or not relation 0 is among them:
    # GW_FIND leaves GW_LOW at the bound even when it reports a miss.
    b.ld_hl_mem_label("NM_DOC")
    b.ld_mem_label_hl("GW_KEY")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("GW_REL")
    b.ld_hl_nn(forward_at)
    b.ld_mem_label_hl("GW_BASE")
    b.call("GW_FIND")
    b.ld_hl_mem_label("GW_LOW")
    b.ld_mem_label_hl("NM_EDGE")

    b.label("NMR_LP")
    b.ld_hl_mem_label("NM_EDGE")
    b.ld_de_nn(num_edges)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("NMR_END")
    b.ld_hl_mem_label("NM_EDGE")
    b.ld_mem_label_hl("GW_MID")
    b.call("GW_FETCH")
    b.ld_hl_mem_label(buffer_label)
    b.ld_de_mem_label("NM_DOC")
    b.or_a()
    b.sbc_hl_de()
    b.jp_nz("NMR_END")                # past the subject's run
    b.ld_a_mem_label(buffer_label, 3)
    b.ld_mem_label_a("NM_REL")
    b.ld_hl_mem_label(buffer_label, 4)
    b.ld_mem_label_hl("NM_DOC2")     # the object, before READ_TITLE clobbers IOBUF
    # The label, or nothing for a relation left out of listings.
    b.ld_a_mem_label("NM_REL")
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.push_hl()
    b.pop_de()
    b.add_hl_hl()
    b.add_hl_de()                    # x3, a pointer a relation
    b.ld_de_label("RELLAB")
    b.add_hl_de()
    b.push_hl()
    b.pop_ix()
    b.ld_hl_ixd(0)                   # the label's address
    b.ld_a_hl()
    b.or_a()
    b.jr_z("NMR_NEXT")
    b.push_hl()
    b.ld_hl_label("MSGINDENT")
    b.call("PRSTR")
    b.pop_hl()
    b.call("PRSTR")
    b.ld_hl_label("MSGCOLON")
    b.call("PRSTR")
    b.ld_hl_mem_label("NM_DOC2")
    b.call("READ_TITLE")
    b.ld_hl_label("TEXTBUF")
    b.call("PRSTR")
    b.call("PRNL")
    b.ld_a_n(1)
    b.ld_mem_label_a("NM_ANY")
    b.label("NMR_NEXT")
    b.ld_hl_mem_label("NM_EDGE")
    b.inc_hl()
    b.ld_mem_label_hl("NM_EDGE")
    b.jp("NMR_LP")

    b.label("NMR_END")
    b.ld_a_mem_label("NM_ANY")
    b.or_a()
    b.ret_nz()
    b.ld_hl_label("MSGNOFACTS")      # findable, readable, and nothing to walk
    b.call("PRSTR")
    b.jp("PRNL")

    # --- data ---------------------------------------------------------------------
    b.label("NM_WORD")
    b.blob(WORD)
    b.label("MSGWHOM")
    b.ascii("Look up whom?")
    b.db(0)
    b.label("MSGNONAME")
    b.ascii("No record under that name.")
    b.db(0)
    b.label("MSGWHICH")
    b.ascii("That name is on more than one record:")
    b.db(0)
    b.label("MSGNOFACTS")
    b.ascii("  The archive holds no facts under that name.")
    b.db(0)
    b.label("MSGINDENT")
    b.ascii("  ")
    b.db(0)
    b.label("MSGCOLON")
    b.ascii(": ")
    b.db(0)
    b.label("RELLAB")
    for index in range(len(labels)):
        b.fixup_word(f"RL{index}")
    for index, label in enumerate(labels):
        b.label(f"RL{index}")
        b.ascii(label)
        b.db(0)
    b.label("NM_DOC2")
    b.ds24(1)


#: Where a line label reads differently from `liboracle.HELD`'s sentence
#: wording: "of Generation 4" is a phrase, "of: Generation 4" is not.
LINE_LABELS = {"generation_is": "generation"}


def labels_for(relations: list[str]) -> list[str]:
    """One label a relation, from `liboracle`'s own wording of a record.

    Empty for the relations a listing leaves out. Anything `liboracle.HELD`
    does not name prints as the relation with its underscores taken out -
    ugly and legible, which is the right failure for a missing entry.
    """
    import liboracle

    return ["" if name in liboracle.REDUNDANT
            else LINE_LABELS.get(name)
            or liboracle.HELD.get(name, name.replace("_", " "))
            for name in relations]
