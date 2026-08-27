#!/usr/bin/env python3
"""
The Agon side of the search demo: an eZ80 program that reads a card index.

Built by buildwikisearch.py, which also writes the card files. This module only
knows the *format*, never the corpus - which is what lets a rebuilt card work
with an unchanged binary.

## What runs on the machine

    read a line
    for each word, up to eight that are actually in the index:
        hash -> bucket -> one seek -> walk that bucket's chain
        add each posting's weight into the accumulator
    scan the pages the query touched for the three largest
    read those three titles and leads off the card and print them

There is no arithmetic beyond addition. BM25's multiply, divide and per-document
length all happened at build time, and each posting arrives as a five-bit weight
that is simply added. That is what keeps the accumulator at one byte per
article - 277KB for 284,000 of them, resident, no sharding - and it is why this
runs in milliseconds rather than seconds.

The accumulator is tiered: one flag byte per 256-article page, set when a
posting lands. The clear and report passes walk the flags and visit only
flagged pages, so a query pays for the pages it touched rather than for the
whole corpus - the two whole-corpus passes were the dominant per-query cost.

## Memory

Neither the accumulator nor the unpacking buffers live in the image. `ds`
emits literal zeros, so a 284,000-byte buffer would put 284,000 zeros in the
.bin; instead they sit at fixed addresses above the program, and the build
asserts the image does not reach them.

## Firmware

Unlike the inference builds, this needs to seek: mos_fopen, mos_flseek,
mos_fread and mos_fclose rather than mos_load alone. That widens the surface
that `libhost.AgonHost` models but real MOS has never been checked against - see
tools/mostest.py and the note in EZ80.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

import libgraphcard
from libez80 import AGON_LOAD_ADDR, AGON_SRAM_TOP, EZ80Builder, agon_header

if TYPE_CHECKING:
    from libinfer import BuildInputs

from libsearch import (
    MAGIC,
    MAX_ARTICLE,
    MAX_BLOB,
    MAX_PACKED_ARTICLE,
    MAX_QUERY_TERMS,
    MAX_WEIGHT,
    NUM_BUCKETS,
    TEXT_MAGIC,
)

MOS_API = 0x08
MOS_OUTCHAR = 0x10
MOS_GETKEY = 0x00
MOS_FOPEN = 0x0A
MOS_FCLOSE = 0x0B
MOS_FREAD = 0x1A
MOS_FLSEEK = 0x1C
FA_READ = 0x01

@dataclass
class OracleSpec:
    """What the oracle build needs beyond the search build.

    Held together rather than passed as five arguments because the three files
    are one card: a graph whose ids mean anything only beside the index they
    were built from.
    """

    #: Card file holding the fact graph.
    graph_name: str
    #: Byte offset of the forward edge table, and how many edges it holds.
    forward_at: int
    num_edges: int
    #: Byte offset of the type table, and how many types it holds.
    types_at: int
    num_types: int
    #: Document count and title digest the .GRF was built against. A mismatch
    #: has no other symptom - every id in the wrong graph is still an article.
    num_docs: int
    digest: int
    #: phrase index -> the (relation, kind) steps it means, or `None` for a
    #: phrase the machine should refuse rather than walk.
    paths: list[list[tuple[int, int]] | None]
    #: The phrasebook model, as buildez80.load_for_build returns it.
    model: BuildInputs | None = None
    #: How many times a climb may step before giving up. A property of the
    #: card and not of the format: it is an immediate in the walk routine, so
    #: the choice costs nothing in bytes and shows up only as probes on the
    #: climbs that use it.
    climb_limit: int = libgraphcard.CLIMB_LIMIT


MAX_INPUT_LEN = 120
MAX_TOKEN_LEN = 32
TOP_K = 3
#: Bytes pulled per postings read. One SD block is 512; this is four of them,
#: which keeps a long postings list to a handful of round trips.
CHUNK = 2048
#: Stack margin below the top of SRAM, matching the inference builds.
STACK_MARGIN = 0x1000

#: Where `PRWRAP` breaks a line. The Agon's default mode is 80 columns and a
#: line printed to exactly 80 makes the terminal wrap it itself, which costs a
#: blank line; this leaves room and is still wider than any prose on the card
#: needs. Not a build parameter because nothing yet knows the screen mode - see
#: the second scope of issue #62, where finding that out is the first item.
WRAP_WIDTH = 76


def accumulator_base(num_docs: int) -> int:
    """Where the score accumulator sits: as high as it can, below the stack."""
    return (AGON_SRAM_TOP - STACK_MARGIN - num_docs) & ~0xFF


#: Offsets into the scratch region, which sits below the accumulator. These are
#: buffers, not data: `ds` would put eleven thousand zeros in the .bin to
#: reserve them, and the file is the thing the user copies onto a card.
PACKBUF_AT = 0
TEXTBUF_AT = PACKBUF_AT + CHUNK + 16
#: Unpacked text is longer than what was read, by as much as the packing saved.
#: A third off is what the corpus measures, so twice the read is room enough.
PAIRTAB_AT = TEXTBUF_AT + 2 * CHUNK + 16
BLOBBUF_AT = PAIRTAB_AT + 3 * 256
SCRATCH_BYTES = BLOBBUF_AT + MAX_BLOB


def scratch_base(num_docs: int) -> int:
    """Where the unpacking buffers sit: directly below the accumulator."""
    return (accumulator_base(num_docs) - SCRATCH_BYTES) & ~0xFF


def num_pages(num_docs: int) -> int:
    """Page-flag bytes, one per 256 articles.

    These are `ds` in the image rather than scratch, so a corpus grows the
    program as well as the accumulator - which is the term that makes the
    ceiling `N * 257/256` rather than `N`, and the one an estimate drops.
    """
    return (num_docs + 255) >> 8


def fixed_bytes(num_docs: int, image_bytes: int) -> int:
    """An image's size with its page table taken out.

    The page table is the only part that moves with the corpus, so this is
    the part `max_docs` can be asked about: everything else is the program.
    """
    return image_bytes - num_pages(num_docs)


def headroom(num_docs: int, image_bytes: int) -> int:
    """Bytes between the top of the image and the unpacking buffers.

    Negative means the corpus cannot be scored in SRAM - which `build`
    asserts, but a caller sizing a corpus wants the distance rather than the
    verdict.
    """
    return scratch_base(num_docs) - (AGON_LOAD_ADDR + image_bytes)


def max_docs(fixed: int) -> int:
    """The largest corpus a program of `fixed` bytes can score in SRAM.

    Both bases are rounded down to a 256-byte boundary, so the accumulator
    and the buffers below it move in whole pages: `scratch_base` falls by
    exactly 256 for each page the corpus adds, while the image rises by one
    byte for the same page. Each page therefore costs 257 bytes, and the
    answer is however many pages the gap holds.

    Solved rather than searched, so that a test bisecting `build` is a second
    implementation rather than the same one twice.
    """
    return 256 * ((scratch_base(0) - AGON_LOAD_ADDR - fixed) // 257)


def build(num_docs: int, index_name: str = "WIKI.IDX",
          text_name: str = "WIKI.DAT",
          org: int = AGON_LOAD_ADDR,
          oracle: OracleSpec | None = None) -> EZ80Builder:
    """Emit the search program for a card holding ``num_docs`` articles.

    With ``oracle``, the same program answers from the fact graph first and
    falls back to listing articles - which is what the search build already
    is, so the fallback costs nothing.
    """
    pages = num_pages(num_docs)
    acc_base = accumulator_base(num_docs)
    b = EZ80Builder(org=org)
    agon_header(b, "START")

    # --- entry ---------------------------------------------------------------
    b.label("START")
    b.ld_hl_label("BANNER")
    b.call("PRSTR")

    b.ld_hl_label("IDXNAME")
    b.call("OPEN")
    b.or_a()
    b.jp_z("NOCARD")
    b.ld_mem_label_a("IDXH")

    b.ld_hl_label("DATNAME")
    b.call("OPEN")
    b.or_a()
    b.jp_z("NOCARD")
    b.ld_mem_label_a("DATH")

    b.call("CHECKMAGIC")
    b.or_a()
    b.jp_nz("BADCARD")

    # Also checks WIKI.DAT's own magic, since the text is packed and a card
    # written before it was would print the pair table as an article.
    b.call("LOADPAIRS")
    b.or_a()
    b.jp_nz("BADCARD")

    if oracle is not None:
        b.ld_hl_label("GRFNAME")
        b.call("OPEN")
        b.or_a()
        b.jp_z("NOCARD")
        b.ld_mem_label_a("GRFH")
        b.call("CHECKGRAPH")
        b.or_a()
        b.jp_nz("BADCARD")

    b.label("MAINLOOP")
    b.call("PRNL")
    b.ld_hl_label("PROMPT")
    b.call("PRSTR")
    b.call("READ_INPUT")

    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("MAINLOOP")
    b.ld_a_mem_label("INPBUF")
    b.cp_n(ord("!"))
    b.jp_z("QUIT")

    b.call("CLEAR_ACC")
    b.call("SCORE_QUERY")
    b.call("ORACLE" if oracle else "REPORT")
    b.jp("MAINLOOP")

    b.label("QUIT")
    b.ld_a_mem_label("IDXH")
    b.call("CLOSE")
    b.ld_a_mem_label("DATH")
    b.call("CLOSE")
    b.call("PRNL")
    b.ret()

    b.label("NOCARD")
    b.ld_hl_label("MSGNOCARD")
    b.call("PRSTR")
    b.ret()

    b.label("BADCARD")
    b.ld_hl_label("MSGBADCARD")
    b.call("PRSTR")
    b.ret()

    # --- file helpers --------------------------------------------------------
    #
    # OPEN: HL = filename -> A = handle (0 on failure).
    b.label("OPEN")
    b.ld_c_n(FA_READ)
    b.ld_a_n(MOS_FOPEN)
    b.rst(MOS_API)
    b.ret()

    b.label("CLOSE")
    b.ld_c_a()
    b.ld_a_n(MOS_FCLOSE)
    b.rst(MOS_API)
    b.ret()

    # SEEK: C = handle, (SEEKOFF) = 32-bit offset.
    b.label("SEEK")
    b.ld_hl_mem_label("SEEKOFF")
    b.ld_a_mem_label("SEEKOFF", 3)
    b.ld_e_a()
    b.ld_a_n(MOS_FLSEEK)
    b.rst(MOS_API)
    b.ret()

    # READ: C = handle, HL = buffer, DE = count -> DE = bytes read.
    b.label("READ")
    b.ld_a_n(MOS_FREAD)
    b.rst(MOS_API)
    b.ret()

    # CHECKMAGIC: the card's magic against the one this binary was built for.
    # A card written by a different format version is refused rather than
    # misread - the failure would otherwise be scores, not an error.
    b.label("CHECKMAGIC")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("SEEKOFF")
    b.xor_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_nn(len(MAGIC))
    b.call("READ")

    b.ld_hl_label("IOBUF")
    b.ld_de_label("MAGICSTR")
    b.ld_b_n(len(MAGIC))
    b.label("CM_LP")
    b.ld_a_hl()
    b.ld_c_a()
    b.ld_a_de()
    b.sub_c()
    b.ret_nz()                       # A is nonzero: mismatch
    b.inc_hl()
    b.inc_de()
    b.djnz("CM_LP")
    b.xor_a()
    b.ret()

    # --- accumulator ---------------------------------------------------------
    #
    # One byte per article, tiered into 256-article pages so the per-query
    # passes visit only the pages scoring touched.
    _emit_clear_acc(b, num_docs, acc_base, pages)

    # --- query ---------------------------------------------------------------
    b.label("SCORE_QUERY")
    b.xor_a()
    b.ld_mem_label_a("TOKPOS")
    b.ld_mem_label_a("NSCORED")

    b.label("SQ_NEXT")
    b.ld_a_mem_label("NSCORED")
    b.cp_n(MAX_QUERY_TERMS)
    b.ret_nc()                       # eight contributing terms is the cap
    b.call("NEXT_TOKEN")
    b.or_a()
    b.ret_z()                        # no more words
    b.call("SCORE_TERM")
    b.jr("SQ_NEXT")

    # NEXT_TOKEN: pull the next word out of INPBUF into TOKBUF, lowercased.
    # Returns its length in A, zero when the line is exhausted.
    b.label("NEXT_TOKEN")
    b.xor_a()
    b.ld_mem_label_a("TOKLEN")
    b.ld_mem_label_a("NTGLUED")

    b.label("NT_SKIP")               # skip anything that is not a letter/digit
    b.ld_a_mem_label("TOKPOS")
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.sub_c()
    b.jr_z("NT_DONE")
    b.call("TOK_CHAR")
    b.call("ALNUM")
    b.jr_c("NT_TAKE")
    b.call("TOK_ADVANCE")
    b.jr("NT_SKIP")

    b.label("NT_TAKE")               # collect the run of letters/digits
    b.ld_a_mem_label("TOKPOS")
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.sub_c()
    b.jr_z("NT_DONE")
    b.call("TOK_CHAR")
    b.call("ALNUM")
    b.jr_nc("NT_DONE")
    b.call("LOWER")
    b.ld_c_a()
    b.ld_a_mem_label("TOKLEN")
    b.cp_n(MAX_TOKEN_LEN)
    b.jr_nc("NT_SKIPCH")             # word too long: stop collecting it
    b.ld_hl_label("TOKBUF")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_hl_c()
    b.ld_a_mem_label("TOKLEN")
    b.inc_a()
    b.ld_mem_label_a("TOKLEN")
    b.label("NT_SKIPCH")
    b.call("TOK_ADVANCE")
    b.jr("NT_TAKE")

    # A single character is an initial, not a word, so it is glued to the run
    # after it: `amanda m wilson` yields `amanda` and `mwilson`. Without this
    # the two Amanda Wilsons are not merely hard to tell apart, they are the
    # *same query* - the index never saw a one-character term either, because
    # `libsearch.tokenize` drops them.
    #
    # `a` and `i` are exempt because they are words. Gluing one eats the word
    # after it, and `what is a black hole` becomes `ablack hole`. Those are the
    # only two single-character stopwords, which `tests/test_wikisearch.py`
    # pins so this pair cannot quietly stop matching the list.
    #
    # NTGLUED makes it a one-shot: without it, a query ending in a lone
    # initial would come back here with TOKLEN still 1 and loop forever.
    b.label("NT_DONE")
    b.ld_a_mem_label("TOKLEN")
    b.cp_n(1)
    b.jr_nz("NT_RET")
    b.ld_a_mem_label("NTGLUED")
    b.or_a()
    b.jr_nz("NT_RET")
    b.ld_a_mem_label("TOKBUF")
    b.cp_n(ord("a"))
    b.jr_z("NT_RET")
    b.cp_n(ord("i"))
    b.jr_z("NT_RET")
    b.ld_a_n(1)
    b.ld_mem_label_a("NTGLUED")
    b.jr("NT_SKIP")

    b.label("NT_RET")
    b.ld_a_mem_label("TOKLEN")
    b.ret()

    # TOK_CHAR: A = INPBUF[TOKPOS]
    b.label("TOK_CHAR")
    b.ld_hl_label("INPBUF")
    b.ld_de_nn(0)
    b.ld_a_mem_label("TOKPOS")
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()
    b.ret()

    b.label("TOK_ADVANCE")
    b.ld_a_mem_label("TOKPOS")
    b.inc_a()
    b.ld_mem_label_a("TOKPOS")
    b.ret()

    # ALNUM: carry set if A is a letter or digit.
    b.label("ALNUM")
    b.cp_n(ord("0"))
    b.jr_c("AN_NO")
    b.cp_n(ord("9") + 1)
    b.jr_c("AN_YES")
    b.cp_n(ord("A"))
    b.jr_c("AN_NO")
    b.cp_n(ord("Z") + 1)
    b.jr_c("AN_YES")
    b.cp_n(ord("a"))
    b.jr_c("AN_NO")
    b.cp_n(ord("z") + 1)
    b.jr_c("AN_YES")
    b.label("AN_NO")
    b.or_a()                         # clear carry
    b.ret()
    b.label("AN_YES")
    b.scf()
    b.ret()

    b.label("LOWER")
    b.cp_n(ord("A"))
    b.ret_c()
    b.cp_n(ord("Z") + 1)
    b.ret_nc()
    b.add_a_n(0x20)
    b.ret()

    # HASH: TOKBUF/TOKLEN -> HL, h = h*31 + c masked to the bucket count.
    # 31 is (h << 5) - h, so this is five adds and a subtract per character.
    # The character count lives in memory, not in B: the multiply needs BC for
    # the shifted copy, so a DJNZ counter would be destroyed on the first pass.
    b.label("HASH")
    b.ld_hl_nn(0)
    b.ld_a_mem_label("TOKLEN")
    b.or_a()
    b.ret_z()
    b.ld_mem_label_a("HCNT")
    b.ld_ix_label("TOKBUF")

    b.label("H_LP")
    b.ld_mem_label_hl("HTMP")
    for _ in range(5):
        b.add_hl_hl()                # h << 5
    b.ld_bc_mem_label("HTMP")
    b.or_a()
    b.sbc_hl_bc()                    # ... minus h, so h * 31
    b.ld_a_ixd(0)
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()                    # ... plus the character

    # Mask to the bucket count. A power of two, so this is one AND on the top
    # byte rather than a 24-bit operation the eZ80 does not have.
    b.ld_mem_label_hl("HTMP")
    b.ld_a_mem_label("HTMP", 2)
    b.and_n(BUCKET_MASK_HI)
    b.ld_mem_label_a("HTMP", 2)
    b.ld_hl_mem_label("HTMP")

    b.inc_ix()
    b.ld_a_mem_label("HCNT")
    b.dec_a()
    b.ld_mem_label_a("HCNT")
    b.jr_nz("H_LP")
    b.ret()

    _emit_score_term(b, acc_base)
    _emit_report(b, num_docs, acc_base, pages, oracle is not None)
    if oracle is not None:
        _emit_oracle(b, oracle)
        _emit_classifier(b, oracle)
    _emit_console(b)
    _emit_data(b, num_docs, acc_base, pages, index_name, text_name)
    if oracle is not None:
        _emit_classifier_data(b, oracle)

    top = b.org + len(b.code)
    scratch = scratch_base(num_docs)
    assert top <= scratch, (
        f"the image reaches {top:06X}h but the unpacking buffers start at "
        f"{scratch:06X}h; the corpus is too large to score in SRAM - "
        f"{num_docs:,} articles against a limit of "
        f"{max_docs(fixed_bytes(num_docs, len(b.code))):,} for this image")
    b.accumulator = acc_base
    b.num_docs = num_docs
    return b


def _emit_clear_acc(b: EZ80Builder, num_docs: int, acc_base: int,
                    pages: int) -> None:
    """Zero the pages the previous query touched, and their flags.

    A whole-corpus memset is ~284,000 stores; scoring touches a handful of
    256-article pages for a typical lookup. Walking the page table and
    clearing only flagged pages is the same resulting state - every article
    byte is zero and so is every flag - for a fraction of the stores. A
    flagged page's flag is reset as it is cleared, so the table needs no
    pass of its own.
    """
    acc_end = acc_base + num_docs
    b.label("CLEAR_ACC")
    b.ld_iy_label("PAGE_TAB")
    b.ld_hl_nn(pages)
    b.ld_mem_label_hl("PGLEFT")
    b.ld_hl_nn(acc_base)
    b.ld_mem_label_hl("PACC")

    b.label("CA_PAGE")
    b.ld_a_iyd(0)
    b.or_a()
    b.jr_z("CA_NEXT")

    # A flagged page: reset its flag, then clear min(256, remaining) bytes -
    # the final page may be short, and writing past the accumulator would
    # reach the stack margin.
    b.xor_a()
    b.ld_iyd_a(0)
    b.ld_hl_nn(acc_end)
    b.ld_de_mem_label("PACC")
    b.or_a()
    b.sbc_hl_de()                    # HL = articles from this page to the end
    b.ld_de_nn(256)
    b.ld_bc_nn(256)
    b.push_hl()
    b.or_a()
    b.sbc_hl_de()
    b.pop_hl()
    b.jr_nc("CA_HAVE")               # a full page survives
    b.push_hl()
    b.pop_bc()                       # the last page: only its remainder
    b.label("CA_HAVE")
    b.ld_hl_mem_label("PACC")
    b.ld_de_mem_label("PACC")
    b.inc_de()
    b.ld_hl_n(0)                     # the first byte, propagated by the LDIR
    b.dec_bc()
    b.ld_a_b()
    b.or_c()
    b.jr_z("CA_NEXT")                # a one-article page needs no copy
    b.ldir()

    b.label("CA_NEXT")
    b.ld_hl_mem_label("PACC")
    b.ld_bc_nn(256)
    b.add_hl_bc()
    b.ld_mem_label_hl("PACC")
    b.inc_iy()
    b.ld_hl_mem_label("PGLEFT")
    b.ld_de_nn(1)
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("PGLEFT")
    b.jp_nz("CA_PAGE")
    b.ret()


def _emit_score_term(b: EZ80Builder, acc_base: int) -> None:
    """Look one term up and add its postings into the accumulator."""
    b.label("SCORE_TERM")
    b.call("HASH")

    # Seek to the bucket's slot in the table: header + 4 * bucket.
    b.add_hl_hl()
    b.add_hl_hl()                    # 4 bytes per entry
    b.ld_bc_nn(TABLE_AT)
    b.add_hl_bc()
    b.ld_mem_label_hl("SEEKOFF")
    b.xor_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.call("SEEK")

    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_nn(4)
    b.call("READ")

    # A zero offset means nothing was ever hashed here.
    b.ld_hl_mem_label("IOBUF")
    b.ld_a_mem_label("IOBUF", 3)
    b.or_a()
    b.jr_nz("ST_GO")
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()
    b.ret_z()
    b.ld_hl_mem_label("IOBUF")

    b.label("ST_GO")
    b.ld_mem_label_hl("SEEKOFF")
    b.ld_a_mem_label("IOBUF", 3)
    b.ld_mem_label_a("SEEKOFF", 3)

    # Walk the chain: length, term, posting count, postings.
    b.label("ST_CHAIN")
    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_nn(MAX_TOKEN_LEN + 8)
    b.call("READ")

    b.ld_a_mem_label("IOBUF")
    b.or_a()
    b.ret_z()                        # end of chain, term is not indexed

    # Same length as the token? If not it cannot be the same word.
    b.ld_c_a()
    b.ld_a_mem_label("TOKLEN")
    b.sub_c()
    b.jr_nz("ST_SKIP")

    b.ld_hl_label("IOBUF", 1)
    b.ld_de_label("TOKBUF")
    b.ld_a_mem_label("TOKLEN")
    b.ld_b_a()
    b.label("ST_CMP")
    b.ld_a_hl()
    b.ld_c_a()
    b.ld_a_de()
    b.sub_c()
    b.jr_nz("ST_SKIP")
    b.inc_hl()
    b.inc_de()
    b.djnz("ST_CMP")
    b.jp("ST_MATCH")

    # A different word in the same bucket - a hash collision. Step over its
    # header and postings and keep walking rather than scoring the wrong docs.
    b.label("ST_SKIP")
    b.call("ST_ADVANCE")
    b.jr("ST_CHAIN")

    b.label("ST_MATCH")
    b.ld_a_mem_label("NSCORED")
    b.inc_a()
    b.ld_mem_label_a("NSCORED")
    b.call("ST_COUNT")               # DF into (NPOST), payload start into SEEKOFF
    b.jp("ST_STREAM")

    # ST_ADVANCE / ST_COUNT: header is 1 + len + 3, payload is 4 * df.
    b.label("ST_COUNT")
    b.ld_a_mem_label("IOBUF")
    b.ld_hl_nn(0)
    b.ld_l_a()
    # The header is length(1) + term(length) + count(3), so the count sits at
    # IOBUF + 1 + length - not +4, which would read three bytes of postings.
    b.ld_de_label("IOBUF", 1)
    b.add_hl_de()
    b.push_hl()
    b.pop_ix()
    b.ld_hl_ixd(0)                   # the 24-bit posting count
    b.ld_mem_label_hl("NPOST")

    b.ld_a_mem_label("IOBUF")
    b.ld_hl_nn(4)
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()                    # 1 + len + 3
    b.ld_de_mem_label("SEEKOFF")
    b.add_hl_de()
    b.ld_mem_label_hl("SEEKOFF")
    b.jp_nc("STC_OK")
    b.ld_a_mem_label("SEEKOFF", 3)
    b.inc_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.label("STC_OK")
    b.ret()

    b.label("ST_ADVANCE")
    b.call("ST_COUNT")
    # Then skip the payload. It is a byte count now rather than a posting
    # count, which is the whole reason the header carries bytes: once postings
    # vary in width, a count no longer says how far a colliding term reaches.
    b.ld_hl_mem_label("NPOST")
    b.ld_de_mem_label("SEEKOFF")
    b.add_hl_de()
    b.ld_mem_label_hl("SEEKOFF")
    b.jp_nc("STA_OK")
    b.ld_a_mem_label("SEEKOFF", 3)
    b.inc_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.label("STA_OK")
    b.ret()

    # Stream the postings in blocks, adding each weight into its slot.
    #
    # A posting is a tag byte and one to three gap bytes, so unlike the flat
    # four it replaced it can straddle the end of a block. Rather than stitch
    # one across the boundary, the file cursor advances by the bytes that held
    # *whole* postings and the straggler is read again at the head of the next
    # block. That re-reads at most three bytes per CHUNK, and cannot stall: the
    # first posting of a block always fits, because a block is either a whole
    # CHUNK - far longer than any posting - or the last few bytes of a payload
    # that ends on a posting boundary by construction.
    b.label("ST_STREAM")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("RUNDOC")      # every gap is measured from zero

    b.label("STS_BLOCK")
    b.ld_hl_mem_label("NPOST")       # payload bytes still unread
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()
    b.ret_z()

    b.ld_hl_mem_label("NPOST")
    b.ld_de_nn(CHUNK)
    b.or_a()
    b.sbc_hl_de()
    b.jr_c("STS_PART")
    b.ld_hl_nn(CHUNK)
    b.jr("STS_HAVE")
    b.label("STS_PART")
    b.ld_hl_mem_label("NPOST")
    b.label("STS_HAVE")
    b.ld_mem_label_hl("NTHIS")

    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("IDXH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_mem_label("NTHIS")
    b.call("READ")

    # Add each (gap, weight) into the accumulator, saturating at 255, and flag
    # the article's 256-doc page so the clear and report passes visit only
    # pages this query touched.
    b.ld_ix_label("IOBUF")
    b.ld_hl_mem_label("NTHIS")
    b.ld_mem_label_hl("NLEFT")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("CONSUMED")

    b.label("STS_ONE")
    b.ld_hl_mem_label("NLEFT")
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("STS_NEXT")

    # Bits 5 and 6 of the tag hold one less than the gap's width, so the three
    # cases are two compares - no shift, and no arithmetic at all.
    b.ld_a_ixd(0)
    b.and_n(0x60)
    b.jp_z("STS_W1")
    b.cp_n(0x20)
    b.jp_z("STS_W2")

    b.ld_hl_mem_label("NLEFT")       # a three-byte gap: four bytes in all
    b.ld_de_nn(4)
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("STS_NEXT")
    b.ld_hl_ixd(1)
    b.ld_de_nn(4)
    b.jp("STS_GAP")

    b.label("STS_W2")
    b.ld_hl_mem_label("NLEFT")
    b.ld_de_nn(3)
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("STS_NEXT")
    b.ld_hl_nn(0)
    b.ld_l_ixd(1)
    b.ld_h_ixd(2)
    b.ld_de_nn(3)
    b.jp("STS_GAP")

    b.label("STS_W1")
    b.ld_hl_mem_label("NLEFT")
    b.ld_de_nn(2)
    b.or_a()
    b.sbc_hl_de()
    b.jp_c("STS_NEXT")
    b.ld_hl_nn(0)
    b.ld_l_ixd(1)
    b.ld_de_nn(2)

    # HL is the gap and DE the posting's size. The running total is the whole
    # of the decoding: an add, which is what this card has always been willing
    # to pay for.
    b.label("STS_GAP")
    b.push_de()
    b.ex_de_hl()
    b.ld_hl_mem_label("RUNDOC")
    b.add_hl_de()
    b.ld_mem_label_hl("RUNDOC")

    b.ld_de_nn(0)
    b.ld_a_mem_label("RUNDOC", 1)
    b.ld_e_a()
    b.ld_a_mem_label("RUNDOC", 2)
    b.ld_d_a()
    b.ld_hl_label("PAGE_TAB")
    b.add_hl_de()
    b.ld_hl_n(1)

    b.ld_hl_mem_label("RUNDOC")
    b.ld_bc_nn(acc_base)
    b.add_hl_bc()
    b.push_hl()
    b.pop_iy()
    b.ld_a_iyd(0)
    b.ld_c_a()
    b.ld_a_ixd(0)
    b.and_n(MAX_WEIGHT)              # the five bits the tag shares
    b.add_a_c()
    b.jr_nc("STS_STORE")
    b.ld_a_n(255)                    # saturate rather than wrap
    b.label("STS_STORE")
    b.ld_iyd_a(0)

    b.pop_de()                       # the posting's size, kept across the above
    b.push_ix()
    b.pop_hl()
    b.add_hl_de()
    b.push_hl()
    b.pop_ix()
    b.ld_hl_mem_label("NLEFT")
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("NLEFT")
    b.ld_hl_mem_label("CONSUMED")
    b.add_hl_de()
    b.ld_mem_label_hl("CONSUMED")
    b.jp("STS_ONE")

    # Only the bytes that held whole postings. Anything left is the front of
    # one that straddles, and the next block starts there rather than after it.
    b.label("STS_NEXT")
    b.ld_hl_mem_label("SEEKOFF")
    b.ld_de_mem_label("CONSUMED")
    b.add_hl_de()
    b.ld_mem_label_hl("SEEKOFF")
    b.jp_nc("STS_NC")
    b.ld_a_mem_label("SEEKOFF", 3)
    b.inc_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.label("STS_NC")
    b.ld_hl_mem_label("NPOST")
    b.ld_de_mem_label("CONSUMED")
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("NPOST")
    b.jp("STS_BLOCK")


def _emit_report(b: EZ80Builder, num_docs: int, acc_base: int,
                 pages: int, split_report: bool = False) -> None:
    """Find the best three scores and print their articles.

    Pages the query never touched hold only zeros, so the scan walks the page
    table and skips unflagged pages wholesale instead of reading 277KB to find
    a handful of nonzero bytes. Order is unchanged - pages in order, articles
    in order within a page - so first-wins ties still match libsearch.
    """
    b.label("REPORT")
    for k in range(TOP_K):
        b.ld_hl_nn(0)
        b.ld_mem_label_hl("BESTID", 3 * k)
        b.xor_a()
        b.ld_mem_label_a("BESTSC", k)

    b.ld_iy_label("PAGE_TAB")
    b.ld_hl_nn(pages)
    b.ld_mem_label_hl("PGLEFT")
    b.ld_hl_nn(acc_base)
    b.ld_mem_label_hl("PACC")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("SCANID")

    b.label("RP_PAGE")
    b.ld_a_iyd(0)
    b.or_a()
    b.jr_nz("RP_DOCPG")

    # An unflagged page is all zeros: skip its 256 articles. The final page
    # may be short, so the skip stops at num_docs, not at a page boundary.
    b.ld_hl_mem_label("SCANID")
    b.ld_de_nn(256)
    b.add_hl_de()
    b.ld_mem_label_hl("SCANID")
    b.ld_de_nn(num_docs)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nc("RP_SCANNED")            # skipped past the final article
    b.jr("RP_PNXT")

    # A flagged page: scan it byte by byte. B counts a full page's 256
    # articles (DJNZ counts 256 from zero); the SCANID check inside ends the
    # final, partial page exactly.
    b.label("RP_DOCPG")
    b.ld_hl_mem_label("PACC")
    b.push_hl()
    b.pop_ix()
    b.ld_b_n(0)

    b.label("RP_SCAN")
    b.ld_a_ixd(0)
    b.or_a()
    b.jr_z("RP_NEXT")
    b.call("RP_OFFER")

    b.label("RP_NEXT")
    b.ld_hl_mem_label("SCANID")
    b.ld_de_nn(1)
    b.add_hl_de()
    b.ld_mem_label_hl("SCANID")
    b.ld_de_nn(num_docs)
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("RP_SCANNED")             # RP_OFFER sits between, past JR range
    b.inc_ix()
    b.djnz("RP_SCAN")

    b.label("RP_PNXT")
    b.ld_hl_mem_label("PACC")
    b.ld_bc_nn(256)
    b.add_hl_bc()
    b.ld_mem_label_hl("PACC")
    b.inc_iy()
    b.ld_hl_mem_label("PGLEFT")
    b.ld_de_nn(1)
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("PGLEFT")
    b.jp_nz("RP_PAGE")
    b.jp("RP_SCANNED")               # unreached: the SCANID checks fire first

    # RP_OFFER: A is a score, (SCANID) its document. Insert if it beats one.
    b.label("RP_OFFER")
    for k in range(TOP_K):
        b.ld_c_a()
        b.ld_a_mem_label("BESTSC", k)
        b.cp_c()
        b.ld_a_c()
        b.jr_nc(f"RP_NEXT{k}")
        # Shift the tail down, then take this slot.
        for j in range(TOP_K - 1, k, -1):
            b.ld_hl_mem_label("BESTID", 3 * (j - 1))
            b.ld_mem_label_hl("BESTID", 3 * j)
            b.ld_a_mem_label("BESTSC", j - 1)
            b.ld_mem_label_a("BESTSC", j)
        b.ld_a_c()
        b.ld_mem_label_a("BESTSC", k)
        b.ld_hl_mem_label("SCANID")
        b.ld_mem_label_hl("BESTID", 3 * k)
        b.ret()
        b.label(f"RP_NEXT{k}")
    b.ret()

    # The scan ends here. An oracle wants the best document without the
    # listing - it prints a fact when it has one - so `split_report` cuts
    # REPORT in two at this line. Without it the label costs no bytes and
    # the scan falls through into the listing exactly as before.
    b.label("RP_SCANNED")
    if split_report:
        b.ret()

    b.label("RP_SHOW")
    b.ld_a_mem_label("BESTSC")
    b.or_a()
    b.jr_nz("RP_LIST")
    b.ld_hl_label("MSGNONE")
    b.call("PRSTR")
    b.ret()

    b.label("RP_LIST")
    b.xor_a()
    b.ld_mem_label_a("SHOWN")

    b.label("RP_ITEM")
    b.ld_a_mem_label("SHOWN")
    b.cp_n(TOP_K)
    b.ret_nc()
    b.ld_hl_label("BESTSC")
    b.ld_de_nn(0)
    b.ld_a_mem_label("SHOWN")
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()
    b.or_a()
    b.ret_z()                        # fewer than three matched

    b.call("SHOW_ONE")
    b.ld_a_mem_label("SHOWN")
    b.inc_a()
    b.ld_mem_label_a("SHOWN")
    b.jr("RP_ITEM")

    # SHOW_ONE: read the article whose id is BESTID[SHOWN] and print it.
    b.label("SHOW_ONE")
    b.ld_hl_label("BESTID")
    b.ld_a_mem_label("SHOWN")
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.add_hl_bc()
    b.add_hl_bc()                    # 3 bytes per id
    b.push_hl()
    b.pop_ix()
    b.ld_hl_ixd(0)
    b.call("READ_ARTICLE")

    b.call("PRNL")
    b.ld_hl_label("TEXTBUF")
    b.call("PRSTR")                  # the title, up to its NUL
    b.call("PRNL")
    # The lead follows immediately after that NUL.
    b.ld_hl_label("TEXTBUF")
    b.label("SO_FIND")
    b.ld_a_hl()
    b.inc_hl()
    b.or_a()
    b.jr_nz("SO_FIND")
    b.call("PRWRAP")                 # the lead, broken between words
    b.call("PRNL")
    b.ret()

    # READ_ARTICLE: HL is a document id; leave its title and lead in TEXTBUF.
    # Split out of SHOW_ONE because an oracle wants the title of an article it
    # walked to, which is not one of the three the search scored.
    b.label("READ_ARTICLE")
    # The offset table sits after the pair table, whose size depends on the
    # corpus - so the base is read from the card at startup rather than built
    # in. A binary that knew it would need rebuilding whenever the text did.
    b.add_hl_hl()
    b.add_hl_hl()
    b.ld_de_mem_label("DATBASE")
    b.add_hl_de()
    b.ld_mem_label_hl("SEEKOFF")
    b.xor_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_nn(4)
    b.call("READ")

    b.ld_hl_mem_label("IOBUF")
    b.ld_mem_label_hl("SEEKOFF")
    b.ld_a_mem_label("IOBUF", 3)
    b.ld_mem_label_a("SEEKOFF", 3)
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.ld_hl_label("PACKBUF")
    b.ld_de_nn(CHUNK)
    b.call("READ")
    # Fall through: what came off the card is packed, and everything
    # downstream wants a title and a lead it can print.

    # UNPACK: PACKBUF -> TEXTBUF, stopping once the lead's NUL has been
    # written. A byte with a zero length in the table stands for itself;
    # anything else is a block move out of the expansion blob. No shifts, no
    # escape, no recursion - the codes are byte values the corpus never uses,
    # so a literal can never be mistaken for one.
    b.label("UNPACK")
    b.ld_ix_label("PACKBUF")
    b.ld_iy_label("TEXTBUF")
    b.ld_a_n(2)
    b.ld_mem_label_a("NULSEEN")      # a title and a lead

    b.label("UP_ONE")
    b.ld_a_ixd(0)
    b.inc_ix()

    # PAIRTAB + 3 * A: the offset and length this byte stands for.
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.push_hl()
    b.pop_de()
    b.add_hl_hl()
    b.add_hl_de()                    # 3 * A
    b.ld_de_label("PAIRTAB")
    b.add_hl_de()
    b.push_hl()
    b.pop_bc()                       # BC -> this byte's slot

    b.ld_hl_nn(0)
    b.ld_l_a()                       # keep the byte itself
    b.push_hl()
    b.push_bc()
    b.pop_hl()
    b.ld_de_nn(2)
    b.add_hl_de()
    b.ld_a_hl()                      # the length
    b.or_a()
    b.jr_nz("UP_PAIR")

    # A literal. Copy it, and stop once the second NUL has gone out.
    b.pop_hl()
    b.ld_a_l()
    b.ld_iyd_a(0)
    b.inc_iy()
    b.or_a()
    b.jr_nz("UP_ONE")
    b.ld_a_mem_label("NULSEEN")
    b.dec_a()
    b.ld_mem_label_a("NULSEEN")
    b.or_a()
    b.jr_nz("UP_ONE")
    b.ret()

    b.label("UP_PAIR")
    b.pop_hl()                       # discard the byte; the table has it now
    # The length has to wait on the stack rather than go straight into C: BC
    # still holds the slot pointer, and C is its low byte.
    b.push_af()
    b.push_bc()
    b.pop_hl()
    b.ld_de_nn(0)
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()                      # DE = offset into the blob
    b.ld_hl_label("BLOBBUF")
    b.add_hl_de()
    b.pop_af()
    b.ld_c_a()                       # C = length, and it is never zero here

    b.label("UP_COPY")
    b.ld_a_hl()
    b.ld_iyd_a(0)
    b.inc_iy()
    b.inc_hl()
    b.dec_c()
    b.jr_nz("UP_COPY")
    b.jr("UP_ONE")

    # LOADPAIRS: the expansion table off the card, once, into RAM. Also settles
    # where the offset table starts, which is the only thing about WIKI.DAT
    # whose position depends on the corpus.
    b.label("LOADPAIRS")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("SEEKOFF")
    b.xor_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_nn(len(TEXT_MAGIC) + 2)
    b.call("READ")

    b.ld_hl_label("IOBUF")
    b.ld_de_label("DATMAGIC")
    b.ld_b_n(len(TEXT_MAGIC))
    b.label("LP_CMP")
    b.ld_a_hl()
    b.ld_c_a()
    b.ld_a_de()
    b.sub_c()
    b.ret_nz()                       # A is nonzero: a card in the old layout
    b.inc_hl()
    b.inc_de()
    b.djnz("LP_CMP")

    b.ld_hl_nn(0)
    b.ld_a_mem_label("IOBUF", len(TEXT_MAGIC))
    b.ld_l_a()
    b.ld_a_mem_label("IOBUF", len(TEXT_MAGIC) + 1)
    b.ld_h_a()
    b.ld_mem_label_hl("BLOBLEN")

    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.ld_hl_label("PAIRTAB")
    b.ld_de_nn(3 * 256)
    b.call("READ")
    b.ld_a_mem_label("DATH")
    b.ld_c_a()
    b.ld_hl_label("BLOBBUF")
    b.ld_de_mem_label("BLOBLEN")
    b.call("READ")

    # Where the offsets begin: the header, the table, the blob, and the
    # document count that follows them.
    b.ld_hl_mem_label("BLOBLEN")
    b.ld_de_nn(len(TEXT_MAGIC) + 2 + 3 * 256 + 4)
    b.add_hl_de()
    b.ld_mem_label_hl("DATBASE")
    b.xor_a()
    b.ret()


def _emit_console(b: EZ80Builder) -> None:
    b.label("PRSTR")
    b.ld_a_hl()
    b.or_a()
    b.ret_z()
    b.rst(MOS_OUTCHAR)
    b.inc_hl()
    b.jr("PRSTR")

    b.label("PRNL")
    b.ld_a_n(13)
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(10)
    b.rst(MOS_OUTCHAR)
    b.ret()

    # PRWRAP: PRSTR, but breaking between words instead of wherever the column
    # runs out.
    #
    # Nothing in this repository has ever emitted a VDU sequence - every one of
    # the fifty-seven print sites pushes one character through `RST 10h` and
    # lets the terminal decide - which was invisible while a lead was 300
    # characters of one paragraph. `data/silo/authored/` put fifteen-hundred-byte
    # documents on the card and the screen started breaking words in half.
    #
    # No lookahead buffer: the whole article is already unpacked in TEXTBUF, so
    # the next word can be measured in place and the decision made before the
    # space is printed. A word longer than a line is not special-cased - it is
    # measured up to the width, fails to fit whatever the column, and starts a
    # line of its own.
    b.label("PRWRAP")
    b.xor_a()
    b.ld_mem_label_a("WRAPCOL")

    b.label("PW_NEXT")
    b.ld_a_hl()
    b.or_a()
    b.ret_z()
    b.cp_n(10)
    b.jr_z("PW_BREAK")
    b.cp_n(32)
    b.jr_z("PW_SPACE")
    b.rst(MOS_OUTCHAR)               # an ordinary character
    b.inc_hl()
    b.ld_a_mem_label("WRAPCOL")
    b.inc_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    # A newline the author wrote: `authored.py` keeps paragraph breaks because
    # they are the only formatting that survives to a screen with no wrap.
    b.label("PW_BREAK")
    b.inc_hl()
    b.call("PRNL")
    b.xor_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    b.label("PW_SPACE")
    b.inc_hl()                       # step over the space
    b.push_hl()
    b.ld_c_n(0)
    b.label("PW_MEAS")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("PW_MEASD")
    b.cp_n(32)
    b.jr_z("PW_MEASD")
    b.cp_n(10)
    b.jr_z("PW_MEASD")
    b.inc_hl()
    b.inc_c()
    b.ld_a_c()
    b.cp_n(WRAP_WIDTH)               # longer than a line: stop counting
    b.jr_nc("PW_MEASD")
    b.jr("PW_MEAS")

    b.label("PW_MEASD")
    b.pop_hl()
    b.ld_a_mem_label("WRAPCOL")
    b.or_a()
    b.jr_z("PW_NEXT")                # at the margin already: swallow the space
    b.add_a_c()
    b.inc_a()                        # and the space itself
    b.cp_n(WRAP_WIDTH + 1)
    b.jr_c("PW_FITS")
    b.call("PRNL")
    b.xor_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    b.label("PW_FITS")
    b.ld_a_n(32)
    b.rst(MOS_OUTCHAR)
    b.ld_a_mem_label("WRAPCOL")
    b.inc_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    b.label("READ_INPUT")
    b.xor_a()
    b.ld_mem_label_a("INPLEN")

    b.label("RI_LOOP")
    b.ld_a_n(MOS_GETKEY)
    b.rst(MOS_API)
    b.or_a()
    b.jr_z("RI_LOOP")
    b.cp_n(13)
    b.jr_z("RI_DONE")
    b.cp_n(8)
    b.jr_z("RI_DEL")
    b.cp_n(127)
    b.jr_z("RI_DEL")
    b.cp_n(32)
    b.jr_c("RI_LOOP")

    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.cp_n(MAX_INPUT_LEN)
    b.jr_nc("RI_LOOP")
    b.ld_hl_label("INPBUF")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_hl_c()
    b.ld_a_mem_label("INPLEN")
    b.inc_a()
    b.ld_mem_label_a("INPLEN")
    b.ld_a_c()
    b.rst(MOS_OUTCHAR)
    b.jr("RI_LOOP")

    b.label("RI_DEL")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("RI_LOOP")
    b.dec_a()
    b.ld_mem_label_a("INPLEN")
    for code in (8, 32, 8):
        b.ld_a_n(code)
        b.rst(MOS_OUTCHAR)
    b.jr("RI_LOOP")

    b.label("RI_DONE")
    b.call("PRNL")
    b.ret()



#: Stride of the resident paths table. A power of two so indexing it is three
#: doublings rather than a multiply, and wide enough for the longest path any
#: phrase means - which is two, with room left over.
PATH_STRIDE = 16

#: `libinfer.NUM_BUCKETS` is 128 trigram buckets; the `NUM_BUCKETS`
#: imported above is libsearch's 1,048,576 hash buckets. Same name, four
#: orders of magnitude apart, so the classifier code names its module.
#: The graph card's magic, from libgraphcard.
GRAPH_MAGIC = libgraphcard.MAGIC


def _emit_oracle(b: EZ80Builder, spec: OracleSpec) -> None:
    """Answer from the fact graph, and fall back to the search when it cannot.

    The four stages, each already measured on its own:

        which article   the BM25 scan that REPORT just did
        which relation  the phrasebook classifier, one forward pass
        the answer      a walk over the graph card
        otherwise       the article listing, which is the search build

    That last line is the whole reason this is one program rather than two: a
    machine that answers "the archive does not record that" and stops is worse
    than one that hands you the article it was reading. `liboracle` calls those
    FACT and SEARCH and marks which it gave you; here the difference is a fact
    printed plainly against a list of articles under a heading.
    """
    # CHECKGRAPH: the graph's magic, then that it was built for this corpus.
    #
    # The second half is the one that matters. A .GRF from a different build
    # has valid ids for a different article list, so it answers fluently and
    # wrongly - there is no other symptom, and no way to notice by reading the
    # output. A refuses on mismatch.
    b.label("CHECKGRAPH")
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("SEEKOFF")
    b.xor_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.ld_a_mem_label("GRFH")
    b.ld_c_a()
    b.call("SEEK")
    b.ld_a_mem_label("GRFH")
    b.ld_c_a()
    b.ld_hl_label("IOBUF")
    b.ld_de_nn(len(GRAPH_MAGIC) + 10)
    b.call("READ")

    b.ld_hl_label("IOBUF")
    b.ld_de_label("GRFMAGIC")
    b.ld_b_n(len(GRAPH_MAGIC))
    b.label("CG_LP")
    b.ld_a_hl()
    b.ld_c_a()
    b.ld_a_de()
    b.cp_c()
    b.jp_nz("CG_BAD")
    b.inc_hl()
    b.inc_de()
    b.djnz("CG_LP")

    # num_docs and digest follow the magic and two count bytes.
    b.ld_hl_mem_label("IOBUF", len(GRAPH_MAGIC) + 2)
    b.ld_de_nn(spec.num_docs & 0xFFFFFF)
    b.or_a()
    b.sbc_hl_de()
    b.jp_nz("CG_BAD")
    b.xor_a()
    b.ret()
    b.label("CG_BAD")
    b.ld_a_n(1)
    b.ret()

    b.label("ORACLE")
    b.call("REPORT")                 # scan only: fills BESTID and BESTSC
    b.ld_a_mem_label("BESTSC")
    b.or_a()
    b.jp_z("RP_SHOW")                # nothing matched at all

    b.call("TOKENIZE")
    b.call("INFER")
    b.call("ARGMAX")                 # -> RESULT, the phrase index

    # RESULT indexes the paths table: a step count, then that many
    # (relation, kind) pairs.
    b.ld_hl_mem_label("RESULT")
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()
    b.add_hl_hl()                    # RESULT * 16
    b.ld_de_label("PATHTAB")
    b.add_hl_de()
    b.ld_a_hl()                      # the step count
    b.cp_n(libgraphcard.REFUSE)
    b.jp_z("RP_IDK")                 # a phrase that is a refusal, not a path
    b.or_a()
    b.jp_z("RP_SHOW")                # a phrase with no walkable path
    b.push_hl()
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("GW_LEFT")
    b.ld_mem_label_a("GW_LEFT")      # widen the count to 24 bits
    b.pop_hl()
    b.inc_hl()
    b.ld_mem_label_hl("GW_STEPS")

    b.ld_hl_mem_label("BESTID")
    b.ld_mem_label_hl("GW_HERE")
    b.ld_hl_nn(spec.forward_at)
    b.ld_mem_label_hl("GW_FWD")
    b.ld_hl_nn(spec.forward_at + spec.num_edges * libgraphcard.EDGE_SIZE)
    b.ld_mem_label_hl("GW_REV")
    b.call("GW_FOLLOW")
    b.jp_c("RP_SHOW")                # no fact: hand over the articles instead

    b.call("PRNL")
    b.ld_hl_mem_label("GW_HERE")
    b.call("READ_ARTICLE")
    b.ld_hl_label("TEXTBUF")
    b.call("PRSTR")                  # the title alone: this is an answer
    b.ld_a_n(ord("."))
    b.rst(MOS_OUTCHAR)
    b.jp("PRNL")

    # A refusal. Not the article list: this corpus has no gaps, so the list is
    # never empty and offering it would be the fluent wrong answer wearing a
    # different hat. The machine says it does not know, and says nothing else.
    b.label("RP_IDK")
    b.call("PRNL")
    b.ld_hl_label("MSGIDK")
    b.call("PRSTR")
    b.jp("PRNL")



def _emit_classifier(b: EZ80Builder, spec: OracleSpec) -> None:
    """The phrasebook classifier and the graph walk, borrowed wholesale.

    Nothing here is new. `libnn.emit_tokenizer` hashes the query into buckets,
    `buildez80` emits the column-major layers and the argmax, and
    `buildgraphwalk` does the hop - this only supplies the buffers they expect
    and the paths table that turns a phrase index into a walk.

    That it works at all is because both builders were written to the same
    convention: `AgonPlatform` reads the query from `INPLEN` and `INPBUF`,
    which is exactly what the wiki program's line editor already fills.
    """
    import buildez80
    import buildgraphwalk
    import libagon
    import libinfer

    assert spec.model is not None
    model = spec.model
    plat = libagon.AgonPlatform()

    # buildez80's tokenizer, not libnn's: an activation is three bytes
    # here and libnn's writes two, which is what Platform.activation_size
    # documents as the boundary of what these two machines can share.
    buildez80._emit_tokenizer_helpers(b, plat, libinfer.FLAT)

    # The compact kernel, not the column one. Column unrolls a block per
    # input and runs ~24x faster, and on a resident model that is the right
    # trade; here it is 269KB of threaded code against 236KB of room once the
    # accumulator has taken its byte per article. Compact keeps the weights as
    # a data blob and interprets them, so the model costs 21KB rather than a
    # quarter of a megabyte - and a classifier runs once per question, beside
    # a search that has already gone to the card several times.
    b.label("INFER")
    b.ld_hl_label("BIASES")
    b.ld_mem_label_hl("BIASP")
    for i in range(model.num_layers):
        in_buf, out_buf = buildez80.layer_buffers(i, model.num_layers)
        b.label(f"LAYER{i + 1}")
        b.ld_hl_label(in_buf)
        b.ld_mem_label_hl("INBASE")
        b.ld_ix_label(out_buf)
        b.ld_bc_label(f"WTS{i + 1}")
        b.ld_a_n(0 if i == model.num_layers - 1 else 1)
        b.ld_mem_label_a("RELUF")
        b.call("LAYER")
    b.ret()

    buildez80._emit_argmax(b)
    buildgraphwalk.emit_walk(
        b, spec.num_edges, spec.types_at, spec.num_types,
        handle_label="GRFH", buffer_label="IOBUF", seekoff_label="SEEKOFF",
        climb_limit=spec.climb_limit)

    # After every routine that uses JR: from here the code is too long for a
    # relative jump to reach across, which is why buildez80 orders it so too.
    buildez80._emit_layer_compact(b)


def _emit_classifier_data(b: EZ80Builder, spec: OracleSpec) -> None:
    """Buffers the borrowed emitters address by name."""
    import buildez80
    import buildgraphwalk
    import libinfer

    assert spec.model is not None

    model = spec.model
    layer_sizes = model.layer_sizes

    # The scratch the borrowed emitters name. INPLEN is the wiki program's
    # own - both builders call the query length that - so it is not repeated.
    for name in ("TOKLEN", "TOKC1", "TOKC2", "TOKC3", "TOKPOS",
                 "CTXPOS", "CTXN", "GENCNT", "RELUF",
                 "TMP0", "TMP1", "TMP2"):
        b.label(name)
        b.db(0)
    for name in ("SPSAV", "SPTMP", "INBASE", "BIASP", "TMPV", "MAXI", "IDX",
                 "RESULT"):
        b.label(name)
        b.d24(0)

    # INBUF and CTXBUF must stay adjacent: layer 1 reads a single vector
    # through the INBUF label, and a phrasebook only fills the first half.
    b.label("INBUF")
    b.ds(libinfer.NUM_BUCKETS * 3)
    b.label("CTXBUF")
    b.ds(libinfer.NUM_BUCKETS * 3)
    assert b.labels["CTXBUF"] == b.labels["INBUF"] + libinfer.NUM_BUCKETS * 3

    hidden = layer_sizes[1:-1] or [layer_sizes[-1]]
    # The compact kernel pops one activation past the last weight, so every
    # buffer it walks carries three bytes of slack.
    b.ds(3)
    b.label("BUF_A")
    b.ds(max(hidden) * 3 + 3)
    b.label("BUF_B")
    b.ds(max(hidden) * 3 + 3)
    b.label("OUTBUF")
    b.ds(model.output_size * 3)
    b.label("OUTEND")
    b.ds(3)

    b.label("BIASES")
    b.blob(b"".join(buildez80.encode_biases(bias) for bias in model.biases()))
    for i, weights in enumerate(model.weights(), start=1):
        b.label(f"WTS{i}")
        b.blob(buildez80.encode_weights(weights))

    # Only the oracle build refuses anything, so the message lives here rather
    # than beside MSGNONE - a search card would otherwise carry 24 bytes it can
    # never print, and every article the accumulator holds is worth a byte.
    b.label("MSGIDK")
    b.ascii("I do not know that one.")
    b.db(0)

    # The paths table: one fixed-width row per phrase, so the index is three
    # doublings rather than a multiply. A phrase whose path this cannot walk -
    # an inverse, for now - gets a zero count and falls back to the search, and
    # a phrase that is a refusal gets REFUSE and falls back to nothing.
    b.label("PATHTAB")
    for steps in spec.paths:
        row = [libgraphcard.REFUSE] if steps is None else [len(steps)]
        for relation, kind in (steps or ()):
            row += [relation, kind]
        assert len(row) <= PATH_STRIDE, f"path too long: {steps}"
        b.emit(*row, *([0] * (PATH_STRIDE - len(row))))

    b.label("GRFMAGIC")
    b.emit(*GRAPH_MAGIC)
    b.label("GRFNAME")
    b.ascii(spec.graph_name)
    b.db(0)
    b.label("GRFH")
    b.db(0)
    buildgraphwalk.emit_cells(b)


def _emit_data(b: EZ80Builder, num_docs: int, acc_base: int, pages: int,
               index_name: str, text_name: str) -> None:
    b.label("BANNER")
    b.ascii(f"Simple English Wikipedia - {num_docs:,} articles")
    b.db(13)
    b.db(10)
    b.ascii("Type a question, or ! to quit.")
    b.db(13)
    b.db(10)
    b.db(0)

    b.label("PROMPT")
    b.ascii("? ")
    b.db(0)
    b.label("MSGNONE")
    b.ascii("Nothing on the card matches that.")
    b.db(13)
    b.db(10)
    b.db(0)
    b.label("MSGNOCARD")
    b.ascii(f"Cannot open {index_name} and {text_name}.")
    b.db(13)
    b.db(10)
    b.db(0)
    b.label("MSGBADCARD")
    b.ascii("That card was written for a different format.")
    b.db(13)
    b.db(10)
    b.db(0)

    b.label("IDXNAME")
    b.ascii(index_name)
    b.db(0)
    b.label("DATNAME")
    b.ascii(text_name)
    b.db(0)
    b.label("MAGICSTR")
    b.ascii(MAGIC.decode())
    b.label("DATMAGIC")
    b.ascii(TEXT_MAGIC.decode())

    for name in ("IDXH", "DATH", "INPLEN", "TOKPOS", "TOKLEN", "NSCORED",
                 "SHOWN", "HCNT", "NULSEEN", "NTGLUED", "WRAPCOL"):
        b.label(name)
        b.db(0)
    b.label("BESTSC")
    b.ds(TOP_K)

    # RUNDOC is the running document id a term's gaps add up to, and CONSUMED
    # is how many bytes of the current block held whole postings - see
    # ST_STREAM for why the second is not simply the block size.
    for name in ("SEEKOFF", "HTMP", "HTMP2", "NPOST", "NTHIS", "NLEFT",
                 "RUNDOC", "CONSUMED", "DATBASE", "BLOBLEN",
                 "SCANID", "PGLEFT", "PACC"):
        b.label(name)
        b.d24(0)
        b.db(0)                      # SEEKOFF's high byte; harmless elsewhere
    b.label("BESTID")
    b.ds24(TOP_K)

    b.label("TOKBUF")
    b.ds(MAX_TOKEN_LEN + 1)
    b.label("INPBUF")
    b.ds(MAX_INPUT_LEN + 1)
    b.label("IOBUF")
    b.ds(CHUNK + 16)
    # The four unpacking buffers live above the program rather than in it, so
    # the .bin does not carry eleven thousand zeros to reserve them. PAIRTAB is
    # the expansion table, read off the card at startup rather than built in, so
    # a card rebuilt from a different corpus still runs on this binary.
    scratch = scratch_base(num_docs)
    b.equ("PACKBUF", scratch + PACKBUF_AT)
    b.equ("TEXTBUF", scratch + TEXTBUF_AT)
    b.equ("PAIRTAB", scratch + PAIRTAB_AT)
    b.equ("BLOBBUF", scratch + BLOBBUF_AT)

    # One flag per 256-article page: small enough to live in the image
    # (1,110 bytes for the full corpus), where a `ds` costs real zeros.
    b.label("PAGE_TAB")
    b.ds(pages)


#: Where the bucket table starts: straight after the header libsearch writes.
TABLE_AT = 6 + 1 + 1 + 4 + 4 + 4

#: The bucket count is a power of two, so masking a hash to it is one AND on
#: the top byte of a 24-bit value rather than a 24-bit operation the eZ80 lacks.
assert NUM_BUCKETS & (NUM_BUCKETS - 1) == 0, "bucket count must be a power of two"
assert NUM_BUCKETS <= 1 << 24, "a bucket index has to fit in 24 bits"

#: `libsearch` refuses an article larger than what READ_ARTICLE reads and
#: UNPACK unpacks into. It has to be told those sizes, and this is the only
#: place that knows them, so the two are pinned against each other here rather
#: than being written down twice and drifting apart once.
assert MAX_PACKED_ARTICLE == CHUNK, "the build refuses what the device reads"
assert MAX_ARTICLE == 2 * CHUNK, "the build refuses what TEXTBUF holds"
BUCKET_MASK_HI = ((NUM_BUCKETS - 1) >> 16) & 0xFF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, required=True,
                        help="Articles on the card, which sizes the accumulator")
    parser.add_argument("--output", "-o", default="WIKI.bin")
    args = parser.parse_args()

    builder = build(args.docs)
    builder.save(args.output)
    print(f"{args.output}: {len(builder.code):,} bytes, "
          f"accumulator at {builder.accumulator:06X}h "
          f"({args.docs:,} bytes)")


if __name__ == "__main__":
    main()
