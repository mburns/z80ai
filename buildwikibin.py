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
    scan the accumulator for the three largest
    read those three titles and leads off the card and print them

There is no arithmetic beyond addition. BM25's multiply, divide and per-document
length all happened at build time, and each posting arrives as a five-bit weight
that is simply added. That is what keeps the accumulator at one byte per
article - 277KB for 284,000 of them, resident, no sharding - and it is why this
runs in milliseconds rather than seconds.

## Memory

The accumulator does not live in the image. `ds` emits literal zeros, so a
284,000-byte buffer would put 284,000 zeros in the .bin; instead it sits at a
fixed address above the program, and the build asserts the two do not overlap.

## Firmware

Unlike the inference builds, this needs to seek: mos_fopen, mos_flseek,
mos_fread and mos_fclose rather than mos_load alone. That widens the surface
that `libhost.AgonHost` models but real MOS has never been checked against - see
tools/mostest.py and the note in EZ80.md.
"""

from __future__ import annotations

import argparse

from libez80 import AGON_LOAD_ADDR, AGON_SRAM_TOP, EZ80Builder, agon_header
from libsearch import MAGIC, MAX_QUERY_TERMS, NUM_BUCKETS

MOS_API = 0x08
MOS_OUTCHAR = 0x10
MOS_GETKEY = 0x00
MOS_FOPEN = 0x0A
MOS_FCLOSE = 0x0B
MOS_FREAD = 0x1A
MOS_FLSEEK = 0x1C
FA_READ = 0x01

MAX_INPUT_LEN = 120
MAX_TOKEN_LEN = 32
TOP_K = 3
#: Bytes pulled per postings read. One SD block is 512; this is four of them,
#: which keeps a long postings list to a handful of round trips.
CHUNK = 2048
#: Stack margin below the top of SRAM, matching the inference builds.
STACK_MARGIN = 0x1000


def accumulator_base(num_docs: int) -> int:
    """Where the score accumulator sits: as high as it can, below the stack."""
    return (AGON_SRAM_TOP - STACK_MARGIN - num_docs) & ~0xFF


def build(num_docs: int, index_name: str = "WIKI.IDX",
          text_name: str = "WIKI.DAT",
          org: int = AGON_LOAD_ADDR) -> EZ80Builder:
    """Emit the search program for a card holding ``num_docs`` articles."""
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
    b.call("REPORT")
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
    # One byte per article, cleared per query. Whole-corpus memset: 284,000
    # stores, a few milliseconds, and far cheaper than tracking which slots the
    # last query touched.
    b.label("CLEAR_ACC")
    b.ld_hl_nn(acc_base)
    b.ld_de_nn(acc_base + 1)
    b.ld_bc_nn(num_docs - 1)
    b.ld_hl_n(0)
    b.ldir()
    b.ret()

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

    b.label("NT_DONE")
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
    _emit_report(b, num_docs, acc_base)
    _emit_console(b)
    _emit_data(b, num_docs, acc_base, index_name, text_name)

    top = b.org + len(b.code)
    assert top <= acc_base, (
        f"the image reaches {top:06X}h but the accumulator starts at "
        f"{acc_base:06X}h; the corpus is too large to score in SRAM")
    b.accumulator = acc_base
    b.num_docs = num_docs
    return b


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
    # Then skip 4 * NPOST bytes of postings.
    b.ld_hl_mem_label("NPOST")
    b.add_hl_hl()
    b.add_hl_hl()
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
    b.label("ST_STREAM")
    b.label("STS_BLOCK")
    b.ld_hl_mem_label("NPOST")
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()
    b.ret_z()

    b.ld_hl_mem_label("NPOST")
    b.ld_de_nn(CHUNK // 4)
    b.or_a()
    b.sbc_hl_de()
    b.jr_c("STS_PART")
    b.ld_hl_nn(CHUNK // 4)
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
    b.add_hl_hl()                    # bytes = 4 * postings
    b.ld_hl_mem_label("NTHIS")
    b.add_hl_hl()
    b.add_hl_hl()
    b.push_hl()
    b.pop_de()
    b.ld_hl_label("IOBUF")
    b.call("READ")

    # Advance the file cursor past what we just consumed.
    b.ld_hl_mem_label("NTHIS")
    b.add_hl_hl()
    b.add_hl_hl()
    b.ld_de_mem_label("SEEKOFF")
    b.add_hl_de()
    b.ld_mem_label_hl("SEEKOFF")
    b.jp_nc("STS_NC")
    b.ld_a_mem_label("SEEKOFF", 3)
    b.inc_a()
    b.ld_mem_label_a("SEEKOFF", 3)
    b.label("STS_NC")

    b.ld_hl_mem_label("NPOST")
    b.ld_de_mem_label("NTHIS")
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("NPOST")

    # Add each (doc, weight) into the accumulator, saturating at 255.
    b.ld_ix_label("IOBUF")
    b.ld_hl_mem_label("NTHIS")
    b.ld_mem_label_hl("NLEFT")

    b.label("STS_ONE")
    b.ld_hl_mem_label("NLEFT")
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()
    b.jp_z("STS_BLOCK")
    b.ld_hl_mem_label("NLEFT")
    b.ld_de_nn(1)
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("NLEFT")

    b.ld_hl_ixd(0)                   # document id
    b.ld_bc_nn(acc_base)
    b.add_hl_bc()
    b.push_hl()
    b.pop_iy()
    b.ld_a_iyd(0)
    b.ld_c_a()
    b.ld_a_ixd(3)                    # the five-bit weight
    b.add_a_c()
    b.jr_nc("STS_STORE")
    b.ld_a_n(255)                    # saturate rather than wrap
    b.label("STS_STORE")
    b.ld_iyd_a(0)

    b.ld_de_nn(4)
    b.push_ix()
    b.pop_hl()
    b.add_hl_de()
    b.push_hl()
    b.pop_ix()
    b.jp("STS_ONE")


def _emit_report(b: EZ80Builder, num_docs: int, acc_base: int) -> None:
    """Find the best three scores and print their articles."""
    b.label("REPORT")
    for k in range(TOP_K):
        b.ld_hl_nn(0)
        b.ld_mem_label_hl("BESTID", 3 * k)
        b.xor_a()
        b.ld_mem_label_a("BESTSC", k)

    # One pass over the accumulator, keeping three. First-wins on ties, which
    # matches libsearch and therefore the reference implementation.
    b.ld_ix_nn(acc_base)
    b.ld_hl_nn(0)
    b.ld_mem_label_hl("SCANID")

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
    b.jp_z("RP_SHOW")                # RP_OFFER sits between, well past JR range
    b.push_ix()
    b.pop_hl()
    b.ld_de_nn(1)
    b.add_hl_de()
    b.push_hl()
    b.pop_ix()
    b.jp("RP_SCAN")

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

    # The offset table sits after a 4-byte count: 4 + 4 * id.
    b.add_hl_hl()
    b.add_hl_hl()
    b.ld_de_nn(4)
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
    b.ld_hl_label("TEXTBUF")
    b.ld_de_nn(CHUNK)
    b.call("READ")

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
    b.call("PRSTR")
    b.call("PRNL")
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


def _emit_data(b: EZ80Builder, num_docs: int, acc_base: int,
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

    for name in ("IDXH", "DATH", "INPLEN", "TOKPOS", "TOKLEN", "NSCORED",
                 "SHOWN", "HCNT"):
        b.label(name)
        b.db(0)
    b.label("BESTSC")
    b.ds(TOP_K)

    for name in ("SEEKOFF", "HTMP", "HTMP2", "NPOST", "NTHIS", "NLEFT",
                 "SCANID"):
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
    b.label("TEXTBUF")
    b.ds(CHUNK + 16)


#: Where the bucket table starts: straight after the header libsearch writes.
TABLE_AT = 6 + 1 + 1 + 4 + 4 + 4

#: The bucket count is a power of two, so masking a hash to it is one AND on
#: the top byte of a 24-bit value rather than a 24-bit operation the eZ80 lacks.
assert NUM_BUCKETS & (NUM_BUCKETS - 1) == 0, "bucket count must be a power of two"
assert NUM_BUCKETS <= 1 << 24, "a bucket index has to fit in 24 bits"
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
