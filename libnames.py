"""
A name index for the card: a typed name to a document, with no BM25 in it.

    SILO.NAM   [ZNAME1][count:4]  then count records of (h1:3, h2:3, doc:3)

Resolving a person by name through the search index costs a resident byte
per article and a floor of `1,910 + 37 x pages` instructions a query, because
the search is built to rank prose. A name is not prose. Ten thousand of
them want a hash: the title, normalised, to two 24-bit hashes, sorted, and
binary-searched on the card in fourteen probes of nine bytes. Zero
accumulator, no tokenizer, and the answer is exact or absent.

## Why two hashes and not one

An eZ80 compares 24-bit numbers, so a key is 24 bits or a chain of them.
One 24-bit hash over thirteen thousand keys collides about five times a
corpus, and a collision here is a record printed about the wrong person.
Two independent 24-bit hashes make that one in a few hundred billion, and
`build` refuses the corpus outright if it happens anyway - a loud failure
at build time over a quiet one on the device.

## What a key is

`normalize` is the whole contract between this file and the eZ80 routine
that hashes what the player typed, so it is written the way the device does
it: byte by byte, letters upper-cased, digits kept, one space between
words, everything else dropped. `Alexander E. Wong` is `ALEXANDER E WONG`,
and so is `alexander e wong`, and so is `ALEXANDER  E.  WONG`.

A title shaped `First M. Last` also keys under `FIRST LAST`, because that is
how people say names. 2,264 of the silo's ten thousand share a first and
last name with somebody, so that key lands on several records - and the
device lists them, which is the honest answer to a name that is not enough.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

MAGIC = b"ZNAME1"
HEADER = struct.Struct("<6sI")
RECORD = 9
MASK = 0xFFFFFF

#: `First M. Last`, the shape a second key is made for.
INITIAL = re.compile(r"^(\S+) (\S)\. (\S+)$")


def normalize(text: str) -> str:
    """The key the device would compute from this text.

    Byte-wise, because the device is: a-z to A-Z, A-Z and 0-9 kept, a run of
    spaces to one space between kept characters, everything else dropped.
    Bytes outside ASCII are "everything else", so an accented name keys on
    its ASCII letters - the same on both sides, which is all that matters.
    """
    out = bytearray()
    pending = False
    for byte in text.encode("utf-8"):
        if 0x61 <= byte <= 0x7A:
            byte -= 0x20
        if 0x41 <= byte <= 0x5A or 0x30 <= byte <= 0x39:
            if pending and out:
                out.append(0x20)
            pending = False
            out.append(byte)
        elif byte == 0x20:
            pending = True
    return out.decode("ascii")


def hashes(key: str) -> tuple[int, int]:
    """Two 24-bit hashes, each a shift-and-add the eZ80 does in a few lines.

    `h1` multiplies by 31 (shift five, subtract) and `h2` by 33 (shift five,
    add), so the device computes both from one loop over the bytes with no
    multiply - `MLT` is 8-bit and would not help.
    """
    h1 = h2 = 0
    for byte in key.encode("ascii"):
        h1 = ((h1 << 5) - h1 + byte) & MASK
        h2 = ((h2 << 5) + h2 + byte) & MASK
    return h1, h2


def keys_for(title: str) -> list[str]:
    """The keys a title is found under: itself, and `First Last` if it has
    a middle initial. Empty keys - a title of punctuation - are none."""
    keys = [normalize(title)]
    if (m := INITIAL.match(title)) is not None:
        keys.append(normalize(f"{m.group(1)} {m.group(3)}"))
    return [k for k in dict.fromkeys(keys) if k]


def build(titles: list[str]) -> list[tuple[int, int, int]]:
    """Sorted (h1, h2, doc) records, one per key per title.

    Raises if two *different* keys share both hashes, which would make the
    device print the wrong person's record with nothing on the screen to
    say so. Duplicate keys - two people called Amanda Wilson - are not a
    collision: they are two records under one name, and the device lists
    both.
    """
    seen: dict[tuple[int, int], str] = {}
    records: set[tuple[int, int, int]] = set()
    for doc, title in enumerate(titles):
        for key in keys_for(title):
            pair = hashes(key)
            other = seen.setdefault(pair, key)
            if other != key:
                raise ValueError(f"{key!r} and {other!r} hash alike; the "
                                 f"device could not tell them apart")
            records.add((*pair, doc))
    return sorted(records)


def write(records: list[tuple[int, int, int]], path: Path) -> dict[str, int]:
    """Header and fixed-width records, so record n is at `HEADER.size + 9n`."""
    with path.open("wb") as fh:
        fh.write(HEADER.pack(MAGIC, len(records)))
        for h1, h2, doc in records:
            fh.write(h1.to_bytes(3, "little"))
            fh.write(h2.to_bytes(3, "little"))
            fh.write(doc.to_bytes(3, "little"))
    return {"bytes": path.stat().st_size, "records": len(records)}


class CardNames:
    """Look a name up exactly as the eZ80 build does: a lower-bound binary
    search over the records, then a scan while the key holds. Literal on
    purpose, so that this and the generated code can disagree."""

    def __init__(self, path: Path) -> None:
        self.fh = path.open("rb")
        magic, self.count = HEADER.unpack(self.fh.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a {MAGIC.decode()} name index")
        self.probes = 0

    def close(self) -> None:
        self.fh.close()

    def _record(self, index: int) -> tuple[int, int, int]:
        self.probes += 1
        self.fh.seek(HEADER.size + RECORD * index)
        raw = self.fh.read(RECORD)
        return (int.from_bytes(raw[0:3], "little"),
                int.from_bytes(raw[3:6], "little"),
                int.from_bytes(raw[6:9], "little"))

    def lookup(self, name: str) -> list[int]:
        """Every document filed under this name, in id order."""
        key = hashes(normalize(name))
        if not normalize(name):
            return []
        low, high = 0, self.count
        while low < high:
            mid = (low + high) // 2
            h1, h2, _ = self._record(mid)
            if (h1, h2) < key:
                low = mid + 1
            else:
                high = mid
        docs = []
        while low < self.count:
            h1, h2, doc = self._record(low)
            if (h1, h2) != key:
                break
            docs.append(doc)
            low += 1
        return docs
