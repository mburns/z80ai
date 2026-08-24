"""
Full-text search over a corpus too large to hold in RAM.

This is the golden model for the Agon search build, the way libinfer is for the
inference builds: it reads the *same two card files* the eZ80 reads, so a
disagreement between them is a bug in one of them rather than a difference of
opinion about the format.

Why not the trigram encoder the models use: it compresses a document into 128
buckets and throws away which words matched, which is the one thing retrieval
needs. Measured over Simple English Wikipedia it answers 2 of 13 probe queries
against BM25's 11, and returns "Bures Hamlet" for "who wrote hamlet". It exists
to feed a fixed 128-input network. This is ordinary inverted-index retrieval,
which suits a 1970s instruction set rather well.

## What the device has to do, and what it does not

All the arithmetic happens here, at build time. BM25 needs floating point, a
per-document length and an inverse document frequency; none of that is welcome
on an eZ80, and the per-document length alone would be another 277KB resident.

So each posting stores its **final contribution**, quantized to five bits. The
device clears an accumulator, adds a byte per posting, and takes the largest -
no multiply, no divide, no per-document table. Five bits is chosen so eight
query terms cannot overflow one byte (8 x 31 = 248), which is what keeps the
accumulator at one byte per article: 277KB for the whole encyclopedia, resident,
no sharding.

## The two files

``WIKI.IDX``  a hashed dictionary and the postings.

    Hashing rather than a sorted term table with a binary search: one seek per
    query term instead of nineteen, and the eZ80 needs no comparison logic. The
    term string is stored with its postings so a hash collision is detected and
    skipped rather than silently scoring the wrong documents.

``WIKI.DAT``  article titles and leads, behind an offset table.

    Two seeks per displayed result, which is three results per query.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"ZWIKI1"
#: Buckets in the hashed dictionary. Comfortably more than the ~427,000 terms
#: Simple English Wikipedia produces, so most buckets hold one term.
NUM_BUCKETS = 1 << 20
#: Bits per posting weight. Eight terms at 31 each is 248, so a one-byte
#: accumulator cannot overflow - which is what keeps it at one byte per article.
WEIGHT_BITS = 5
MAX_WEIGHT = (1 << WEIGHT_BITS) - 1
#: Query terms scored. The eighth is already noise, and more would overflow.
MAX_QUERY_TERMS = 8

#: A title says what an article *is*; a lead merely mentions things.
TITLE_WEIGHT = 3
K1, B = 1.5, 0.75

WORD = re.compile(r"[a-z0-9]+")

#: Dropped before indexing. BM25's idf already demotes them, but they are a
#: third of every query and skipping them saves the device a seek each.
STOPWORDS = frozenset([
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "there", "here",
    "what", "who", "whom", "how", "why", "when", "where", "which",
    "do", "does", "did", "make", "made", "makes",
    "have", "has", "had", "will", "shall", "may", "might", "must",
    "me", "you", "i", "we", "they", "he", "she", "him", "her", "his",
    "my", "your", "our", "their",
    "about", "tell", "can", "could", "would", "should",
    "from", "with", "as", "by", "into", "over", "under", "than",
    "not", "no", "yes", "but", "if", "then",
    "so", "such", "other", "some", "any", "all", "most", "more", "much", "many",
])


def tokenize(text: str) -> list[str]:
    """Words worth indexing, lowercased. Single characters carry nothing."""
    return [w for w in WORD.findall(text.lower())
            if len(w) > 1 and w not in STOPWORDS]


def term_hash(term: str) -> int:
    """Multiply-by-31 rolling hash, 24-bit, masked to the bucket count.

    The same shape as libinfer.hash16, and for the same reason: 31 is
    ``(h << 5) - h``, which on an eZ80 is five ``ADD HL,HL`` and a subtract.
    FNV would need a 32-bit multiply by 0x01000193 per character, which is a
    lot of code to save nothing - the only job here is spreading 427,000 terms
    over a million buckets, and collisions are detected and skipped anyway.
    """
    h = 0
    for ch in term.encode("utf-8"):
        h = ((h * 31) + ch) & 0xFFFFFF
    return h & (NUM_BUCKETS - 1)


@dataclass
class Posting:
    doc: int
    weight: int


@dataclass
class Index:
    """A built index, before it is written to card files."""

    titles: list[str]
    leads: list[str]
    #: term -> its postings, already weighted and quantized.
    postings: dict[str, list[Posting]]

    @property
    def num_docs(self) -> int:
        return len(self.titles)


def build(titles: list[str], leads: list[str],
          aliases: dict[int, list[str]] | None = None,
          report=lambda _msg: None) -> Index:
    """Index a corpus, returning postings weighted by BM25 and quantized.

    ``aliases`` are alternate names for a document - Wikipedia's redirects -
    indexed at title weight. They are why `jane austin` finds Jane Austen:
    nothing here does fuzzy matching, so every spelling that should work has to
    be a term someone wrote down.
    """
    aliases = aliases or {}
    raw: dict[str, dict[int, int]] = {}
    length = [0] * len(titles)

    def add(doc: int, text: str, weight: int) -> None:
        for term in tokenize(text):
            bucket = raw.setdefault(term, {})
            bucket[doc] = bucket.get(doc, 0) + weight
            length[doc] += weight

    for doc, (title, lead) in enumerate(zip(titles, leads, strict=True)):
        add(doc, title.replace("_", " "), TITLE_WEIGHT)
        add(doc, lead, 1)
        for alias in aliases.get(doc, ()):
            add(doc, alias.replace("_", " "), TITLE_WEIGHT)
        if doc % 50_000 == 0 and doc:
            report(f"  indexed {doc:,} documents")

    num_docs = len(titles)
    avg_len = sum(length) / num_docs if num_docs else 1.0
    report(f"  {len(raw):,} terms, mean document length {avg_len:.0f}")

    # BM25, then quantized. The scale is set by the largest weight in the whole
    # index so the quantization is one global mapping the device need not know.
    scored: dict[str, list[tuple[int, float]]] = {}
    largest = 0.0
    for term, docs in raw.items():
        idf = math.log(1 + (num_docs - len(docs) + 0.5) / (len(docs) + 0.5))
        entries = []
        for doc, tf in docs.items():
            norm = 1 - B + B * length[doc] / avg_len
            weight = idf * tf * (K1 + 1) / (tf + K1 * norm)
            largest = max(largest, weight)
            entries.append((doc, weight))
        scored[term] = entries

    scale = MAX_WEIGHT / largest if largest else 1.0
    postings = {
        term: [Posting(doc, max(1, min(MAX_WEIGHT, round(w * scale))))
               for doc, w in entries]
        for term, entries in scored.items()
    }
    return Index(titles=titles, leads=leads, postings=postings)


# --- the card files -----------------------------------------------------------


def _u24(value: int) -> bytes:
    return value.to_bytes(3, "little")


def write_index(index: Index, path: Path) -> dict:
    """Write WIKI.IDX: a bucket table, then a chain of terms per bucket.

    Each bucket's chain is contiguous, so a query term is one seek and one
    sequential read - no second lookup, no pointer chasing across the card.
    """
    chains: dict[int, list[str]] = {}
    for term in sorted(index.postings):           # sorted: reproducible output
        chains.setdefault(term_hash(term), []).append(term)

    blocks: dict[int, bytes] = {}
    for bucket, terms in chains.items():
        out = bytearray()
        for term in terms:
            entries = index.postings[term]
            encoded = term.encode("utf-8")[:255]
            out += bytes([len(encoded)]) + encoded + _u24(len(entries))
            for posting in entries:
                out += _u24(posting.doc) + bytes([posting.weight])
        out += b"\x00"                            # end of chain
        blocks[bucket] = bytes(out)

    header = struct.calcsize("<6sBBIII") + 4 * NUM_BUCKETS
    offsets = {}
    cursor = header
    for bucket in sorted(blocks):
        offsets[bucket] = cursor
        cursor += len(blocks[bucket])

    with path.open("wb") as fh:
        fh.write(struct.pack("<6sBBIII", MAGIC, WEIGHT_BITS, MAX_QUERY_TERMS,
                             index.num_docs, NUM_BUCKETS, len(index.postings)))
        table = bytearray(4 * NUM_BUCKETS)
        for bucket, offset in offsets.items():
            struct.pack_into("<I", table, 4 * bucket, offset)
        fh.write(table)
        for bucket in sorted(blocks):
            fh.write(blocks[bucket])

    return {
        "bytes": path.stat().st_size,
        "terms": len(index.postings),
        "postings": sum(len(p) for p in index.postings.values()),
        "buckets_used": len(blocks),
    }


def write_text(index: Index, path: Path) -> dict:
    """Write WIKI.DAT: an offset table, then title and lead per document."""
    body = bytearray()
    offsets = []
    for title, lead in zip(index.titles, index.leads, strict=True):
        offsets.append(len(body))
        body += title.encode("utf-8", "replace") + b"\x00"
        body += lead.encode("utf-8", "replace") + b"\x00"

    table_bytes = 4 * len(offsets)
    with path.open("wb") as fh:
        fh.write(struct.pack("<I", len(offsets)))
        for offset in offsets:
            fh.write(struct.pack("<I", offset + 4 + table_bytes))
        fh.write(body)
    return {"bytes": path.stat().st_size}


# --- the reference searcher ---------------------------------------------------


class CardSearch:
    """Search the card files exactly as the eZ80 build does.

    Deliberately literal: one seek per term, a byte accumulator with saturating
    adds, first-wins on ties. If this and the generated code disagree, one of
    them is wrong - which is the whole point of having it.
    """

    def __init__(self, index_path: Path, text_path: Path) -> None:
        self.index = index_path.open("rb")
        self.text = text_path.open("rb")

        magic, weight_bits, max_terms, num_docs, buckets, terms = struct.unpack(
            "<6sBBIII", self.index.read(struct.calcsize("<6sBBIII")))
        if magic != MAGIC:
            raise ValueError(f"{index_path} is not a {MAGIC.decode()} index")
        self.weight_bits = weight_bits
        self.max_terms = max_terms
        self.num_docs = num_docs
        self.num_buckets = buckets
        self.num_terms = terms
        self._table_at = struct.calcsize("<6sBBIII")

        self.text.seek(0)
        self.text_count = struct.unpack("<I", self.text.read(4))[0]

    def close(self) -> None:
        self.index.close()
        self.text.close()

    def _bucket_offset(self, bucket: int) -> int:
        self.index.seek(self._table_at + 4 * bucket)
        return struct.unpack("<I", self.index.read(4))[0]

    def _postings(self, term: str) -> list[Posting]:
        offset = self._bucket_offset(term_hash(term))
        if offset == 0:
            return []
        self.index.seek(offset)
        while True:
            length = self.index.read(1)[0]
            if length == 0:                        # end of chain
                return []
            found = self.index.read(length).decode("utf-8", "replace")
            count = int.from_bytes(self.index.read(3), "little")
            payload = self.index.read(4 * count)
            if found == term:
                return [Posting(int.from_bytes(payload[i:i + 3], "little"),
                                payload[i + 3])
                        for i in range(0, len(payload), 4)]
            # A different term in the same bucket: a collision, so keep walking.

    def article(self, doc: int) -> tuple[str, str]:
        self.text.seek(4 + 4 * doc)
        offset = struct.unpack("<I", self.text.read(4))[0]
        self.text.seek(offset)
        chunk = self.text.read(4096).split(b"\x00")
        return (chunk[0].decode("utf-8", "replace"),
                chunk[1].decode("utf-8", "replace"))

    def search(self, query: str, top: int = 3) -> list[tuple[int, int]]:
        """(document, score) for the best matches, largest first."""
        accumulator = bytearray(self.num_docs)
        for term in tokenize(query)[:self.max_terms]:
            for posting in self._postings(term):
                total = accumulator[posting.doc] + posting.weight
                accumulator[posting.doc] = min(255, total)   # saturating

        best: list[tuple[int, int]] = []
        for doc, score in enumerate(accumulator):
            if score and (len(best) < top or score > best[-1][1]):
                best.append((doc, score))
                best.sort(key=lambda pair: (-pair[1], pair[0]))
                del best[top:]
        return best
