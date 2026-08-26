"""
The fact graph as a card file, and the reader an eZ80 will imitate.

`libgraph` walks the graph in SQL. This is the same graph laid out so a machine
with no arithmetic beyond addition can walk it: fixed-width records in sorted
order, so a hop is a binary search and nothing else.

## Why this fits a Z80 when reading prose does not

Every capability this project has measured runs into the same wall - answering
from an article means comprehension, which is out of reach. A graph hop does
not. It is a comparison and a seek, repeated about eighteen times for 150,335
edges, and the eZ80 is perfectly capable of comparing two 24-bit numbers.

    forward   (subject, relation, object)   sorted   1,028 KB
    reverse   (object, relation, subject)   sorted   1,028 KB
    types     sorted doc ids per kind                   0.5 KB

Two copies of the same edges, because "who was born in Edinburgh" is the same
question read from the other end and a second sort is cheaper than any index.
Roughly 2MB, on a card holding 32GB.

## Ids, and the way this goes wrong

An edge names articles by **document id**, which is a position in the search
card's corpus - not `article.id`. `buildwikisearch.py` renumbers from zero, and
`--limit` reorders the corpus by notability, so the same title is a different
id in a limited card than in a full one.

Nothing detects that by reading a wrong answer: id 4,102 is a valid article
either way, so a mismatched pair of files answers fluently and wrongly. The
header therefore carries `num_docs` and a digest of the title list, and
`CardGraph.check` refuses a graph built against a different corpus. That is the
only guard, because there is no other symptom.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"ZGRAF1"

#: subject/object are 24-bit document ids, relation is an index into the
#: relation table. Fixed width is the whole point: record n is at n * EDGE_SIZE,
#: so a binary search needs no offset table and no second file.
EDGE_SIZE = 7

HEADER = struct.Struct("<6sBBIIIIIII")

#: A step in a path. `kind` is 0xFF for an ordinary hop, or a type id for a
#: climb - "repeat this relation until the value is of that type", which is
#: what "what country was X born in" actually asks for.
PLAIN = 0xFF
STEP = struct.Struct("<BB")

#: How many times a climb may step before giving up, and the *default* for a
#: card rather than a property of one: `buildwikisearch --climb-limit` picks
#: the number a given card is built with, and the walk routine carries it as an
#: immediate.
#:
#: It lives here because three walkers have to agree about it - `libgraph` in
#: SQL, `CardGraph.follow` reading the card, and the eZ80 routine
#: `buildgraphwalk` emits - and until it did, all three held their own literal
#: `6` with a comment in one of them pointing at a constant that did not exist.
#:
#: Six is a containment depth. It is *not* a pedigree depth: a climb counts
#: hops rather than nodes, so a corpus seven generations deep needs seven, and
#: `data/silo/` has spent its whole life reporting generation 6 as unanswerable
#: for that reason. What bounds it at all is a cycle - two places each inside
#: the other - which must terminate whatever the data says.
CLIMB_LIMIT = 6

#: Set on a step's relation byte to walk the reverse table instead: "who was
#: born in Edinburgh" is the born_in row read from the other end.
#:
#: The flag rides in the high bit because there are eleven relations and the
#: byte holds 256, so direction costs nothing - and the reverse table was
#: already being written and read by nothing at all.
INVERSE = 0x80
RELATION = 0x7F


def _u24(value: int) -> bytes:
    return value.to_bytes(3, "little")


def _pack_edge(left: int, relation: int, right: int) -> bytes:
    return _u24(left) + bytes([relation]) + _u24(right)


def _unpack_edge(raw: bytes) -> tuple[int, int, int]:
    return (int.from_bytes(raw[0:3], "little"), raw[3],
            int.from_bytes(raw[4:7], "little"))


def corpus_digest(titles: list[str]) -> int:
    """A 32-bit digest of the corpus this graph's ids refer to.

    Cheap to recompute on either side and specific enough that two different
    `--limit` values cannot collide, which is the one confusion that produces
    confident wrong answers rather than an error.
    """
    h = hashlib.sha256()
    h.update(str(len(titles)).encode())
    for title in titles:
        h.update(title.encode("utf-8", "replace"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:4], "little")


@dataclass
class Graph:
    """What goes on the card, before it is laid out."""

    num_docs: int
    digest: int
    #: (subject_doc, relation_id, object_doc)
    edges: list[tuple[int, int, int]]
    #: relation id -> name, so a card can be read without this module
    relations: list[str]
    #: type id -> (name, sorted doc ids)
    types: list[tuple[str, list[int]]]
    #: phrase index -> the path it means, as (relation_id, kind) steps
    paths: list[list[tuple[int, int]]]


def build(titles: list[str],
          edges: Iterable[tuple[int, int, int]], relations: list[str],
          types: dict[str, list[int]], paths: list[list[tuple[int, int]]]) -> Graph:
    return Graph(
        num_docs=len(titles),
        digest=corpus_digest(titles),
        edges=sorted(edges),
        relations=list(relations),
        types=sorted((name, sorted(docs)) for name, docs in types.items()),
        paths=paths,
    )


def write(graph: Graph, path: Path) -> dict[str, int]:
    """Write the card file: header, relation names, types, paths, then edges.

    The two edge tables go last and in that order because they are the only
    part streamed rather than read whole - everything before them is a few
    kilobytes the program keeps resident.
    """
    names = b"".join(n.encode("ascii") + b"\x00" for n in graph.relations)
    type_names = b"".join(n.encode("ascii") + b"\x00" for n, _ in graph.types)

    type_body = bytearray()
    type_table = bytearray()
    for _name, docs in graph.types:
        type_table += struct.pack("<II", len(type_body), len(docs))
        for doc in docs:
            type_body += _u24(doc)

    path_body = bytearray()
    path_table = bytearray()
    for steps in graph.paths:
        path_table += struct.pack("<I", len(path_body))
        path_body += bytes([len(steps)])
        for relation, kind in steps:
            path_body += STEP.pack(relation, kind)

    forward = b"".join(_pack_edge(s, r, o) for s, r, o in graph.edges)
    reverse = b"".join(_pack_edge(o, r, s)
                       for o, r, s in sorted((o, r, s) for s, r, o in graph.edges))

    cursor = HEADER.size
    sections = []
    for blob in (names, type_names, bytes(type_table), bytes(type_body),
                 struct.pack("<I", len(graph.paths)) + bytes(path_table)
                 + bytes(path_body)):
        sections.append((cursor, blob))
        cursor += len(blob)
    forward_at, reverse_at = cursor, cursor + len(forward)

    with path.open("wb") as fh:
        fh.write(HEADER.pack(
            MAGIC, len(graph.relations), len(graph.types),
            graph.num_docs, graph.digest, len(graph.edges),
            sections[0][0], sections[2][0], sections[4][0],
            forward_at))
        for _at, blob in sections:
            fh.write(blob)
        fh.write(forward)
        fh.write(reverse)

    return {"bytes": path.stat().st_size, "edges": len(graph.edges),
            "forward_at": forward_at, "reverse_at": reverse_at}


class CardGraph:
    """Walk the card exactly as the eZ80 build will.

    Deliberately literal, for the same reason `libsearch.CardSearch` is: if
    this and the generated code disagree, one of them is wrong, and having two
    implementations is the only way to find out which.
    """

    def __init__(self, path: Path) -> None:
        self.fh = path.open("rb")
        raw = self.fh.read(HEADER.size)
        (magic, num_relations, num_types, self.num_docs, self.digest,
         self.num_edges, names_at, types_at, paths_at,
         self.forward_at) = HEADER.unpack(raw)
        if magic != MAGIC:
            raise ValueError(f"{path} is not a {MAGIC.decode()} graph")
        self.reverse_at = self.forward_at + self.num_edges * EDGE_SIZE

        self.fh.seek(names_at)
        self.relations = self._names(num_relations)
        self.type_names = self._names(num_types)

        self.fh.seek(types_at)
        table = [struct.unpack("<II", self.fh.read(8)) for _ in range(num_types)]
        self._types_at = self.fh.tell()
        self._type_span = {self.type_names[i]: table[i] for i in range(num_types)}

        self.fh.seek(paths_at)
        count = struct.unpack("<I", self.fh.read(4))[0]
        offsets = [struct.unpack("<I", self.fh.read(4))[0] for _ in range(count)]
        body_at = self.fh.tell()
        self.paths = []
        for offset in offsets:
            self.fh.seek(body_at + offset)
            steps = self.fh.read(1)[0]
            self.paths.append([STEP.unpack(self.fh.read(2)) for _ in range(steps)])

    def _names(self, count: int) -> list[str]:
        out: list[str] = []
        for _ in range(count):
            chars = bytearray()
            while (ch := self.fh.read(1)) != b"\x00":
                chars += ch
            out.append(chars.decode("ascii"))
        return out

    def close(self) -> None:
        self.fh.close()

    def check(self, num_docs: int, digest: int) -> None:
        """Refuse a graph built against a different corpus than the card.

        The failure this prevents is silent: every id in a mismatched graph is
        still a valid article, so the machine answers fluently and wrongly.
        """
        if (num_docs, digest) != (self.num_docs, self.digest):
            raise ValueError(
                f"graph is for a different corpus: it holds {self.num_docs:,} "
                f"documents (digest {self.digest:08x}), the card holds "
                f"{num_docs:,} (digest {digest:08x})")

    # --- the walk -------------------------------------------------------------

    def _edge(self, base: int, n: int) -> tuple[int, int, int]:
        self.fh.seek(base + n * EDGE_SIZE)
        return _unpack_edge(self.fh.read(EDGE_SIZE))

    def _find(self, base: int, key: int, relation: int) -> int | None:
        """First record matching (key, relation), by binary search.

        Eighteen seeks for 150,335 edges, each one a 24-bit compare - which is
        the entire reason the graph is laid out this way rather than as
        anything cleverer.
        """
        low, high = 0, self.num_edges
        want = (key, relation)
        while low < high:
            mid = (low + high) // 2
            left, rel, _right = self._edge(base, mid)
            if (left, rel) < want:
                low = mid + 1
            else:
                high = mid
        if low >= self.num_edges:
            return None
        left, rel, _right = self._edge(base, low)
        return low if (left, rel) == want else None

    def objects(self, subject: int, relation: int, limit: int = 8) -> list[int]:
        at = self._find(self.forward_at, subject, relation)
        return self._run(self.forward_at, at, subject, relation, limit)

    def subjects(self, obj: int, relation: int, limit: int = 8) -> list[int]:
        at = self._find(self.reverse_at, obj, relation)
        return self._run(self.reverse_at, at, obj, relation, limit)

    def _run(self, base: int, at: int | None, key: int, relation: int,
             limit: int) -> list[int]:
        if at is None:
            return []
        out: list[int] = []
        while at < self.num_edges and len(out) < limit:
            left, rel, right = self._edge(base, at)
            if (left, rel) != (key, relation):
                break
            out.append(right)
            at += 1
        return out

    def count(self, obj: int, relation: int) -> int:
        """How many subjects point at `obj` - free, once the reverse table
        exists, and the capability most likely to be overlooked."""
        return len(self.subjects(obj, relation, limit=1 << 24))

    def is_a(self, doc: int, kind: str) -> bool:
        span = self._type_span.get(kind)
        if span is None:
            return False
        at, count = span
        low, high = 0, count
        while low < high:
            mid = (low + high) // 2
            self.fh.seek(self._types_at + at + mid * 3)
            here = int.from_bytes(self.fh.read(3), "little")
            if here < doc:
                low = mid + 1
            elif here > doc:
                high = mid
            else:
                return True
        return False

    def follow(self, subject: int, steps: list[tuple[int, int]],
               climb_limit: int = CLIMB_LIMIT,
               ) -> tuple[int | None, list[int], int | None]:
        """Walk `steps`, returning (answer, path, the step that had no edge).

        A step whose kind is not PLAIN is a climb: repeat the relation until
        the value is of that type, checking before stepping so a value that is
        already what was asked for is returned rather than stepped past.

        A step whose relation carries INVERSE walks the reverse table: "who was
        born in Edinburgh" is the same row read from the other end. It names
        *one* of them - 536 people were born in London - which is an answer
        rather than a list, and listing them is a separate thing to build.
        """
        here = subject
        walked = [subject]
        for index, (step, kind) in enumerate(steps):
            relation = step & RELATION
            hop = self.subjects if step & INVERSE else self.objects

            if kind == PLAIN:
                found = hop(here, relation, limit=1)
                if not found:
                    return None, walked, index
                here = found[0]
                walked.append(here)
                continue

            name = self.type_names[kind]
            for _ in range(climb_limit):
                if self.is_a(here, name):
                    break
                found = hop(here, relation, limit=1)
                if not found:
                    return None, walked, index
                here = found[0]
                walked.append(here)
            else:
                return None, walked, index
        return here, walked, None
