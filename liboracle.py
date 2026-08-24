"""
An oracle that is wrong about half the time, and says so usefully.

Three parts, each measured separately and each doing only what it wins at:

    which entity        the BM25 search index          works on names
    which relation      the phrasebook classifier      92.5% macro, 9 relations
    which path          the same classifier            43.8% on unseen phrasings
    the answer          a walk over libgraph edges     a lookup

The two classifier numbers come from the same model and are not comparable.
The first is over crowdsourced questions; the second is over multi-hop
phrasings this repo wrote, scored only on phrasings withheld from training.
`data/questions/relations.py` explains why the second is the honest one and why
it is so much lower.

Nothing here reads prose. Extracting an answer from a paragraph is
comprehension and out of reach on this hardware; a fact table is a lookup, and
that is the whole reason facts were pulled out of the infoboxes.

## Why it fails, and why that is the interesting part

Two-hop chains complete about 45% of the time over Simple English Wikipedia.
Not because the traversal is unreliable - it is three index lookups - but
because 54% of articles have no infobox, so a hop lands somewhere with no edges
to continue from.

An oracle that is right half the time is either a broken product or a
character, depending entirely on whether it can say *which* half. So a failed
walk reports where it stopped:

    "He was born in Steventon. The archive does not record what country that is."

rather than "I don't know" - the difference between a machine with gaps and a
machine that is simply unreliable. `Response.said` carries what was learned
before the walk stopped, which is what a narrator would use.

## The fallback

When there is no fact - no relation recognised, or no edge to walk - the
question goes to the search index instead and comes back with an article. That
is a strictly worse answer and it is marked as such: `kind` says whether you
are being told a fact or handed something related.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import libgraph

#: How the answer was reached, worst to best.
UNKNOWN, SEARCH, PARTIAL, FACT = "unknown", "search", "partial", "fact"


@dataclass
class Response:
    """An answer, and an honest account of how much of one it is."""

    kind: str
    #: The answer, when there is one.
    value: str | None = None
    #: The article the question was taken to be about.
    subject: str | None = None
    #: The relations walked, and how far the walk got.
    relations: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    #: The relation that had no edge, when the walk stopped short.
    missing: str | None = None
    #: What was learned before it stopped - a partial answer is worth saying.
    said: str | None = None

    @property
    def answered(self) -> bool:
        return self.kind == FACT


class Oracle:
    """Question in, fact out - or an account of why not."""

    def __init__(self, db: sqlite3.Connection, source: str = "simplewiki",
                 relations=None, search=None) -> None:
        self.db = db
        self.source = source
        #: A libinfer.Model whose phrases are relation names, or None to skip
        #: straight to search. Trained by classify.py on data/questions.
        self.relations = relations
        #: A libsearch.CardSearch, or None. Used to find the entity a question
        #: is about, and as the fallback when no fact answers it.
        self.search = search

    # --- the three steps ------------------------------------------------------

    def entity(self, question: str) -> str | None:
        """The article a question is about.

        BM25 over titles does this without help: the entity mention carries the
        rare words, and the frame around it - "where was", "who wrote" - is
        common enough that idf discounts it to nothing.
        """
        if self.search is None:
            return None
        best = self.search.search(question, top=1)
        return self.search.article(best[0][0])[0] if best else None

    def relation(self, question: str) -> list[str] | None:
        """The relations a question is asking to walk, in order.

        A phrase may name several - "born_in located_in" is the two-hop chain
        for "what country was X born in" - so the model's vocabulary is paths
        rather than single relations, and a one-hop path is just the short case.
        """
        if self.relations is None:
            return None
        import libinfer

        phrase = libinfer.classify(self.relations, question,
                                   accum_bits=self.relations.accum_bits)
        # The phrasebook is uppercase because the Z80 targets have an uppercase
        # charset; the graph is lowercase. Bridging here rather than at either
        # end keeps both conventions intact - and a mismatch is silent, since
        # every lookup simply finds no edge and the oracle falls back to search
        # looking for all the world like a corpus gap.
        return phrase.lower().split()

    def ask(self, question: str) -> Response:
        subject = self.entity(question)
        relations = self.relation(question)

        if subject and relations:
            walk = self._walk(subject, relations)
            if walk is not None:
                return walk

        if self.search is not None:
            best = self.search.search(question, top=1)
            if best:
                title, lead = self.search.article(best[0][0])
                return Response(SEARCH, value=lead, subject=title)
        return Response(UNKNOWN, subject=subject, relations=relations or [])

    def _walk(self, subject: str, relations: list[str]) -> Response | None:
        """Follow the relations, forwards or backwards as each one asks."""
        # An inverse relation is asked from the other end: "who was born in
        # Berlin" walks the object index rather than the subject one.
        if len(relations) == 1 and relations[0].endswith("_of"):
            forward = relations[0][:-3]
            found = libgraph.inverse(self.db, self.source, subject, forward,
                                     limit=3)
            if found:
                return Response(FACT, value=", ".join(found), subject=subject,
                                relations=relations, path=[subject])
            return None

        walk = libgraph.follow(self.db, self.source, subject,
                               [r for r in relations if not r.endswith("_of")])
        if walk.complete:
            return Response(FACT, value=walk.value, subject=subject,
                            relations=relations, path=walk.path)
        if len(walk.path) > 1:
            # Got somewhere before stopping. Saying so is the difference
            # between a machine with gaps and one that is merely unreliable.
            return Response(PARTIAL, subject=subject, relations=relations,
                            path=walk.path, missing=walk.missing,
                            said=walk.path[-1])
        return None


# --- the voice ----------------------------------------------------------------
#
# Kept apart from the machinery so the fiction can be replaced without touching
# what it reports. These phrasings are one possible register - an archive that
# is certain about what it holds and blank about what it does not.

VOICE = {
    FACT: "{value}.",
    PARTIAL: "{said}. The archive does not record {missing_phrase}.",
    SEARCH: "I have no record of that. The archive offers: {value}",
    UNKNOWN: "The archive holds nothing on that subject.",
}

READABLE = {
    "born_in": "where that is", "died_in": "where that is",
    "located_in": "what contains it", "capital_is": "its capital",
    "created_by": "who made it", "spouse_of": "who they married",
    "member_of": "what they belong to", "language_is": "its language",
    "genre_is": "its kind", "preceded_by": "what came before",
    "followed_by": "what came after",
}


def speak(response: Response) -> str:
    """Render a response as the machine would say it."""
    template = VOICE[response.kind]
    return template.format(
        value=response.value,
        said=response.said,
        missing_phrase=READABLE.get(response.missing or "", "any more than that"),
    )
