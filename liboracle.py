"""
An oracle that is wrong about half the time, and says so usefully.

Three parts, each measured separately and each doing only what it wins at:

    which entity        the BM25 search index          works on names
    which relation      the phrasebook classifier      93.8% macro, 9 relations
    which path          the same classifier            ~50% on unseen phrasings
    the answer          a walk over libgraph edges     a lookup

The two classifier numbers come from the same model and are not comparable.
The first is over crowdsourced questions; the second is over multi-hop
phrasings this repo wrote, scored only on phrasings withheld from training.
`data/questions/relations.py` explains why the second is the honest one and why
it is so much lower.

The second carries no decimal place on purpose. Over five seeds it runs 43.1%
to 56.2% on its 320 held-out questions - the figure this file used to give,
51.6%, is one draw from that and reproduces exactly at seed 0. So is the 43.8%
that another module used to give. They were never in disagreement.

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
from typing import Protocol

import libgraph
import libinfer


class Search(Protocol):
    """What the oracle needs from a search backend, and nothing more.

    Two things satisfy this: :class:`libsearch.CardSearch`, over the index an
    Agon would actually read off its card, and oracle.py's database equivalent
    for when no card has been built. Naming the interface is what lets the
    second exist - liboracle cannot import the script that defines it.
    """

    def search(self, query: str, top: int = 3) -> list[tuple[int, int]]: ...

    def article(self, doc: int) -> tuple[str, str]: ...


#: How the answer was reached, worst to best.
UNKNOWN, SEARCH, PARTIAL, FACT = "unknown", "search", "partial", "fact"


def mask(question: str, subject: str) -> str:
    """The question with the entity's words taken out of it.

    **Measured, and not used.** It is here because the negative result is worth
    keeping reproducible, not because anything calls it in `ask`.

    The idea was sound and the mechanism is real: the encoder hashes character
    trigrams over the whole query, so a name is most of a short question and
    the classifier reads the subject as though it were part of the verb.
    `tools/name_sensitivity.py` measures what that costs - hold a phrasing
    fixed, vary only the name, and twelve of Wikipedia's thirty-two chain
    phrasings change their answer, against 117 of the silo's 240. The entity is
    known before the relation is, on the device as well as here, so the words
    could be removed rather than reasoned about.

    What it buys, measured on the silo where the subject of every question is
    recorded:

        steady phrasings      123/240  ->  239/240
        trained phrasings       96.1%  ->    95.0%
        unseen phrasings        44.5%  ->    53.3%  at one seed, +1.1% at three

    So it does exactly what it was aimed at and nothing else. Consistency was
    not what limited accuracy: a model fails an unseen wording because it never
    saw the wording, and taking the name out does not teach it that "grandad"
    means two hops. The third row is the one that decided it - +8.8% at seed 0
    became -5.4% at seed 1, which is the same spread `data/questions/relations.py`
    already documents for this measurement.

    It cannot be tested on Wikipedia's one-hop classes at all. SimpleQuestions
    records a subject as a Wikidata Q-id rather than as the words that appear
    in the question, so there is nothing to remove.

    Word-level rather than substring, because the question is whatever somebody
    typed and the title is canonical: "wong's" and "Wong" are the same word and
    not the same string. Removing a possessive is the one rule with a judgement
    in it, and without it "alexander e wong's father" keeps `wong's`, which is
    most of the name and all of the problem.
    """
    drop = {_word(w) for w in subject.split()} - {""}
    kept = [w for w in question.split() if _word(w) not in drop]
    return " ".join(kept)


def _word(token: str) -> str:
    """A token reduced to what two spellings of one name have in common."""
    token = token.lower().removesuffix("'s").removesuffix("s'")
    return "".join(c for c in token if c.isalnum())


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
                 relations: libinfer.Model | None = None,
                 search: Search | None = None) -> None:
        self.db = db
        self.source = source
        #: A libinfer.Model whose phrases are relation names, or None to skip
        #: straight to search. Trained by classify.py on data/questions.
        self.relations = relations
        #: Anything satisfying Search, or None. Used to find the entity a
        #: question is about, and as the fallback when no fact answers it.
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

        A phrase may name several - "born_in in_country" is what "what country
        was X born in" asks for - so the model's vocabulary is paths rather
        than single relations, and a one-hop path is just the short case.

        `in_country` is a climb rather than a step: it repeats `located_in`
        until the value is a country. The question asks for a type, and how
        many hops that takes is the graph's business, not the model's.

        The whole question goes to the encoder, name and all. Removing the
        subject first is an obvious-looking improvement that was tried and
        measured - see `mask` for what it was worth, which was nothing.
        """
        if self.relations is None:
            return None

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
    "in_country": "what country that is",
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
