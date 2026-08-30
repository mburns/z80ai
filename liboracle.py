"""
An oracle that is wrong about half the time, and says so usefully.

Three parts, each measured separately and each doing only what it wins at:

    which entity        the BM25 search index          works on names
    which relation      the phrasebook classifier      93.8% macro, 9 relations
    which path          the same classifier            84.0% on unseen phrasings
    the answer          a walk over libgraph edges     a lookup

The two classifier numbers come from the same model and are not comparable.
The first is over crowdsourced questions; the second is over multi-hop
phrasings this repo wrote, scored only on phrasings withheld from training.

That second number was ~50% for a long time, and the fix was not a better
model. It was writing more ways to ask the question: eight more phrasings per
path took it from 59.2% to 84.0%, three seeds, against an unchanged held-out
set. `data/questions/relations.py` has the curve and the one-hop macro it cost.

Do not compare 84.0% with the 43.1%-to-56.2% spread that used to be quoted
here: that measurement held out two phrasings of eight and this holds out three
of sixteen, so they are answers to different questions.

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
#:
#: `record` sits above `search` and below `partial` on purpose. All three are
#: failures to answer the question; what separates them is how much of it the
#: machine got right before giving up. A search result is prose about whatever
#: the index liked. A record is everything the graph holds about the person the
#: question named - so the subject was understood and only the relation was
#: not, which is a strictly better failure and reads as one.
UNKNOWN, SEARCH, RECORD, PARTIAL, FACT = (
    "unknown", "search", "record", "partial", "fact")


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


def residual(question: str, subject: str) -> str:
    """The question with *one copy* of each of the subject's words removed.

    `mask` removes every copy, which is right for what it was written to do -
    take the subject out before the encoder sees it - and wrong for finding a
    second subject. "Is Alexander E. Wong related to Corey W. Wong" masked
    against the first name loses **both** surnames, and the second search then
    goes looking for a man called `corey w`, who is somebody else entirely.

    Removing one copy of each word leaves the second Wong his surname. That is
    the whole difference, and it only shows up on a corpus where 2,264 people
    share a first and last name with somebody - which is to say, on a corpus
    that took the trouble to be realistic about families.
    """
    drop: dict[str, int] = {}
    for word in subject.split():
        key = _word(word)
        drop[key] = drop.get(key, 0) + 1
    kept: list[str] = []
    for word in question.split():
        key = _word(word)
        if drop.get(key):
            drop[key] -= 1
            continue
        kept.append(word)
    return " ".join(kept)


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
    #: How far the classifier's first choice was ahead of its second. Small
    #: means the machine nearly asked a different question, which is worth
    #: saying out loud rather than hiding behind a full stop - see `speak`.
    margin: int | None = None
    #: True when the answer came from the runner-up because the first choice
    #: had no edge to walk.
    second_choice: bool = False
    #: (relation, object) for everything the graph holds about the subject,
    #: when nothing could be walked but the subject was found.
    held: list[tuple[str, str]] = field(default_factory=list)
    #: True for a question about two people, where `value` is where they meet
    #: and None means they do not. A pair question has a yes and a no and both
    #: are answers, which is why it cannot be told by `value` alone.
    pair: bool = False
    #: The subject's `entity_type`, when it is one the voice can use: `man` or
    #: `woman` become a pronoun. A **type** and not the `sex` fact, because
    #: types are on the card and facts are not - a pronoun the eZ80 cannot
    #: reach would be a register the device could never speak in.
    kind_of: str | None = None

    @property
    def answered(self) -> bool:
        return self.kind == FACT


class Oracle:
    """Question in, fact out - or an account of why not."""

    def __init__(self, db: sqlite3.Connection, source: str = "simplewiki",
                 relations: libinfer.Model | None = None,
                 search: Search | None = None, backoff: int = 0,
                 records: bool = True) -> None:
        self.db = db
        self.source = source
        #: A libinfer.Model whose phrases are relation names, or None to skip
        #: straight to search. Trained by classify.py on data/questions.
        self.relations = relations
        #: Anything satisfying Search, or None. Used to find the entity a
        #: question is about, and as the fallback when no fact answers it.
        self.search = search
        #: Try the classifier's runner-up when the first path finds no edge and
        #: the top two were closer together than this. **Zero, which is off, is
        #: the default and the shipped behaviour.**
        #:
        #: It is a dial rather than a flag because what it trades is measured
        #: and the right setting depends on what the machine is for. Over 600
        #: held-out silo questions, 168 had a first path with no edge:
        #:
        #:     gate            backed off   answered what was asked
        #:     never (0)                0                         -
        #:     margin < 25             22                     27.3%
        #:     margin < 75             54                     27.8%
        #:     always                 111                     21.6%
        #:
        #: Ungated it turns 17.8 points of dead end into facts and **four out of
        #: five of those facts answer a different question fluently**, which is
        #: the failure this repository has argued against from the start.
        #:
        #: The reasoning for the gate is that a confident first choice which
        #: finds nothing is more likely a real gap - asking when somebody still
        #: alive died - than a misroute. The measurement is weaker than the
        #: reasoning: on the previous card the gate doubled the hit rate, on
        #: this one it adds about six points, and each is one held-out split.
        #: What survives both is the direction and the shape of the trade -
        #: gating answers fewer questions and is right about more of them - not
        #: any particular size.
        #:
        #: So a demo that wants an answer to everything sets it high, and a
        #: machine that would rather say what it does not know leaves it at 0.
        self.backoff = backoff
        #: Report what the graph holds about the subject when no path walks,
        #: rather than going straight to the search index. Unlike `backoff`
        #: this invents nothing - every word of a record is an edge - so it is
        #: on by default. It only ever fires where the alternative was prose.
        self.records = records

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

    def subjects(self, question: str) -> list[str]:
        """The one or two articles a question is about.

        "Is X related to Y" names two, and `libgraph.common` answers questions
        of that shape by walking the same path from both ends - a capability
        `data/silo/README.md` measures at 400/400 on crews and classes and
        cannot use, because the pipeline had nowhere to put a second name.

        This is that place, and it costs one idea: **search, take the words of
        what was found out of the question, and search again.** `residual` does
        the taking-out, removing one copy of each word rather than all of them
        - see its docstring for the two Wongs that decided it.

        The second search has to be allowed to fail, and BM25 will not fail on
        its own: given `where was born` it happily returns whoever those words
        touch. So a second subject is kept only if its own name is still in
        what is left of the question, which is a check on whether a second
        person was *named* rather than on how strongly one scored - and it
        needs nothing from the backend, whose second tuple element means
        different things in the two implementations of `Search`.

        Two searches and no new machinery. The card cannot do this yet: it
        resolves one document per question, and that is a change to the eZ80
        program rather than to the graph.
        """
        if self.search is None:
            return []
        first = self.entity(question)
        if first is None:
            return []
        rest = residual(question, first)
        if not rest.strip():
            return [first]
        second = self.entity(rest)
        if second is None or second == first:
            return [first]
        left = {_word(w) for w in rest.split()} - {""}
        if not {_word(w) for w in second.split()} & left:
            return [first]
        return [first, second]

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
        ranked = self.ranked(question, top=1)
        return ranked[0] if ranked else None

    def ranked(self, question: str, top: int = 2) -> list[list[str]]:
        """The paths the classifier considered, best first.

        `libinfer.rank` measures why this is worth having: over the silo's
        held-out phrasings the top choice is right 55.6% of the time and the
        top *two* contain the answer 69.4% of the time, so a first path with no
        edge to walk has somewhere better to look than the search index.
        """
        if self.relations is None:
            return []
        # The phrasebook is uppercase because the Z80 targets have an uppercase
        # charset; the graph is lowercase. Bridging here rather than at either
        # end keeps both conventions intact - and a mismatch is silent, since
        # every lookup simply finds no edge and the oracle falls back to search
        # looking for all the world like a corpus gap.
        return [phrase.lower().split() for phrase, _ in libinfer.rank(
            self.relations, question, accum_bits=self.relations.accum_bits,
            top=top)]

    #: Entity types the voice can turn into a pronoun. A corpus that does not
    #: carry them loses nothing but the pronoun: `speak` falls back to *their*,
    #: which is also what it uses for anybody the corpus does not type.
    GENDERED = ("man", "woman")

    def pronoun(self, subject: str) -> str | None:
        """The subject's type, when it is one the voice knows how to use."""
        for kind in self.GENDERED:
            if libgraph.is_a(self.db, self.source, subject, kind):
                return kind
        return None

    def shared(self, question: str, path: str) -> Response | None:
        """Whether the two people a question names meet on one relation.

        `libgraph.common` walks the same path from both ends and compares, and
        is measured at 400/400 on crews and classes with no false positive over
        1,500 unconnected pairs. What it never had was a second name;
        `subjects` supplies one.

        None when only one name resolved, which sends the question back to the
        ordinary single-subject route - a pair question with one person in it
        is a misroute, and the machine should fail it the usual way rather
        than inventing a second person to be wrong about.
        """
        both = self.subjects(question)
        if len(both) < 2:
            return None
        relation = path[len(libgraph.SHARED):]
        meeting = libgraph.common(self.db, self.source, both[0], both[1],
                                  [relation])
        return Response(FACT, value=meeting, subject=both[0], pair=True,
                        relations=[path], path=both)

    def confidence(self, question: str) -> int | None:
        """How far the winning path was ahead of the runner-up."""
        if self.relations is None:
            return None
        return libinfer.margin(libinfer.rank(
            self.relations, question, accum_bits=self.relations.accum_bits))

    def ask(self, question: str) -> Response:
        relations = self.relation(question)

        # A path asked about two people needs both of them, and finding the
        # second costs a search - so it is only paid for when the classifier
        # has already said the question is of that shape.
        if relations and len(relations) == 1 \
                and relations[0].startswith(libgraph.SHARED):
            pair = self.shared(question, relations[0])
            if pair is not None:
                return pair

        subject = self.entity(question)
        candidates = [relations] if relations else []
        gap = None

        # Nothing above this line changes when `backoff` is 0, which is the
        # default: `relation` stays the one seam a caller overrides, and the
        # runner-up is not even asked for.
        #
        # The second choice is tried only when the first found no edge at all.
        # It is not a better answer than a fact, and preferring it when the
        # first path completed would be trading a right answer for a confident
        # one.
        if self.backoff:
            gap = self.confidence(question)
            if gap is not None and gap < self.backoff:
                candidates = self.ranked(question, top=2)

        for rank, relations in enumerate(candidates):
            if not subject:
                break
            walk = self._walk(subject, relations)
            if walk is not None:
                walk.margin = gap
                walk.second_choice = rank > 0
                walk.kind_of = self.pronoun(subject)
                return walk

        # Before handing over prose: the subject resolved, so say what is held
        # about it. This answers a question that was not asked either, but it
        # is about the right person and every word of it is an edge.
        #
        # `candidates` has to be non-empty. With no relation model nothing was
        # ever asked of the graph, so there is nothing to fall back *from* -
        # that machine is a search engine and should answer like one.
        if subject and candidates and self.records:
            held = libgraph.record(self.db, self.source, subject)
            if held:
                return Response(RECORD, subject=subject, margin=gap,
                                held=held,
                                relations=candidates[0] if candidates else [])

        if self.search is not None:
            best = self.search.search(question, top=1)
            if best:
                title, lead = self.search.article(best[0][0])
                return Response(SEARCH, value=lead, subject=title, margin=gap)
        return Response(UNKNOWN, subject=subject, margin=gap,
                        relations=candidates[0] if candidates else [])

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
    RECORD: "Not that I hold. On {subject} the archive has: {held}.",
    SEARCH: "I have no record of that. The archive offers: {value}",
    UNKNOWN: "The archive holds nothing on that subject.",
}

#: How a relation reads inside a record listing. Anything absent here is
#: printed as its own name with the underscores taken out, which is ugly and
#: legible - the failure mode of a missing entry should be a plain word, not a
#: silent omission that makes the archive look emptier than it is.
HELD = {
    "father_is": "father", "mother_is": "mother", "spouse_of": "married",
    "born_in_year": "born", "died_in_year": "died", "fate_is": "fate",
    "generation_is": "of", "moved_in_year": "there since",
    "born_on": "born on", "lives_at": "lives at", "works_in": "works in",
    "job_is": "trade", "shift_is": "shift", "class_is": "schooled with",
    "crew_is": "crew", "sits_on": "sits on", "child_of": "child of",
    "located_in": "within", "part_of": "part of",
}


#: Relations left out of a record because another relation already says it.
#: `child_of` duplicates `father_is` and `mother_is` in the graph on purpose -
#: it is the edge a question about "a parent" needs - but a listing that names
#: both says everyone's parents twice.
REDUNDANT = frozenset({"child_of"})

#: The order a person reads a record in, which is not the order the edges come
#: back in. Anything unlisted follows, alphabetically, so a relation added to
#: the corpus appears at the end rather than vanishing.
HELD_ORDER = (
    "born_in_year", "born_on", "generation_is", "father_is", "mother_is",
    "spouse_of", "job_is", "works_in", "shift_is", "crew_is", "class_is",
    "sits_on", "lives_at", "moved_in_year", "died_in_year", "fate_is",
)


def _listing(held: list[tuple[str, str]]) -> str:
    """A record as one line: `born Year 135, father Larry O. Wilson, …`.

    Repeated relations keep their order rather than being merged - two
    `spouse_of` edges are two marriages, and collapsing them would lose one.
    """
    def where(row: tuple[str, str]) -> tuple[int, str]:
        relation = row[0]
        rank = (HELD_ORDER.index(relation) if relation in HELD_ORDER
                else len(HELD_ORDER))
        return rank, relation

    rows = sorted((r for r in held if r[0] not in REDUNDANT), key=where)
    return ", ".join(f"{HELD.get(r, r.replace('_', ' '))} {o}" for r, o in rows)

#: What a fact reached through the classifier's *second* choice sounds like.
#: Four out of five of those answer a question other than the one asked, so
#: saying them in the same register as a fact is the one thing this file has
#: argued against throughout - a confident wrong answer with nothing on the
#: screen to say so. The hedge is the something.
SECOND = "{value}, if I have your meaning."

#: A pair question has two answers and both are answers. The negative names
#: what was checked, because "no" on its own hides how narrow the check was -
#: a shared paternal line is not the same as unrelated.
PAIR_YES = "Yes. {value}."
PAIR_NO = "Not by anything the archive holds."

#: What a fact sounds like, by the path that found it. `FACT` prints
#: `{value}.` for anything not listed - a bare name and a full stop, which was
#: the whole register until this table existed.
#:
#: Keyed by the joined path, so a two-hop answer can say what it walked:
#: `father_is works_in` is *"His father works in Mechanical"* and not
#: *"Mechanical."*, which is the difference between an answer and a word. The
#: card would carry this as one string per class beside the phrase table; the
#: classifier already emits the index it would be looked up by.
#:
#: `{they}` and `{their}` are filled from the subject's entity type, or from
#: *they* and *their* when the corpus does not say.
SAYS = {
    "father_is": "{their} father is {value}.",
    "mother_is": "{their} mother is {value}.",
    "spouse_of": "{they} married {value}.",
    "born_in_year": "{they} was born in {value}.",
    "died_in_year": "{they} died in {value}.",
    "born_on": "{they} was born on {value}.",
    "generation_is": "{they} is of {value}.",
    "lives_at": "{they} lives at {value}.",
    "moved_in_year": "{they} has held that door since {value}.",
    "job_is": "{they} is a {value}.",
    "works_in": "{they} works in {value}.",
    "shift_is": "{they} works {value}.",
    "crew_is": "{they} serves on {value}.",
    "class_is": "{they} was schooled with the {value}.",
    "father_is father_is": "{their} grandfather is {value}.",
    "mother_is mother_is": "{their} grandmother is {value}.",
    "father_is works_in": "{their} father works in {value}.",
    "spouse_of job_is": "{their} spouse is a {value}.",
    "lives_at in_section": "{they} lives in {value}.",
    "works_in located_in": "{their} department is on {value}.",
    "founding_father": "{they} descends from {value}.",
    "child_of_of": "{their} children: {value}.",
    "count_child_of": "{they} has {value} children.",
    "class_is class_is_of": "{they} was schooled with {value}.",
    "lives_at next_along lives_at_of": "{value} lives next door.",
    # Wikipedia's classes. The ones whose subject is a place or a work name it
    # rather than taking a pronoun: `simplewiki` types nobody, so every subject
    # there falls back to its own name and "Their capital is Paris" would be
    # the result of pretending otherwise.
    "born_in": "{they} was born in {value}.",
    "died_in": "{they} died in {value}.",
    "member_of": "{they} belongs to {value}.",
    "born_in in_country": "{they} was born in {value}.",
    "born_in located_in": "{they} was born in {value}.",
    "died_in in_country": "{they} died in {value}.",
    "capital_is": "The capital of {subject} is {value}.",
    "created_by": "{subject} was made by {value}.",
    "located_in": "{subject} is in {value}.",
    "in_country": "{subject} is in {value}.",
    "language_is": "{subject} is in {value}.",
    "genre_is": "{subject} is {value}.",
    "preceded_by": "{subject} came after {value}.",
    "followed_by": "{subject} was followed by {value}.",
}

#: The two words `SAYS` needs, for a subject the corpus has typed.
#:
#: When it has not, the fallback is **the subject's own name** rather than
#: *they*: `simplewiki` types nobody, and "They was born in Steventon" is the
#: kind of sentence that makes a machine sound broken rather than terse. A name
#: agrees with every verb these templates use and reads better than a pronoun
#: in a one-line answer anyway.
PRONOUNS = {
    "man": ("He", "His"),
    "woman": ("She", "Her"),
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
    """Render a response as the machine would say it.

    The order of these branches is the register. A hedge outranks a per-path
    sentence, because a fluent sentence is exactly what a second-choice answer
    should *not* get - saying "His father is Larry O. Wilson" about a question
    that asked something else is the confident wrong answer this file keeps
    warning about, dressed better than before.
    """
    if response.pair:
        template = PAIR_YES if response.value else PAIR_NO
    elif response.second_choice and response.answered:
        template = SECOND
    elif response.answered:
        template = SAYS.get(" ".join(response.relations), VOICE[FACT])
    else:
        template = VOICE[response.kind]
    name = response.subject or "They"
    they, their = PRONOUNS.get(response.kind_of or "", (name, f"{name}'s"))
    return template.format(
        value=response.value,
        said=response.said,
        subject=response.subject,
        they=they,
        their=their,
        held=_listing(response.held),
        missing_phrase=READABLE.get(response.missing or "", "any more than that"),
    )
