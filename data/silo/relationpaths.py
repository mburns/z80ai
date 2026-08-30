#!/usr/bin/env python3
"""
Questions about the silo, and the path each one is asking for.

    python data/silo/relationpaths.py > silo-relations.txt
    python data/silo/relationpaths.py --emit held-out --held-out-templates 3 \
           > silo-held-out.txt

The card's oracle is three parts, and this is the middle one. The search index
turns a name into a document; a **phrasebook classifier** turns a question into
a path; the graph walks it. `classify.py` trains the classifier on what this
prints, and `buildwikisearch.py --relations` bakes it into the binary along
with the path table `buildwikigraph.paths_for` derives from these labels.

A label *is* a path: `father_is works_in` is two hops, `child_of_of` is
`child_of` read backwards, and `founding_father` is a climb. Anything this file
emits that the corpus has no edges for becomes an inert row in the card's path
table, so `main()` asserts against the database rather than trusting the list.

## These questions are templated, and that is a problem to be measured

`data/README.md` says it plainly: a `--val-frac` split holds out unique
*queries*, so with templated data a held-out "who is X's father" still has
"who is Y's father" in the training half, and the score measures interpolation
inside a grammar this file wrote. The Wikipedia classifier avoids this for its
one-hop classes by training on SimpleQuestions - real questions, written by
people who had never heard of `libgraph`. There is no crowdsourced question set
about a silo that does not exist.

So the honest number here comes from holding out whole *phrasings*:
`--held-out-templates 3` reserves three wordings per path, and `--emit
held-out` prints only those. A model trained without them and scored on them is
being asked whether it learned the question or the vocabulary. Both numbers are
in `data/silo/README.md`, and the difference between them is the point.

Twelve phrasings per path is few enough that the held-out three are a quarter
of the grammar rather than a rounding error, which is deliberate: the failure
this is trying to catch is a classifier that has memorised "grandfather" and
falls over on "father's father".

## Six paths have twenty-four, and it is not because they were short

Three pairs lost to each other far more than anything else did, always in the
same shape - a two-hop path answered as its own one-hop prefix:

    class_is class_is_of  ->  class_is           70
    mother_is mother_is   ->  mother_is          51
    father_is           ->  father_is father_is  35

The cause is lexical. "grandmother" *contains* "mother" - five of the six
trigrams in the shorter word are inside the longer one - and almost every
wording of the longer path used it. The word that was supposed to distinguish
them was mostly the word they shared.

So those six paths have twelve more wordings apiece, chosen to say the same
thing **without the shared token**: *gran*, *nan*, *granny* for the grandmother
path, *schoolfellow* and *taught with* for the classmate one, and terse forms
like "{s}'s dad" for the one-hop paths that were losing upward. That is a
hypothesis about the encoder rather than a stylistic preference, and
`data/silo/README.md` reports what it was worth: prefix confusions fell from
261 questions to 153, and the worst pair left the table entirely.

`tools/phrasebook_diversity.py` is the check that they broadened rather than
padded - mean self-similarity within those six paths fell, which is what
adding genuinely different wordings looks like and what adding more of the
same would not.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: F401  - registers the silo's climbs into libgraph.CLIMB
from schema import SOURCE

import libgraph
import libgraphcard
import liboracle

DB_PATH = Path(__file__).resolve().parent.parent / "silo.db"

#: path -> the ways somebody might ask for it. `{s}` is the subject, filled
#: from the corpus so the rare words in a question are real even where the
#: frame is not.
#:
#: Ordered longest-path-last, because that is the order they get hard in: one
#: hop is a lookup with a synonym problem, and three hops is a question whose
#: surface form shares almost no words with the path it means.
PATHS: dict[str, tuple[str, ...]] = {
    "father_is": (
        "who is {s}'s father", "who was {s}'s father", "name {s}'s father",
        "who fathered {s}", "who is the father of {s}", "{s}'s dad is who",
        "which man is {s}'s father", "who sired {s}",
        "tell me {s}'s father", "{s} is the child of which man",
        "whose son or daughter is {s} on the father's side",
        "who is the male parent of {s}",
        # Twelve more, and the other half of the same repair: this path loses
        # *into* `father_is father_is` 38 times, so it wants wordings a
        # grandparent question would never use. Short ones especially - a
        # terse query shares little with a long one whatever the words.
        "{s}'s dad", "father of {s}", "{s} dad", "dad of {s}",
        "who is dad to {s}", "name the dad of {s}",
        "who is {s}'s papa", "{s} was raised by which man",
        "the man on {s}'s birth record",
        "look up the father for {s}",
        "which man is entered as {s}'s parent",
        "{s} is son or daughter to whom",
    ),
    "mother_is": (
        "who is {s}'s mother", "who was {s}'s mother", "name {s}'s mother",
        "who is the mother of {s}", "{s}'s mum is who",
        "which woman is {s}'s mother", "who gave birth to {s}",
        "tell me {s}'s mother", "{s} is the child of which woman",
        "who bore {s}", "who is the female parent of {s}",
        "{s} was mothered by whom",
        # Twelve more, the mirror of the `father_is` additions.
        "{s}'s mum", "mother of {s}", "{s} mum", "mum of {s}",
        "who is mum to {s}", "name the mum of {s}",
        "who is {s}'s mama", "{s} was carried by which woman",
        "the woman on {s}'s birth record",
        "look up the mother for {s}",
        "which woman is entered as {s}'s parent",
        "{s} is son or daughter to which woman",
    ),
    "spouse_of": (
        "who did {s} marry", "who is {s} married to", "who is {s}'s wife",
        "who is {s}'s husband", "name {s}'s spouse", "who is the partner of {s}",
        "{s} is married to whom", "who did {s} wed",
        "tell me who {s} married", "whose husband or wife is {s}",
        "who shares a flat and a name with {s}",
        "who is {s} paired with",
    ),
    "works_in": (
        "which department does {s} work in", "where does {s} work",
        "what department is {s} in", "who does {s} work for",
        "which section employs {s}", "{s} works in what department",
        "name the department of {s}", "which office is {s} attached to",
        "tell me where {s} is employed", "{s} is on the books of which department",
        "what outfit does {s} belong to", "which department has {s}",
    ),
    "job_is": (
        "what does {s} do", "what is {s}'s job", "what is {s}'s trade",
        "what does {s} do for a living", "name {s}'s occupation",
        "what work does {s} do", "{s} is employed as what",
        "what is the title of {s}", "tell me {s}'s trade",
        "what does {s} do all day", "what post does {s} hold",
        "{s} works as a what",
    ),
    "shift_is": (
        "which shift does {s} work", "what shift is {s} on",
        "when does {s} work", "name {s}'s shift", "{s} works which shift",
        "what hours does {s} keep", "which rotation is {s} on",
        "tell me {s}'s shift", "is {s} first second or third shift",
        "what watch does {s} stand", "which shift has {s}",
        "{s} is rostered to which shift",
    ),
    "lives_at": (
        "where does {s} live", "what is {s}'s address",
        "which flat is {s} in", "where is {s}'s apartment",
        "name the address of {s}", "{s} lives where",
        "what is the apartment number of {s}", "tell me where {s} sleeps",
        "which dwelling belongs to {s}", "where would i find {s} at night",
        "what door is {s} behind", "{s} is quartered where",
    ),
    "born_on": (
        "which level was {s} born on", "where was {s} born",
        "what floor was {s} born on", "name the level {s} was born on",
        "{s} was born on which level", "on what level did {s} arrive",
        "tell me where {s} was born", "which floor did {s} come from",
        "what level is {s} native to", "{s}'s birth level is what",
        "where did {s} first draw breath", "which deck was {s} born on",
    ),
    # Three classes over what used to be `fact` rows with nothing to point at.
    # `born_in_year` shares its central token with `born_on` above and there is
    # no honest way around that - "born" is the word - so the separation has to
    # come from the rest of the sentence: everything here asks *when* and
    # nothing here says level, floor or deck. That is the same repair the
    # grandmother paths got, made in advance rather than after the confusion
    # matrix showed it, and `tools/phrasebook_diversity.py` is the check.
    "born_in_year": (
        "what year was {s} born", "when was {s} born",
        "which year did {s} arrive", "name the year {s} was born",
        "{s} was born in which year", "what is {s}'s birth year",
        "tell me the year of {s}'s birth", "{s}'s year of birth is what",
        "{s} was born when", "the year of {s}'s birth",
        "what does the record give for {s}'s birth year",
        "which year is entered against {s}'s birth",
    ),
    "died_in_year": (
        "what year did {s} die", "when did {s} die",
        "which year did {s} die in", "name the year {s} died",
        "{s} died in which year", "what is {s}'s year of death",
        "tell me when {s} died", "{s}'s death year is what",
        "when did {s} pass", "the year {s} was lost",
        "what year is entered against {s}'s death",
        "when was {s} taken off the roll",
    ),
    # Two more values the corpus already held with nothing pointing at them:
    # `generation` was a `fact` and a `category` and not an edge, and the year
    # a tenancy started has been in `residence` since the first build. Ten
    # paired seeds put the cost of both at -0.6 +/- 0.6, which is nothing.
    #
    # They score **47.4%** held out against a 52.7% mean, the weakest classes
    # here. `moved_in_year` is a year question next to `born_in_year` and a
    # dwelling question next to `lives_at`, and it shares vocabulary with both;
    # `generation_is` asks something nothing else asks and has no near
    # neighbour to lose to, which makes the pair's average the wrong summary of
    # either. They ship anyway: without a class these questions do not fail,
    # they land on whatever they resemble and get answered fluently.
    "generation_is": (
        "which generation is {s}",
        "what generation does {s} belong to",
        "how far down from the founders is {s}",
        "name {s}'s generation",
        "{s} is of which generation",
        "how many generations after the founders is {s}",
        "what is {s}'s remove from the founding",
        "which cohort of descent is {s}",
        "tell me {s}'s generation",
        "{s} stands how far from the founders",
        "what generation is entered against {s}",
        "how deep in the line is {s}",
    ),
    "moved_in_year": (
        "when did {s} move in",
        "what year did {s} take that flat",
        "how long has {s} lived there",
        "when did {s} take up residence",
        "name the year {s} moved",
        "{s} moved in which year",
        "since when has {s} been at that address",
        "what year does {s}'s tenancy start",
        "when was {s} housed there",
        "tell me when {s} took the flat",
        "what year is entered against {s}'s move",
        "{s} has held that door since when",
    ),
    # Deliberately short of "died" and "death", which belong to the class above
    # and are most of it. What distinguishes these two questions in English is
    # `how` against `when`, and three characters is not much for an encoder
    # that hashes trigrams - so the vocabulary does the work instead.
    "fate_is": (
        "how did {s} die", "what happened to {s}",
        "was {s} sent to clean", "what became of {s}",
        "how did {s} end", "tell me the manner of {s}'s end",
        "did {s} go out to clean", "what does the record give as {s}'s fate",
        "{s}'s fate is what", "was it a cleaning for {s}",
        "what is written against {s}'s fate", "what finished {s}",
    ),
    "class_is": (
        "which class was {s} in", "what class did {s} attend",
        "name {s}'s class", "{s} was in which class",
        "which school class did {s} belong to", "what year group was {s} in",
        "tell me {s}'s class", "which class list has {s} on it",
        "what class did {s} sit in", "{s} was schooled with which class",
        "which nursery class was {s} in", "what cohort did {s} study with",
        # Twelve more that ask for a *name* rather than for people, which is
        # the distinction `class_is class_is_of` kept losing.
        "{s}'s class", "class of {s}", "{s} class",
        "name the year group {s} belonged to",
        "what was {s} taught in", "under which teacher was {s} taught",
        "which room did {s} learn in", "{s}'s year group",
        "look up the class for {s}",
        "what is entered as {s}'s schooling",
        "name the cohort {s} was registered to",
        "{s} was registered to which group",
    ),
    "crew_is": (
        "which crew is {s} on", "what crew does {s} serve on",
        "name {s}'s crew", "{s} is on which crew",
        "which work crew has {s}", "what team does {s} work with",
        "tell me {s}'s crew", "which gang does {s} belong to",
        "what crew is {s} rostered to", "{s} works alongside which crew",
        "which crew list has {s}", "what shift team is {s} part of",
    ),
    "father_is father_is": (
        "who is {s}'s grandfather", "who is {s}'s paternal grandfather",
        "who is the father of {s}'s father", "name {s}'s grandfather",
        "{s}'s father's father is who", "who fathered {s}'s father",
        "tell me {s}'s grandfather on the father's side",
        "which man is two generations above {s} on the father's line",
        "who is grandad to {s}", "{s} is the grandchild of which man",
        "who is the father of the father of {s}",
        "name the man whose son fathered {s}",
        # Twelve more without "father" in them, for the reason set out under
        # `mother_is mother_is`.
        "who is {s}'s grandad", "who is {s}'s grandpa",
        "name {s}'s grandpa", "{s}'s grandad is who",
        "who is grandpa to {s}",
        "which man is {s}'s parent's parent on the male side",
        "name the older man {s} descends from through the men",
        "{s} is grandchild to which man",
        "who is two above {s} on the male line",
        "the man whose son sired {s}",
        "{s} calls which man grandad",
        "name the elder man on {s}'s paternal line",
    ),
    "mother_is mother_is": (
        "who is {s}'s grandmother", "who is {s}'s maternal grandmother",
        "who is the mother of {s}'s mother", "name {s}'s grandmother",
        "{s}'s mother's mother is who", "who bore {s}'s mother",
        "tell me {s}'s grandmother on the mother's side",
        "which woman is two generations above {s} on the mother's line",
        "who is granny to {s}", "{s} is the grandchild of which woman",
        "who is the mother of the mother of {s}",
        "name the woman whose daughter bore {s}",
        # Twelve more that do not contain "mother" at all.
        #
        # This path loses to `mother_is` more often than any other pair loses
        # to anything, and the reason is lexical rather than grammatical:
        # "grandmother" *contains* "mother", so most of the twelve above are
        # the shorter path's word with a prefix stuck on it. Five of the six
        # trigrams in "mother" are inside "grandmother", and 128 buckets could
        # not tell them apart at all. These say the same thing without sharing
        # the token, which is a hypothesis about the encoder and not a stylistic
        # preference - see `data/silo/README.md`.
        "who is {s}'s gran", "who is {s}'s nan", "name {s}'s granny",
        "{s}'s gran is who", "who is nana to {s}",
        "which woman is {s}'s parent's parent on the female side",
        "name the older woman {s} descends from through the women",
        "{s} is grandchild to which woman",
        "who is two above {s} on the female line",
        "the woman whose daughter gave birth to {s}",
        "{s} calls which woman gran",
        "name the elder woman on {s}'s maternal line",
    ),
    "father_is works_in": (
        "which department does {s}'s father work in",
        "where does {s}'s father work", "what department is {s}'s dad in",
        "name the department of {s}'s father",
        "{s}'s father works where", "who employs {s}'s father",
        "tell me where {s}'s father is employed",
        "which office does the father of {s} report to",
        "what outfit does {s}'s father belong to",
        "the man who fathered {s} works in which department",
        "which department has {s}'s father on its books",
        "what does {s}'s father's department call itself",
    ),
    "spouse_of job_is": (
        "what does {s}'s wife do", "what does {s}'s husband do",
        "what is the trade of {s}'s spouse",
        "name the occupation of whoever {s} married",
        "{s}'s partner works as what",
        "what job does {s}'s spouse hold",
        "tell me the trade of {s}'s husband or wife",
        "the person {s} married does what for a living",
        "what is the title of {s}'s spouse",
        "what work does {s}'s partner do",
        "whoever {s} wed is employed as what",
        "what post does the spouse of {s} hold",
    ),
    "lives_at in_section": (
        "which section does {s} live in", "is {s} up top or down deep",
        "what part of the silo does {s} live in",
        "name the section {s} lives in", "{s} lives in which section",
        "which third of the silo is {s} in",
        "tell me the section of {s}'s home",
        "whereabouts in the silo does {s} sleep",
        "which stretch of levels is {s}'s flat on",
        "{s}'s address falls in which section",
        "what section contains the home of {s}",
        "which run of the silo does {s} live along",
    ),
    "works_in located_in": (
        "which level is {s}'s department on",
        "what floor does {s} work on", "where is {s}'s department",
        "name the level {s} works on", "{s} works on which level",
        "which floor houses the department of {s}",
        "tell me the level of {s}'s workplace",
        "what level would i climb to to find {s} working",
        "the department that employs {s} sits on which floor",
        "which deck is {s}'s office on",
        "what floor is {s}'s workplace on",
        "{s}'s department is headquartered where",
    ),
    "founding_father": (
        "which founder is {s} descended from",
        "who is {s}'s founding ancestor",
        "trace {s} back to a founder",
        "name the founder in {s}'s male line",
        "{s} descends from which founder",
        "which original settler fathered {s}'s line",
        "tell me the founder {s} comes from",
        "who started the line that ends at {s}",
        "which of the first generation is {s} descended from",
        "{s}'s family goes back to which founder",
        "name the founding man behind {s}",
        "which founder began {s}'s father's line",
    ),
    "child_of_of": (
        "who are {s}'s children", "name a child of {s}",
        "who did {s} raise", "{s} is the parent of whom",
        "which children does {s} have", "who calls {s} their parent",
        "tell me a child of {s}", "who was born to {s}",
        "name the offspring of {s}", "{s} has which children",
        "who are the sons and daughters of {s}",
        "which people list {s} as a parent",
    ),
    # The two counts. A count step is `child_of` read backwards with a tally
    # instead of a hop, so the subject picker draws parents - the objects of
    # the relation - exactly as it does for `child_of_of` above.
    #
    # These deliberately sit a word away from refusals: "how many children" is
    # answerable and "how many cousins" is not, "born on {s}'s level" is
    # answerable and "live on {s}'s floor" is not. The seam was measured before
    # either was written - see `data/silo/README.md` - and the answer was that
    # the collision is not where it looked like it would be.
    "count_child_of": (
        "how many children does {s} have", "count {s}'s children",
        "how many kids has {s} got", "what is {s}'s child count",
        "{s} has how many children",
        "how many sons and daughters does {s} have",
        "the number of children {s} has is what",
        "give me a number for {s}'s children",
        "how many offspring does {s} have",
        "tell me how many children {s} has",
        "how many did {s} raise",
        "how many births are down to {s}",
    ),
    "class_is class_is_of": (
        "who was in {s}'s class", "name a classmate of {s}",
        "who was schooled with {s}", "who sat in class with {s}",
        "which pupils shared {s}'s class",
        "tell me somebody from {s}'s class",
        "who studied alongside {s}", "who else was in the class of {s}",
        "name another pupil from {s}'s year group",
        "which children learned beside {s}",
        "who shared a class list with {s}",
        "{s} went to school with whom",
        # Twelve more without "class" in them. This pair is the worst in the
        # phrasebook - 70 misses to `class_is` - and the two questions are
        # genuinely near-identical on the surface: one asks *which class*, the
        # other asks *who was in it*, and both said "class" in almost every
        # wording. The distinguishing word was carrying the whole burden.
        "who were {s}'s schoolfellows", "name a schoolmate of {s}",
        "who sat beside {s} at school", "which children were taught with {s}",
        "who shared a teacher with {s}", "name somebody {s} learned beside",
        "who was in the same year as {s}", "which pupils knew {s} at school",
        "name a child schooled at the same time as {s}",
        "who grew up in the nursery with {s}",
        "{s} learned to read beside whom",
        "name one of the children taught alongside {s}",
    ),
    "lives_at next_along lives_at_of": (
        "who lives next door to {s}", "name a neighbour of {s}",
        "who lives beside {s}", "who is {s}'s neighbour",
        "which people live next to {s}",
        "tell me who lives along from {s}",
        "who is the next flat round from {s}",
        "name somebody living next to {s}",
        "who shares a wall with {s}",
        "which neighbour does {s} have",
        "who lives thirty minutes round the ring from {s}",
        "who is {s}'s nearest neighbour along the ring",
    ),
    # `born_on count_born_on` - "how many people were born on X's level" - is a
    # path the card can walk and is deliberately **not** here. It costs the
    # refusal class 11.7 +/- 5.6 points where `count_child_of` costs 6.9 +/-
    # 6.8, which is nothing; `data/silo/README.md` has the three arms.
    #
    # Not because it steals the refusals: it barely gets any. It displaces
    # them. Separating "born on {s}'s level" from `refuse`'s "live on {s}'s
    # floor" pushes the ring-count wordings out of the refusal region
    # altogether and into `lives_at`, which answers them with an address.
    #
    # Not a path, and not a path that this corpus happens to be missing edges
    # for: `data/silo/README.md` sets out why composition stops at aggregation,
    # ranking and set intersection, and these are the four shapes it stops at.
    #
    # Without a class of their own they are not refused, they are *misrouted* -
    # every one of them lands on some path, every path completes on a corpus
    # with no gaps, and the machine answers a different question fluently. That
    # is the failure this class exists to convert into a "no".
    libgraphcard.REFUSE_PATH: (
        # a count of a union: four hops, two inverted, then a tally over a *set*
        "how many cousins does {s} have",
        "count {s}'s cousins",
        "how many first cousins has {s} got",
        "what is the number of {s}'s cousins",
        "how many cousins is {s} down for",
        "tell me the cousin count for {s}",
        "how many children do {s}'s aunts and uncles have",
        "{s} has how many cousins",
        "count everybody who is a cousin of {s}",
        "the size of {s}'s cousinhood is what",
        "how many cousins altogether for {s}",
        "give me a number for {s}'s cousins",
        # a maximum over a set the walk can enumerate but not rank
        "who is the oldest person on {s}'s crew",
        "which of {s}'s crew is eldest",
        "name the youngest on {s}'s shift",
        "who is the senior member of {s}'s crew",
        "which of {s}'s classmates was born first",
        "who on {s}'s crew has been there longest",
        "name the eldest of {s}'s siblings",
        "which of {s}'s children is oldest",
        "who is the youngest in {s}'s class",
        "rank {s}'s crew by age",
        "who came first among {s}'s crew",
        "the oldest of {s}'s neighbours is who",
        # an intersection of two recursive ancestor sets
        "is {s} related to me",
        "are {s} and i related",
        "is {s} any relation to the sheriff",
        "does {s} share an ancestor with me",
        "am i kin to {s}",
        "is there any blood between {s} and me",
        "are {s} and the sheriff cousins",
        "do {s} and i have family in common",
        "is {s} a relative of mine",
        "tell me whether {s} and i are related",
        "what relation is {s} to me",
        "how is {s} related to the sheriff",
        # a count around the ring: a program rather than a query
        "how many people live on {s}'s floor",
        "count the residents of {s}'s level",
        "how many live on the same level as {s}",
        "what is the population of {s}'s level",
        "how many flats are occupied on {s}'s floor",
        "how many neighbours does {s} have altogether",
        "count everybody housed on {s}'s deck",
        "how many souls on {s}'s level",
        "the level {s} lives on holds how many",
        "tell me how many share {s}'s floor",
        "how big is the population where {s} lives",
        "number of residents on {s}'s level",
    ),
}

#: Labels written down but **not shipped**, kept so the measurement that
#: rejected them can be run again. `tools/class_cost.py --added` reads this as
#: well as `PATHS`, and puts anything it finds here in the *with* arm only.
#:
#: This exists because [#89](../../pull/89) deleted `born_on count_born_on`'s
#: wordings along with the class, which left `data/silo/README.md` asserting a
#: cost - 11.7 +/- 5.6 points off the refusal class - that nothing could
#: reproduce. A negative result whose inputs are gone is an anecdote.
#:
#: The wordings below are **not** the ones that measurement used, because those
#: are not in the history either. They are written to the same brief: re-use
#: `born` and `level`, which are exactly the tokens `refuse`'s ring count uses,
#: since the collision under test is between "how many were *born* on X's
#: level" and "how many *live* on X's floor".
#: Two of these name **two** people, and `{s}` is filled with one name. They
#: are here rather than in `PATHS` for a reason the card enforces: it resolves
#: one document per question, so a `shared_` step has nowhere to put the second
#: subject and `paths_for` would write an inert row. `liboracle.subjects` does
#: the two searches on the Python side and finds both names 99.5% of the time;
#: what is missing is the eZ80's half.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "shared_crew_is": (
        "are {s} and the sheriff on the same crew",
        "do {s} and i serve together",
        "is {s} on my crew",
        "does {s} share a crew with the sheriff",
        "are {s} and i crewmates",
        "is {s} rostered with me",
        "do {s} and the sheriff work the same gang",
        "is {s} on the same work team as me",
        "are {s} and i on one crew",
        "does {s} crew with the sheriff",
        "is {s} a crewmate of mine",
        "tell me if {s} and i share a crew",
    ),
    "shared_founding_father": (
        "is {s} related to me",
        "are {s} and i related",
        "is {s} any relation to the sheriff",
        "do {s} and i share a founder",
        "am i kin to {s}",
        "is there any blood between {s} and me",
        "are {s} and the sheriff of one line",
        "do {s} and i come from the same founder",
        "is {s} a relative of mine",
        "tell me whether {s} and i are related",
        "what relation is {s} to me",
        "how is {s} related to the sheriff",
    ),
    "born_on count_born_on": (
        "how many people were born on {s}'s level",
        "how many were born on the same level as {s}",
        "count the births on {s}'s level",
        "how many share {s}'s birth level",
        "what is the birth count for {s}'s level",
        "how many people were born where {s} was born",
        "number of births on {s}'s deck",
        "how many arrived on {s}'s level",
        "tell me how many births {s}'s level has seen",
        "count everybody born on {s}'s floor",
        "the level {s} was born on saw how many births",
        "how many births are recorded for {s}'s level",
    ),
}

#: How many questions to write per phrasing. Twelve phrasings times this is the
#: class size, and `classify.py --balance` levels them anyway.
PER_TEMPLATE = 40

#: Twelve more wordings for five paths, used for **training only**.
#:
#: `data/silo/README.md` ends the phrasing-curve section saying "extending
#: every path to twenty-four and holding out six would settle it", and the
#: reason nobody has is that the comparison is easy to get wrong. A path given
#: twelve more wordings while still holding out three has a held-out set with
#: more neighbours to learn from, so its score rises for a reason that is not
#: grammar - `tools/phrasebook_diversity.py` measured novelty falling from
#: 0.188 to 0.100 on `mother_is` when the six paths were extended.
#:
#: Keeping these out of `PATHS` is what removes that confound. `build(extra=
#: True)` appends them to the *training* half and never reserves them, so both
#: arms are scored on an identical held-out set and the only thing moving is
#: how much English the model was shown. That is the same design the learning
#: curve already used, carried past the nine wordings it stopped at.
#:
#: The five are chosen to span the range rather than to flatter it: at the
#: last measurement `child_of_of` scored 3.3% held out, `works_in` 24.2%,
#: `shift_is` 29.2%, `job_is` 40.8% and `crew_is` 56.7%. If more grammar is
#: worth something it should be worth most at the bottom.
#:
#: They are frames rather than synonyms, which is the distinction the six-path
#: repair turned on. `child_of_of` is the clearest case in the corpus: its
#: three held-out wordings were all "name/tell me a child of X", a frame none
#: of the nine it trained on used, and they scattered over seven classes with
#: no winner. That is not a collision with a neighbour, it is having no region
#: at all for a sentence shape the model never saw.
EXTRA: dict[str, tuple[str, ...]] = {
    "child_of_of": (
        "{s}'s children",
        "list the children of {s}",
        "who did {s} bring up",
        "name everybody born to {s}",
        "whose parent is {s}",
        "{s} raised whom",
        "who does the record give as {s}'s issue",
        "which young ones are {s}'s",
        "the children of {s} are who",
        "say who {s} fathered or mothered",
        "give me the kids of {s}",
        "who looks up {s} as a parent",
    ),
    "works_in": (
        "{s}'s department",
        "under which department is {s} filed",
        "{s} answers to which department",
        "what does {s}'s badge say",
        "which arm of the silo has {s}",
        "{s} draws pay from where",
        "where is {s} posted",
        "the department of {s} is what",
        "which staff list carries {s}",
        "{s} reports to which department",
        "what is {s}'s posting",
        "which department claims {s}",
    ),
    "shift_is": (
        "{s}'s shift",
        "{s} stands which watch",
        "{s} clocks on when",
        "which of the three does {s} work",
        "{s} is down for which watch",
        "what are {s}'s hours",
        "the shift of {s} is what",
        "when is {s} on duty",
        "which turn does {s} take",
        "{s} keeps which hours",
        "what time does {s} start",
        "which roster has {s}",
    ),
    "job_is": (
        "{s}'s trade",
        "{s} is what by trade",
        "what does the ledger call {s}",
        "which trade is {s} trained to",
        "{s} is down as what",
        "under what title does {s} serve",
        "the occupation of {s} is what",
        "what skill does {s} have",
        "{s} was apprenticed to what",
        "what is {s} by profession",
        "give me {s}'s post",
        "what trade did {s} learn",
    ),
    "crew_is": (
        "{s}'s crew",
        "{s} musters with whom",
        "which detail is {s} assigned to",
        "{s} turns out with which lot",
        "who does {s} labour beside",
        "which working party has {s}",
        "the crew of {s} is what",
        "{s} is posted to which gang",
        "which squad does {s} run with",
        "name the party {s} works with",
        "{s} is one of which crew",
        "which band of workers has {s}",
    ),
    # --- the second five ------------------------------------------------------
    #
    # Disjoint from the first, and picked to match its difficulty rather than
    # its subject matter: the first five averaged 39.2% held out and these five
    # average 38.2%. `tools/grammar_pilot.py --group second` runs them, and
    # `--group both` is the measurement that matters - whether the first five
    # keep their seventeen points once somebody else is growing too.
    "works_in located_in": (
        "{s}'s department is on which level",
        "how far up is {s}'s work",
        "which landing do i want for {s}'s department",
        "the floor {s} works on is what",
        "{s} goes to which level for work",
        "at what depth does {s} work",
        "which level holds the place {s} works",
        "give me the floor for {s}'s department",
        "how deep is {s}'s workplace",
        "{s} climbs to which level",
        "what level is entered for {s}'s department",
        "where in the stair is {s}'s work",
    ),
    "count_child_of": (
        "{s}'s child count",
        "how large is {s}'s family",
        "how many young did {s} have",
        "put a number on {s}'s children",
        "{s} is down for how many",
        "how many names list {s} as a parent",
        "the size of {s}'s brood is what",
        "how many entries has {s} fathered or mothered",
        "reckon up the children of {s}",
        "how many does {s}'s line come to",
        "what number of children is entered for {s}",
        "how many did {s} put on the register",
    ),
    "founding_father": (
        "{s}'s founder",
        "which of the first is {s} out of",
        "run {s}'s male line to its head",
        "who stands at the top of {s}'s line",
        "the founder behind {s} is who",
        "which first-generation man leads to {s}",
        "give me the head of {s}'s line",
        "{s} comes down from whom",
        "who is at the root for {s}",
        "what founding name is {s} under",
        "trace the fathers above {s}",
        "which founder owns {s}'s line",
    ),
    "lives_at": (
        "{s}'s address",
        "which door is {s}'s",
        "where is {s} housed",
        "give me {s}'s flat",
        "{s} is at which address",
        "look up where {s} is quartered",
        "the dwelling of {s} is what",
        "which apartment is entered for {s}",
        "where does the register put {s}",
        "{s} keeps which rooms",
        "what address is down for {s}",
        "which flat has {s} in it",
    ),
    "spouse_of": (
        "{s}'s wife or husband",
        "who stood up with {s}",
        "give me the name {s} married",
        "which match did {s} make",
        "who is joined to {s}",
        "the marriage of {s} was to whom",
        "who took {s} to wife or husband",
        "whom is {s} bound to",
        "what name is entered beside {s} as married",
        "who did {s} take",
        "{s} is wed to whom",
        "look up {s}'s match",
    ),
    # --- and the rest of them -------------------------------------------------
    #
    # The ten paths still on twelve wordings, which is what "finishing" means:
    # every class except `refuse` then trains on twenty-one. The six that were
    # given a second dozen by the prefix repair already do, because their extra
    # twelve went into `PATHS` rather than here - a path with 24 wordings holding
    # out 3 trains on 21, and so does a path with 12 holding out 3 plus 12 here.
    "born_on": (
        "{s}'s birth level",
        "on which floor did {s} start",
        "which level does the register give for {s}'s birth",
        "give me the level {s} came from",
        "{s} is a child of which floor",
        "what level bore {s}",
        "which landing was {s} born on",
        "the level of {s}'s birth is what",
        "how deep was {s} born",
        "which floor does {s} count as home",
        "what level is entered for {s}'s birth",
        "{s} began life on which level",
    ),
    "born_in_year": (
        "{s}'s birth year",
        "in what year did {s} first appear",
        "give me the year {s} came into the silo",
        "which year does the register give for {s}",
        "{s} dates from when",
        "what year stands beside {s}'s name",
        "how long ago was {s} born",
        "the year {s} arrived is what",
        "{s} was entered in which year",
        "look up the year of {s}'s birth",
        "what year does {s} count from",
        "{s}'s first year is which",
    ),
    "died_in_year": (
        "{s}'s death year",
        "in what year was {s} lost",
        "give me the year {s} ended",
        "which year does the register close {s}",
        "{s} was struck off in which year",
        "the year of {s}'s death is what",
        "look up when {s} was buried",
        "what year did {s} stop",
        "{s} lasted until when",
        "which year ends {s}'s record",
        "when was {s}'s name closed",
        "what year is down for {s}'s death",
    ),
    # Numbers and depth rather than founders and lines, because
    # `founding_father` has just been given twelve wordings of its own and half
    # of them are about running a line to its head.
    "generation_is": (
        "{s}'s generation",
        "how many removes is {s}",
        "which cohort does {s} count in",
        "give me the generation number for {s}",
        "{s} sits at which depth of descent",
        "what number generation is {s}",
        "which of the seven is {s}",
        "how many steps down is {s}",
        "the generation of {s} is what",
        "what generation number is entered for {s}",
        "{s} belongs to which numbered generation",
        "look up {s}'s generation",
    ),
    "moved_in_year": (
        "{s}'s move-in year",
        "what year was {s} given that flat",
        "how long has that door been {s}'s",
        "which year does {s}'s tenancy begin",
        "{s} took those rooms when",
        "give me the year {s} was housed",
        "the year {s} moved is what",
        "since which year has {s} lived there",
        "when was that address entered for {s}",
        "look up when {s} was quartered there",
        "what year did {s} settle",
        "{s} came to that flat in which year",
    ),
    "fate_is": (
        "{s}'s fate",
        "what does the roll say became of {s}",
        "was {s} put out",
        "give me the manner of {s}'s ending",
        "what is {s} recorded as",
        "was {s} one of the cleanings",
        "the fate of {s} is what",
        "in what way did {s} go",
        "what ending is entered for {s}",
        "did they send {s} out",
        "look up how {s} finished",
        "what is set down for {s}'s ending",
    ),
    # Six of these twelve lead with the father, which is the repair the
    # grandparent paths needed: the pair differs by two words and the shorter
    # one has just grown, so the longer one has to say *whose* department it
    # means before it says department.
    "father_is works_in": (
        "{s}'s father's department",
        "the man who fathered {s} works where",
        "{s}'s dad is posted where",
        "give me the department of {s}'s father",
        "which department employs the father of {s}",
        "{s}'s father draws pay from where",
        "what is {s}'s father's posting",
        "where does the father of {s} spend his shift",
        "which staff list carries {s}'s father",
        "{s}'s father answers to which department",
        "look up the department for {s}'s father",
        "the department of {s}'s father is what",
    ),
    "spouse_of job_is": (
        "{s}'s spouse's trade",
        "what is the one {s} married by trade",
        "give me the post of {s}'s wife or husband",
        "which trade does {s}'s match hold",
        "the occupation of {s}'s spouse is what",
        "what does the ledger call {s}'s spouse",
        "what skill has the person {s} wed",
        "under what title does {s}'s spouse serve",
        "{s}'s husband or wife is what by trade",
        "look up the trade of {s}'s spouse",
        "what is {s}'s partner by profession",
        "what trade did {s}'s spouse learn",
    ),
    "lives_at in_section": (
        "{s}'s section",
        "which end of the silo is {s} at",
        "give me the section of {s}'s address",
        "is {s} housed high or low",
        "the section {s} lives in is what",
        "which band of levels holds {s}'s home",
        "what quarter of the silo does {s} live in",
        "which section is entered for {s}'s flat",
        "{s}'s home falls in which part",
        "look up the section for {s}",
        "in which reach of the silo does {s} live",
        "{s} is quartered in which section",
    ),
    "lives_at next_along lives_at_of": (
        "{s}'s neighbours",
        "who is round the ring from {s}",
        "give me the name next door to {s}",
        "which flat adjoins {s} and who is in it",
        "who is {s}'s closest door",
        "name whoever lives alongside {s}",
        "the neighbour of {s} is who",
        "who would {s} hear through the wall",
        "which household sits beside {s}",
        "look up who is next to {s}",
        "who occupies the flat by {s}",
        "who is {s}'s door neighbour",
    ),
}

#: The three groups `EXTRA` was written in, so the pilot can grow them one at a
#: time and watch what happens to everybody else. The point of the second five
#: was that the first five's gain came 80-88% out of their neighbours, and two
#: arms cannot tell a redistribution that would vanish if everybody grew from
#: one that would not. It did vanish, mostly: at ten grown the share taken from
#: the rest fell to 55% and the corpus gained 2.3 points.
#:
#: `REMAINING_TEN` is the rest of them. Every path except `refuse` trains on
#: twenty-one wordings with all three grown - the six the prefix repair already
#: took to twenty-four are there by a different route and arrive at the same
#: number.
FIRST_FIVE = frozenset({
    "child_of_of", "works_in", "shift_is", "job_is", "crew_is"})
SECOND_FIVE = frozenset({
    "works_in located_in", "count_child_of", "founding_father", "lives_at",
    "spouse_of"})
REMAINING_TEN = frozenset({
    "born_on", "born_in_year", "died_in_year", "generation_is",
    "moved_in_year", "fate_is", "father_is works_in", "spouse_of job_is",
    "lives_at in_section", "lives_at next_along lives_at_of"})


def resolve(word: str, have: set[str]) -> tuple[str, bool]:
    """(relation, is it read backwards) - the same reading `paths_for` makes.

    Written once and used twice, by the subject picker and by the assertion in
    `main`, because two readings of a path vocabulary that drift apart produce
    a card whose path table is silently inert.
    """
    if word.startswith(libgraph.COUNT):
        # "How many children does X have" counts `child_of` records pointing at
        # X, so the people worth asking it about are the *objects* of that
        # relation - the parents - which is the same direction an inverse hop
        # reads.
        return resolve(word[len(libgraph.COUNT):], have)[0], True
    if word in libgraph.CLIMB:
        return libgraph.CLIMB[word][0], False
    if word in have:
        return word, False
    if word.endswith("_of") and word[:-3] in have:
        return word[:-3], True
    raise SystemExit(f"no edges for {word!r}; the card would ignore that path")


def subjects(db: sqlite3.Connection, path: str, have: set[str], wanted: int,
             rng: Random) -> list[str]:
    """Names the question can sensibly be asked about, drawn from the corpus.

    Asking who is on somebody's crew when they died two centuries ago and have
    no crew teaches the classifier a phrasing and teaches the graph nothing, so
    a subject is drawn from the entities that actually carry the path's first
    relation.
    """
    if path == libgraphcard.REFUSE_PATH:
        # No first relation to draw from, and none wanted: a question the
        # machine must refuse is one it must refuse about anybody, so the
        # subjects come from the whole corpus rather than from the entities
        # that carry some relation.
        rows = [r for (r,) in db.execute(
            "SELECT name FROM person WHERE source = ?", (SOURCE,))]
        rng.shuffle(rows)
        return [rows[i % len(rows)] for i in range(wanted)]

    relation, inverse = resolve(path.split()[0], have)
    column = "object" if inverse else "subject"
    rows = [r for (r,) in db.execute(
        f"SELECT DISTINCT {column} FROM edge "
        "WHERE source = ? AND relation = ?", (SOURCE, relation))]
    rng.shuffle(rows)
    return [rows[i % len(rows)] for i in range(wanted)]


def _ask(template: str, name: str, masked: bool) -> str:
    """One question, with the subject either in it or taken back out.

    Masked questions are what the classifier would see if the pipeline removed
    the entity before encoding - the oracle resolves the subject first, so it
    could. Generated with the *true* subject, so this is the ideal case: what
    masking is worth when the search was right, which it is 88.6% of the time.
    """
    question = template.format(s=name.lower())
    return liboracle.mask(question, name) if masked else question


def build(db: sqlite3.Connection, have: set[str], per_template: int,
          hold_out: int, seed: int,
          masked: bool = False, phrasings: int | None = None,
          extra: bool | Collection[str] = False,
          ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(training pairs, held-out pairs), split by phrasing rather than by row.

    ``phrasings`` keeps only that many of the wordings left after the held-out
    ones are reserved, which is how the learning curve in
    `data/silo/README.md` is drawn: the evaluation set is identical at every
    point on it, so the only thing moving is how much of the grammar the model
    was shown.

    ``extra`` adds `EXTRA`'s wordings to the training half - `True` for every
    path that has them, or a collection of path names for some of them. They
    are never reserved and never counted against ``phrasings``, so the held-out
    set is **identical** with and without, which is the whole reason they live
    outside `PATHS`. Shuffling twenty-four wordings reserves a different three
    than shuffling twelve, and comparing two arms scored on different sentences
    would measure the sentences.
    """
    grow = set(EXTRA) if extra is True else set(extra or ())
    rng = Random(seed)
    # A stream per extended path, and neither of those two words is spare.
    #
    # *A stream of its own*, because drawing the extra wordings' subjects from
    # `rng` would advance it, so every path after the first extended one would
    # shuffle differently and reserve different wordings - precisely the
    # confound `extra` exists to avoid, and it does not announce itself: both
    # arms still look like they hold out three apiece.
    #
    # *Per path*, because one shared spare stream is consumed in `PATHS` order,
    # so growing `spouse_of` as well would change which names `works_in`'s
    # extra rows are asked about. Harmless in expectation and still a variable
    # nobody needs: keyed this way, a path's extra rows depend on the seed and
    # on nothing else that happens in the same build.
    extra_order = sorted(EXTRA)
    train: list[tuple[str, str]] = []
    unseen: list[tuple[str, str]] = []
    for path, templates in PATHS.items():
        order = list(templates)
        rng.shuffle(order)
        reserved = set(order[:hold_out]) if hold_out else set()
        rest = order[hold_out:]
        kept = set(rest if phrasings is None else rest[:phrasings])
        names = subjects(db, path, have, per_template * len(templates), rng)
        for i, template in enumerate(templates):
            if template not in reserved and template not in kept:
                continue
            block = names[i * per_template:(i + 1) * per_template]
            rows = [(_ask(template, name, masked), path) for name in block]
            (unseen if template in reserved else train).extend(rows)
        if path in grow:
            # Drawn separately so the subjects are the corpus's own for this
            # path, the same way the block above draws them - a wording trained
            # against names that cannot carry the relation teaches the phrasing
            # against an answer of nothing.
            more = EXTRA[path]
            spare = Random((seed + 1) * 1_000_003 + extra_order.index(path))
            drawn = subjects(db, path, have, per_template * len(more), spare)
            for i, template in enumerate(more):
                block = drawn[i * per_template:(i + 1) * per_template]
                train.extend((_ask(template, name, masked), path)
                             for name in block)
    rng.shuffle(train)
    rng.shuffle(unseen)
    return train, unseen


@dataclass
class Audit:
    """What a trained classifier does to a phrasing it *was* trained on."""

    #: questions asked, and how many routed to the right path
    asked: int = 0
    right: int = 0
    #: phrasings where every subject got the same answer, and how many there are
    steady: int = 0
    phrasings: int = 0
    #: (path, phrasing, share right, the class it mostly gave instead)
    worst: list[tuple[str, str, float, str]] = field(default_factory=list)


def audit(model_path: str, db: sqlite3.Connection, have: set[str],
          per_template: int, seed: int) -> Audit:
    """Ask each trained phrasing about many different people, and compare.

    This exists because of one observation that a per-question accuracy hides:
    `who is alexander e wong's father` and `who is corey w wong's father` do
    not classify the same way. The encoder hashes the *whole* question into 128
    trigram buckets, and a name is most of a short question - so the subject is
    not something the model ignores on its way to the verb, it is the bulk of
    the signal.

    "Steady" counts phrasings where changing only the name never changes the
    answer. It is the number that says whether a card can be relied on for a
    question it has already been taught.
    """
    import libinfer

    model = libinfer.Model.load(model_path)
    rng = Random(seed)
    out = Audit()
    for path, templates in PATHS.items():
        names = subjects(db, path, have, per_template * len(templates), rng)
        for i, template in enumerate(templates):
            block = names[i * per_template:(i + 1) * per_template]
            got = [libinfer.classify(model, template.format(s=n.lower()), 24)
                   for n in block]
            hits = sum(1 for g in got if g.lower() == path)
            out.asked += len(got)
            out.right += hits
            out.phrasings += 1
            out.steady += len(set(got)) == 1
            if hits < len(got):
                instead = Counter(g.lower() for g in got if g.lower() != path)
                out.worst.append((path, template, hits / len(got),
                                  instead.most_common(1)[0][0]))
    out.worst.sort(key=lambda row: row[2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--per-template", type=int, default=PER_TEMPLATE)
    ap.add_argument("--held-out-templates", type=int, default=0, metavar="N",
                    help="Reserve N phrasings per path, unseen in training")
    ap.add_argument("--emit", choices=("train", "held-out"), default="train",
                    help="'held-out' prints only the reserved phrasings, to "
                         "score generalisation rather than recall")
    ap.add_argument("--phrasings", type=int, metavar="K",
                    help="Train on only K of the wordings left after the "
                         "held-out ones, for the learning curve")
    ap.add_argument("--mask", action="store_true",
                    help="Take the subject back out of each question before "
                         "printing it - see liboracle.mask")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-extra", action="store_true",
                    help="Leave out EXTRA's twelve-per-path, which is what "
                         "every measurement before it was made against")
    args = ap.parse_args()

    if args.emit == "held-out" and not args.held_out_templates:
        ap.error("--emit held-out needs --held-out-templates N")
    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}\n"
                         f"  python data/silo/generate.py")

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (SOURCE,))}
    # Every word of every path, checked against the corpus before anything is
    # printed. A label the card cannot walk is not an error at build time - it
    # becomes an empty row in the path table and a question the machine
    # classifies correctly and then answers with silence.
    for path in PATHS:
        if path == libgraphcard.REFUSE_PATH:
            continue                 # a refusal is not a path to check
        for word in path.split():
            resolve(word, have)

    # `extra` on by default, because it is what the card should carry: 55.4% to
    # 65.3% held out, and every path on twenty-one wordings rather than six of
    # them on twenty-one and twenty on nine. `--no-extra` reproduces every
    # number in `data/silo/README.md` that predates it.
    train, unseen = build(db, have, args.per_template, args.held_out_templates,
                          args.seed, masked=args.mask,
                          phrasings=args.phrasings, extra=not args.no_extra)
    pairs = unseen if args.emit == "held-out" else train
    counts = Counter(path for _, path in pairs)

    print(f"# Templated questions about the silo, over {len(counts)} paths.")
    print(f"# {len(pairs):,} questions, {args.per_template} per phrasing, "
          f"subjects drawn from {args.db.name}.")
    if args.emit == "held-out":
        print(f"# Held-out phrasings only: {args.held_out_templates} of "
              f"{len(next(iter(PATHS.values())))} wordings per path, which the "
              f"training half never saw.")
    else:
        print("# Templated - a --val-frac split flatters this. See the module "
              "docstring, and score on --emit held-out.")
    for question, path in pairs:
        print(f"{question.replace('|', ' ').strip()}|{path}")

    for path, n in counts.most_common():
        print(f"  {path:<34} {n:>6,}", file=sys.stderr)


if __name__ == "__main__":
    main()
