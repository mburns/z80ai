# A silo, synthesized

Ten thousand people under one lid, seven generations deep, with enough facts
about each of them that most of what you would want to ask is not written down
anywhere — it has to be worked out.

```bash
pip install -r data/silo/requirements.txt
python data/silo/generate.py           # data/silo.db, ~3 seconds
python data/silo/generate.py --stats
python data/silo/questions.py          # what can be worked out, and by what
python data/silo/buildcard.py          # the Agon card, classifier and all
python data/silo/benchcard.py          # what it costs, in the emulator
```

The corpus tables are the ones `data/wikipedia/ingest.py` writes, and the
`source` column says `silo`, so everything downstream reads it unchanged:
`libgraph` walks it, `oracle.py` answers from it, `buildwikisearch.py` turns it
into a card. `schema.py` adds six tables and eleven views on top of that and
explains why it is a separate database file.

If you only read one section, read [where a question's time actually
goes](#where-a-questions-time-actually-goes): on the real card the graph walk
is 1.0% of a query and the classifier is 56% — and that is *after* halving the
classifier, which halved the query with it.

## Why bother inventing a corpus

The Wikipedia oracle is limited by **coverage**, not by the walk. 46% of
articles carry an infobox, so a chain that hops onto one of the other 54%
stops, and `data/wikipedia/coverage.py` spends its time measuring where the
road ends rather than how far a machine could travel on it.

That makes it the wrong instrument for the question underneath: *given facts
that are all there, how much can a machine that does nothing but compare
24-bit numbers actually work out?*

This corpus has no accidental gaps and the answers are known for all ten
thousand people. What it costs is the thing worth being loud about — **a
synthetic dataset is the easiest place in the world to publish a flattering
number**, so `questions.py` prints a trivial baseline beside every result and
two of them are deliberately strong.

## What is in it

| | |
|---|---:|
| people | 10,000 |
| articles | 13,072 |
| facts | 134,048 |
| edges | 105,404 |
| dwellings | 2,088 |
| classes, crews and committees | 757 |
| memberships | 13,698 |
| the database | 39 MB |

Everyone has two parents, a birth year, a death year or none, a dwelling, a
job, a department, a shift and a school class. The living also have a work
crew, an address in the graph, and neighbours. A tenth of the corpus has sat on
one of the silo's twenty committees, with a term that started and mostly ended.

Names come from [Faker](https://github.com/joke2k/faker), seeded and pinned, so
the corpus is believable and reproducible at the same time. Surnames descend
from the father, which is a decision with consequences — see the baselines.

### Addresses

`FLOOR TIME RING`, after Burning Man's clock: **`42 600 A`** is the
forty-second floor, six o'clock, innermost ring. A silo floor is a circle, so a
bearing is a position on a twelve-hour face — 24 of them at half-hour spacing,
three rings deep, 72 dwellings to a floor.

It is stored as `(floor, bearing, ring)` with the printable form as a
**generated column**, so nothing can spell an address two ways. Adjacency is
stored as `next_along` and `next_out` **edges** rather than computed from the
bearings, because the machine that has to walk it has no modulo: the
arithmetic happens once, here.

## Stored, derivable, walkable

Three different things, and the corpus is built to keep them apart.

**Stored** — one row each, no conclusions: `father_is`, `mother_is`,
`child_of`, `spouse_of`, `lives_at`, `born_on`, `works_in`, `job_is`,
`shift_is`, `crew_is`, `class_is`, `sits_on`, and the geography.

**Derivable** — a view in `schema.py`, never a table. 105,404 stored edges
imply **506,543 relationships** nobody wrote down:

| view | rows | view | rows |
|---|---:|---|---:|
| `classmate` | 227,280 | `sibling` | 16,078 |
| `neighbour` | 91,496 | `committee_mate` | 14,600 |
| `cousin` | 55,752 | `coworker` | 24,792 |
| `grandparent` | 30,000 | `housemate` | 19,426 |
| `aunt_or_uncle` | 27,119 | `ancestor` | recursive |

```sql
SELECT other, relation FROM relative WHERE person = 'Alexander E. Wong';
-- 69 rows in 7ms: 2 siblings, 4 grandparents, 4 aunts and uncles, 9 cousins,
--                 4 housemates, 11 neighbours, 10 coworkers, 25 classmates
```

None of those 69 rows is stored. What is stored about Alexander E. Wong is
twelve edges — his parents, his wife, his flat, his job, his shift, his
department, his class, his crew.

**Walkable** is a third and smaller set, and finding its edge is what
`questions.py` is for.

## Can you reason over it?

`python data/silo/questions.py` asks twenty-two questions three ways — a
`libgraph` graph walk, the SQL views, and a guess that ignores the question —
over a sample of 2,000 people. Ground truth is computed in Python from `fact`,
`residence` and `membership`; the walk reads `edge`.

### A value at the end of a forward path

This is what an eZ80 with a card can do, and it does it exactly:

| | hops | walk | best guess |
|---|---:|---:|---:|
| who is X's father | 1 | 100% | 48.9% — nearest man of the same surname |
| which department does X work in | 1 | 100% | 8.2% — always the commonest |
| who is X's paternal grandfather | 2 | 100% | 45.1% — same surname, 56 years older |
| which department does X's **father** work in | 2 | 100% | 28.8% — *whatever X does* |
| which section does X live in | 2 | 100% | 60.6% — always the Mids |
| which shift does X's paternal grandfather work | 3 | 100% | 35.7% — always First |
| which section does X's father's department sit in | 4 | 100% | — |
| what is X's spouse's trade | 2 | 98.0% | — |

The guess column is the point of the table. **Two of those baselines are not
straw men**: children take their father's surname, so "who is X's father"
is guessable half the time from the name alone, and 45% of people follow a
parent into their department, so "what does X's father do" is guessable from
what X does. The corpus was built that way on purpose. A walk that could not
beat them would not be worth the card it walks on.

The 2% that "what is X's spouse's trade" misses is the remarried:
`spouse_of` has two edges and `follow` takes the first.

### The climb, and what a hop limit costs

`libgraph.CLIMB` repeats a relation until the value has a given type — what
*"what country was X born in"* really asks, since the number of hops is a
property of the graph and not of the question. This corpus adds three, and one
of them runs into the limit:

```
generation      asked  hops needed   reached
1                 234            1    100.0%
2                 290            2    100.0%
3                 299            3    100.0%
4                 299            4    100.0%
5                 324            5    100.0%
6                 307            6      0.0%
```

`CLIMB_LIMIT` is 6 and counts hops taken rather than nodes checked, so
generation 5 reaches its founder on the last hop it is allowed and generation 6
falls exactly one short. Nothing is wrong; that is the price of a walk that
must not loop forever, and it is only visible on a corpus where the true answer
is known.

### A set, reached through an inverse hop

`follow` returns a value. Half the questions a person would ask have an answer
*set*, and a path reaches those only backwards — which the card supports, since
it stores the reverse table anyway:

| | any | recall | precision |
|---|---:|---:|---:|
| who are X's children | 100% | 100% | 100% |
| who was in X's class | 100% | 100% | 100% |
| who lives next door to X | 100% | 100% | 100% |
| who is X's sibling | 98.9% | **89.7%** | 100% |

The gap between `any` and `recall` on siblings is the honest reading, and it
has a specific cause rather than a shrug: `follow` takes `LIMIT 1` on
`child_of`, so the walk goes up through one parent and every half-sibling is
on the other one. Measured, not reasoned — **0 of 1,914 full siblings missed,
254 of 540 half-siblings missed.**

Neighbours reaching 100% is the ring-adjacency decision paying off. Four walks
(`next_along` and `next_out`, each read forwards and backwards), then read off
who lives at each door. No arithmetic anywhere.

### Not a path at any length

Reported, not scored — a 0% here would suggest the walk got them wrong, and
the truth is that it cannot be asked:

- **how many cousins does X have** — four hops, two of them inverses, then a
  count. A count is an aggregate; a path ends in a value.
- **who is the oldest person on X's crew** — a maximum over a set the walk can
  enumerate but not rank.
- **is X related to Y** — an intersection of two ancestor sets. `ancestor` is a
  recursive CTE and there is no such thing as a recursive path.
- **how many people live on X's floor** — the walk can circle the ring in 24
  hops counting as it goes, which is a program rather than a query.

That list is the actual finding. The reasoning an eZ80 can do is *composition*
— follow this, then that, then that, in either direction — and it stops at
aggregation, ranking, and set difference.

## Coverage is a decision here, not an accident

```
child_of    8,800 subjects   88.0%  everyone with a parent
works_in   10,000 subjects  100.0%  everyone
class_is    9,983 subjects   99.8%  everyone who reached six
lives_at    2,655 subjects   26.6%  the living only
crew_is     2,561 subjects   25.6%  the living of working age
sits_on     1,078 subjects   10.8%  committee members, past and present
```

`lives_at` covers a quarter of the corpus because **the graph carries the
present and the `residence` table carries all 220 years**. A graph has no
notion of time, and "who lives next door" answered across two centuries of
occupants is wrong rather than merely broad — 2,088 flats hold 10,000 people
over the corpus's history, so a flat has several successive households. The
`housemate` and `neighbour` views test that the tenancies overlapped; the graph
sidesteps the question by only knowing about now.

## On the machine

```bash
python data/silo/buildcard.py        # dist/SILO.{bin,IDX,DAT,GRF}, ~2 min
python data/silo/benchcard.py        # runs it in the emulator, ~2 min
```

| | |
|---|---:|
| `SILO.bin` | 38.9 KB — program, path table and classifier |
| `SILO.IDX` | 5.0 MB — 6,019 terms, 333,278 postings |
| `SILO.DAT` | 1.2 MB — titles and leads, byte-pair packed |
| `SILO.GRF` | 1.6 MB — 105,404 edges over 16 relations |
| accumulator | 13 KB resident, one byte per article |

All 20 phrases the classifier knows are paths the card can walk.

### Where a question's time actually goes

`benchcard.py` asks the same question of people at different pedigree depths, so
the only thing varying is the number of hops, and compares against a card built
without `--relations` — one that searches and neither classifies nor walks:

| | share of a query |
|---|---:|
| the classifier — one forward pass, 30,592 two-bit weights | **55.8%** |
| the search — BM25 over 13,072 articles | 44.2% |
| the graph walk — four hops | 1.0% |

**The graph is the cheap part, by a factor of fifty.** What the card pays for
is deciding which question it was asked, not answering it. That is the opposite
of where the effort has gone in this repository, and it is the useful thing to
know before optimising anything.

Those shares are after [shrinking the
classifier](#the-classifier-was-two-and-a-half-times-larger-than-it-needed-to-be).
Before that the classifier alone was 78% of a query.

A hop moves about **178 bytes** off the card: a binary search over 105,404
fixed-width records is 17 probes of 7 bytes, which is what the measurement
recovers. In instructions a hop is around 3,600 — a slope over five hop counts,
and smaller than the spread of any single query, so it is quoted as *under 2% of
a question* rather than as a constant. Quoting such a number to four digits is
the mistake this repository already made once.

### And the search half is already doing the right thing

With the classifier halved, search became the co-equal cost, so the obvious next
move was to stop feeding it the question. An oracle query is mostly frame —
"who is … father" — which the entity lookup does not need. Measured on the
search-only card:

| query | instructions | finds |
|---|---:|---|
| `who is alexander e wong's father` | 175,838 | Alexander E. Wong |
| `alexander e wong` | 173,966 | Alexander E. Wong |
| `father` | 3,826 | *nothing* |

**The frame costs 1.1% and the idea is dead.** `libsearch.STOPWORDS` already
drops `who`, `is`, `what` and thirty more, and the words left over — `father`,
`descended` — are not in this corpus's dictionary at all, so they are refused at
the index for 1,975 bytes exactly the way `the` is on the Wikipedia card.

What the measurement did show is that the tiering works as designed. Cost tracks
how many of the 52 pages a term touches, not how many terms there are:

| | documents | instructions |
|---|---:|---:|
| `davies` | 3 | 33,103 |
| `wong` | ~80 | 164,761 |
| `zzqqxx` | 0 | 3,826 |
| `alexander wong` | two terms | 173,706 |

A second term adds 10k to a 164k query. **What sets the price is how widely one
term's postings scatter** — the same conclusion `data/wikipedia/README.md`
reaches, reproduced on a corpus a twentieth of the size.

(Those two tables were measured before initials were joined, which changed the
absolute numbers and none of the ratios.)

Which leaves one lever, and it is a corpus decision rather than a card one:
every lead here names the subject's parents and spouse, so a surname held by 21
people appears in about 80 documents and flags nearly every page. Shortening the
leads would make lookups both cheaper and more accurate, and would cost the
property that makes this corpus useful for the comprehension argument — that
every fact in the graph is also there in the prose, for a reader. Not taken.

### The same corpus at four sizes, and what actually gets dearer

Everything above was measured on one corpus, where "cost tracks how widely a
term scatters" and "cost tracks how big the corpus is" cannot be told apart —
every term that is rare in a silo of ten thousand is rare in a silo of that one
size. `sweep.py` builds the same corpus at four sizes to separate them.

```
python data/silo/sweep.py            # search-only cards, no classifier, ~7 min
```

| people | articles | IDX MB | DAT MB | acc KB | of the card | absent | rare, 3 docs | common | docs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 6,754 | 4.6 | 0.6 | 6.6 | 1.3% | 2,902 | 33,955 ±857 | 578,007 | 6,692 |
| 10,000 | 13,072 | 5.0 | 1.2 | 12.8 | 2.6% | 3,834 | 35,510 ±1,239 | 1,116,258 | 13,010 |
| 20,000 | 25,796 | 5.7 | 2.3 | 25.2 | 5.1% | 5,647 | 37,843 ±1,359 | 2,199,945 | 25,734 |
| 37,000 | 47,141 | 6.9 | 4.4 | 46.0 | 9.4% | 8,755 | 40,813 ±1,207 | 4,018,058 | 47,079 |

Three columns, and the middle one is the experiment: `absent` is a word no
corpus holds, `rare` is the mean of five words holding at exactly three
documents, and `common` is the widest-scattering term there is. Holding the
posting count fixed at three while the corpus grows sevenfold is the only way
to ask whether an article nobody searched for costs anything.

**It does, and the amount is exact.** Subtracting the floor from each row:

| | |
|---|---|
| a query that finds nothing | **1,910 + 37 per page**, to the instruction at all four sizes |
| three documents, above that floor | 31,053 / 31,676 / 32,196 / 32,058 — flat |
| the widest term, per document | 85.9 / 85.5 / 85.3 / 85.2 |

So the cost of a question is `1,910 + 37 × pages + 85 × documents touched`, and
only the middle term knows how large the corpus is. **The conclusion the
one-corpus tables reached survives — what sets the price is how widely a term's
postings scatter — but it is not the whole sentence.** There is a floor that
grows with the corpus whatever is asked, because both passes still walk the
page table to find out which pages to skip. Skipping a page is cheap. Deciding
to skip it is 37 instructions, and there is one of those per 256 articles
however empty they are.

At the card's 502,016-article limit that floor is about **74,000 instructions a
query** — 1,961 pages of tiering overhead before a single posting is read. Set
against the 7.6 M the Wikipedia card's worst query costs it is nothing, and it
is the reason the tiering exists at all. It is still the one cost in this
design that a bigger corpus cannot avoid.

`IDX` barely moves: 4.6 MB at 6,754 articles and 6.9 MB at 47,141, because
`libsearch.NUM_BUCKETS` is 1 << 20 and the empty bucket table is 4 MB before a
single posting is written. At this scale the index is mostly the table.

### The corpus cannot reach the card, by a factor of ten

The sweep stops at 37,000 people because `generate.py` will not go further. Two
walls, and the near one is the calendar: the seventh cohort is born in year 220
and `NOW` is 220, so `populate` refuses at **37,559** rather than housing people
who have not been born yet. Behind it is the geometry, at roughly 57,500 —
`LEVELS` is 144, a floor holds `BEARINGS × RINGS` = 72 dwellings, and 10,368
homes is all there are.

A search card scores **502,016 articles** ([#62](../../issues/62)). 37,558
people is about 47,000 articles, which is 9.4% of it. So the sizing question
this corpus was built to answer has an answer it cannot demonstrate: the card
is not what stops this corpus, and nothing here reaches the point where it
would be. `tests/test_silo.py` pins the inequality rather than the two numbers,
which move with the seed — what must not change quietly is which side of the
card they fall on.

### The classifier was two and a half times larger than it needed to be

`classify.py` defaults to 256,192 hidden units. Nothing had ever asked whether a
twenty-way decision needs them. Swept over the silo's paths, with accuracy on
trained phrasings and on unseen ones averaged over three held-out splits:

| hidden | weights | trained | unseen |
|---|---:|---:|---:|
| 256,192 | 85,760 | 96.1% | 45.0% |
| **128,96** | **30,592** | **95.3%** | **45.8%** |
| 64 | 9,472 | 90.1% | 43.8% |
| 32 | 4,736 | 76.1% | 39.1% |
| (none) | 2,560 | 81.4% | 37.8% |

128,96 gives up 0.8 points for 2.8× fewer weights and is no worse on unseen
phrasings — that column is noise at either width. Below it the loss is real: 64
costs six points, 32 costs twenty, and **a 32-wide bottleneck is worse than no
hidden layer at all**, which is the shape of a layer too narrow to carry 128
buckets rather than a model too small to learn.

Measured on the card, not extrapolated — both cards built from the same corpus
on the same commit, since [#51](../../pull/51) repacked `.DAT` and comparing
across it would have measured that instead:

| | 256,192 | 128,96 |
|---|---:|---:|
| binary | 94.4 KB | **38.9 KB** |
| a question, at one hop | 756,277 | **383,117** instructions |
| card bytes | 4,961 | 4,961 |
| routes correctly, trained phrasings | 95.8% | 95.6% |

Both arms were measured before initials were joined, which is why neither
matches the 366,655 the card costs now. The comparison is between them.

Near enough half the work for 0.2 points. **Card bytes do not move at all**,
which is the cleanest confirmation that the classifier is arithmetic and not
I/O — and therefore that shrinking it is free everywhere except accuracy.

### It is short of grammar, not short of data

Nothing tried against the ~45% unseen-phrasing number moved it — not masking,
not position bands, not halving the model. All three were changes to the
encoder or the architecture. So the remaining question was whether the limit is
the machine at all, and it is not.

Train on **k** of the nine wordings left after the held-out three, evaluate on
the same three at every point, three seeds each:

| phrasings | rows grow with k | rows held at 7,200 |
|---:|---:|---:|
| 1 | 17.3% | 17.4% |
| 2 | 22.7% | 22.7% |
| 3 | 31.4% | 32.7% |
| 4 | 36.3% | 36.7% |
| 6 | 39.2% | 40.6% |
| 9 | **45.8%** | **45.8%** |

Two things, and the second is the one that matters.

**The curve is still climbing at nine** — 39.2% to 45.8% for the last three
wordings, with three-seed spreads that do not overlap. Whatever this classifier
is short of, it has not run out of it.

**The two columns are the same.** The right-hand arm holds the training set at
7,200 rows however few wordings it is split across, so three phrasings get 120
questions each instead of 40. It buys nothing: 32.7% against 31.4%, inside the
noise at every k. **More examples of the same sentence are worth nothing; more
sentences are worth everything.** That is a fact about the phrasebook and not
about the corpus, and it is the first thing measured here that says what to do
next rather than what not to.

What it does not say is where the curve ends. Nine points of grammar is what
`relationpaths.py` happens to contain, and extrapolating past the last
measurement is how the numbers in this repository have gone wrong before. The
test is to write more wordings and re-run this — with the caveat that a second
dozen written by the same hand on the same afternoon will be more like the
first dozen than a stranger's would be, which would understate the gain.

### Position bands lose, and now they lose with error bars

`--position-bands` seeds each trigram's hash with where in the query it
appeared, so word order stops being discarded ([ENCODING.md](../../ENCODING.md)).
It looks like the obvious lever on the ~45% unseen-phrasing number, since
`who is the father of X's father` and `who is X's father` differ by structure
and share every word.

| bands | trained | steady | unseen, three seeds |
|---|---:|---:|---|
| **flat** | 95.3% | **114/240** | **45.8%** — 45.4 / 46.6 / 45.5 |
| 2 | 95.4% | 93/240 | 35.6% — 35.6 / 36.5 / 34.6 |
| 4 | 93.8% | 81/240 | 29.5% — 26.4 / 33.5 / 28.7 |
| 8 | 91.4% | 48/240 | 21.5% — 17.6 / 22.5 / 24.3 |

Monotone, and the three-seed spreads do not overlap — this is well outside the
noise that killed the masking result. `data/questions/relations.py` already
reported bands losing on the Wikipedia class set and said why: no class there
is another class reversed, so word order carries no signal while the banding tax
on the buckets still applies. The silo's twenty paths have the same property,
and behave the same way. That note asks for a multi-seed re-run before the
figure is quoted, and this is it.

Two things worth taking from the table beyond the verdict. **Trained accuracy
barely moves at two bands — 95.4% against 95.3% — while unseen phrasings lose
ten points**, which is a clean demonstration that the in-grammar number hides
the damage. And **banding makes the classifier more name-sensitive, not less**:
steadiness falls from 114/240 to 48/240, because the same name in a different
position now hashes differently.

### The hop limit, on the actual machine

```
  hops   instructions      +/-  card bytes    +/-  answered
     1        366,655   19,950       4,701    161     20/20
     2        366,092   25,275       4,840    166     20/20
     3        371,171   27,931       5,095    271     20/20
     4        373,118   26,852       5,219    177     20/20
     5        367,864   20,850       5,266    168     20/20
     6        384,938   26,845       9,419    175      0/20  <- past the hop limit
```

Generation 6 needs a seventh hop and `CLIMB_LIMIT` allows six. On the eZ80 that
is not an error message: the walk returns nothing, the program falls back to
listing articles, and the card bytes double because a fallback reads article
text and a graph answer does not.

That column read `2/20` until the entity lookup was fixed, and the two were not
successes — they were two different people resolving to one document.

### Two stages that are not the graph

**Entity lookup: 100% first**, over 500 people, up from 88.6%. Only *first* is
usable — the walk follows the top hit and nothing else.

It was 88.6% because **2,264 of the 10,000 share a first and last name** with
somebody and differ only by a middle initial. That was not a weak signal, it
was *no* signal: `libsearch.tokenize` dropped single characters at both ends,
so `Amanda M. Wilson` and `Amanda X. Wilson` were not similar queries, they
were **the same query**, and the tie-break decided who you meant. 95% of every
lookup miss was that.

The fix is to glue an initial to the name after it — `mwilson` — which makes a
rare, highly specific term out of exactly the character that tells them apart.
Indexing the 26 letters on their own would do the opposite: each would land on
hundreds of scattered documents and flag every page. Titles and aliases only,
because that is where a name is an identity rather than a mention.

It cost 2.9% more postings, changed nothing on Wikipedia's twenty probes, and
made queries *cheaper* — `alexander e wong` went from 173,966 instructions to
154,276, because `ewong` is rarer than `wong`.

And it deleted the two generation-6 "answers" in the table above, which is the
part worth pausing on. `Amanda M. Wilson` and `Amanda X. Wilson` had both
resolved to the same document and both answered `Kyle I. Wilson.` at
byte-identical cost — the graph right about the wrong person, with nothing on
the screen to say so. With the initials joined, generation 6 fails **0/20**:
the machine now reports its hop limit honestly for every one of them.

**The classifier is where the accuracy goes.** Three numbers, and the first one
means nothing:

| | |
|---|---:|
| a `--val-frac` split over queries | 96.8% |
| **phrasings it was never trained on** | **44.5%** |
| questions it *was* trained on | 95.8% |
| phrasings that answer the same way whatever the name | **124/240** |

The first is the trap `data/README.md` warns about, measured: these questions
are templated, so a held-out "who is X's father" still has "who is Y's father"
in the training half. Holding out whole *phrasings* instead costs 52 points.

The last row is the one worth staring at. For **half the phrasings, changing
only the subject's name changes the classification** — the encoder hashes the
whole question into 128 trigram buckets and a name is most of a short question,
so who you ask about is most of what is asked. `who is alexander e wong's
father` routes to the grandfather path; `who is corey w wong's father` routes
correctly. Same question, different person, different answer.

### Masking the name out: tried, measured, not shipped

If the subject is most of what the encoder sees, take it out. The oracle
resolves which document a question is about *before* it needs the relation —
`ask` searches first, and so does the eZ80 program — so the entity's words can
be removed rather than reasoned about. `liboracle.mask` does it and
`relationpaths.py --mask` regenerates the training set that way.

| | steady phrasings | trained phrasings | unseen phrasings |
|---|---:|---:|---:|
| as shipped | 123/240 | **96.1%** | 44.5% |
| masked | **239/240** | 95.0% | 53.3% *(one seed)* |

It does exactly what it was aimed at and **nothing else**. Consistency was not
what limited accuracy — a model fails an unseen wording because it never saw
the wording, and taking the name out does not teach it that "grandad" means two
hops.

The third column is what decided it. +8.8 points at seed 0 became **−5.4 at
seed 1** and 0.0 at seed 2: mean +1.1%, which is noise of exactly the size
`data/questions/relations.py` already documents for this measurement. Quoting
the seed-0 run would have been the same mistake as the two disputed values of
the multi-hop number, made deliberately.

So the trade is a consistent 95.0% in place of an inconsistent 96.1%, plus new
eZ80 code to read a title back off the card and strip its tokens before
`TOKENIZE`. Not worth it. The code stays so the negative result is
reproducible, and nothing calls it.

It also cannot be tested on Wikipedia's one-hop classes at all: SimpleQuestions
records a subject as a Wikidata Q-id rather than as the words appearing in the
question, so there is nothing to remove. This corpus could answer the question
only because it knows who every question is about.

**`data/questions/relations.py` had already rejected masking**, which was found
after the fact and is the more interesting half. Its reason is stronger than
"it did not help": there, masking *actively destroys* the question, because
`what country is X in` is only a place question because X is a place, and with
X removed `in_country` collapsed to 0%.

That mechanism cannot fire here. Every subject in this corpus is a person, so
the entity's type distinguishes nothing and masking is merely neutral rather
than harmful. Two corpora, one repair, two different reasons to refuse it —
and the general one is Wikipedia's, not this one's.

### A dense graph never says "I don't know"

Wikipedia's oracle falls back to listing articles when the graph has no edge,
and 54% of its articles carry no infobox, so it falls back often and visibly.
This corpus has no gaps. Every walk the classifier asks for completes, so a
misrouted question produces a fluent, confident, wrong answer with no symptom
at all — and the one failure the machine *can* report is the hop limit.

Completeness takes away the machine's ability to say it does not know. That is
worth having measured before wishing a corpus were denser.

## What SQLite is doing

Not decoration — each of these earns its place:

| | |
|---|---|
| **generated column** | `apartment.address`, so `42 600 A` has one spelling |
| **foreign keys** | `residence → apartment`, `membership → cohort`, with `PRAGMA foreign_keys=ON` |
| **`STRICT`, `WITHOUT ROWID`** | typed columns, and the primary key *is* the storage |
| **`CHECK` constraints** | a bearing is a multiple of 30 below 720; a term cannot end before it starts |
| **views** | every derived relationship, so a conclusion is never a row |
| **recursive CTE** | `ancestor`, where the depth is a property of the pedigree |
| **`FILTER`** | the `person` view, pivoting `fact` rows back into columns |
| **FTS5** | full text over the leads, for finding a person by name |
| **`ANALYZE`** | see below |

**`ANALYZE` is not housekeeping.** Every kinship view is a three- or four-way
self-join of `edge`, and with no statistics the planner cannot tell a lookup
returning 2 rows from one returning 20,000: it chose a cross product and
`cousin` had not finished after three minutes. With `sqlite_stat1` in front of
it the same query takes 0.15s. Four hundred milliseconds, once, at the end of a
build.

The other planner lesson is in `schema.py`: **an aggregate is the end of a view
chain, never the middle.** `cousin` was originally written in terms of the
`sibling` view, which groups; SQLite will not push a join condition into a
`GROUP BY`, and the whole query fell off a cliff. Both `aunt_or_uncle` and
`cousin` are now written as chains of `child_of` for that reason.

## Reproducing it

The database is not in git — 39 MB of derived data that three seconds rebuilds
exactly. What is committed is the generator, and `--seed` is the whole
provenance:

```bash
python data/silo/generate.py --seed 18 --people 10000   # the default
python data/silo/generate.py --seed 7  --people 2000    # same shape, smaller
```

Generation sizes scale with `--people`, so a small corpus is the same shape as
the full one rather than a prefix of it — a prefix would be all founders and
would answer no kinship question at all.

The card is six files in this directory, in the order they run:

| | |
|---|---|
| `schema.py` | the database — tables, views, and what is not a table |
| `generate.py` | the simulation, and the writer |
| `questions.py` | what a graph walk can reach, against ground truth |
| `relationpaths.py` | templated questions, labelled with the path they mean |
| `buildcard.py` | classifier and card — the one place that knows the order |
| `benchcard.py` | the emulator, and what a hop costs |

`sweep.py` is beside them but not part of a build: it makes its own corpora, at
several sizes, to measure what changes between them.

`tests/test_silo.py` builds a 600-person corpus and checks that it is coherent
(nobody is their own ancestor, no parent died before their child was born,
nothing is dated after the present year) and that the three readings of it
agree — the graph edges, the SQL views, and the fact tables. It skips itself if
Faker is not installed, which is why the rest of the repository still has no
third-party dependency.
