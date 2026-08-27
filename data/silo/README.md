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
of them used to run into the limit:

```
generation      asked  hops needed   reached at 6   reached at 8
1                 234            1         100.0%         100.0%
2                 290            2         100.0%         100.0%
3                 299            3         100.0%         100.0%
4                 299            4         100.0%         100.0%
5                 324            5         100.0%         100.0%
6                 307            6           0.0%         100.0%
```

`CLIMB_LIMIT` counts the values a climb may *examine* rather than the hops it
may take, so a limit of n buys n − 1 hops. At 6 this corpus stopped one short of
its deepest generation, and that was the price of a walk that must not loop
forever — visible here and nowhere else, because this is the corpus where the
true answer is known.

**It is 8 now, and this corpus no longer reaches it.** The reason came from
somewhere else entirely: Wikidata supplied Simple English Wikipedia's
containment, and the chains it added were longer than the ones it replaced —
`Cannes` reaches France through an arrondissement, a department, a region and
`Metropolitan France`. 4,643 answers went past a limit that had one step of
spare. Eight recovers them and nine buys nothing, and seven generations happens
to be exactly what eight covers.

What that cost on the machine is [measured below](#the-hop-limit-on-the-actual-machine),
and the answer is nothing: the climbs that never reach the limit are
byte-identical, and the one it newly answers got *cheaper*.

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

## Written entries, beside the generated ones

Every one of the ten thousand generated leads is the same sentence with
different nouns in it. That is what makes this corpus good for measuring a
graph walk and useless as something to read, and the card has room for both:
the sweep below puts the whole thing at 2.6% of what the machine can hold.

`authored.py` reads a directory of text files into the same `article` table,
with the same `source`, so `buildwikisearch.py` takes no new argument and
cannot tell them apart:

```bash
python data/silo/generate.py
python data/silo/authored.py --report      # <- and re-run after any generate
python buildwikisearch.py --db data/silo.db --source silo --out dist/SILO
```

Ten entries ship in `authored/` — incident reports, committee minutes, a
maintenance log, a judicial direction on what the archive may be asked. They
carry no `edge`, no `fact` and no `entity_type`: a written entry is findable
and readable, and the graph can neither walk to it nor from it. The oracle
answers *about people* from the graph, and this is the archive it is sitting
on.

**Re-run it after any re-generate.** `generate.write` opens with
`DELETE FROM article WHERE source = ?`, which takes these too — the same shape
as `data/wikipedia/birthplaces.py` step 2a.

### What a written entry costs, against a generated one

Codes are learned per corpus, so #51's 30.6% on Wikipedia leads said nothing
about either of these. Learned over the corpus and packed one entry at a time,
which is what `write_text` does:

| | entries | raw | packed | saving | bytes each |
|---|---:|---:|---:|---:|---:|
| written | 10 | 15,240 | 10,721 | **29.7%** | 1,524 |
| generated | 4,000 | 450,020 | 99,077 | **78.0%** | 112 |

**Written prose packs like Wikipedia's — 29.7% against 30.6% — and the
generator's leads pack like a template, because that is what they are.** The
78% is not a compression result, it is a measurement of how little the
generated corpus says: a lead that is one sentence with the nouns swapped is
mostly a sentence the packer has already seen.

So the two are not interchangeable on the card. A written entry costs about
1,070 packed bytes and a generated one about 25 — **forty of them for every
document somebody wrote.**

Which still does not make prose the constraint. The ceiling is a count and not
a size: the accumulator is one byte per article whatever that article holds, so
a card of 502,016 *written* entries is as legal as one of 502,016 generated
ones and comes to about 540 MB of `.DAT`. The format has offsets for 4 GB and
the machine has an SD slot. Filling it would take fifteen hundred novels'
worth of writing, and the writing is the part that runs out.

### An article is capped by what the device reads, and it was not

`READ_ARTICLE` reads exactly one `CHUNK` — 2,048 packed bytes — and `UNPACK`
walks it until it has seen the two NULs ending the title and the lead. An
article packing to more than that **does not truncate**: the second NUL is not
in what was read, so the decoder carries on into whatever the last query left
in SRAM.

Nothing checked this, because every lead had been 300 characters since the
format was written. Asked for a 3,000-character lead the machine printed about
two and a half thousand of them and stopped, and looked entirely well while
doing it — emulated SRAM starts zeroed, so the decoder ran into zeros and took
them for the terminators it was waiting on. On hardware it would run into
whatever the previous query left.

`libsearch.write_text` now refuses an article past `MAX_PACKED_ARTICLE` or past
the `2 * CHUNK` it unpacks into, naming the article. `buildwikibin` asserts the
two constants against its own `CHUNK`, so the build refuses exactly what the
device cannot finish.

`CardSearch.article` was reading 4,096 packed bytes against the device's 2,048,
which is the more embarrassing half: a card the machine could not finish was
one the reference finished fine, so every test that compares the two would have
agreed with the wrong one. It reads `MAX_PACKED_ARTICLE` now.

The cap on a written entry is therefore the device's and not a taste. Byte-pair
packing only ever replaces a pair with one byte, so packed text is never longer
than what went in, and capping the *raw* entry at the packed limit is a cap no
prose can get past however badly it compresses. That is why `authored.MAX_BODY`
is derived rather than guessed: a character count would have needed a
compression ratio, and the ratio is a property of the corpus.

All ten entries come back off a real card byte-for-byte identical to the
reference, at 1,414 to 1,536 bytes each.

### And then the screen broke them in half

Nothing in this repository has ever emitted a VDU sequence. All fifty-seven
print sites push one character through `RST 10h` and let the terminal decide
where a line ends, which was invisible for as long as a lead was 300 characters
of one paragraph. Fifteen hundred bytes of prose is not.

`PRWRAP` measures the next word before printing the space in front of it, and
breaks the line instead when it will not fit. No lookahead buffer is needed:
the whole article is already unpacked in `TEXTBUF`, so the word can be measured
in place. A paragraph break the author wrote is honoured rather than treated as
another space, which is the only formatting `authored.py` keeps.

```
Incident Report 214-11: Cistern Pump Failure, Level 142
At approximately 0340 on the eleventh day of the two hundred and fourteenth
year, the primary cistern pump on Level 142 stopped without warning. The
Third Shift pump operator on duty logged the silence before the pressure
alarm reached her, which is the only reason the loss was held to eleven
hours of supply rather than the full reserve.

Water Treatment attended within the hour. The fault was traced to the lower
bearing housing, where a seal had been weeping for long enough to leave a
salt line the width of a thumb...
```

**106 bytes, and it cost nothing at all.** The article ceiling moves in whole
256-article pages — an article is 257 bytes of budget and the gap holds a whole
number of pages — so 106 bytes of program did not cross a boundary and the
limit is still 502,016. That is the granularity from [#65](../../pull/65) paying
for something rather than just being a fact about the arithmetic.

`AgonHost` models no screen; it collects what `RST 10h` was given. That is the
right level to assert at, because the *program* is what has to decide where a
line ends — so the tests read the character stream the way a terminal would and
check that no line runs past the width and no word is split across two.

### Two of them lose their own name

`Ration Appeals Panel, Case 2196` asked for by its own title returns the
generator's `Ration Appeals Panel` — the committee stub — and
`Relic Disposal Committee, Minutes of Year 217` likewise. Both written entries
are on the card and both are found; they are not first.

This is BM25 doing what it is for. The stub is eleven words long and the query
terms are most of it, so it scores above a fifteen-hundred-byte document that
mentions the committee once in its title. Nothing is broken and nothing needs a
setting changed.

It is worth knowing because it is the shape of the problem authored entries
have generally: **a written entry named after something the generator already
writes about competes with it, and usually loses.** Naming an entry for the
thing it is *about* is the instinct, and the instinct is wrong here. An exact
title collision is refused outright by `authored.py`, because taking the name
would delete the generated article — but a near miss like these two is legal,
findable, and second.

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

### Before writing more wordings, the encoder had 128 buckets

The next move was obvious and wrong to make first. Writing more wordings meant
trusting the caveat above, so `tools/phrasebook_diversity.py` measured it
instead: how much a path's wordings resemble each other in the 128 buckets the
model actually sees, and how far a held-out wording is from the nearest one
trained on.

**The caveat does not hold.** Per-path held-out accuracy against per-path
novelty correlates **−0.126** across the twenty-one paths — nothing. The 44.5%
is a fact about the model, not an artifact of a phrasebook that repeats itself,
and more varied wordings would not have depressed it.

What the instrument did find was worse. Held-out wordings sit only **0.225**
away from their nearest training twin — 77.5% similar to something the model
has already seen — and it still gets more than half of them wrong. And of the
misses, **a quarter are a path losing to its own prefix**:

```
class_is class_is_of  ->  class_is           71
mother_is mother_is   ->  mother_is          64
father_is father_is   ->  father_is          61
```

Those paths already say *grandmother* in five of their twelve wordings. The
distinguishing word is there and the encoder is losing it, so no amount of
English fixes that quarter.

**859 distinct trigrams, 128 buckets, 85% of them sharing one.** Every bucket
occupied, 6.7 trigrams apiece — so a trigram that separates two paths arrives
on top of six that do not. `libinfer.NUM_BUCKETS` had been 128 since the first
commit; [#54](../../pull/54) swept the classifier's *hidden* width and nothing
had ever swept its *input* width.

| buckets | held out | three seeds | trigrams colliding |
|---:|---:|---|---:|
| 128 | 45.0% | 42.7 / 46.8 / 45.4 | 85% |
| **256** | **52.5%** | 54.5 / 51.2 / 51.8 | 71% |
| 512 | 51.7% | 53.4 / 48.4 / 53.1 | 49% |
| 1,024 | 51.2% | 51.7 / 50.8 | 27% |

**+7.5 points from one constant**, with three-seed spreads that do not overlap
— against masking (noise), position bands (worse) and halving the model (no
change). All of it arrives at 256 and none of it after, which is the happier
half of the result: **the device takes the bucket index from the hash's low
byte and puts it in one register, so 256 is also the most it can address**
without a wider index everywhere. The sweep and the hardware agree on the same
number for unrelated reasons.

Collisions keep falling past 256 and accuracy does not, so collision is the
mechanism up to a point and not the whole story. What is left at 256 is the
prefix quarter, which did not move: it is 19.6% of misses at 128 and 23.9% at
512 — fewer in absolute terms because there are fewer misses, but a larger
share of what remains. That is a different problem and it is still open.

### What the width costs on the card

Measured on a real silo card, same corpus and same 128,96 hidden layers:

| | 128 | 256 |
|---|---:|---:|
| held-out phrasings | 44.5% | **49.9%** |
| weights | 30,592 | 47,072 |
| `SILO.bin` | 38.9 KB | 55.9 KB |
| articles this image can score | 467,200 | 449,792 |

**17,408 articles for five points.** On a corpus of 13,082 that is free; on the
whole of Simple English Wikipedia at 283,997 the ceiling is still 1.6× the
corpus.

One figure worth not misreading: `classify.py` reports the model as "11,768
bytes packed 2-bit", and that is the `.npz`. The eZ80's compact kernel
interprets **a byte per weight**, so what the card carries is 47,320 — which is
where sixteen of the seventeen kilobytes went.

The count travels in the model file the way `position_bands` does, and is
written only when it differs from 128, so every model trained before this
existed still loads as what it is and no shipped artifact moved.

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

Measured when `CLIMB_LIMIT` was 6, which is why generation 6 fails here. It is
8 now, and this is the measurement that said raising it was safe:

```
  hops   instructions      +/-  card bytes    +/-  answered
     1        366,655   19,950       4,701    161     20/20
     2        366,092   25,275       4,840    166     20/20
     3        371,171   27,931       5,095    271     20/20
     4        373,118   26,852       5,219    177     20/20
     5        367,864   20,850       5,266    168     20/20
     6        384,938   26,845       9,419    175      0/20  <- past the hop limit
```

Generation 6 needs a sixth hop and a `CLIMB_LIMIT` of 6 allows five. On the eZ80
that is not an error message: the walk returns nothing, the program falls back
to listing articles, and the card bytes double because a fallback reads article
text and a graph answer does not.

That column read `2/20` until the entity lookup was fixed, and the two were not
successes — they were two different people resolving to one document.

### The limit counts values examined, not hops

Everything above, and every comment in this repository, described `CLIMB_LIMIT`
as a count of hops. It is not, and the difference is exactly one.

Both walkers test the type at the **top** of the loop and give up when the count
runs out, so the value the last hop reached is never tested at all:

```python
for _ in range(climb_limit):
    if self.is_a(here, name):
        break
    here = hop(here)          # <- the value this lands on is only tested
else:                         #    if there is another iteration left
    return None
```

The eZ80 does the same thing — check, decrement, give up on zero, and only then
hop — which is precisely why nothing caught it. Two implementations agreeing
tells you they match, not that they are right, and this is the failure mode that
argument has.

So **a limit of n buys n − 1 hops.** Generation g is exactly g hops from its
founder, measured over all 10,000 people, so the limit of 6 this was written
against bought generations 1 to 5 and generation 6 fell one short. Three
separate files reached that correct conclusion through a backwards explanation,
and `tests/test_silo.py` asserted `range(1, CLIMB_LIMIT)` and was right for a
reason nobody had written down.

At 8 it buys seven hops and this corpus has seven generations, so the limit is
no longer what stops anything here. The test that demonstrated it now lowers the
limit to do so, rather than relying on the default to bite — otherwise the only
test of the bound would be one that passes because nothing exercises it.

### Which makes it worth choosing

`buildwikisearch --climb-limit`, threaded through `buildcard.py`. The limit is
an immediate in `GW_CLIMBTO` rather than an unrolled loop, so the two cards are
the same size — **one byte differs between them, at offset 3291, and it is the
number itself.** Both binaries are 39,865 bytes.

Eight subjects a generation, on both cards:

| gen | hops | limit 6 | instr | card bytes | limit 7 | instr | card bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8/8 | 368,649 | 4,633 | 8/8 | 368,649 | 4,633 |
| 2 | 2 | 8/8 | 361,798 | 4,686 | 8/8 | 361,798 | 4,686 |
| 3 | 3 | 8/8 | 368,153 | 5,053 | 8/8 | 368,153 | 5,053 |
| 4 | 4 | 8/8 | 369,687 | 5,096 | 8/8 | 369,687 | 5,096 |
| 5 | 5 | 8/8 | 355,772 | 5,097 | 8/8 | 355,772 | 5,097 |
| 6 | 6 | **0/8** | 375,877 | 9,310 | **8/8** | 360,646 | 5,354 |

Two things worth taking from it. **Generations 1 to 5 are byte-identical across
the two cards** — same instructions, same card bytes — so a deeper limit costs
nothing whatever on the climbs that do not reach it. The loop is bounded by the
answer, not by the bound.

And **answering generation 6 is cheaper than failing it**: 360,646 instructions
against 375,877, and 5,354 card bytes against 9,310. Failing means falling back
to listing articles, and reading article text costs more than a binary search
does. The hop limit was never buying speed.

What it buys is termination. A cycle in the data — two places each inside the
other — has to stop somewhere, and that is the whole reason there is a number.
Six was a containment depth for Wikipedia's `in_country`, where places are not
nested six deep; it was never a pedigree depth, and this corpus is seven
generations tall. It stays at 6 by default because that is Wikipedia's number
and this card is the one that should ask for something else.

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

> The second row is **49.9%** on the card this now builds. It was 44.5% for as
> long as the encoder had 128 trigram buckets, which nothing had ever swept —
> see [before writing more wordings](#before-writing-more-wordings-the-encoder-had-128-buckets).
> The paragraphs below are the diagnosis made at 128 and every word of it still
> holds; the number moved for a reason none of them names.

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

### Giving it back, as a class rather than a threshold

`libinfer.classify` is a bare argmax. There is no score, no margin and nowhere
to put a confidence cut-off, so the only way this machine can decline is to be
*taught* declining — the way `examples/smalltalk` carries CLINC's out-of-scope
utterances as an `IDK` class rather than as a rule.

`relationpaths.PATHS` now holds a twenty-first label, `refuse`, with twelve
phrasings covering the four shapes above: a count over a set, a maximum over
one, an intersection of two ancestor sets, and a count around the ring.

On the card it needs its own step count. Zero was already taken — it means "no
edges for this phrase", which sends the machine to the article list — so
`libgraphcard.REFUSE` is 0xFF and the walk routine jumps to a message instead
of a walk. `None` and `[]` are different things all the way from `paths_for`
through the `.GRF` to the eZ80.

```
? how many cousins does amanda m wilson have
I do not know that one.

? who is amanda m wilson's father
Larry O. Wilson.
```

**A refusal is the cheapest thing the card does** — 2,573 card bytes against
4,663 for an answer, because it reads no article text and walks no graph.

### What the class costs, and what it does not fix

| | 20 paths | 21 with `refuse` |
|---|---:|---:|
| binary | 39,865 | 40,030 |
| trained phrasings | 95.6% | 95.4% |
| **phrasings never seen** | **44.5%** | **44.3%** |
| steady phrasings | 114/240 | 116/252 |

A twenty-first class costs 165 bytes and nothing else. Both accuracy columns
move less than the seed-to-seed noise this file has already documented, so the
class is free in the only budget that was in question.

Held out, **`refuse` scores 47.5%** — above the 44.3% mean and mid-table among
the twenty-one. Two caveats, and the second is the interesting one.

Its number is not comparable to the others. For a path, twelve phrasings are
twelve ways of asking one question; for `refuse` they are three ways each of
asking four *different* questions, so holding out three at random can remove a
whole shape rather than a wording of a familiar one. It is a harder split, not
a better classifier.

And **when a refusal is missed it goes where the words point**: 32 of the 63
misses land on `crew_is`, because "who is the oldest person on X's crew" shares
almost every term with "who is on X's crew". The rest scatter over `father_is
father_is`, `job_is` and `works_in located_in`.

That is exactly the misrouting the class exists to prevent, still happening on
the wordings where an answerable path is a near neighbour in trigram space. The
class converts about half of these questions from a confident wrong answer into
a refusal. It does not convert them all, and the ones it misses are the ones
that look most like questions this corpus can answer.

### And the class is fragile to anything else being added

Counting became a walkable step, so the obvious next move was to teach the
classifier to ask for one. Measured first, three seeds, before writing a line
of it into the vocabulary:

| | classes | held out | `refuse` | the counts | refusals answered as a count |
|---|---:|---:|---:|---:|---:|
| as shipped | 21 | 44.9% | **56.7%** | — | 0 |
| with two count classes | 23 | 45.7% | **30.0%** | 73.3% | 8.7 |

The count classes work — 73.3%, well above the 45.7% mean. **They cost the
refusal class twenty-seven points**, which is why they are not in
`relationpaths.PATHS` and why the card still cannot be asked for a count.

The prediction that motivated the measurement was **wrong**, and being wrong is
what makes it worth writing down. The expected collision was between `how many
people live on {s}'s floor`, which is refused, and `how many people were born
on {s}'s level`, which is now answerable — two sentences a word apart, one
reachable and one not. That is not what happened. Breaking the leak down by
shape, at one seed:

| refusals answered instead of refused | 21 classes | 23 classes |
|---|---:|---:|
| `which of {s}'s crew is eldest` → `crew_is` | 32 | 32 |
| `is {s} any relation to the sheriff` → *seven different classes* | 26 | 57 |
| total, of 120 | 63 | 89 |

The `crew_is` leak is **identical in both** — it is the pre-existing one
measured above and counting did not touch it. Every one of the twenty-six extra
failures is the *intersection* shape, which simply found new places to go, and
twenty of them went to a count.

So the fault is not a wording collision. **`refuse` is four unrelated question
shapes under one label, and the intersection shape has no stable region in
trigram space** — `is X any relation to the sheriff` shares almost no
vocabulary with `how many cousins does X have`, yet they are the same class.
Adding *any* vocabulary gives those questions somewhere new to land; counting
was the occasion, not the cause.

### The split was the obvious fix, and it is worth nothing

So `refuse` was split by shape — four labels all encoding to
`libgraphcard.REFUSE`, which costs the card nothing because the step table only
needs to know a phrase *is* a refusal — and each was written twelve wordings,
the same as every answerable class.

It works. It is also not why it works, and the control is the whole story:

| | classes | refusal recall | answerable |
|---|---:|---:|---:|
| one class, 12 wordings *(what shipped)* | 21 | 56.7% | 44.6% |
| **one class, all 48 wordings** | 21 | **77.2%** | 42.2% |
| four classes, the same 48 wordings | 24 | 73.7% | 42.5% |

Refusal is scored as **did it refuse at all**, not as did it pick the right one
of four, because that is the only distinction the eZ80 can make.

**One class holding forty-eight wordings beats four classes holding the same
forty-eight.** The split contributes nothing; the wordings contribute
everything. The diagnosis above — that the class had no coherent region — was
wrong, or at least was not the binding constraint. It was short of grammar, which
is the same answer [#63](../../pull/63) reached about the chain classes and the
same one the [phrasing curve](#it-is-short-of-grammar-not-short-of-data) reached
about everything else. Three wordings per shape was simply too few.

So the shipped arrangement is one class with forty-eight wordings, and the four
labels are gone.

### What it costs, in the currency this repository cares about

| | refusal recall | answerable | **answerable questions refused** |
|---|---:|---:|---:|
| 12 wordings | 56.7% | 44.6% | 3.2% |
| 48 wordings | **77.2%** | 42.2% | **10.0%** |

It catches twenty more points of the questions it should decline, and wrongly
declines seven more points of the ones it could answer. That is a judgement
rather than a measurement, and the judgement is that it is worth it: a wrong
refusal is visible and unhelpful, a missed refusal is a confident wrong answer
with nothing on the screen to say so, and this file has argued from the start
that a machine with gaps that says so beats one that is merely unreliable.

### And it makes counting affordable

The count classes cost the old arrangement twenty-seven points of refusal. They
cost this one fourteen, and land it **above where it started**:

| | classes | refusal recall | answerable |
|---|---:|---:|---:|
| 12 wordings, no counts *(what shipped)* | 21 | 56.7% | 44.6% |
| 48 wordings, plus two count classes | 23 | **63.1%** | 43.3% |

Same answerable accuracy, better refusal, and two capabilities that were not
there. Counting is no longer blocked by the classifier — what remains is card
support for a count step and the eZ80 routine to walk one, which is
[scoped](#counting-was-not-an-aggregate-after-all) and not built.

## Using it as an oracle for authored fiction

Everything above measures the card. This is what an author of a Silo-like
Interactive Fiction would need to know before writing against one, and most of
it is a constraint rather than a capability.

### What it can be asked

Two things, and they do not overlap.

**About people, from the graph.** Twenty question shapes — parent, spouse, job,
shift, crew, class, dwelling, neighbour, section, birth year, and the
compositions of those. A question in one of those shapes about somebody in the
corpus is answered with a name and a full stop, in about 370,000 instructions,
whatever the corpus size.

**About anything, from the text.** Any entry — generated or written — is found
by the words in it. `data/silo/authored/` holds ten documents nobody generated,
and asking for a phrase in one returns it. That is a search engine, not an
oracle: it hands over prose and makes no claim about it.

### The graph knows nothing about written entries, and that is the point

A written entry carries no `edge`, no `fact` and no `entity_type`. So a *path*
question that lands on one cannot be answered, and what the machine does then is
fall back to showing the text:

```
? who is the cistern pump failure's father
Incident Report 214-11: Cistern Pump Failure, Level 142
At approximately 0340 on the eleventh day of...
```

The classifier routed that to `father_is`, the search resolved it to the
incident report, and the walk found no edge. The fallback is the honest answer,
and it is only available because the entry has no edges to walk. **An authored
entry can never be the subject of a fabricated fact**, because there is nothing
there to fabricate from. Giving written entries their own edges would take that
away, which is the argument against doing it.

### Two different ways of saying no

Worth keeping apart when writing dialogue for the machine, because they mean
different things and the order they fire in is fixed:

| | |
|---|---|
| `Nothing on the card matches that.` | it does not know **who** you mean |
| `I do not know that one.` | it knows who, but not **that question** |

The search runs before the classifier, so a refusal only fires once a subject
has resolved. `how many cousins does zzqqxx have` gets the first message, not
the second, at 7,101 instructions — the cheapest thing the card does.

### The four questions a path cannot express

Composition — follow this, then that, in either direction — is the whole of what
this machine reasons with. It stops at aggregation, ranking and set
intersection, and these are the four shapes it stops at:

| | why |
|---|---|
| how many cousins does X have | a count over a set; a path ends in a value |
| who is the oldest on X's crew | a maximum over a set it can enumerate but not rank |
| is X related to Y | an intersection of two recursive ancestor sets |
| how many live on X's floor | a count around the ring: a program, not a query |

They now route to `refuse` and are declined about half the time on wordings the
classifier never saw. **The other half still misroute**, and predictably: a
refusal whose words overlap an answerable path lands on that path. Anything
phrased around a crew tends to reach `crew_is`.

For an author that is a rule about *phrasing*, not about content. A question the
machine must decline is safest when it shares as few words as possible with one
it can answer — and "who is the oldest person on X's crew" is about as unsafe as
it gets.

### Rules for writing entries

- **Do not name an entry after something the generator writes about.** An exact
  collision is refused by `authored.py`; a near miss is legal and loses. Two of
  the ten shipped entries do exactly this — `Ration Appeals Panel, Case 2196`
  asked for by its own title returns the committee stub, because BM25 prefers an
  eleven-word document the query terms are most of.
- **1,952 bytes an entry**, derived from what the device reads rather than
  chosen. Longer is refused at build time.
- **A written entry costs about forty generated ones** on the card — 1,070
  packed bytes against 25 — and the ceiling is still a count, not a size, so
  neither is the constraint.
- **Re-run `authored.py` after every `generate.py`**, which deletes them.
- The classifier knows twenty shapes and one refusal. A question outside all
  twenty-one does not fail; it lands on whichever of them it looks most like.

### What this is not

It is an oracle you query, not a world you are inside. There is no state, no
turn loop, no parser for verbs, and nothing here writes to the card. Those are
the second half of [#62](../../issues/62) and none of them exist.
## Counting was not an aggregate after all

"How many children does X have" was listed here as a question no path could
express. That was right about `follow` and **wrong about the graph**. The
reverse table is sorted, so every record for one object is contiguous: a count
is a binary search followed by a scan — a loop and a counter, not an aggregate.

> **This sits next to the `refuse` class above, and somebody should measure the
> seam before counting reaches the card.** Checked rather than assumed: none of
> the twelve `refuse` phrasings is a count over a single relation. They are
> counts of a *union* ("how many cousins"), a maximum, an intersection, and a
> count around the ring — all still out of reach, so the class is exactly as
> defensible as it was.
>
> The hazard is trigram distance, not logic. `how many people live on {s}'s
> floor` is trained as a refusal and `how many people were born on X's level`
> is now answerable, and those two share almost every term. That is the same
> near-neighbour misrouting the section above measures, where 32 of 63 missed
> refusals landed on `crew_is` for exactly this reason.
>
> It is only a hazard so far: these count questions are in `questions.py`,
> which measures the graph, and **not** in `relationpaths.PATHS`, which is what
> the classifier learns. The card cannot ask them yet and so cannot confuse
> them yet. Adding them is the point at which this has to be measured.

`libgraph.COUNT` makes it a step. `count_child_of` is "how many children";
`born_on count_born_on` hops to a level and counts what points back at it.

| | steps | walk |
|---|---:|---:|
| how many children does X have | 1 | 100% |
| how many people were born on X's level | 2 | 100% |

A count ends the walk, because a number has no edges — there is nowhere to hop
from three.

What is genuinely out of reach is narrower than "aggregates", and worth stating
precisely now that one of them has moved:

- **a maximum** — "who is the oldest on X's crew" needs the values compared
  rather than tallied, and the card reads ids, not birth years.
- **a count of a union** — "how many cousins" tallies the children of *two*
  parents' siblings, and one scan tallies one relation.
- **"related on any line"** — the paternal line is answered below; any line
  needs the ancestor sets rather than their tops.

**The eZ80 does not carry counting yet**, and what it needs is specific: a
routine beside `GW_HOP` that scans from the record `GW_FIND` lands on instead
of returning it, one of the spare `kind` bytes in the step encoding to mark the
step, and something that can print a decimal number — the program has only ever
printed titles it read off the card.

## Two subjects, not one

Every question above names one person. An investigation asks about pairs —
*is X related to Y*, *did X and Y work together*, *were they at school
together* — and a lookup is what you do once you already know whose record to
open.

`libgraph.common` is the smallest mechanism that answers them: **walk the same
path from both ends and compare the answers.** `founding_father` climbs a male
line until it reaches a founder, so running it twice and comparing settles
descent with no `sibling` table, no ancestor set and no join. Two walks and one
comparison of two 24-bit ids — and the walk is 1% of a query, so there is room
for a second one.

| | pairs | found | missed | false, over 1,500 unrelated |
|---|---:|---:|---:|---:|
| share a founding father | 400 | 341 | 59 | 0 |
| are on the same crew | 400 | 400 | 0 | 0 |
| were in the same class | 400 | 400 | 0 | 0 |

The pairs are drawn *because* they are connected. Two people picked out of ten
thousand share a crew about once in two thousand tries, so "always say no"
scores above 99% here and means nothing — the column that matters is `missed`.

**And the misses are the climb limit, not the comparison.** At `CLIMB_LIMIT` 6
it finds 342 of 400; at 8 it finds all 400. A pair fails when *either* person is
a generation too deep to reach a founder, so **one walk running out costs the
answer for two people** — the hop limit is twice as expensive on a pair question
as on a single one.

It also answers something narrower than "related", and the narrowness is the
honest part: a shared `founding_father` is a shared *paternal* line. Two people
with the same mother's mother do not have one, and this says they are unrelated.

**The card cannot ask this yet**, and the blocker is not the graph. The search
resolves one document and the classifier emits one path, so a question naming
two people has nowhere to put the second. That is a pipeline change — two
searches, two walks, one comparison — rather than a new capability, and the
capability is what the table above measures.

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

## Something to find

```bash
python data/silo/generate.py --plant 12     # and data/silo.key.json beside it
```

A corpus where everything is consistent is a phone book. Every query returns a
fact, no fact is more interesting than another, and the only reason to ask a
second question is that you wanted to know a second thing. That is the right
shape for measuring a machine and the wrong shape for using one.

`--plant N` seeds contradictions and writes down exactly what it seeded. **Off
by default** — every number above was measured without them, and a flag that
quietly changed the data under a measurement would be worse than no flag.

| kind | what is wrong |
|---|---|
| `impossible_father` | a father recorded as dying before his child was born |
| `purge` | a committee whose members were all sent to clean, within three years |
| `altered_parentage` | the `fact` table and the graph name different fathers |

Each inverts an invariant `tests/test_silo.py` already asserted, which is the
design rather than a coincidence: a corpus is interesting exactly where it
violates something a reader assumes, and the assumptions were already written
down as tests. The test that says *no parent died before their child was born*
is the detector.

The third is the one worth having. `libgraph` walks edges and the `person` view
reads facts, and everywhere else they are written from one pass and **cannot**
disagree. Here they do, for a handful of people — which is what a falsified
record looks like from the inside: the card answers confidently, the database
answers differently, and neither is malfunctioning.

```
[impossible_father] Dylan W. Torres is recorded as dying in year 119,
                    1 year before Steven A. Torres was born.
[altered_parentage] The record gives Rachel W. Mathews's father as
                    John O. Wilson; the relations still lead to Eric T. Mathews.
```

### The machine cannot find any of it, and that is the point

Finding an impossible father means asking when someone died, asking when their
child was born, and *noticing*. Three steps, of which the card does two. This
repository has argued from the start that comprehension is out of reach on the
hardware; a planted corpus turns that from a limitation into a division of
labour.

Two things the planter learned the hard way, both now pinned by tests. Moving a
death earlier makes **every** child born after it impossible, so aiming at a
random child planted three contradictions and created ten — it aims at the
youngest now, and stops short of the one before. And the purge moved deaths
earlier too, manufacturing four more impossible fathers indistinguishable from
the planted ones; it respects its victims' children now.

One consequence is left in deliberately. A purged member who sat on a second
committee thins that one as well, so counting cleanings per committee finds
more clusters than were planted. A corpus where only the planted thing is
findable would teach a player to stop looking.

The key is a file beside the database rather than a table inside it, for the
obvious reason.

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

The card is seven files in this directory, in the order they run:

| | |
|---|---|
| `schema.py` | the database — tables, views, and what is not a table |
| `generate.py` | the simulation, and the writer |
| `authored.py` | written entries, into the same tables — optional, and re-run after `generate.py` |
| `questions.py` | what a graph walk can reach, against ground truth |
| `relationpaths.py` | templated questions, labelled with the path they mean |
| `plant.py` | contradictions to find, and the key that says which |
| `buildcard.py` | classifier and card — the one place that knows the order |
| `benchcard.py` | the emulator, and what a hop costs |

`sweep.py` is beside them but not part of a build: it makes its own corpora, at
several sizes, to measure what changes between them.

`authored/` holds the entries themselves, one document per `.txt` file with its
title on the first line. They are the only prose in this corpus a person wrote,
and they are in git because nothing regenerates them.

`tests/test_silo.py` builds a 600-person corpus and checks that it is coherent
(nobody is their own ancestor, no parent died before their child was born,
nothing is dated after the present year) and that the three readings of it
agree — the graph edges, the SQL views, and the fact tables. It skips itself if
Faker is not installed, which is why the rest of the repository still has no
third-party dependency.
