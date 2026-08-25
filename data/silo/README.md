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
is 1.9% of a query and the classifier is 78%.

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
| `SILO.bin` | 96.2 KB — program, path table and classifier |
| `SILO.IDX` | 4.9 MB — 1,525 terms, 323,732 postings |
| `SILO.DAT` | 3.5 MB — titles and leads |
| `SILO.GRF` | 1.6 MB — 105,404 edges over 16 relations |
| accumulator | 13 KB resident, one byte per article |

All 20 phrases the classifier knows are paths the card can walk.

### Where a question's time actually goes

`benchcard.py` asks the same question of people at different pedigree depths, so
the only thing varying is the number of hops, and compares against a card built
without `--relations` — one that searches and neither classifies nor walks:

| | share of a query |
|---|---:|
| the classifier — one forward pass, 85,760 two-bit weights | **78.1%** |
| the search — BM25 over 13,072 articles | 21.9% |
| the graph walk — four hops | 1.9% |

**The graph is the cheap part, by a factor of forty.** What the card pays for is
deciding which question it was asked, not answering it. That is the opposite of
where the effort has gone in this repository, and it is the useful thing to know
before optimising anything.

A hop moves about **178 bytes** off the card: a binary search over 105,404
fixed-width records is 17 probes of 7 bytes, which is what the measurement
recovers. In instructions a hop is around 3,600 — a slope over five hop counts,
and smaller than the spread of any single query, so it is quoted as *under 1% of
a question* rather than as a constant. Quoting such a number to four digits is
the mistake this repository already made once.

### The hop limit, on the actual machine

```
  hops   instructions      +/-  card bytes    +/-  answered
     1        752,372   17,509       2,996    430     20/20
     2        753,648   20,539       3,394  1,316     19/20
     3        758,356   15,628       3,391    280     20/20
     4        767,983   10,531       3,845    978     19/20
     5        763,128   10,173       3,659    251     20/20
     6        775,518   18,872       7,520  1,289      2/20  <- past the hop limit
```

Generation 6 needs a seventh hop and `CLIMB_LIMIT` allows six. On the eZ80 that
is not an error message: the walk returns nothing, the program falls back to
listing articles, and the card bytes double because a fallback reads article
text and a graph answer does not. The two that answered are the collisions
below.

### Two stages that are not the graph

**Entity lookup: 88.6% first, 98.0% in the top three** over 500 people. Only
*first* is usable — the walk follows the top hit and nothing else. Wikipedia's
equivalent probe scores 85% first, so this is comparable, and it is harder than
it looks: Faker draws from a few hundred first names and a few hundred
surnames, so **1,292 of the 10,000 share a first and last name with somebody
else** and differ only by a middle initial — one search term, one letter.

That is exactly how the two generation-6 "answers" happened. `Amanda M. Wilson`
and `Amanda X. Wilson` both resolved to the same document and both answered
`Kyle I. Wilson.` at byte-identical cost. The graph was right; it was right
about the wrong person, and nothing on the screen says so.

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

`tests/test_silo.py` builds a 600-person corpus and checks that it is coherent
(nobody is their own ancestor, no parent died before their child was born,
nothing is dated after the present year) and that the three readings of it
agree — the graph edges, the SQL views, and the fact tables. It skips itself if
Faker is not installed, which is why the rest of the repository still has no
third-party dependency.
