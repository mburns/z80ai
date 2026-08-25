# Simple English Wikipedia, as a search card

The whole encyclopedia on an Agon Light: 283,997 articles, searched in plain
English, from an SD card.

## Building it from nothing

Three commands, about twenty minutes, most of it the download.

```bash
# 1. Fetch a dump (~340MB). Any snapshot works; this is the one it was built
#    against. https://dumps.wikimedia.org/simplewiki/ lists the rest, and
#    `latest/` always points at the newest.
curl -O https://dumps.wikimedia.org/simplewiki/20260801/simplewiki-20260801-pages-articles.xml.bz2

# 2. Into the database (~5 minutes). Nothing else needs the dump afterwards.
python data/wikipedia/ingest.py simplewiki-20260801-pages-articles.xml.bz2

# 3. Into a card (~4 minutes). Add --limit 20000 for a small one that builds
#    in seconds, ranked by how many redirects point at each article.
python buildwikisearch.py --out dist/WIKI
```

For a card that answers questions rather than only finding articles, train the
relation classifier and pass it to step 3:

```bash
python data/questions/relations.py > relations.txt
python classify.py --file relations.txt -o relations.npz \
       --accum-bits 24 --balance
python buildwikisearch.py --out dist/WIKI --relations relations.npz
```

That writes a fourth file, `WIKI.GRF`, and a binary that walks it. **Pass the
same `--limit` to every build of one card.** A document id is a position in the
article list, so a limited card renumbers everything; the header carries a
digest of the titles and the program refuses a mismatched pair, because a wrong
one has no other symptom — every id in it is still some article.

| | full corpus |
|---|---|
| `WIKI.IDX` | 33.1 MB |
| `WIKI.DAT` | 74.5 MB |
| `WIKI.GRF` | 2.4 MB — 167,922 edges |
| `WIKI.bin` | 95.6 KB |

`data/simple_english_wikipedia.db` is **not in git** — it is ~500MB of derived
data, and step 2 rebuilds it from any snapshot. The dump is not in git either.
What *is* committed is everything needed to turn one into the other.

The database records where its contents came from, so a card can be traced
back to a snapshot without asking anyone:

```console
$ python data/wikipedia/ingest.py --stats
  schema_version               7
  simplewiki.articles          283997
  simplewiki.digest            adf8cbb46aabe719
  simplewiki.dump              simplewiki-20260801-pages-articles.xml.bz2
  simplewiki.edges             167922
  simplewiki.facts             2086920
  simplewiki.ingested          2026-08-24T21:49:30
  simplewiki.redirects         114771
  simplewiki.url               https://dumps.wikimedia.org/simplewiki/20260801/...

  simplewiki: 283,997 articles, 114,771 redirects (97.6% resolve), 68 MB of lead
              980,928 category filings over 272,022 articles (96%), 76,102 categories
              2,086,920 facts over 129,732 subjects (46% of articles), 9,664 properties
              values: text 72%, number 20%, date 7%, url 1%
              47 properties map to a relation; biggest unmapped: name (95,698),
              birth_date (44,872), subdivision_type (40,782), years (34,994), ...
```

Two things to read there. The coverage lines go together — 96% of articles file
themselves under a category against the 46% that carry an infobox, which is why
the graph reads both. And the last line is the corpus telling you what it knows
that nothing yet understands; the biggest entries are a footballer's career
table, which is a fair summary of what Simple English Wikipedia is mostly made
of.

The URL is reconstructed from the filename, since a Wikimedia dump name
carries its wiki and its date. A dump named anything else records no URL
rather than a guessed one.


That writes three files. Copy all three onto the card and run `WIKI`:

| | |
|---|---|
| `WIKI.bin` | 6 KB — the program |
| `WIKI.IDX` | 38 MB — hashed dictionary and postings |
| `WIKI.DAT` | 80 MB — titles and leads |

```
Simple English Wikipedia - 283,997 articles
Type a question, or ! to quit.

? mount everest

Mount Everest
Mount Everest is the highest mountain on Earth. Mount Everest is in the
Himalayas, a tall mountain range in Asia. It is about high and one of the
Seven Summits.
```

## Keeping it current

`ingest.py` takes a dump and **replaces** that source's rows in one
transaction. A dump is a complete snapshot, so that is a sync rather than a
merge: articles that were deleted upstream disappear, and an interrupted run
leaves the previous corpus intact. Re-run `buildwikisearch.py` afterwards.

A schema change bumps `PRAGMA user_version`, and a database written by an older
one refuses to open rather than being read as though it were current. Ingest
may rebuild it, since it replaces every row regardless.

The binary contains no corpus and no counts except the article total, so a
rebuilt card drops in beside an unchanged `WIKI.bin` unless the *format*
changed — and a card written for a different format is refused by its magic
rather than misread.

`--source` lets several corpora share one database, so Wiktionary or Wikibooks
could be ingested alongside later.

## What it is, and what it is not

**It is a search engine.** Ask for a thing by name, including a misspelling
Wikipedia has a redirect for, and you get the article:

```
? jane austin        ->  Jane Austen
? zilog z80          ->  Z80
? zx spectrum        ->  ZX Spectrum
```

**It is a search engine, and — given `--relations` — an oracle as well.**
Measured on thirteen probe queries, the right article is in the top three
eleven times.

"Who wrote hamlet" used to return *Hamlet*, and this paragraph used to explain
that this was a search engine working correctly: extracting the answer from the
article is comprehension, which is out of reach here. That is still true. What
changed is that the answer no longer has to come from the article.

```
? who wrote hamlet                     ->  William Shakespeare.
? where was alexander graham bell born ->  Edinburgh.
? what country is warsaw in            ->  Poland.
```

Those come off `WIKI.GRF`, a card file holding the same fact graph `libgraph`
walks, in a layout a machine with no arithmetic beyond addition can read: a hop
is a binary search over fixed-width records. Comprehension is still out of
reach; a comparison and a seek are not. Each of those questions moved between
2,857 and 11,857 bytes off the card.

When the graph has no answer the program lists articles, which is what it did
before, so nothing is lost by asking.

The trigram encoder the language models use scores **2 of 13** on the same
probes and returns *Bures Hamlet* for "who wrote hamlet". It compresses a
document into 128 buckets and discards which words matched, which is the one
thing retrieval needs. This uses an ordinary inverted index instead — no model,
no training, and a better fit for a 1970s instruction set.

## Why it fits

The device does no arithmetic beyond addition. BM25's multiply, divide,
inverse document frequency and per-document length all happen at build time,
and each posting arrives as a five-bit weight that is simply added.

Five bits is not arbitrary: eight query terms at 31 each is 248, so the
accumulator can be **one byte per article**. That is 277 KB for the whole
encyclopedia — resident, no sharding, no routing.

| | |
|---|---|
| accumulator | 277 KB in SRAM, plus a 1,110-byte page table in the image |
| card read per query | ~23 KB → 0.09 s at 250 KB/s |
| instructions per query | 5.4 M before the page tier → ~0.74 s at 18.432 MHz |
| program | 7,450 bytes |

The two passes over the accumulator — clearing it and scanning it for the best
three — used to dominate every query at 284,000 bytes apiece, whatever the
query. The accumulator is now tiered: one flag per 256-article page, set when
a posting lands, and both passes visit only flagged pages. A lookup that names
its subject touches a handful of pages and retires a fraction of the
instructions (7x on a two-term entity query, measured on a synthetic 100k-article
corpus); a term common enough to flag every page pays the whole-corpus scan as
before, plus the table's overhead. The 5.4 M figure is the pre-tier measurement
on this corpus; the tiered number waits on a rebuilt card to be measured
honestly.

## Facts, for the oracle this is not

The database also carries **2,086,920 facts** pulled from infoboxes — an
infobox is a hand-curated set of typed key/value pairs, which is to say a set
of facts about its article. The lead throws them away as furniture; the `fact`
table keeps them.

```sql
SELECT value FROM fact WHERE subject = 'Alexander Graham Bell'
                         AND property = 'birth_place';
-- Edinburgh, Scotland
```

Alongside it, **980,928 filings** into 76,102 categories — what a page files
itself under rather than what it tabulates. 95.8% of articles carry at least
one, against the 46% that carry an infobox, which is why they are here: for a
great many pages a category is the only place a containment is written down.

```sql
SELECT name FROM category WHERE source = 'simplewiki' AND title = 'Michigan';
-- 1837 establishments in the United States
-- Michigan
```

### What the ingest normalizes, and what it refuses to

An infobox is hand-typed by thousands of people, so what arrives is a
folksonomy: before any cleaning, 13,387 distinct property names, 3,740 of them
used exactly once. The ingest normalizes **form** and leaves **meaning** to
`libgraph`, and the line matters — form is mechanical and decidable from the
data, while *"`subdivision_name` and `country` are the same question"* is a
judgement about this corpus that belongs where the judgement is made.

| | |
|---|---|
| comments stripped before splitting | a comment between two pipes ran into the next key |
| keys cleaned and shape-checked | only values went through the cleaner before |
| `subdivision_name1..7` → `(property, ordinal)` | **27.6% of facts** were positional variants |
| values typed `text`/`number`/`date`/`url` | 22% are numbers, which sorted lexically as text |
| known value templates read before cleaning | `{{birth date\|1847\|3\|3}}` cleaned to nothing, and the fact went with it |

That takes the vocabulary from 13,387 properties to **9,664**, and costs
5,103 facts — 0.26%, all of them keys with no letters in them, keys longer
than 64 characters, or fields that only existed because a comment ran into
the key after them.

### The templates were the expensive rule

`clean()` strips every `{{...}}`, contents and all, because a lead has to
survive an infobox that ran past the window we captured. Applied to a *value*
that is a deletion, and `templates.py` measures the bill: **301,306 values are
written as a template and 232,947 of them were dropped — 8.0% of every named
field.** Dates being 4% of values in an encyclopedia largely made of people is
the fingerprint of it.

So a value now gets one pass of expansion first, for templates whose meaning is
unambiguous — dates, `{{convert}}`, `{{url}}`, the list templates, and the two
that carry relations: `{{marriage}}` on `spouse`, and `{{flag}}` on
`subdivision_name`. Everything else still falls to `clean()`, because inventing
a reading is worse than dropping a field.

Two rules were removed again after measuring, which is the only reason they are
worth writing down:

| | |
|---|---|
| `{{small}}`, `{{big}}`, `{{nobold}}` | `successor = Osman Hussein {{small\|(Acting)}}` — reading the annotation turned a title that resolved into one that did not, and cost 308 edges |
| `{{flag}}` beside a value rather than as one | `birth_place = {{flagicon\|IRI}} Urmia` is an icon next to a place; reading it gave "IRI Urmia" and cost 155 more |

A flag is read only when it is the whole value. Position is the only thing that
says whether it is the country or a picture of one.

Some of the biggest losses cannot be repaired here at all: the 15,471
`{{france metadata wikidata}}` values hold no text, because the template
fetches a population from Wikidata at render time. Nothing is inside them to
expand, which makes them an argument for ingesting Wikidata rather than a rule
this table could carry.

The index split is the one that needed care. `subdivision_name2` is the second
subdivision; `area_km2` is square kilometres, and splitting it invents a field
called `area_km` that nobody wrote. Nothing in the name tells you which — only
whether the rest of the vocabulary agrees there is a series. Requiring more
than one index under a base separates 743 real series from 137 lone names, and
every one of those 137 ends in `km2`.

What cannot be normalized is recorded. The `property` table holds the
vocabulary with its counts and the relation `libgraph` reads it as, so *"what
is used often and mapped to nothing"* is a query rather than an afternoon with
a dump:

```sql
SELECT name, uses FROM property
 WHERE source = 'simplewiki' AND relation IS NULL
 ORDER BY uses DESC LIMIT 10;
```

The last property that query would have surfaced was `subdivision_name`, and
adding it took chaining from 1.7% to 40.7%.

They are here because answering *"where was Bell born"* is a **lookup**, not
reading. That is the difference between a search box and an oracle, and it is
the only route to one on this hardware: extracting an answer from prose is
comprehension, and out of reach.

The pieces, and where they stand — `oracle.py` puts them together:

| | mechanism | measured |
|---|---|---|
| "bell" → the article | the search index here | works |
| "where was … born" → `born_in` | the phrasebook classifier | **93.8% macro**, 9 relations |
| "what country was … born in" → a path | the same classifier | **~50%** on unseen phrasings |
| `(subject, property, ordinal)` → value | this table's primary key | a lookup |

The relation number is from [SimpleQuestions](https://github.com/askplatypus/wikidata-simplequestions)
mapped to Wikidata — real human-written questions. The path number is over
phrasings this repo wrote and then withheld from training, which is why it is
so much lower and why it is the one to believe.

It has no decimal place on purpose. Over five seeds the same measurement runs
43.1% to 56.2% on its 320 held-out questions — a spread of 13 points, on a set
where four more right in one class moves the figure by one and a quarter. Two
different values of it were quoted in this repository for months, 51.6% here
and 43.8% in `oracle.py`, and the interesting part is that neither was wrong:
both are ordinary draws from that spread, and 51.6% reproduces to the decimal
at seed 0. A number quoted more precisely than it can be measured invites
exactly that kind of disagreement, and resolving it by picking a side would
have been the wrong repair.

**What stops it being better is coverage, not accuracy.** Only 46% of articles
have an infobox, so a chain that hops onto one of the other 54% cannot
continue. The oracle is confidently right about where someone was born and
silent about a great deal else — and the design leans into that, reporting how
far a walk got rather than only that it failed.

### Where the chains were actually breaking

That paragraph was right about the cause and wrong about the remedy, which
`coverage.py` was written to settle. Of the birthplace climbs that failed,
**41.7% ran out of `located_in` edges and 2.2% hit the hop limit** — so the
graph was running out of road, not failing to recognise a country when it
arrived at one. The 193 entities it calls countries are very nearly the 195
there are.

The places it died on were not obscure: New York, Washington, Moscow,
Maryland, Michigan. `Infobox U.S. state` has no country field at all — the
template implies it — so there was nothing to map and nothing to normalize.
Michigan records that it is in the United States in exactly one place, which
is that it is filed under `1837 establishments in the United States`.

So `libgraph` reads categories too, for containment and only for containment,
with three guards that each exist because the rule without it produced
nonsense:

| guard | what it stops |
|---|---|
| the target must be a place | `Bands established in 2022` parses, and this encyclopedia has an article on 2022 |
| the subject must be a place | otherwise every band formed in California is filed inside it — 70,844 edges, of which 3,763 were about places |
| the subject must not be a person | `Presidents of France` parses exactly like `Cities in France` |

"A place" is not a judgement: it is anything already on a `located_in`,
`born_in`, `died_in` or `capital_is` edge. An infobox always outranks a
category, so this fills gaps and never replaces — which makes it monotonic,
and means no chain that completed before stops completing.

3,945 edges, and they are the ones that were being asked for. Taken with the
value templates and the rank fallback:

| | before | after |
|---|---:|---:|
| edges | 150,335 | **167,922** |
| `born_in in_country` | 45.7% | **76.9%** |
| — questions it answers | 19,238 | **32,518** |
| `died_in in_country` | 48.1% | **81.0%** |
| climbs that never reach a country | 43.8% | **23.1%** |

### Measuring the coverage, rather than remembering it

Every number in this section used to be measured by hand and quoted, which is
how `libgraph.py` came to claim that `birth_place -> country` completes for
40.7% of subjects with nothing to check it against. `coverage.py` prints them
from whatever database is in front of it:

```bash
python data/wikipedia/coverage.py                     # the table
python data/wikipedia/coverage.py --json > before.json
python data/wikipedia/coverage.py --baseline before.json   # with a delta column
```

It walks with `libgraph.follow` rather than joining, so it measures the
traversal the oracle actually ships, and it reports `startable`, `answered` and
`rate` separately **because they can move in opposite directions.** Letting
`libgraph.build` fall through to a lower-ranked field when the best one names
no article is worth 6,332 edges on an unchanged corpus, and on that change
`in_country` answered 781 more questions while its *rate* fell 1.6 points —
because everything that makes more subjects startable adds the ones that were
failing for a reason, and they complete at less than the existing average by
construction. A scoreboard of rates alone would have called it a regression.

It is also what caught two changes that looked right and were not. Reading
`{{small|(Acting)}}` turned resolvable titles into unresolvable ones and cost
308 edges; reading `{{flagicon|IRI}} Urmia` as a value gave "IRI Urmia" and
cost 155 more. Both were measured, rejected and pinned by tests, and both
would have shipped as improvements without a before and after.

## Redirects earn their place

Wikipedia's 114,771 redirects are indexed as alternate names scoring into their
target's slot: they cost card space and nothing in RAM. `jane austin` finds
Jane Austen **only** because a redirect says so — nothing here does fuzzy
matching, so every spelling that should work has to be one somebody wrote down.

97.6% of them resolve to an article in the corpus.

## Source

[Simple English Wikipedia](https://simple.wikipedia.org), CC BY-SA 4.0.
Dumps from <https://dumps.wikimedia.org/simplewiki/>; this was built against
the 2026-08-01 snapshot. The database is derived data and is not in git —
`ingest.py` rebuilds it from any snapshot in about four minutes.
