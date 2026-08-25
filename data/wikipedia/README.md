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

`data/simple_english_wikipedia.db` is **not in git** — it is 337MB of derived
data, and step 2 rebuilds it from any snapshot. The dump is not in git either.
What *is* committed is everything needed to turn one into the other.

The database records where its contents came from, so a card can be traced
back to a snapshot without asking anyone:

```console
$ python data/wikipedia/ingest.py --stats
  schema_version               6
  simplewiki.articles          283997
  simplewiki.digest            adf8cbb46aabe719
  simplewiki.dump              simplewiki-20260801-pages-articles.xml.bz2
  simplewiki.edges             150335
  simplewiki.facts             1945061
  simplewiki.ingested          2026-08-24T02:33:39
  simplewiki.redirects         114771
  simplewiki.url               https://dumps.wikimedia.org/simplewiki/20260801/...

  simplewiki: 283,997 articles, 114,771 redirects (97.6% resolve), 68 MB of lead
              1,945,061 facts over 129,725 subjects (46% of articles), 9,509 properties
              values: text 74%, number 22%, date 4%, url 0%
              47 properties map to a relation; biggest unmapped: name (95,626),
              subdivision_type (40,770), years (34,988), clubs (34,323), ...
```

That last line is the one to read. It is the corpus telling you what it knows
that nothing yet understands - and the biggest entries are a footballer's
career table, which is a fair summary of what Simple English Wikipedia is
mostly made of.

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

The database also carries **1,945,061 facts** pulled from infoboxes — an
infobox is a hand-curated set of typed key/value pairs, which is to say a set
of facts about its article. The lead throws them away as furniture; the `fact`
table keeps them.

```sql
SELECT value FROM fact WHERE subject = 'Alexander Graham Bell'
                         AND property = 'birth_place';
-- Edinburgh, Scotland
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

That takes the vocabulary from 13,387 properties to **9,509**, and costs
5,103 facts — 0.26%, all of them keys with no letters in them, keys longer
than 64 characters, or fields that only existed because a comment ran into
the key after them.

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
| "what country was … born in" → a path | the same classifier | **51.6%** on unseen phrasings |
| `(subject, property, ordinal)` → value | this table's primary key | a lookup |

The relation number is from [SimpleQuestions](https://github.com/askplatypus/wikidata-simplequestions)
mapped to Wikidata — real human-written questions. The path number is over
phrasings this repo wrote and then withheld from training, which is why it is
so much lower and why it is the one to believe.

**What stops it being better is coverage, not accuracy.** Only 46% of articles
have an infobox, so a chain that hops onto one of the other 54% cannot
continue. The oracle is confidently right about where someone was born and
silent about a great deal else — and the design leans into that, reporting how
far a walk got rather than only that it failed.

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
