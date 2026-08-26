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
| `WIKI.IDX` | 23.1 MB |
| `WIKI.DAT` | 51.7 MB |
| `WIKI.GRF` | 2.4 MB — 167,868 edges |
| `WIKI.bin` | 94.0 KB |

`data/simple_english_wikipedia.db` is **not in git** — it is ~500MB of derived
data, and step 2 rebuilds it from any snapshot. The dump is not in git either.
What *is* committed is everything needed to turn one into the other.

The database records where its contents came from, so a card can be traced
back to a snapshot without asking anyone:

```console
$ python data/wikipedia/ingest.py --stats
  schema_version               8
  simplewiki.articles          283997
  simplewiki.digest            adf8cbb46aabe719
  simplewiki.dump              simplewiki-20260801-pages-articles.xml.bz2
  simplewiki.edges             167868
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
| `WIKI.bin` | 5.6 KB — the program |
| `WIKI.IDX` | 23.1 MB — hashed dictionary and postings |
| `WIKI.DAT` | 51.7 MB — titles and leads, byte-pair packed |

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
`tools/probe_entities.py` scores twenty question probes against a built card:
the right article is **first 85% of the time and in the top three every time**.

First is the number that matters. A search engine is judged on the top three
because a person reads all of them; an oracle walks only the first, and its
mistakes have no symptom — the graph answers correctly about the wrong subject
and what comes back is fluent and wrong.

An initial is glued to the name after it — `amanda m wilson` is indexed and
queried as `amanda mwilson` — because a single character was dropped at both
ends, which made two people who differ only by a middle initial into the *same
query* rather than two similar ones. That is worth nothing here, where nobody
is searched for by initial, and the probes are unchanged by it. It is worth
88.6% → 100% on [`data/silo/`](../silo/), where the population is closed and
thousands of people share a first and last name. The cost is 3% more postings;
`a` and `i` are exempt because they are words, and gluing one ate `black` out
of "what is a black hole".

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
| program | 7,584 bytes |

The two passes over the accumulator — clearing it and scanning it for the best
three — used to dominate every query at 284,000 bytes apiece, whatever the
query, for 5.4 M instructions a query however rare the word. The accumulator is
now tiered: one flag per 256-article page, set when a posting lands, and both
passes visit only flagged pages.

`benchwiki.py` runs the real card in the emulator and counts what a query
costs, which is how the numbers below stopped being provisional. They were
measured on the full 283,997-article card, not a synthetic one:

| query | instructions | card bytes | s @ 18.432 MHz | finds |
|---|---:|---:|---:|---|
| `z80` | 63,011 | 6,221 | 0.00 | Z80 |
| `zilog z80` | 66,533 | 6,275 | 0.00 | Z80 |
| `everest` | 206,753 | 6,353 | 0.01 | Mount Everest |
| `jane austen` | 1,279,822 | 7,610 | 0.07 | Jane Austen |
| `mount` | 1,796,552 | 8,220 | 0.10 | Mount Everest |
| `mount everest` | 1,827,941 | 8,411 | 0.10 | Mount Everest |
| `world war` | 4,903,859 | 41,475 | 0.27 | World War I |
| `the united states of america` | 7,577,578 | 126,202 | 0.41 | United States |
| `the` | 42,640 | 90 | 0.00 | *nothing* |

**What sets the cost is the commonest word in the query, not how many words it
has.** `mount everest` costs what `mount` costs on its own — the rare word is
nearly free and the common one flags most of the page table either way. That is
a better description than the one this file used to give, which counted terms:
`zilog z80` and `mount everest` are both two-term entity lookups and one is
27 times dearer than the other.

So the tiering is worth **86x on a query of rare words** and nothing at all on a
common one, where 6.4 M against the old 5.4 M is the whole-corpus scan it always
paid plus the page table's overhead. Both cases are in the table above because
the second is the one the design does not help.

The last row is the cheap kind of failure: a word that common is not in the
dictionary at all, so the query is refused at the index for 90 bytes rather than
scored against the corpus.

### The postings are gaps, not document ids

Which is where the index's other 10 MB went. A posting used to be a flat
three-byte document id and a one-byte weight, and three quarters of the file
was postings — the hashed dictionary everyone assumes is the bloated part is an
eighth of it.

Document ids ascend within a term and cluster, because a term's articles are
related: **65.8% of the gaps between them fit in one byte and 33.1% in two.**
So a posting stores the gap from the one before, and the width rides in bits 5
and 6 of the weight byte — a weight is five bits and those two were spare, so
the tag costs nothing.

    index          33,083,799 -> 23,136,084 bytes    -30.1%
    whole card    107,637,689 -> 97,690,103          -9.2%
    program             7,455 -> 7,584               +129

Decoding is a running add and a compare, and the add is the operation this
design has always been willing to pay for — the accumulator that scores a query
is a byte and an `add` too. It is not free: a query retires up to 18% more
instructions and reads up to 49% fewer bytes, and on this machine the card is
slower than the processor, so the trade is worth taking.

| query | instructions | card bytes | total at 18.432 MHz and 250 KB/s |
|---|---|---|---|
| `z80` | 62,906 → 63,011 | 6,226 → 6,221 | unchanged |
| `mount everest` | 1,809,585 → 1,827,941 | 9,806 → 8,411 | 0.14 s → 0.13 s |
| `world war` | 4,571,035 → 4,903,859 | 76,430 → 41,475 | 0.55 s → **0.43 s** |
| `the united states of america` | 6,444,411 → 7,577,578 | 245,518 → 126,202 | 1.31 s → **0.90 s** |

The queries that were cheap stay cheap and the ones that were slow get faster,
which is the shape you want: the extra 0.06 s of decoding on the last row buys
back 0.47 s of reading.

A card in the old layout is refused rather than misread — the magic went from
`ZWIKI1` to `ZWIKI2` — because every byte in it is still a plausible posting
and the failure would otherwise be wrong answers rather than an error.

### The text is byte pairs, and the codes were already lying around

With the index down to 23 MB, `WIKI.DAT` was **76% of the card** — 98.5% of it
title and lead text, averaging 258 bytes an article. It is English prose, so it
compresses; the question was only what a Z80 can afford to decompress.

Two measurements decided the format. **0.69% of the bytes are non-ASCII, and 49
byte values never occur in the corpus at all.** Forty-nine unused values is
forty-nine codes available for free — no escape byte, no shift state, no
reserved range stolen from real characters. So a byte is either itself or it
stands for a short string, and nothing in the file has to say which.

Decoding is a table lookup and a block move:

    read a byte -> index PAIRTAB + 3 * byte -> length 0? emit it
                                            -> otherwise copy `length` from BLOBBUF + offset

No arithmetic, no shifts, no recursion — the pairs are learned by BPE, so a
code can stand for a merge of merges, but the table stores every expansion
**flattened**, so one lookup finishes the job. The delta postings above at
least needed an add; this needs nothing.

    WIKI.DAT       74,546,435 -> 51,730,093 bytes    -30.6%
    whole card     97,690,103 -> 74,871,954          -23.4%
    program             7,584 -> 5,777               -1,807

The program got *smaller* while gaining a decoder, because the buffers moved
out of the image — see **Memory** in `buildwikibin.py`. A `ds` reserves space by
emitting that many zeros into the .bin, and four buffers declared that way had
quietly added 11 KB to a file whose whole point is being small.

The table is read off the card at startup rather than compiled in, which keeps
the property the rest of the build depends on: **a rebuilt card drops in beside
an unchanged `WIKI.bin` unless the format itself changed.** A corpus with
different common digraphs simply ships a different table.

That read is 892 bytes — the 6-byte magic, the length, a 768-byte slot table
and a 116-byte blob — and the benchmark charges exactly that, on every query:

| query | instructions | card bytes |
|---|---|---|
| `zilog z80` | 66,533 → 86,414 | 6,275 → 7,167 |
| `jane austen` | 1,279,822 → 1,303,649 | 7,610 → 8,502 |
| `mount everest` | 1,827,941 → 1,853,192 | 8,411 → 9,303 |
| `world war` | 4,903,859 → 4,916,382 | 41,475 → 42,367 |
| `the united states of america` | 7,577,578 → 7,600,796 | 126,202 → 127,094 |

**+892 bytes every time, and between 12,523 and 25,251 instructions.** The
bytes are flat because the table is the only extra read — the leads themselves
got *smaller*, which is why a 30% smaller file costs nothing to read. The
instructions vary with how much lead text the top three results unpack, and the
worst of them is 1.4 ms; with the read, about 5 ms once per session, against
0.41 s for the query it rides along with.

22.9 MB off the card, then, for a cost that does not scale with anything. Note
that it is the cheap query that pays most in relative terms — `zilog z80` grew
by a third — because there is nothing else happening to hide it.

### One pair the corpus offers and the format must refuse

No code may stand for a pair containing a NUL. NUL ends the title and ends the
lead, and the device counts those two to know when to stop — but it copies an
expansion with a block move it does not inspect. A NUL inside an expansion is
one the device never sees, so it reads on into the article that follows.

This is not hypothetical, and it is not rare. Most leads end in a full stop, so
`.\0` is one of the commonest pairs in the file, and it was duly learned: **the
first full card built this way had exactly one bad code, 248, and every article
whose lead ended in a period ran past its own terminator.** The card still
printed the right answer, because printing stops at a NUL for its own reasons,
which is precisely what made it worth pinning with a test rather than an eye.

Excluding NUL when the pairs are learned is enough for every later merge too, a
code expanding to a NUL only if a pair behind it held one. It costs 117,802
bytes — 0.23% — and `write_text` refuses to write a table that breaks the rule
rather than trusting that it held.

`ZWDAT1` is the first magic this file has carried; before it, the text file
began with its article count.

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

### Some of that spread is the name, not the phrasing

```console
$ python tools/name_sensitivity.py --model relations.npz
   97.7%  of 1,280 questions route to the right path
   62.5%  of phrasings (20/32) answer the same way whatever the subject
```

Accuracy over a question set cannot separate a phrasing the model never learned
from a phrasing it *did* learn whose answer depends on who is being asked
about. The second is invisible in an accuracy figure, and it is here: hold one
of the four chain phrasings fixed, vary only the entity, and **twelve of the
thirty-two change their answer.**

That is the encoder doing what it is documented to do. A query is hashed into
128 trigram buckets, a name is most of a short question, and the subject is
therefore most of the input — it is not something the model steps over on its
way to the verb.

The confusions are not random, which is what makes them worth reporting:

| | | |
|---|---|---|
| `what country is X in` | 77.5% | falls to `located_in` |
| `what country is X located in` | 85.0% | falls to `located_in` |
| `what country was X born in` | 95.0% | falls to `born_in` |

Every one of them drops the climb and answers with the step before it. That is
the shape of the bug this corpus already fixed twice — a region returned where
a country was asked for — arriving by a different route, and it means the
remaining cases cannot all be repaired in `libgraph`.

The measurement came from the synthetic corpus in [`data/silo/`](../silo/),
where the same tool reports 124 of 240. That is a worse figure, and it should
be: those names come from a pool of a few hundred. The point of running it here
was to find out whether the effect belongs to that corpus or to the encoder,
and it belongs to the encoder.

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
arrived at one. It called 193 entities countries at the time, against the 195
there are — a resemblance that turned out to be a coincidence covering for
about fifty bad entries in both directions. See below.

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
| edges | 150,335 | **167,868** |
| `born_in in_country` | 45.7% | **77.7%** |
| — questions it answers | 19,238 | **32,842** |
| `died_in in_country` | 48.1% | **82.3%** |
| climbs that never reach a country | 43.8% | **22.3%** |

### Asking for a country and being told California

The rule above — a claimed country sitting inside another claimed country is
not a country — was right, landed in #40, and **did not cover its own examples**.
It was still answering Chicago 445 times and California 555.

Two faults, and they had to be fixed together because either one alone makes
things worse.

**The demotion ran before the categories were read.** California is placed
inside the United States by its categories, not by its infobox, so when the
types were settled it was contained by nothing and kept its claim. The
ordering — types before categories — is deliberate and documented above,
because `from_categories` needs to know what a country is to prefer Denmark
over Europe. So the types are now settled **twice**: once for that, and again
once the containment is complete. Only demotions can change on the second pass,
since a category never says `country = X`.

**Dropping the contained one is the wrong rule.** Three infoboxes say
`country = Asia`, which is exactly `TYPE_FLOOR` — so Asia was a country, and
Japan, China, Iran and forty more became things "inside another country". Fixing
the ordering alone would have demoted every one of them. Antarctica and the
Caribbean clear the floor the same way; Europe does not, which is why the France
example worked and hid all of it.

So the rule now keeps whichever of the two the corpus calls a country **more
often**, and the counts are not close:

```
Asia            3   against  Japan 257, China 74, Iran 35
United States 4155  against  California 16, Chicago 5, Massachusetts 4
Canada        283   against  Ontario 3
```

| | before | after |
|---|---:|---:|
| entities typed a country | 193 | **143** |
| — of those, inside another country | 44 | **0** |
| `in_country` | 76.2% | **86.6%** |
| — questions it answers | 37,888 | **43,137** |
| climbs that never reach a country | 23.8% | **13.4%** |
| birthplace climbs landing on the United States | 6,786 | **10,600** |

That last row is the point. Those 3,814 were not failures — they were answers,
and the answer was "California". **A missing answer is silent and a wrong one
is fluent**, which is the failure this whole file is arranged against.

Greenland and Hong Kong are demoted by the same rule and I am content with
that. Antarctica and the Caribbean survive, containing nothing that
contradicts them — nothing is filed inside either, so nothing outvotes them.
No birthplace climbs to Antarctica and twelve climb to the Caribbean, which is
the same fault as California at a five-hundredth of the size.

### The one chain that looked broken was the scoreboard

`created_by born_in` sat at **51.7%** while every comparable chain was near
77%, which made it the obvious next thing to fix. It was not a coverage
problem, and two plausible repairs died on measurement before the real cause
turned up.

**Mapping `nationality` would have made it worse.** It is the largest unmapped
property that looks like a relation — 10,936 uses, and `coverage.py` scores
86.5% of its values as naming an article. But only 19 of the blocked creators
carry one, and the values are demonyms: `American` *does* name an article, just
not the United States. That column checks a title exists, not that it is the
right kind of thing, and it flatters every demonym-shaped property the same
way.

**The creators are not people.** Of the 1,558 creators with an article and no
birthplace, 692 are companies and groups accounting for 3,318 of the works the
chain cannot finish. The biggest single blockers are Microsoft, Apple Inc.,
Google, U2, The Beatles and Capcom. A band has no birthplace to be missing —
the chain was being marked down for declining to say where ABBA was born.

So a walk that stops at `born_in`, `died_in` or `spouse_of` on something the
corpus does not call a person is now counted as **moot** rather than as a miss,
and `rate` is over what is left:

| | before | after |
|---|---:|---:|
| `created_by born_in` | 51.7% | **74.3%** |
| — walks with no possible answer | counted as misses | **3,370** |
| — stalls that are real gaps | 5,346 | **1,976** |

Nothing about the graph changed and no answer changed: the oracle already
declined these and fell back to listing articles. The only thing that was wrong
was the number, and it was wrong in the direction that invites work on the
wrong problem.

Who counts as a person comes from birth and death dates in the infobox, plus
the `1935 births` / `Living people` categories Wikipedia files people under
almost without exception — 78,594 of them. That tail is kept deliberately
tight; `Deaths from cancer` is not a birth-year category and the articles in it
are diseases.

### Reading the birthplaces the infoboxes never had

78,594 people are in this corpus and 42,288 have a birthplace. The other
**36,191 have a lead and no birthplace** — and Wikipedia's house style puts the
birth in the first clause, so a great many of those leads say it anyway.

`birthplaces.py` reads them. It is the only part of this pipeline that reads a
sentence rather than a table, which is exactly why it is the only part with an
evaluation harness attached to it.

**Nothing it produces reaches `fact`.** Rows land in `derived`, keyed by the
method that wrote them, so a reader wanting only what the encyclopedia
tabulated reads `fact` and never sees them. `method` is part of the primary key
rather than a column beside it, so two extractors can disagree about the same
person and both be kept — which is what makes one measurable against the other.
The graph build ignores `derived`, so the card is unchanged.

**It is scored before it is trusted**, and the ground truth is free: the 42,288
people who *do* have a birthplace also have leads, and their infoboxes say the
answer.

| | regex |
|---|---:|
| the lead said something | 20.4% |
| and it named an article | 18.3% |
| and it matched the infobox exactly | **13.8%** |
| and it climbs to the same country | **93.4%** |

That gap is the finding. **Agreement is the obvious measure and it is the wrong
one**, because the infobox and the lead routinely name *different places that
are both true* — College Park against Georgia, Ontario against Canada, Brooklyn
against New York City, Whitby against Toronto. Marking those wrong measures
granularity, not correctness. The oracle is asked which *country*, so what
matters is whether the two climb to the same one, and 93.4% is where that lands.

93.4% is itself a floor. Several of the 352 disagreements are the climb's own
typing rather than the extraction: `Ontario` climbs to Ontario and `California`
to California, both being called countries by enough infoboxes to clear
`TYPE_FLOOR`. That is a separate bug and it is not this file's.

**The eval set and the target set are not the same population, in the useful
direction.** The regex resolves 18.3% of people who have an infobox birthplace
and **37.5% of people who do not** — because a person missing one frequently
has no infobox at all, and the opening sentence states the birth regardless. So
the yield does not transfer, and the figure to quote is the one measured on the
population it ran on:

```
13,585 rows over 36,191 people    37.5%
```

Which is the answer to *"could a model fill in the missing facts"*: **measure
the trivial thing first.** A regex over the leads recovers 13,585 birthplaces
for nothing, and any model has to beat that number, on that population, before
it is worth its cost. `--method` is where one would go; the harness that would
score it is already here.

#### Putting them on the card is a separate decision

`libgraph.build` ignores `derived` unless given a method, and `birthplaces.py`
writes the table without touching the graph. Getting these onto a device takes
a third command, which exists so that somebody takes the decision rather than
inheriting it:

```bash
python data/wikipedia/birthplaces.py --write --rebuild-graph
```

Everything else on the card comes from something a Wikipedia author tabulated
or filed. These come from a sentence, read by a regex, and a card built with
them asserts things no infobox states. They fill gaps only — written after
every other edge and skipping any subject that already has the relation, so a
sentence cannot overrule a table even if the table is stale.

What it buys, measured:

| | tabulated only | with the leads read |
|---|---:|---:|
| edges | 167,868 | **181,453** |
| subjects on the graph | 37.3% | **42.0%** |
| `born_in in_country` startable | 42,288 | **55,873** |
| — questions it answers | 32,842 | **42,996** |
| — rate | 77.7% | 77.0% |
| `created_by born_in` | 74.3% | **83.1%** |

**+10,154 answered birthplace questions**, and the rate barely moves. That last
part is the interesting one: newly startable subjects complete below the
existing average by construction, so a fall of 0.7 points across 13,585 new
subjects says the derived edges chain about as well as the tabulated ones. It
is not evidence that they are *right* — `--write`'s own scoring is what speaks
to that — but a bad batch would have shown up here as a much steeper fall.

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
