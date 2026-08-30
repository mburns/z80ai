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

#    Optionally, the Wikidata id of each article (~40MB, ~15 seconds). Not
#    needed for a card; needed for anything that wants to join this corpus to
#    Wikidata. Same date as the XML dump, or it is refused.
curl -O https://dumps.wikimedia.org/simplewiki/20260801/simplewiki-20260801-page.sql.gz
curl -O https://dumps.wikimedia.org/simplewiki/20260801/simplewiki-20260801-page_props.sql.gz
python data/wikipedia/ingest.py --sitelinks \
       simplewiki-20260801-page.sql.gz simplewiki-20260801-page_props.sql.gz

# 3. Into a card (~4 minutes). Add --limit 20000 for a small one that builds
#    in seconds, ranked by how many redirects point at each article.
python buildwikisearch.py --out dist/WIKI
```

For a card that answers questions rather than only finding articles, read the
birthplaces out of the lead text, then train the relation classifier and pass
it to step 3:

```bash
# 2a. Birthplaces for the 36,191 people whose infobox has none (~2 minutes).
#     `--rebuild-graph` is what puts them on the card; see below.
python data/wikipedia/birthplaces.py --write --rebuild-graph

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

**Step 2a is the one step here that puts something read out of prose on the
device**, and it is worth 10,154 more answered birthplace questions — the
reasoning, the scoring against ground truth, and what it costs are all
[below](#reading-the-birthplaces-the-infoboxes-never-had). Leave it out and the
card carries only what a Wikipedia author tabulated or filed; everything else
works the same either way.

**Re-run 2a after any re-ingest.** `ingest.py` rebuilds the graph from the
facts, which drops the derived edges — the `derived` table survives, so 2a is
cheap the second time, but the card is back to tabulated-only until you do.

| | full corpus |
|---|---|
| `WIKI.IDX` | 23.3 MB |
| `WIKI.DAT` | 51.7 MB |
| `WIKI.GRF` | 2.5 MB — 181,453 edges |
| `WIKI.bin` | 94.0 KB |

`data/simple_english_wikipedia.db` is **not in git** — it is ~500MB of derived
data, and step 2 rebuilds it from any snapshot. The dump is not in git either.
What *is* committed is everything needed to turn one into the other.

The database records where its contents came from, so a card can be traced
back to a snapshot without asking anyone:

```console
$ python data/wikipedia/ingest.py --stats
  schema_version               10
  simplewiki.articles          283997
  simplewiki.digest            adf8cbb46aabe719
  simplewiki.dump              simplewiki-20260801-pages-articles.xml.bz2
  simplewiki.edges             181453
  simplewiki.facts             1995246
  simplewiki.ingested          2026-08-26T02:01:54
  simplewiki.redirects         114771
  simplewiki.sitelinks         283865
  simplewiki.sitelinks.dump    simplewiki-20260801-page_props.sql.gz
  simplewiki.url               https://dumps.wikimedia.org/simplewiki/20260801/...

  simplewiki: 283,997 articles, 114,771 redirects (97.6% resolve), 68 MB of lead
              283,861 sitelinks (100.0% of articles), 4 name no article
              980,928 category filings over 272,022 articles (96%), 76,102 categories
              1,995,246 facts over 129,504 subjects (46% of articles), 9,633 properties
              values: text 70%, number 21%, date 7%, url 1%
              47 properties map to a relation; biggest unmapped: birth_date (44,871),
              subdivision_type (40,782), years (34,994), clubs (34,323), ...
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
| `WIKI.bin` | 5.7 KB — the program |
| `WIKI.IDX` | 23.3 MB — hashed dictionary and postings |
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

### Twenty probes were not enough to notice

They still read 85%. What they could not see is that **less than half of all
articles were found first by their own exact title** — 47.8%, on the shipped
card, for months.

`libsearch.FAME` boosts an article by how many redirects point at it, and was
set to 1.0 by a measurement over those same twenty probes, with a note saying
it had never been swept. Swept now, on the whole corpus, against probe sets
built from the corpus itself — `--sample N`, every article asked for by its own
name and every redirect asked for by itself:

| FAME | by title | by redirect | the twenty |
|---:|---:|---:|---:|
| 0.0 | 93.9% | 88.1% | 11/20 |
| **0.25** | 87.8% | **91.7%** | 17/20 |
| 0.5 | 73.3% | 89.7% | 17/20 |
| 1.0 *(was)* | **47.8%** | 85.1% | 17/20 |

0.25 beats the old value by forty points by title, by six on redirects — a set
biased *toward* fame, since redirect count is what the knob scores — and ties
it on the twenty. It dominates, so there is no trade to weigh, and it is now
the default.

The useful half of this is not the number. **The twenty probes were assembled
from the misses of one particular failure** — a derived article beating the
thing it derives from — so they are blind by construction to any failure its
repair introduces, and they approved of one that broke half the corpus. A probe
set built from a bug's symptoms will endorse any fix for that bug.

It also got worse as the corpus grew: 77.7% by title at 40,000 articles, 53.3%
at 120,000, 47.8% at 283,997. A card built with `--limit` was hurt least, which
is exactly backwards from where anyone would notice.

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
| program | 5,817 bytes, of which 1,110 is that page table |
| the most it could be | **502,016 articles**, and this card is 57% of it |

That last row was a guess until it was measured. `buildwikisearch.py` warned
above 380 KB of accumulator "because that leaves under 130 KB of Agon SRAM for
the program", and the program is 4.7 KB — the allowance had been sized against
the *oracle* binary, which carries a classifier. `buildwikibin.build` takes an
article count and no corpus, so the real boundary costs milliseconds to find,
and it is 29% higher than the warning nobody had ever hit.

A page of 256 articles costs 257 bytes rather than 256, and that is the part an
estimate drops: both bases round down to a 256-byte boundary, so the buffers
below the accumulator fall by a whole page for each page of articles added,
while the page table in the image rises by one byte for the same page.
`max_docs` solves that; `tests/test_wikisearch.py` bisects `build` to check it,
because two implementations are the only way to notice.

The trade it prices is what an oracle card costs in articles. Every byte of
image is a byte the accumulator cannot have, so the silo's two classifier
widths — 94.4 KB and 38.9 KB — were also a choice between 55,000 articles.

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

The database also carries **1,995,246 facts** pulled from infoboxes — an
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
folksonomy: before any cleaning, 19,117 distinct property names as typed, 5,266 of them
used exactly once. The ingest normalizes **form** and leaves **meaning** to
`libgraph`, and the line matters — form is mechanical and decidable from the
data, while *"`subdivision_name` and `country` are the same question"* is a
judgement about this corpus that belongs where the judgement is made.

| | |
|---|---|
| comments stripped before splitting | a comment between two pipes ran into the next key |
| keys cleaned and shape-checked | only values went through the cleaner before |
| `subdivision_name1..7` → `(property, ordinal)` | **25.0% of facts** were positional variants |
| values typed `text`/`number`/`date`/`url` | 22% are numbers, which sorted lexically as text |
| known value templates read before cleaning | `{{birth date\|1847\|3\|3}}` cleaned to nothing, and the fact went with it |

That takes the vocabulary to 13,268 names and then to **9,633**, and costs
3,500 facts — 0.18%, all of them keys with no letters in them, keys longer
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
| "what country was … born in" → a path | the same classifier | **84.0%** on unseen phrasings |
| `(subject, property, ordinal)` → value | this table's primary key | a lookup |

The relation number is from [SimpleQuestions](https://github.com/askplatypus/wikidata-simplequestions)
mapped to Wikidata — real human-written questions. The path number is over
phrasings this repo wrote and then withheld from training, which is why it is
the one to believe.

### It was ~50%, and the fix was writing more questions

Not a better model. Every repair tried on this number for months was to the
encoder or the architecture — class weighting, order-sensitive bands, entity
masking, two heads — and this note used to conclude that "sweeping the number
of training phrasings from one to six moved the score 20.3% → 39.4%, so writing
more of them would have bought a few points at best."

Nineteen points over five wordings is not a few points. That sentence was
reading a steep curve as a flat one, and the same curve on
[`data/silo/`](../silo/) was still climbing at nine. So eight more wordings per
path were written. The held-out three are unchanged, so every row is scored
against the same 480 questions:

| phrasings | held out | one-hop macro |
|---:|---:|---:|
| 1 | 35.8% | |
| 3 | 51.8% | |
| 5 *(what shipped)* | 59.2% | 92.1% |
| 8 | 78.3% | |
| **13** | **84.0%** | 90.8% |

It costs **1.3 points of one-hop macro**, which makes it a trade rather than a
free win — and a cheap one by the bar the rejected repairs set, since bands
cost 7.6 points *for a loss*.

Do not compare 84.0% against the old figure. That measurement held out two
phrasings of eight and this holds out three of sixteen; they are answers to
different questions. What the old figure is still good for is its warning about
precision: over five seeds it ran 43.1% to 56.2%, a spread of 13 points, and
two different draws from it — 51.6% and 43.8% — sat in two files for months
looking like a disagreement. Neither was wrong. A number quoted more precisely
than it can be measured invites exactly that.

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
value templates and the rank fallback — on the graph the facts and categories
support, before step 2a reads any lead text:

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

#### Putting them on the card is a decision, and it has been taken

**The oracle build runs this** — it is step 2a above:

```bash
python data/wikipedia/birthplaces.py --write --rebuild-graph
```

It is still two flags rather than a default, and `libgraph.build` still ignores
`derived` unless given a method, because the distinction is worth keeping
legible: everything else on the card comes from something a Wikipedia author
tabulated or filed, and these come from a sentence read by a regex. A card
built this way asserts things no infobox states. Dropping 2a gives a card that
does not, and nothing else changes.

They fill gaps only — written after every other edge and skipping any subject
that already has the relation, so a sentence cannot overrule a table even if
the table is stale.

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

### The country list has junk in it, and no rule finds it

`Sydney` climbs to **England**. Its infobox says it is in "Cumberland", which
resolves to the English one, and `Grafton, New South Wales` reaches the United
States by way of "Clarence" the same way.

Chasing that turned up the smaller and sharper version: four entities on the
country list are not countries.

```
Baku        36 climbs land here    the capital of Azerbaijan
Victoria    82                     a state, and a capital of Seychelles
CA           1                     a country code
World        0
```

`demote` cannot reach them. It compares two claimed countries in a
containment, and nothing contains Baku — so nothing contradicts it. Three
rules were tried:

| rule | what it did |
|---|---|
| demote anything named as a capital | took **China, Angola and Mongolia** with it — their `capital` fields resolve to the modern country, so `Dzungar Khanate capital_is China` is an edge |
| keep whichever claim is stronger | the corpus calls Baku a country **6** times and a capital **2**, so Baku stays |
| a threshold sparing China and catching Baku | fitted to two data points, which is not a rule |

So it is not fixed. The four cost **118 answers out of 42,996** — 0.27% — and
a rule that risks China to recover that is a bad trade even in the version
where it works.

What is shipped instead is the list, because a person reading 143 names finds
`World` and `CA` in seconds where no curve does:

```bash
python data/wikipedia/coverage.py --countries
```

The floor curve above says how *many* countries a setting yields. This says
which, and that turns out to be the reviewable part.

### Three things that turned out not to be wrong

`created_by born_in` was being marked down for questions with no answer, so the
obvious next move was to look for the same fault everywhere else. It is not
there. Recorded here because a negative result nobody wrote down gets
re-investigated, and each of these cost an afternoon.

**The other chains are clean.** Of the walks that stop short, 98–99% stop on a
genuine place — something the corpus places, or places things inside — and
under 1% on a person:

```
born_in in_country    a place nothing places 99%   a person 1%
died_in in_country    a place nothing places 99%   a person 1%
in_country            a place nothing places 98%   a place that records where it is 2%
created_by born_in    a person 100%
```

Their shortfall is real missing containment, not category error. That last row
is the fix from the previous section seen from the other side: everything left
in `created_by born_in` is a person with no recorded birthplace, which is a gap
worth filing rather than a question worth declining.

That table is now printed by `coverage.py` on every run rather than worked out
by hand, which is the whole argument of this file applied to itself.

**The 93 creators with no categories are mostly bands** — Bon Jovi, The Police,
One Direction, Panic! at the Disco. `people` already excludes them, since they
carry neither a birth date nor a birth-year category, so they were being
counted moot all along and no change was needed.

**Disambiguation pages are not the problem they look like.** Several of those
93 turned out to be list pages: `created_by` pointing at *Drake* reaches "Drake
may mean:" and 27 works stop there. Roughly 6% of all edges land on a page like
that, which sounds alarming until you ask the question that matters — how often
a walk carries *on* through one and answers with whatever the list was filed
under. **Fourteen times, out of 32,842**, and all fourteen are right: Oshawa to
Canada, Thimphu to Bhutan, Eskişehir to Turkey. A list page almost never has a
`located_in` of its own, so it blocks a chain rather than misdirecting one.

Worth adding that the 6% is unreliable in its own right. Detecting these from
the lead catches `Cereal usually refers to a type of grass` and `A combination
puzzle can be solved`, and tightening the pattern to only `may refer to` and
`may mean` trades those for different false positives. The corpus files just 24
pages under a disambiguation category, so there is nothing to check the
detector against. A number that cannot be validated is not worth acting on, and
the harm it would be measuring is fourteen correct answers.

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

## The sitelink is the only exact key to Wikidata

Everything above reads facts out of an encyclopedia written for people, and the
ceiling on it is coverage: 46% of articles carry an infobox, the other 54% say
what they say in prose or in a category or not at all. Wikidata has the same
facts as a table, which makes "join this corpus to Wikidata" the obvious next
move and the identity of each article the thing that decides whether it works.

**The XML dump does not carry that identity.** It says what an article
contains and never says what it is *about*. `page_props` does, as
`wikibase_item`, keyed by page id — so it takes `page.sql.gz` too, to say which
title a page id is. Both are small next to the 340MB of markup:

```console
$ python data/wikipedia/ingest.py --sitelinks \
      simplewiki-20260801-page.sql.gz simplewiki-20260801-page_props.sql.gz

reading sitelinks from simplewiki-20260801-page_props.sql.gz
283,865 sitelinks, 283,861 of them joined to an article (100.0% of the corpus)
```

That is what it printed on the first run, and **the 728 that joined to nothing
were an ampersand.** Finding them is the only reason the orphan count is
printed at all.

### The titles were escaped, and nothing else could have said so

`page.sql.gz` says `Dungeons & Dragons`. The XML dump says
`Dungeons &amp; Dragons`, because that is what XML does to an ampersand, and
`article` stored what the XML said. `clean()` resolves entities in the *lead* —
to a fixed point, with a comment about `&amp;amp;` — and a title never went
through it, so **726 articles were called `AT&amp;T` and
`&quot;Weird Al&quot; Yankovic`** in the database, on the card and in every
table keyed on a title.

The fix is one call in `raw_pages`, at the single place a title enters the
program, and it decodes **once** rather than to a fixed point. That difference
is the whole reason it cannot reuse `clean()`: an article whose name literally
contains `&lt;` arrives written `&amp;lt;`, and a second pass would turn it
into `<` and invent a title nobody wrote. Only the five entities an XML escaper
emits are decoded, so a title containing the text `&ndash;` keeps it.

The redirect target is decoded with it. It has to be — 114,771 alternate names
resolve by matching a target against a title, and fixing one side alone would
have broken every redirect pointing at one of those 726 pages.

| | before | after |
|---|---:|---:|
| titles holding `&amp;` or `&quot;` | 726 | **0** |
| sitelinks joined to an article | 283,137 | **283,861** |
| sitelinks naming no article | 728 | **4** |
| edges, tabulated only | 167,868 | **168,306** |
| facts | 1,995,435 | **1,995,246** |

Both sides of the edge row are a plain `ingest.py` run, before the derived
birthplaces of step 2a are written — which is why it is 167,868 and not the
181,453 above. Comparing a tabulated-only build against one that had 2a run on
it would have credited this change with 13,585 edges somebody else wrote.

**Two of those rows are not the sitelinks.** The graph gained 438 edges,
because a fact's *value* was already decoded by `clean()` and could never match
the escaped title it named — `Lilo & Stitch` as a value had no article called
`Lilo & Stitch` to resolve to. And 189 facts went away, all of them from
`name` (153), `title` (22) and the other fields that repeat the article's own
name: the rule that drops a fact saying a thing is called what it is called
could not fire while the two spellings differed. Both numbers moved because
the same mismatch was costing them, which is the argument that this was one bug
and not three.

`SCHEMA_VERSION` goes to 10 for it. The table definitions are unchanged, but a
database written by 9 holds different strings under the same column, and a
version that only tracks columns would let a stale corpus pass for a current
one.

**Four orphans survive**, and they are honest: `Amaury Vassili`,
`Honey Come Back (song)` and two more exist in `page.sql.gz` and not in the XML
dump, which is skew between two jobs of the same day's dump. 136 articles still
have no sitelink and never will — `Czechia`, `Kingdom of Prussia`, pages with
no Wikidata item.

Escaping the sitelink titles to match the broken ones would have made the
coverage read 100% on the first run, and left the encyclopedia calling it
AT&amp;T with nothing anywhere to say otherwise.

**Matching on the title instead does not work**, and it is worth being precise
about how it fails, because it does not fail loudly. Against the same corpus,
a casefolded exact match of the article title to the English Wikidata label:

| | titles | |
|---|---:|---:|
| unambiguously right | 123,445 | 43.5% |
| no match at all | 84,545 | 29.8% |
| ambiguous, right one somewhere in the pile | 69,443 | 24.5% |
| unique and wrong | 2,806 | 1.0% |
| ambiguous and all wrong | 3,550 | 1.3% |

91.6M Wikidata entities share 83.5M labels. `Paris` is 236 of them, `Hamlet`
184, `California` 171; even `Barack Obama` is 4. The 29.8% that match nothing
are recoverable — the label just differs from the title, as it does for
`Chinese language` against `Chinese`. **The 2.3% that match one wrong thing are
not.** `creative commons` resolves to Q114734814 rather than Q43449 and there
is no signal anywhere that it went wrong; the fact arrives, reads fluently, and
is about something else. That is the same failure as answering "California" to
*what country*, arriving from a new direction.

So the join is a table with a key in it, not a heuristic. `sitelink` is
`(source, title) -> qid`, indexed both ways — the reverse index is the one that
matters, because reading a Wikidata dump means arriving with a Q-id and asking
which article it is.

**Two guards, both for silent failures.** Mixing snapshots is refused outright:
page ids are stable enough between dumps that it mostly works, and it fails
only on the pages deleted and recreated in between, which is precisely the kind
of error nothing would notice. And `--stats` reports how many sitelinks name no
article, because a stale pair of dumps parses perfectly and writes rows that
match nothing — which is the guard that caught the ampersands above, on its
first run against a real corpus. It prints a decimal place for the same reason:
`.0%` rounds 99.7% to 100% and the whole point of the number is the remainder.

Nothing downstream reads the table yet — no card file changed and no answer
changed. It is the key, not the facts.

## Reading the facts, now that the key exists

`wikidata.py` cuts a Wikidata graph dump down to the statements this
encyclopedia could use — 22GB and 766.5M edges in, **3.1MB out** — and imports
them into `derived` under method `wikidata`. Two programs, because the export
needs `ladybug` and 22GB of disk and nothing else here does; the file it writes
is read with the standard library.

```bash
# Once per Wikidata snapshot (~25GB down, hours, needs `ladybug`).
curl -O https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2
python data/wikipedia/wikidata.py --build latest-truthy.nt.bz2 -o wikidata.lbdb

python data/wikipedia/wikidata.py --export wikidata.lbdb -o wikidata.tsv.gz
python data/wikipedia/wikidata.py --survey wikidata.tsv.gz         # needs no corpus
python data/wikipedia/wikidata.py --score wikidata.tsv.gz          # writes nothing
python data/wikipedia/wikidata.py --write wikidata.tsv.gz --rebuild-graph
```

### The graph the export reads has to be buildable

For a while it was not. `--export` was documented against `wikidata.lbdb` and
nothing in the repository made one — the `wikidata_node` / `wikidata_rel`
schema was simply assumed to exist, which meant the pipeline worked exactly as
long as one 22GB file survived on one disk.

`--build` reads the **truthy** N-Triples dump rather than the full JSON export,
because `export` reads a subject, a property and an object and nothing else:
truthy is those same statements with the qualifiers, references and deprecated
ranks already dropped. It keeps entity-to-entity statements and discards labels,
descriptions, aliases, sitelinks and every literal value — a birth *date* is a
fact about an article rather than an edge to another one, and the corpus already
reads dates out of the infobox.

Most of the runtime is bzip2. It shells out to `lbzip2` or `pbzip2` when the
machine has one and to `grep` for the first filter, because both of those loops
are C ones and the pure-Python fallback is the same work an order of magnitude
slower. Installing `lbzip2` is worth more than any other tuning here.

The node list comes from a bitmap over the qids the edges mention rather than a
set of them: a Python set of 91.6M integers is several GB, and the ids are
dense enough that one bit each is 16MB. Both tables load by `COPY` from
parquet — 766M edges inserted one at a time is not a thing that finishes.

### Mapping a relation does not cost a pass over the dump

The export used to filter on `PROPERTY` in the query, which made the property
map part of the *dump-scanning* step: adding P57 meant going back to 22GB to
answer a question the previous scan had already read past and discarded. Since
format 2 it keeps **every** property with both ends in the corpus, so the cost
of a new relation is an edit to `PROPERTY` and a re-read of a file already on
disk — seconds, and no `ladybug`.

That leaves three reasons to re-export, none of which is "we want one more
relation": a newer Wikidata snapshot, a corpus whose article set changed, or a
bug in `export`. The scan is unfiltered now and the scratch parquet is
correspondingly larger, which is the right trade — building the database is
where the hours go, and that is unchanged.

`--survey` prints what the file holds per property, unmapped first and biggest
first, which is the same question `ingest.py --stats` answers for infobox
fields it does not understand. It reads only the export, so it works on a
machine that has the file and not the 500MB corpus.

An older format 1 export still imports. It was cut against a fixed list of
nine, so `--score` says which newly mapped properties are absent from the
*file* rather than from Wikidata — the two look identical otherwise, and the
symptom of confusing them is a relation that silently never appears.

**A relation costs more than a `PROPERTY` entry.** It needs a question class in
`data/questions/relations.py` to be askable, and `MIN_EXAMPLES = 150` is a real
floor — a class the crowdsourced corpus cannot fill is one the classifier
answers from the prior. On the card a relation id is one byte with `INVERSE` in
the high bit, so **127 is the hard ceiling** and eleven are used; edges are 7
bytes each, stored twice, so the graph file grows linearly with what is mapped.

Keyed by Q-id, not by title. A title is a fact about one snapshot — 726 of them
changed the day the escaping was fixed — so the export outlives the corpus it
was cut against and the join is redone from `sitelink` every time.

| path | before | after |
|---|---:|---:|
| `born_in in_country` | 42,288 startable, 77.7% | **67,510, 88.0%** |
| `died_in in_country` | 17,277, 82.3% | **33,645, 90.9%** |
| `in_country` | 49,784, 86.6% | **86,289, 94.5%** |
| `created_by born_in` | 11,107, 74.3% | **13,406, 93.3%** |
| subjects on the graph | 37.4% | **62.8%** |

**The rates went up as well as the counts**, which is not what happens here.
Every previous change that made more subjects startable added the ones that
were failing for a reason and pulled the rate down — this file says so twice.
This one adds the containment those subjects needed to climb in the same pass
that adds the subjects.

### Four rules, each because the version without it was wrong

**A country is only taken for something Wikidata puts administratively inside
something else.** `country` on a place is where it is; on a *language* it is
where it is spoken, which is how `English language` collects ninety of them.
Rather than carry a list of what counts as a place, this asks Wikidata the
question it already answers: only a place has a `P131`. 36,103 subjects refused.

**Values that do not nest are declined, not picked.** `derived` holds one object
per subject and relation. Where they nest — `Sialkot` inside `Punjab Province`
inside `British Raj` — that is one answer at three depths and the innermost is
it. Where they do not, Everest is in China *and* Nepal and a band is nine
genres, and choosing would put a fluent half-truth on a card with nothing
marking it as one. 14,680 declined, about ten thousand of them genres.

**A refinement has to keep its ability to climb.** Where Wikidata's value is
provably inside what the corpus already said it replaces it — `Mississippi` to
`Carrollton, Mississippi`. That is only an improvement if the finer answer still
reaches a country, so 924 are refused because it would not.

**Where several properties mean one relation, they are ranked rather than
merged.** Containment is the exception — P131 and P17 are one question asked
twice, so they union and the innermost wins. The eight that land on `created_by`
are not: a film's director, producer and composer are three different people,
and unioning them hands `choose` three values that do not nest, so it declines
and the film gets nothing. The infobox path never had this problem because
`libgraph.CANONICAL` ranks its fields — director outranks producer — and
`PROPERTY` is now written in that same order, with the ranking *being* the
order. An outranked property is not a fallback: two directors is an ambiguous
answer to "who directed", and the producer is not the repair for it.

### The importer fetched fewer properties than the questions asked about

`relations.py` builds a question class per Wikidata property and mapped sixteen
of them; `wikidata.py` exported nine. The seven in the gap were **P57**
(director), **P86** (composer), **P162** (producer), **P676** (lyricist),
**P84** (architect), **P178** (developer) and **P364** (original language) — so
the classifier was trained to answer "who directed X" against a graph whose
only `created_by` edge on a film was whatever that film's infobox tabulated,
which is the 46% this import exists to get past. `language_is` had no Wikidata
source at all; **P37** (official language) is mapped too, which is the same
relation asked of a country rather than of a work.

They are in `PROPERTY` now, and `test_wikidata.py` asserts the two maps agree
rather than leaving it to be noticed again.

The graph as the 2026-08-01 snapshot leaves it, before any Wikidata import, is
what the seventeen have to improve on:

| relation | edges | | relation | edges |
|---|---:|---|---|---:|
| `located_in` | 49,784 | | `member_of` | 10,507 |
| `born_in` | 42,288 | | `language_is` | 9,909 |
| `died_in` | 17,277 | | `preceded_by` | 6,783 |
| `genre_is` | 11,429 | | `followed_by` | 6,538 |
| `created_by` | 11,107 | | `spouse_of` | 1,836 |
| | | | `capital_is` | 848 |

168,306 edges over 283,997 articles. **`created_by` is 11,107 and every one of
them is an infobox `director` or `author` field** — the nine-property import
added P170 and P50 and reached 13,406, so the eight that land there now are
being asked to move the number that has moved least. `language_is` is 9,909 and
had no Wikidata source at all until P364 and P37.

**The numbers in the tables above were measured against the nine**; re-running
`--export` is what says what the seventeen are worth.

### What a country is, asked rather than voted on

`TYPE_FLOOR` decides a country by counting infobox fields that say `country = X`,
which is a vote that once elected California and still elects `CA`, `FRA`,
`Baku` and `World`. Wikidata states it instead, and knows **94 countries this
corpus did not** — so `wikidata.py` writes them into `derived` under the
relation `type_is`, and `libgraph.types` reads them.

They are entered *at* the floor rather than above it. A statement is worth more
than an opinion, but where the two contradict each other the existing
containment guard should arbitrate on the corpus's own numbers rather than be
outranked — and it still runs, so a claimed country inside another claimed
country is demoted exactly as California was. 244 claimed, **217 after
containment**, against 143 before.

A `type_is` row is never admitted as an edge. "England is a country" belongs in
`entity_type`; as an edge it would be a hop a walk could take, out of the graph
and into a word.

### The cost of a refinement was measured against the wrong graph

Refinement looked to cost **6,879** country answers. `Carl Wieman: Oregon ->
Corvallis, Oregon` — and Corvallis is in this encyclopedia while nothing in it
says Corvallis is in Oregon, because that article has no infobox at all and
`from_categories` will not read `Cities in Benton County, Oregon` for a subject
it does not already know to be a place. Chicken and egg: it cannot be recognised
as a place because nothing yet says where it is.

But Wikidata has `Corvallis -> Benton County, Oregon`, and **this import adds
it**. Measured against the graph the import leaves rather than the one it found,
the cost is **909**, against 1,319 subjects that could not climb before and now
can. A cost measured before the repair that removes it is not the cost.

### Which is also why the hop limit moved

The chains got longer. `Carl Wieman` reached the United States in two hops and
now reaches it in four, through a county and a state that were not there before.
`Cannes` needs six:

```
Cannes -> Grasse -> Arrondissement of Grasse -> Alpes-Maritimes
       -> Provence-Alpes-Côte d'Azur -> Metropolitan France -> France
```

`CLIMB_LIMIT` was 6, which buys five hops, and it had exactly one step of
headroom before this. 4,643 answers went past it. **Eight recovers every one and
nine buys nothing** — no chain in this corpus is deeper — and the eZ80 cost of
the deeper limit was already measured in [`data/silo/`](../silo/): climbs that
never reach the limit are byte-identical, and a climb the limit newly answers
gets *cheaper*, because failing one falls back to reading article text.

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
