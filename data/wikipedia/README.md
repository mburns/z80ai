# Simple English Wikipedia, as a search card

The whole encyclopedia on an Agon Light: 283,997 articles, searched in plain
English, from an SD card.

```bash
# once, from a MediaWiki dump (~4 minutes)
python data/wikipedia/ingest.py ~/Downloads/simplewiki-20260801-pages-articles.xml.bz2

# then, whenever the database changes
python buildwikisearch.py --out dist/WIKI
```

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

```bash
python data/wikipedia/ingest.py --stats     # what is in there, and from when
```

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

**It is not an oracle.** "Who wrote hamlet" returns *Hamlet*, not
*Shakespeare* — that is a search engine working correctly, and extracting the
answer from the article is comprehension, which is out of reach here. Measured
on thirteen probe queries, the right article is in the top three eleven times.

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
| accumulator | 277 KB in SRAM |
| card read per query | ~23 KB → 0.09 s at 250 KB/s |
| instructions per query | 5.4 M → ~0.74 s at 18.432 MHz |
| program | 6,132 bytes |

Most of that time is the two passes over the accumulator — clearing 277 KB and
then scanning it for the best three. The retrieval itself is a few thousand
postings.

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
