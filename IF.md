# A world, rather than an oracle

```bash
python buildif.py -o SILO.bin
```

Six rooms down the stair of Silo 18, four things to carry between them, and a
turn loop. It is small on purpose: the question issue #62 asks is whether the
machine can *hold* an Interactive Fiction and what a turn costs, and neither
needs three hundred rooms to answer.

```
Level 1 Landing
The top of the stair. A sealed hatch above you has not been opened in living
memory, and the light through the screen is the colour of dust.

> down
The Cafeteria
Long tables, and the great screen along the far wall showing the hills
outside. Nobody sits at the tables nearest it.
You can see ledger.

> x ledger
A ration ledger from year 188. The back of it has been drawn on.

> take ledger
Taken.
```

## A turn reads nothing

That is the whole reason this is a separate program from the search card.

| | a question | a move | asking a person |
|---|---:|---:|---:|
| card bytes | ~4,600 | **0** | **0** |
| instructions | ~470,000 | ~4,700 | ~7,000 |

The oracle's figures are fine for a question and hopeless for a step. A player
takes a step every few seconds and most steps are `DOWN`, so a move has to be
free, and the only way it is free is if nothing about it touches the card.

So the world is tables in the image and a small mutable overlay in RAM. Nothing
is read, because there is nothing to read: it is all already there.

**Talking is in the same column as walking, and that was the design
constraint.** A game about asking questions in which asking a person cost what
asking the archive costs would be a game nobody talked in - so a person is a
table in the image and answers for nothing, and the card is reserved for the
one thing that is supposed to feel expensive.

The parser grew a third word slot and a noise-word table, so every command now
pays a table scan per word that it did not before. Measured on the same world
either side of the change, a move went from **4,603 instructions to 4,746** —
143 instructions, about 3%, to buy `ASK MARNES ABOUT ALLISON` and `TAKE THE
LEDGER`. It was worth checking rather than assuming; a scan per word per turn
sounds like it should cost more than that, and the reason it does not is that
the noise table has seven entries and most commands are two words.

## The parser is a word table, and that was measured

`examples/parser/` scores both approaches on the same commands:

```
           encoder   char acc    train     eval
    flat (current)     99.3%    97.9%    85.9%
  8 position bands    100.0%   100.0%    98.4%
        word table         -   100.0%   100.0%
```

Accuracy is not what decides it. Asked about a word neither was given, both
models answer with a confident `OK` or `NO` and the table declines:

```
> xyzzy
I do not know the word 'XYZZY'.

> take zorkmid
I do not know the word 'ZORKMID'.
```

A player types a noun the author never wrote about every few turns. Naming it
back is the only useful reply, and a bare argmax cannot produce it — the same
gap `data/silo/` closed by [teaching a refuse class](data/silo/README.md).

### What a thing may be called

`SPLIT` takes two words and stops copying either at twelve characters, so a
thing named `ration book` or `identification` is one the parser can never
resolve. Nothing said so, and the failure was quiet twice over — the build
succeeded, and `DESCRIBE` then listed the thing in the room every turn:

```
> take identification
I do not know the word 'IDENTIFICATI'.
```

The truncation in that reply is the only evidence on screen that the *world*
was built wrong rather than the player spelling it wrong. `World.check()` now
refuses both, and `libworld.MAX_WORD_LEN` is where the limit lives because it
bounds what an author may write as much as what the parser may read.

That limit now covers a person's name and a topic's words too, for the same
reason and with the same message: both are looked up in the same `LOOKUP` over
the same twelve-character slots.

### Three words, and a table of the ones that mean nothing

`ASK MARNES ABOUT ALLISON` is four words and the shortest natural phrasing of
the only command that names two things. So the splitter fills three slots and
drops the noise between them — `ABOUT`, `THE`, `A`, `AN`, `TO`, `AT`, `FOR` —
by looking each word up in a table and writing over it if it hits.

Dropping them in the splitter rather than in `DO_ASK` is what keeps the count
at three. Every natural wording puts a preposition between the person and the
topic, so either the splitter loses it or every slot in the program widens by
one to carry a word only one command ever uses. It pays for itself elsewhere
too: `TAKE THE LEDGER` now works, and did not before.

The model is not wrong there in a way more training would fix. It is being
asked a question with no answer and returning its best guess, because that is
the only thing it can do.

## What is mutable, which is what a save file is

| | |
|---|---|
| `rooms` | name, description, six exits — 12 bytes, in the image |
| `things` | name, description, where it starts, portable — 8 bytes, in the image |
| a thing's name | **one word, at most `MAX_WORD_LEN`** — the parser's limit, not a style rule |
| `SUBJECTS` | one pointer a thing, to the line `CONSULT` types at the archive — in the image |
| `people` | description, default line, where they start — 7 bytes, in the image |
| `lines` | person, topic, gate, flag to set, text — 7 bytes, in the image |
| `HERE` | the room the player is in — **1 byte, in RAM** |
| `WHERE[]` | where each thing is now — **1 byte each, in RAM** |
| `FLAGS[]` | one bit a proposition — **in RAM** |
| `ASKED[]` | what has been asked about — **1 byte a topic, in RAM** |
| `HEAT` | how much attention that has cost — **1 byte, in RAM** |
| `CLOCK` | how many turns have been taken — **1 byte, in RAM** |
| `SEALED[]` | which records the archive is declining — **1 byte a topic, in RAM** |
| `ALTERED[]` | which records it is serving rewritten — **1 byte a topic, in RAM** |
| `PWHERE[]` | where each person is now — **1 byte each, in RAM** |

Only the bottom nine change. The image is identical on every copy, so a saved
game is the overlay and nothing else: **13 bytes** for the six-room world as it
was, **85** for the mystery with its four people and five topics, one
contiguous run so that writing it is a single `mos_fwrite` rather than three
and a format.

`ASKED` is inside that run rather than beside it, and the reason is not
symmetry. A restore that put the player back on the stair but forgot what they
had already been told would re-explain everything and fire every `C_ASKED`
rule a second time — a worse bug than losing the save, because it looks like
the game working.

It is also the one thing in the overlay that only grows. No action clears it
and no opcode could: `A_CLEAR` can put a flag back, and a mystery whose record
of what the player has learned can be rewound is not a fair one. Monotone
state is enforced by there being no instruction for the alternative rather
than by nobody having written one.

`SAVE` and `RESTORE` move exactly that run, [below](#save-restore-and-the-one-file-that-outlives-a-game).

## The free-SRAM figure, re-measured

Issue #62 budgeted the second scope against **388 KB** of free SRAM. That was
arithmetic over the *oracle's* memory map — 512 KB less the program, the
accumulator, the unpacking scratch and the stack.

A world binary does not share that map. It has no accumulator, no unpacking
buffers and no classifier, so:

```
SILO.bin  4,713 bytes   6 rooms, 4 things, 0 people, 0 lines
  overlay 76 bytes - the whole saved game
  image ends at 041269h, 515,479 bytes of SRAM unused
  108 reachable states
```

**503 KB**, not 388.

This file quoted 3,124 bytes and 13 of overlay for the same six rooms, and
both had drifted before any of this: the tree built 4,242 bytes and 73 of
overlay immediately before the change that added people. Measured against
that rather than against the prose, **people, topics, attention and the third
word slot cost this world 471 bytes and 3 bytes of overlay** — and it has
nobody to ask and nothing to ask about, so that is the price of the machinery
existing at all rather than of anything in `worlds.py`.

The lesson is the smaller one and worth writing down: a number in a document
is not a measurement, and the delta that matters was 471 rather than the 1,589
the stale figure would have implied. Both are nothing against half a megabyte
spare. Only one of them is true. The two programs were always going to be separate
binaries and the estimate quietly assumed one. What the 388 KB figure is
actually about is a world that lives *inside* the oracle card's program, which
is the "terminal found in the world" item and a different arrangement.

At 12 bytes a room and 8 a thing, 505 KB is more rooms than anybody will write.
Prose dominates and [#67](data/silo/README.md) already measured that a card
holds far more of it than an author will produce.

## Rules, and the three shapes of four they close

`data/silo/README.md` names four things a graph path cannot express, and the
reason is principled rather than accidental: composition works in both
directions and stops at aggregation, ranking and set intersection.

A rule is a flat list of conditions that must all hold, and a list of actions
to take when they do. That is the smallest step past a path, and it is worth
being exact about what the step buys:

| a path cannot | a rule | |
|---|---|---|
| a count over a set | **yes** | `CARRYING 2` — a question about a set, not about any one thing |
| an intersection of two sets | **yes** | two `HAVE` conditions ANDed: not two things, *these* two |
| state that outlives the question | **yes** | `FLAG` set in one room and read in another |
| a maximum over a set | **no** | ranking needs a loop, and a flat condition list has none |

Three of four. The fourth is not an oversight — a condition list is evaluated
once per rule and never iterates, so "the oldest person on X's crew" is exactly
as far out of reach as it was. Closing it wants a loop opcode, which is a
different instrument and a bigger one.

**And a fair-play mystery does not want it.** Ranking a set is what the
*player* does; the machine serves the clues and checks one answer. `ACCUSE
<person>` is a byte compare, and the deduction happens in a head the hardware
does not have to model. The division of labour the eZ80 forces turns out to be
the one the genre already wanted — a librarian, not a detective — which is why
the loop opcode is still not here and is no longer the obvious next thing.

```
> take badge
> take wrench
Your hands are full. Whatever else you find down here is going to have to
wait, or something you already have is going down the stair without you.
The badge and the wrench together look like a story you would rather not
have to tell a deputy.
```

Both of those are rules. The first is a count and the second a conjunction, and
neither is a question the silo card could be asked at any length.

A rule is length-prefixed so that skipping one is an addition rather than a
walk over its parts. Flags and the one-shot markers are a **byte apiece rather
than a bit**: bits are eight times smaller and want a shift and a mask at four
call sites, and there is half a megabyte of SRAM spare — the sixty bytes are
not worth four places to be wrong, and a restore that replayed every event the
player had already seen would be the bug that saved them.

## The question is the plot

Three of the four shapes a path cannot express were closed by a condition
list. The one the condition list could not reach at any length is not
aggregation — it is *what the player wanted to know*, which was not state at
all until now.

```
> ask jahns about allison
Jahns looks at the screen rather than at you. 'She went out. That is the
whole of the record and it is enough.'
Jahns does not answer that one. She looks at you for a moment longer than
she needs to, and then back at the hills.
```

The second paragraph is a rule, and its conditions are `ASKED allison` and
`WITH jahns`. Neither is a fact about the map or the inventory. Together they
are a question and who was standing there when it was asked, which is the
first thing in this program that the card could not have been asked and the
map could not have recorded.

| | | |
|---|---|---|
| `C_ASKED n` | topic `n` has come up | of the archive **or** of a person |
| `C_HEAT n` | attention stands at `n` or above | one byte, saturating |
| `C_WITH n` | person `n` is in the room | |
| `A_HEAT n` / `A_COOL n` | attention up and down | |
| `A_SEND p r` | move a person | not `A_MOVE`: a person is not a thing |

Both directions of the counter exist because a world with only `A_HEAT` is one
the player can only lose. There has to be somewhere to lie low, or the counter
is a countdown wearing a different name.

It saturates at both ends rather than wrapping. 200 and 200 is 255, not 144 —
a counter that rolled over would hand the player an escape from every
consequence by asking enough questions, which is exactly backwards.

### One table, two ways of asking

A topic is one index whether it is put to a person or to the card. That is the
point rather than a saving: a player who reads the incident report about the
cistern pump and a player who asks Knox about it have learned the same thing,
and a world that recorded those separately would need every rule written
twice.

### Why the hook is the document, not the classifier

The obvious place to notice a question is where the question is understood.
It is the wrong place, and the repository already has the number that says so.

`liboracle` gets the relation right **84.0%** of the time on phrasings the
model has not seen. A plot that advanced only when the classifier agreed would
stall one turn in six, for a reason the player cannot see and cannot act on —
which is not an unreliable oracle, it is a broken one.

The *document* is not a guess in the same way. BM25 over titles resolves the
entity without help, and `liboracle.entity` says why: the mention carries the
rare words while the frame around it — "where was", "who wrote" — is common
enough that idf discounts it to nothing. So `NOTICE` hangs the plot off which
article was reached, which is the reliable half of the machine, and leaves the
classifier doing the job it is actually measured at.

That is the whole of the wiring: one scan of a table of `(article, topic,
attention, seal)` rows, between the search and the answer.

## A record that declines, and declines the same way twice

```
archive> pump
Incident Report 214-11: Cistern Pump Failure
The cistern pump on Level 142 stopped without warning.

archive> allison
RECORD SEALED BY ORDER OF JUDICIAL. THIS ACCESS HAS BEEN LOGGED.
```

Unreliability was accidental before this: 54% of Simple English Wikipedia
articles carry no infobox, so a chain that hops onto one of them stops. That
is a machine with gaps, and `liboracle` already does the honest thing with it
by reporting where the walk stopped.

A seal is the other kind, and it is chosen. A topic can carry text the archive
prints *instead of* the record — and the topic is still marked asked and still
charged its attention, because the refusal is the thing the player learned.

`data/silo/plant.py` established the principle for the corpus and this is the
same one at the terminal: **a record that is wrong in a fixed, discoverable
way is interesting, and one that is unreliable at random is noise.** A stable
lie is a clue. That is also the fair-play contract, and it is worth stating as
a rule rather than a hope — the archive may decline, but when it states a fact
that fact holds, except where it has been sealed and says so.

## Fair play, checked rather than promised

A fair-play mystery makes a promise the author cannot keep by reading their
own source: that the ending can be reached, and that every clue it rests on
can be found first. The state space is where that promise lives, so it is
checked there.

```bash
python buildif.py --world mystery -o MYST.bin
```
```
MYST.bin  7,335 bytes   6 rooms, 4 things, 4 people, 8 lines
  overlay 85 bytes - the whole saved game
  30,688 reachable states
  solved in 6: down, ask marnes about allison, down, take badge, east,
               ask walk about allison
```

`World.explore` walks the game rather than the map. A state is where the
player is, what everything is holding, which flags are set, which rules have
fired, what has been asked and what that cost. Every command is an edge.

**It models the device rather than an idealisation of it.** Rules are one pass
a turn, not a fixpoint, because `RULES_RUN` walks the table once and a rule
made true by a later rule does not fire until the next turn. `LOOK` is
therefore a move — it is the turn that costs nothing and lets a cascade finish
— and a walkthrough that needs one will contain one. That fidelity is what
makes `test_the_mystery_can_be_won` meaningful: it takes the solver's own
answer and plays it through the emulator, and the two have to agree.

### The reduction that makes it finish, and why it is sound

The first run of this did not finish. The state space is dominated by
inventory permutations: three portable things in six rooms multiplies
everything else by 343.

Dropping a thing changes exactly three conditions. `C_HAVE` goes false,
`C_CARRYING` falls, and `C_HERE` goes true. **The first two can only ever stop
a rule firing, never start one** — so the only way putting something down can
*open* anything is through a `C_HERE` that names it, and for every other thing
which floor it is lying on is a distinction the rule language cannot make.

So drops are modelled for exactly the things some `C_HERE` observes. On the
mystery, which has none, that is the difference between not finishing and
30,688 states in under a second.

| | states |
|---|---:|
| `worlds.silo()` — six rooms, four things | 108 |
| `worlds_mystery.mystery()` — plus four people, five topics, a counter | 30,688 |

### What it finds

An unwinnable game gets noticed in playtesting. The bugs this is for are the
quiet ones:

| | |
|---|---|
| a goal no state satisfies | the game plays for an hour and cannot be won |
| a line behind a gate nothing sets | reads as the author cheating |
| a room, thing or person nothing reaches | authored content nobody can see |
| **a rule no state fires** | no error, no output, nothing to notice at all |

The last one is not hypothetical. A rule in `worlds_mystery` was written to
charge attention for reading a sealed record and keyed on a flag that nothing
set. It had no error and no symptom, and this is what found it.

`libworld.World.check` catches the rest before anything is emitted — a person
standing in a room that does not exist, two topics claiming one word, a
person with no default line, and the one that actually gets made: an ungated
line written *above* a gated one for the same pair, so the fallback always
wins and the specific line can never be spoken.

## People are a table, and the table is the oracle's shape

```
> ask walk about allison
'Allison who,' says Walk, in the voice of somebody who knows exactly which
Allison.

> ask marnes about allison
'She was IT,' says Marnes, and puts the cup down.

> ask walk about allison
The boots go still. 'She fitted the landing screen the week before. On her
own. You have read the order on that wall.'
```

Three lookups and a linear scan over `(person, topic, gate, sets, text)` rows,
first match wins. That is the oracle's own shape — resolve the subject,
resolve what is being asked, look it up — at a hundredth of the cost, because
the table is in the image and the card is not touched.

Ordering is the whole of the conditional mechanism. The author writes the most
specific line first; there is no condition list on a line, because a line that
needed one is a rule.

`sets` is how a conversation teaches the world something. `C_ASKED` records
that a subject came up; `sets` records that a *particular person answered it*,
which is the difference between having raised a name and having been told
something. In the transcript above it is what makes the third command print a
different sentence from the first.

### Every person needs a refuse class

```
> ask knox about allison
'Down here we fix what is in front of us,' says Knox. 'You want somebody who
reads.'
```

That is the person's default line, and it is not optional — `check` refuses a
person without one. It is the same finding as the parser's, in a voice: a
model asked about a word it was never given answers with a confident `OK`, and
a person who answered confidently about a topic the author never considered is
worse, because the player cannot tell invention from testimony and will act on
it. A deflection in character is worth more than a guess.

### A person is not a `Thing` with `portable=False`

It would have saved eight bytes. A thing that cannot be carried is scenery and
is listed as "You can see screen."; a person is listed by the sentence that
puts them in a room, is asked rather than taken, and moves under `A_SEND`
rather than `A_MOVE` — so nothing can pick one up. Sharing the table would
have cost every message that mentions one.

## The card, standing in a room

```bash
python buildwikisearch.py --db data/silo.db --source silo --out dist/SILO
# and a binary that carries the world as well as the card:
#   buildwikibin.build(num_docs, world=worlds.silo())
```

```
> east
IT, Level 34
Racks of machines behind glass, and a bench where a screen sits with its
back off.

> use
The screen wakes. Type a name to look it up, or LEAVE to stand up again.

archive> cistern pump failure
Incident Report 214-11: Cistern Pump Failure, Level 142
At approximately 0340 on the eleventh day of...

archive> leave
IT, Level 34
```

### Carrying a name to it

`CONSULT <thing>` is the other way in, and the one the world is built around.
Ten thousand people are on the card and the world can carry none of them — but
it can carry a *name*, on a ledger or a work order or a death notice, and a
`Thing.subject` is that piece of paper:

```
> consult ledger
Pump Failure
The cistern pump on Level 142 stopped without warning.

> take screen
That is not something you can carry.
```

**The player never sat down.** That is the load-bearing difference: after `USE`
the classifier owns `INPBUF` and `TAKE` is not a command until you `LEAVE`,
whereas after `CONSULT` the world is still listening. The thing *is* the
question rather than a way into a prompt, so the entries a player can reach are
the ones they have physically found a reference to.

Two of the four shipped things name something and two do not, which is the
distinction the verb exists to make — a wrench is a tool. And there are two
refusals, at different layers:

| | |
|---|---|
| `The screen has nothing to say about that.` | the **thing** names nothing |
| `Nothing on the card matches that.` | the **archive** has never heard the name |

The world answers the first and hands the second over, because they are
different facts and it would be wrong for the world to answer the second.

One `World` is built twice — standalone and carried — and only the second has a
card, so `check()` deliberately allows a subject in a world with no terminal.
It refuses an empty one (the emitted table cannot tell that from *none*) and
one longer than the console reads, since `CONSULT` copies it into the very
buffer a player types into.

The wiring is one parameter: `emit_world_routines(..., ask_label=...)`, the
card's ask path in the merged binary and a stub that says there is no terminal
in the standalone one. Passed in rather than defined twice, because
`Z80Builder.label` overwrites silently.

**It goes the other way round from how #62 pictured it.** The issue described a
world with a terminal inside it. The world is 4,050 bytes and the oracle
program is 38,912, so the terminal is not the small thing — the world is, and
`buildwikibin.build(..., world=...)` is what carries it. A world costs the
oracle binary about 5.5 KB, [re-measured below](#what-a-world-costs-the-oracle-binary-re-measured).

### What "the two input paths can coexist" turned out to mean

Not that both fit. They share `INPBUF`: one line, read by whichever parser a
mode byte selects, and neither writes to it. `ATTERM` is that byte, and the
whole of the switch.

What nearly went wrong is narrower and worth recording. `Z80Builder.label`
assigns into a dict:

```python
def label(self, name: str) -> None:
    self.labels[name] = self.addr()
```

A name defined twice resolves to whichever was emitted last, and **nothing says
so**. Both programs had a `PROMPT` and a `BANNER`, so the merged binary printed
the oracle's `?` while the player was walking about — a bug with no error, in a
build that otherwise worked. `test_the_two_programs_define_no_label_twice`
spies on `label` during a merged build and asserts the count, because that is
the only way to see it.

## The map was already in the database

`worlds.py` hand-authors six rooms. `data/silo/buildworld.py` reads a world out
of `data/silo.db` instead, and it is short because the map has been in there
since the schema was written:

```bash
python data/silo/buildworld.py --floors 2 -o SILO.bin
```

| in the corpus | in the world |
|---|---|
| `next_along` — thirty minutes clockwise, and it wraps | `EAST`, and `WEST` back |
| `next_out` — one ring outward, and it does not | `NORTH` out, `SOUTH` in |
| `located_in` — a dwelling's level, a department's level | the stair, and the door off it |
| `article.lead` | every room description |
| `residence.until IS NULL` | the name beside the door |

`data/silo/schema.py` stores those two adjacencies as edges rather than
arithmetic **because the machine that walks them has no modulo**. A card walks
them to answer "who lives next door"; a world walks them to go east. Nothing
here recomputes `(bearing + 30) % 720`, and
`test_the_geometry_is_read_rather_than_recomputed` is the test that can tell
the difference — it deletes one edge and checks the exit went with it.

Nothing here writes prose either. Every description is a lead the corpus
already carries, so editing an article moves a room.

### The wall is the room id, not the memory

The section above measured 505 KB of SRAM free and observed that at 12 bytes a
room, that is more rooms than anybody will write. It is, and **it is not what
stops you**: `libworld.NOWHERE` is `0xFF`, so a room id is one byte.

```
144 landings + 14 departments        158 rooms    12,347 bytes
one residential floor                 72 rooms    24,492 bytes
                                     ---
                                     230 rooms, and 255 is the ceiling
```

One floor fits and two do not — 302 rooms, refused rather than truncated,
because a world quietly missing its bottom forty levels walks perfectly well.
The silo has twenty-nine opened floors. Reaching all of them wants a two-byte
room id, which costs every exit a byte in the image, or a world that streams
floors off the card, which costs a turn the one thing a turn must never cost.

A 230-room world still reads nothing: `io_bytes == 0` holds at two hundred
rooms exactly as it did at six, because a move is a table lookup whatever the
size of the table.

### Ten things, and nine of them are one case

The corpus has no objects, so a seed cannot be derived — only *placed*, and
derived **about**. `data/silo/items.py` is ten hand-written sentences with
holes in them, and the corpus fills the holes:

```
> x notice
A carbon of a cleaning notice, year 148. Sent out: Alexandra H. Anderson.
There is no reason given on it, which is the form.

> x key
A flat key on a wire loop. The paper tag is soft with handling and reads
107 800 A.
```

Each names the next place to stand. The notice is in Judicial and names a
person; the key is in the Sheriff's Office and names their flat; the photograph
*in* that flat names who they married; the slate in the Nursery names their
class. Consulting any of them at the terminal on Level 34 asks the card.

The case is the alphabetically first cleaning in the corpus rather than a
sample, so the same database always seeds the same silo — two builds of one
card must not disagree about what is in the drawer.

**The constraint that shapes the list is smaller than it looks.** A thing's
name is one word and every name in a world must be unique, so there is no such
thing as seventy-two ledgers, one per flat. Ten distinct objects is not a small
version of a big idea; it is the only shape this parser has.

Two items can legitimately fail to be placed — a corpus with no cleaning in it,
and the photograph when that floor was not opened with `--floors` — and both go
in a build log rather than quietly, because nine things out of ten is
indistinguishable from a world that was always meant to hold nine:

```
158 rooms, 9 things (6 of them name something on the card)
  not placed - photo: nowhere to put it - 107 800 A is not a room in this world
```

One of the ten is the control. A wrench names nothing, because `CONSULT WRENCH`
has to be able to say so.

### The chain, walked

Every other test holds one link — the seed is placed where it belongs, the case
fills the holes, `CONSULT` copies a subject into `INPBUF` — and none of them
says the links join. That is the arrangement that passes while the thing it
describes does not work, so one test builds the whole apparatus and plays it:

```
> east            take notice     Judicial, level 5
> east            take key        Sheriff's Office, level 4
> east            IT, level 3 — the terminal
> consult notice  Alexandra Anderson is an entry in the archive...
> consult key     2 100 A is an entry in the archive...
> west > east > east              round ring A to the flat the key named
> take photo
> consult photo   Ronald Gordon is an entry in the archive...
```

Three questions and twenty-five moves. The cost assertion is an **equality**
rather than a bound: the same three questions asked after twelve more moves
read the same number of bytes, so the moves cost nothing at all rather than
nearly nothing — a bound would pass a world that paged.

The card is made by the test rather than being `data/silo.db`, which is 42 MB,
gitignored and wants Faker. That is a real limit and worth stating plainly:
this walks the *mechanism* over a corpus invented for it, and nothing in CI
walks the corpus that ships.

## Two checks the prose needed

**A whole playthrough, kept.** `tests/test_if.py` asserts phrases — fifty-three
tests, out of a world that says several thousand words — and is blind to the
half of an Interactive Fiction that *is* prose. `tools/transcript.py` replays a
session and compares the lot:

```bash
python tools/transcript.py                       # replay every one
python tools/transcript.py --update tests/transcripts/silo.txt
```

The file is the game's own output. `READ_INPUT` echoes what it accepts, so a
run already comes back looking like a session at a terminal, and the commands
are recovered by reading the `> ` lines back — there is no second file of
commands to keep in step. A diff is not by itself a bug; it is the change being
*seen*.

It found one thing on its first run: `DO_INV` prints "You are carrying" once
per item. Consistent with `DESCRIBE`, not what a player expects, and now
pinned.

**And it made a second one visible by omission.** Every thing row has carried a
description pointer since this file was written, and no code path ever read
offset 3 of one — four descriptions sat in the image, indexed and unreachable,
because there was no verb that showed them. `EXAMINE` / `X` / `READ` is the
cheaper of the two resolutions; the other was to stop emitting the text, and
`worlds.py` had already written four descriptions worth reading. It works on
what you are carrying as well as what is in the room: picking a thing up has
not stopped you being able to look at it.

**A rule that can never fire.** `World.check()` refuses every argument that
indexes nothing, and none of those is the bug an author ships. That one is a
key in the room it unlocks: arguments all in range, conditions that can never
all hold, no error. `World.reach()` is a monotone fixpoint whose result is
deliberately a **superset** of what a player can bring about, so a rule outside
it is certainly dead and a rule inside it is only probably live. Erring the
other way would report locked doors that are not locked, and an author who has
seen one false alarm stops reading the report.

Some contradictions are exact rather than searched — `AT 3` with `AT 5`,
`FLAG 2` with `NFLAG 2`, and `HAVE k` with `HERE k`, which reads as a plausible
sentence and is impossible because `where[k]` is `CARRIED` *or* a room.

## What a world costs the oracle binary, re-measured

| | bytes | |
|---|---:|---:|
| the search program, no world | 4,812 | |
| carrying `worlds.silo()` | 10,458 | +5,646 |
| carrying `worlds_mystery.mystery()` | 13,155 | +8,343 |

The first delta was 4,434 for as long as this file claimed a world costs the
oracle binary under 5 KB. Save, restore and the archive's log took it to
5,512 — a kilobyte, most of it the four routines that talk to the card and
the buffer a save goes through — and the Voice's four actions and two
conditions another 134. The claim is now "about 5.5 KB", which is still the
small half by a factor of seven. The second is what people,
topics, dialogue and the attention counter add on top, and most of it is
prose rather than code.

## A clock, and a deadline the solver can be held to

Attention moves only when the player asks. Everything above fires because of
something the player did, and a mystery needs the other kind of pressure -
the thing that happens because turns passed, whether or not anybody was
there. The cleaning is at the end of the week. The suspect leaves on the
next shift.

`CLOCK` is one byte in the overlay and `C_TURN n` is one condition: the
clock stands at `n` or above. It reads as **the number of commands already
taken** - the opening pass sees zero, the pass after the third command sees
three - and `RULES_RUN` ticks it after the pass, saturating at 255 for the
reason `HEAT` saturates: a clock that rolled over would hand back every
deadline that had passed.

```
> look
> look
> look
There are boots on the stair.
```

That is `TURN 3`, on the third command and not the second, and
`tests/test_clock.py` holds the device and `explore` to the same count.

It is in the overlay rather than beside it for the reason `ASKED` is. A
restore that reset the clock would give the player every deadline a second
time, which is a worse bug than losing the save because it looks like the
game being generous.

**What it changes about fair play is the question.** `solve` used to ask
whether the goal could be reached. With a rule that sets `LOST` on the
deadline and a goal that needs `NFLAG LOST`, it asks whether the goal can be
reached *in time*, and returns `None` when the clues do not fit inside the
clock - which is the promise a Golden Age mystery makes and the one no author
can check by reading their own source.

The price is the state space. `explore` clamps the clock at the latest
deadline any condition reads, exactly as it clamps attention, so a world with
no deadline has a clock that never leaves zero and `worlds_mystery` is still
30,688 states. A world with a deadline at turn N multiplies its space by about
N. That is fine for six rooms and the reason [#108](../../issues/108) wants
a different instrument for two hundred.

Shifts, and where ten thousand people are at a given hour, are the next two
steps of [#101](../../issues/101) and are not here. See [ROADMAP.md](ROADMAP.md).

## The Voice, which does things to the record

```
archive> order
Standing Order 11: Screen Fitting (as amended, year 218). A screen may be
fitted by one person where a second is not available. Judicial.
```

That is not the standing order. The one on Walk's wall says two, the player
has read it, and the archive started saying otherwise on the turn the two
clues met. Until now a seal was authored — `Topic.censor` was a fact about
the topic, fixed for the whole game — and the archive's unreliability was
something that had happened before the player arrived. This is the other
kind: the Voice reacting to what the player did.

| | |
|---|---|
| `A_SEAL n` / `A_UNSEAL n` | the archive declines topic `n`, or stops declining it |
| `A_ALTER n` / `A_TRUTH n` | the archive serves topic `n` rewritten, or as written |
| `C_SEALED n` / `C_ALTERED n` | a rule can read either, so a person can react to a seal the player has not yet run into |
| `Topic.alter` | what the archive prints while the topic is altered |
| `Topic.sealed` | whether it starts sealed; `None` means "if it has censor text", which is what every earlier world meant |
| `World.seal` | what a sealed topic prints when it has no censor text of its own |

Two bytes a topic in the overlay, because a restore that unsealed everything
the Voice had closed would be the Voice forgetting it had been threatened. A
seal wins over an alteration: a record that is closed cannot also be read
wrong. And both have an undo, for the reason `A_COOL` sits beside `A_HEAT` —
a world where the archive can only close is one the player can only lose.

**The altered text is in the image, not on the card.** The obvious design
was a second article and a swapped id, and it is wrong: a false record that
can be found by searching for it is not a false record, it is a second
record. Like a seal, an alteration is prose the archive prints *instead*,
and nothing the search does can reach it.

**And it is still fair.** `plant.py` set the rule for the corpus — a record
that is wrong in a fixed, discoverable way is a clue, and one that is
unreliable at random is noise — and the mystery keeps it: the amendment
fires on a rule the player caused, it contradicts a sentence they have
already read in a room description, and it is the same amendment every
time. The archive may decline, and when it states a fact that fact holds,
*except where a rule has changed it and the world says so somewhere*.

The mystery uses both. At five on the attention counter the deputy comes up
the stair and the pump report is sealed — a record that was never about
Allison, because the Voice reacts to the *asking*. With both clues in hand
the standing order is rewritten. Neither costs the search a state: both
ride on rules that already fired in states the search already told apart,
so `worlds_mystery` is still 30,688 states, and `tests/test_voice.py` pins
that beside a world of its own where the clock drives all four actions in
a fixed order.

What is not here is the hedge — the margin between the classifier's top two
logits, surfaced on the device as *"if I have your meaning"*, which
`liboracle` already renders on the host. That wants the classifier in the
binary, which is the silo card and not this fixture, and it is the rest of
[#102](../../issues/102).

## Save, restore, and the one file that outlives a game

```
> save 2
Saved.
> restore 2
Restored.
The Mids Stair
```

A saved game is the overlay and a four-byte header, `SV` and a stamp, in one
`mos_fwrite` to `SILO1.SAV` through `SILO9.SAV`. The stamp is the *shape* of
the world hashed — rooms, things, flags, rules, topics, people — and not its
prose, because editing a room description must not invalidate every saved
game, while a save from a world with a different shape would load into this
one without complaint and put the player somewhere that does not exist. A
restore that finds any other header, or fewer bytes than the file should
hold, says so and touches nothing.

**Neither verb is a turn.** Both come back through `NOTURN`, which is also
where an empty line goes now — the rules do not run and the clock does not
tick — so a game saved and restored plays on *exactly* as one that was not.
`tests/test_save.py` holds it to that by playing the same commands both ways
and comparing the output from the restore onward, byte for byte. `FIRED` and
`ASKED` coming back with the rest is what makes that true: a restore that
forgot either would re-explain everything and send the deputy up the stair a
second time, and it would look like the game working.

### The archive's log

The archive says it is logged. It is:

| | |
|---|---|
| `SILO.LOG` | two bytes a question the card saw — the clock, and the topic, or `0xFF` for a record the world has no name for |
| `LOGGED` | its length, read when a game starts and kept in step by every question |
| `C_LOGGED n` | a condition: the log holds `n` questions or more, this game or before |

A walk writes nothing and a question writes two bytes, appended with
`FA_OPEN_APPEND` — the one FatFs mode `libhost` had to learn for this, and
one `tools/mostest.py` now probes on hardware beside `mos_load`.

**`LOGGED` is outside the overlay on purpose.** The file is the truth and the
byte is its length, so a restore leaves it alone: the archive does not forget
what it was asked because the player wound the clock back, and the rule that
noticed the second question fires again after a restore to before it. That is
the Voice's memory and it is the shape series memory wants — a game that
opens on a card another game has already written to starts with `LOGGED` at
that game's count, and a rule on `LOGGED 2` fires before the first prompt.
The standalone world binary reads the same file, so what the oracle binary
was asked is known to a world that has no terminal to ask.

`explore` counts the log from zero, which is a fresh card, and clamps it at
the largest count any condition reads. A rule keyed on it is therefore one
the solver reaches by asking, and one the device may also fire on the opening
pass — both are what the author meant.

## What it does not do yet

No daemons, no containers, no ranking. No screen mode and no status line:
`PRWRAP` decides where a line ends, and nothing here has ever told the
terminal anything.

People are still not `Thing`s and cannot be: `where[]` is a byte apiece, which
would turn a 13-byte save into 10 KB. They stay on the card and are reached
through the objects that name them.

`libworld.Person` does not change that arithmetic and is not an attempt to.
A `Person` is an authored character with lines written for them, and four of
them cost four bytes of overlay; the corpus's ten thousand are records, and
they stay on the card where they can be read and not carried. The two are
different kinds of thing that the word "person" happens to cover, and the
resident one is bounded by how much dialogue anybody writes rather than by how
large the silo is.

Attention is not shown to the player. It moves the world and the player infers
it from boots on the stair, which is the right register for this fiction and
is also the reason there is no status line to put it on.

`C_ASKED` tests a topic against zero. The count is stored, because the byte is
spent either way and a threshold opcode would want it already there, but
nothing exposes it — "asked three times" is one `cp` away and has not been
needed.

The state search grows with the flags a world actually uses, and `explore`
raises rather than guesses when it passes 200,000 states. Three hundred rooms
would want a different instrument; six and four people want this one.

Those are what is left of #62's second scope.
