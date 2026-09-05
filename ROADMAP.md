# Roadmap: a series of fair-play mysteries on a card

Where this is going, and in what order. The target is an Interactive Fiction
set in a Silo-like community: a series of fair-play mysteries in the Golden
Age shape, an archive with a Voice that is an unreliable narrator on purpose,
and ten thousand people any of whom can be asked about - on an Agon Light
with an SD card.

The starting point is measured rather than hoped, and it says where the
effort goes. A question on the silo card is about 480,000 instructions and
4,600 card bytes; a move is 4,700 instructions and none. Neither is what a
player will feel. What they will feel is the classifier being wrong one time
in three on a wording it has not seen, a world that stops at 255 rooms, and
four people to talk to. So the roadmap is accuracy, scale and content, and
the engine changes are the ones those need.

Each item is an issue. The order is roughly dependency order; the case
generator is last because it is what the rest is for.

## Engine

- [x] **A clock the world can read** - `C_TURN`, one overlay byte, a deadline
      the solver can be held to. Step one of [#101](../../issues/101) is in
      this tree; shifts and presence by schedule are steps two and three.
- [x] **The Voice acts** - `A_SEAL`, `A_UNSEAL`, `A_ALTER`, `A_TRUTH`, and
      rules that read what it has done; the mystery seals a record and
      rewrites another. [#102](../../issues/102)
- [ ] **The Voice hedges** - the margin between the classifier's top two,
      surfaced on the device as its tell. The rest of #102; wants the
      classifier in the binary.
- [x] **The whole silo on one byte** - a dwelling is a door on its floor's
      ring, knocked on by its number and never entered; 187 rooms and 2,088
      doors, no overlay. Paging the stair turned out not to be needed.
      [#103](../../issues/103)
- [x] **Save, restore, and a card the world writes to** - `SAVE n` and
      `RESTORE n` move the overlay in one write, neither is a turn, and the
      archive's log is the series memory: a game opens knowing what the card
      was asked before. [#104](../../issues/104)
- [x] **The archive as records** - `LOOKUP <name>` on the card: a name index
      of two 24-bit hashes a title, binary-searched, and the record printed
      from the graph with no search and no classifier. 12,904 instructions
      against a question's 465,179. [#105](../../issues/105)
- [ ] **Two subjects on the eZ80** - the second search, a `SHARED` step, a walk
      that runs twice. [#106](../../issues/106)
- [x] **Testimony from records** - `ASK <door> ABOUT <name>`: the household
      behind any door answers about any name from the graph within two hops,
      in its department's voice, or says it does not know. Nothing is written
      per person. [#107](../../issues/107)
- [ ] **The VDP is idle** - status line, the stair, the wall screen, sound.
      [#111](../../issues/111)

## Tooling

- [x] **The accusation** - `ACCUSE <person | door>`, one shot, `won` and
      `lost` as flags the goal reads. The mystery is won by accusing the
      mayor, and not by guessing. [#112](../../issues/112), first step
- [~] **Fair play at scale** - `libplan` builds a walkthrough backwards from
      the goal and steps it through the exact model: 200 rooms in under a
      second where `explore` refuses. Unique culprit and difficulty are the
      generator's, still to come. [#108](../../issues/108)
- [x] **Wordings from strangers** - `tools/wordings.py` asks a model for
      wordings in character and writes a review file. Playtest telemetry is
      the other half. [#109](../../issues/109)
- [ ] **Paperwork by the ton** - a document generator per department, and the
      Legacy as a Wikipedia subset. [#110](../../issues/110)

## Content

- [ ] **The Fair Play Case Generator** - victim, closed circle, motive, means,
      one planted contradiction, and a solver that refuses the unfair.
      [#112](../../issues/112)

## How it stays honest

The habits already in the repository, kept:

- Every number beside a trivial baseline (`data/baseline.py`, the guess
  column in `data/silo/questions.py`).
- Every world checked before it is emitted, and every mystery solved by the
  state search before anybody plays it (`libworld.World.check`, `explore`).
- Every prose change seen, not asserted (`tools/transcript.py`).
- Every measurement paired over seeds when it is about the classifier
  (`tools/class_cost.py`), because three seeds could not find an 18-point
  effect.
- Hardware before a release, because CI cannot see MOS (`tools/mostest.py`).

And one to add: **the players are the dataset.** Unmatched input from a
playtest is the next dozen wordings, and it comes from strangers.
