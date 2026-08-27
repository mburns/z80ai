"""Things in the archive that are not true, and a key saying which.

    python data/silo/generate.py --plant 12

A corpus where everything is consistent is a phone book. Every query returns a
fact, no fact is more interesting than another, and the only reason to ask a
second question is that you wanted to know a second thing. That is the right
shape for measuring a machine and the wrong shape for using one.

So this plants a fixed number of contradictions and writes down exactly what it
planted. **Off by default**: every number in `data/silo/README.md` was measured
on a corpus with none of this in it, and a flag that quietly changed the data
under a measurement would be worse than no flag.

## The detector already existed

Each kind inverts one of the invariants `tests/test_silo.py` asserts. That is
not a coincidence, it is the design: a corpus is interesting exactly where it
violates something a reader would assume, and the assumptions were already
written down as tests.

    impossible_father   a father who died before his child was born
    purge               a committee whose members were all sent to clean
    altered_parentage   the fact table and the graph name different fathers

The last is the one worth having. `libgraph` walks edges and the `person` view
reads facts, and until now they could not disagree - they are written from one
pass over one simulation. Here they do, for a handful of people, which is what
a falsified record looks like from the inside: the card answers confidently,
the database answers differently, and neither is malfunctioning.

## What the machine can and cannot do about it

Nothing, and that is the point. Finding an impossible father means asking when
someone died, asking when their child was born, and noticing - three steps, of
which the machine does two. `data/wikipedia/README.md` has argued from the
beginning that comprehension is out of reach on this hardware; a planted corpus
turns that from a limitation into a division of labour.

The key is a separate file rather than a table, for the obvious reason.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path
    from random import Random

    from generate import World

#: Cleanings that make up a purge, and the window they happen in. Small enough
#: that a reader who lists a committee notices, large enough that it is not one
#: coincidence.
PURGE_SIZE = (4, 7)
PURGE_YEARS = 3


@dataclass(frozen=True)
class Anomaly:
    """One planted contradiction, in enough detail to check it was planted."""

    kind: str
    #: The person or cohort the anomaly is about - where a reader would start.
    subject: str
    #: What is wrong, in a sentence.
    detail: str
    #: Everyone else it touches, so a purge lists its victims.
    involves: tuple[str, ...] = ()


def plant(rng: Random, world: World, count: int) -> list[Anomaly]:
    """Plant ``count`` anomalies, in roughly equal parts of each kind."""
    kinds = (_impossible_father, _purge, _altered_parentage)
    out: list[Anomaly] = []
    used: set[str] = set()
    attempts = 0
    while len(out) < count and attempts < count * 40:
        attempts += 1
        made = kinds[len(out) % len(kinds)](rng, world, used)
        if made is not None:
            out.append(made)
    if len(out) < count:
        raise SystemExit(
            f"could only plant {len(out)} of {count} anomalies; the corpus is "
            f"too small for that many - try --people 10000 or fewer --plant")
    return out


def _impossible_father(rng: Random, world: World,
                       used: set[str]) -> Anomaly | None:
    """Move a father's death to before his youngest child was born.

    The youngest, and not one picked at random, because a death date moved
    earlier makes every child born after it impossible too. Aiming at the last
    one and stopping short of the one before leaves **exactly one**
    contradiction, which is what the key claims. The first version of this
    planted three and created ten.

    The dates are the record and nothing else changes. A reader who asks both
    questions has the contradiction; a reader who asks one has a fact.
    """
    for _ in range(80):
        father = rng.choice(world.people)
        # `male`, because this is the *father* anomaly and the detector reads
        # `father_is`. Without the check it happily made a woman's death
        # impossible, recorded it as a father, and the detector found somebody
        # else's father instead - a planted anomaly nobody could confirm.
        if (father.died is None or not father.male or not father.children
                or father.name in used):
            continue
        children = sorted((world.people[i] for i in father.children),
                          key=lambda c: c.born)
        child = children[-1]
        if child.name in used:
            continue
        # Late enough that his other children are still possible, and late
        # enough that he was an adult.
        floor = max([father.born + 16] + [c.born for c in children[:-1]])
        if floor >= child.born:
            continue
        father.died = rng.randint(floor, child.born - 1)
        father.fate = "Natural causes"
        used.update({child.name, father.name})
        return Anomaly(
            "impossible_father", child.name,
            f"{father.name} is recorded as dying in year {father.died}, "
            f"{child.born - father.died} years before {child.name} was born.",
            (father.name,))
    return None


def _purge(rng: Random, world: World, used: set[str]) -> Anomaly | None:
    """Send a committee to clean, within a few years of each other.

    A death is never moved earlier than the victim's last child's birth, or the
    purge quietly manufactures impossible fathers - it made four, on top of the
    three this file meant to plant, and they were indistinguishable from them.

    Two consequences are left in on purpose. A victim who sat on another
    committee thins that one too, so a reader counting cleanings per committee
    finds more clusters than were planted; and the key names the committee that
    was purged rather than every committee that looks thin. That is what a
    purge does to a small population, and a corpus where only the planted thing
    is findable would teach a player to stop looking.
    """
    committees = sorted({name for name, group in world.cohorts.items()
                         if group.kind == "committee"})
    for _ in range(40):
        committee = rng.choice(committees)
        if committee in used:
            continue
        sitting = [p for p in world.people
                   if any(seat == committee for seat, _, _ in p.seats)
                   and p.died is not None and p.name not in used]
        if len(sitting) < PURGE_SIZE[0]:
            continue
        victims = rng.sample(sitting, min(len(sitting), rng.randint(*PURGE_SIZE)))
        year = max(v.born + 25 for v in victims)
        for victim in victims:
            last_child = max((world.people[i].born for i in victim.children),
                             default=victim.born)
            victim.died = max(year + rng.randint(0, PURGE_YEARS),
                              victim.born + 25, last_child + 1)
            victim.fate = "Cleaning"
        used.add(committee)
        used.update(v.name for v in victims)
        return Anomaly(
            "purge", committee,
            f"{len(victims)} of the {committee} were sent to clean between "
            f"years {min(v.died or 0 for v in victims)} and "
            f"{max(v.died or 0 for v in victims)}.",
            tuple(sorted(v.name for v in victims)))
    return None


def _altered_parentage(rng: Random, world: World,
                       used: set[str]) -> Anomaly | None:
    """Make the fact table name a different father from the graph.

    Everywhere else in this corpus the two are written from one pass and cannot
    disagree. Here the *record* says one man and the *relations* say another,
    which is what somebody changing an entry looks like afterwards.
    """
    for _ in range(60):
        child = rng.choice(world.people)
        if child.father is None or child.name in used:
            continue
        real = world.people[child.father]
        stand_in = rng.choice(world.people)
        if (stand_in.name in {real.name, child.name} or not stand_in.male
                or stand_in.name in used
                or stand_in.born > child.born - 16):
            continue
        child.recorded_father = stand_in.name
        used.update({child.name, real.name, stand_in.name})
        return Anomaly(
            "altered_parentage", child.name,
            f"The record gives {child.name}'s father as {stand_in.name}; "
            f"the relations still lead to {real.name}.",
            (real.name, stand_in.name))
    return None


def write_key(path: Path, anomalies: list[Anomaly], seed: int) -> None:
    """The answers, beside the database rather than inside it."""
    path.write_text(json.dumps(
        {"seed": seed, "planted": len(anomalies),
         "anomalies": [asdict(a) for a in anomalies]}, indent=2) + "\n")
