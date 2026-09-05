"""
A walkthrough for a world too large to search, built backwards and checked
forwards.

    python -c "import libplan, worlds_mystery; print(libplan.plan(worlds_mystery.mystery()))"

`World.explore` is exact and pays an exponential search for it: 92,064
states for six rooms, and a cap at 200,000 that the whole silo passes in
the first corridor. A fair-play mystery on 187 rooms still has to be
proved winnable, and a proof needs a witness - a walkthrough the emulator
can replay - rather than a state count.

This finds one by asking the goal what it needs. A flag needs the rule or
the line that sets it; a rule needs its conditions; a line needs the person,
their room, and the gate; a thing needs the room it lies in and a `take`;
a room needs a route; an accusation needs the culprit. Each need becomes a
few commands, and **every command is stepped through the exact model** -
`World.step` is the same transition `explore` follows - so what comes out
is not a plan that ought to work but a sequence that did.

## What it is and is not

It is sound: a walkthrough it returns reaches the goal in the model, and
`test_plan.py` replays it through the emulator to hold the model to the
device. It is not complete: a goal that needs something this cannot
express - a flag only an `A_CLEAR` can produce, attention that only a
sealed record raises - comes back as `None` with the need it could not
meet, which is a report and not a verdict. `explore` is still the verdict
where it fits, and this is the instrument where it does not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import libworld
from libworld import CARRIED, NONE, World, _State


class Unplannable(Exception):
    """A need the planner could not meet, and which."""


@dataclass
class Plan:
    """The walkthrough, and the state it reaches."""

    world: World
    state: _State
    commands: list[str] = field(default_factory=list)
    #: Flags being sought right now, to stop a rule that needs the flag it
    #: sets from recursing forever.
    seeking: set[tuple[int, int]] = field(default_factory=set)

    def do(self, command: str) -> None:
        after = self.world.step(self.state, command)
        if after is None:
            raise Unplannable(f"{command!r} is not a legal turn at "
                              f"{self.world.rooms[self.state.here].name!r}")
        self.state = after
        self.commands.append(command)
        if len(self.commands) > MAX_COMMANDS:
            raise Unplannable("the walkthrough passed "
                              f"{MAX_COMMANDS} commands")


#: A walkthrough longer than this is a loop, not a solution.
MAX_COMMANDS = 2_000


def plan(world: World) -> list[str] | None:
    """A walkthrough that satisfies the goal, or None with nothing found.

    `explain(world)` says what could not be met.
    """
    try:
        return solve(world).commands
    except Unplannable:
        return None


def explain(world: World) -> str:
    """Why `plan` gave up, or an empty string if it did not."""
    try:
        solve(world)
    except Unplannable as why:
        return str(why)
    return ""


def solve(world: World) -> Plan:
    world.check()
    p = Plan(world, world.start_state())
    for op, arg in world.goal:
        _achieve(p, op, arg)
    # A cascade of rules takes a turn each; let it finish before judging.
    for _ in range(len(world.rules) + 1):
        if world.satisfied(p.state, world.goal):
            return p
        _look(p)
    if world.satisfied(p.state, world.goal):
        return p
    missing = [libworld.CONDITION_NAMES[op] + f" {arg}"
               for op, arg in world.goal
               if not world._holds_in(p.state, op, arg)]
    raise Unplannable(f"the goal still needs {', '.join(missing)} after "
                      f"everything the planner could do")


# --- needs ---------------------------------------------------------------------


def _achieve(p: Plan, op: int, arg: int) -> None:
    w = p.world
    if w._holds_in(p.state, op, arg):
        return
    if op == libworld.C_AT:
        _go(p, arg)
    elif op == libworld.C_HAVE:
        _take(p, arg)
    elif op == libworld.C_HERE:
        _bring(p, arg)
    elif op == libworld.C_CARRYING:
        for thing in _portable_nearest(p):
            if w._holds_in(p.state, op, arg):
                return
            _take(p, thing)
        raise Unplannable(f"CARRYING {arg}: not enough can be picked up")
    elif op == libworld.C_FLAG:
        _set_flag(p, arg)
    elif op == libworld.C_NFLAG:
        _clear_flag(p, arg)
    elif op == libworld.C_ASKED:
        _raise_topic(p, arg)
    elif op == libworld.C_HEAT:
        _heat(p, arg)
    elif op == libworld.C_WITH:
        _go(p, p.state.pwhere[arg])
    elif op == libworld.C_TURN:
        while p.state.turn < arg:
            _look(p)
    elif op == libworld.C_LOGGED:
        _log(p, arg)
    elif op in (libworld.C_SEALED, libworld.C_ALTERED):
        action = libworld.A_SEAL if op == libworld.C_SEALED else libworld.A_ALTER
        _fire_rule_doing(p, action, arg, f"{libworld.CONDITION_NAMES[op]} {arg}")
    else:
        raise Unplannable(f"no way to bring about condition {op}")
    if not w._holds_in(p.state, op, arg):
        raise Unplannable(f"{libworld.CONDITION_NAMES[op]} {arg} did not hold "
                          f"after trying to bring it about")


def _look(p: Plan) -> None:
    if p.state.at_terminal:
        p.do("leave")
    else:
        p.do("look")


def _go(p: Plan, room: int) -> None:
    """The shortest route by the exits, stepped one direction at a time."""
    if p.state.at_terminal:
        p.do("leave")
    if p.state.here == room:
        return
    parents: dict[int, tuple[int, str]] = {}
    seen = {p.state.here}
    queue = deque([p.state.here])
    while queue:
        here = queue.popleft()
        if here == room:
            break
        for direction, target in p.world.rooms[here].exits.items():
            if target not in seen:
                seen.add(target)
                parents[target] = (here, direction.lower())
                queue.append(target)
    if room not in parents:
        raise Unplannable(f"no route from {p.world.rooms[p.state.here].name!r} "
                          f"to {p.world.rooms[room].name!r}")
    route: list[str] = []
    at = room
    while at != p.state.here:
        at, direction = parents[at]
        route.append(direction)
    for direction in reversed(route):
        p.do(direction)


def _take(p: Plan, thing: int) -> None:
    where = p.state.where[thing]
    if where == CARRIED:
        return
    if not p.world.things[thing].portable:
        raise Unplannable(f"{p.world.things[thing].name!r} cannot be carried")
    if where == libworld.NOWHERE:
        raise Unplannable(f"{p.world.things[thing].name!r} is nowhere")
    _go(p, where)
    p.do(f"take {p.world.things[thing].name}")


def _bring(p: Plan, thing: int) -> None:
    """`C_HERE`: the thing in the room the player is in, which after a
    `C_AT` in the same rule means carrying it here and putting it down."""
    if p.state.where[thing] == p.state.here:
        return
    _take(p, thing)
    if thing in p.world.droppable():
        p.do(f"drop {p.world.things[thing].name}")
    else:
        raise Unplannable(f"HERE {thing}: no rule observes where "
                          f"{p.world.things[thing].name!r} is put down")


def _portable_nearest(p: Plan) -> list[int]:
    return [i for i, t in enumerate(p.world.things)
            if t.portable and p.state.where[i] != CARRIED]


def _set_flag(p: Plan, flag: int) -> None:
    w = p.world
    key = (libworld.C_FLAG, flag)
    if key in p.seeking:
        raise Unplannable(f"flag {flag} needs itself")
    p.seeking.add(key)
    try:
        # The accusation, if this is its flag.
        if w.culprit is not None and flag in (w.won, w.lost):
            _accuse(p, guilty=(flag == w.won))
            return
        # A line of dialogue that sets it: somebody to find, and a gate.
        for line in w.lines:
            if line.sets != flag:
                continue
            if line.gate != NONE:
                _set_flag(p, line.gate)
            person, topic = w.people[line.person], w.topics[line.topic]
            _go(p, p.state.pwhere[line.person])
            p.do(f"ask {person.name} about {topic.words[0].lower()}")
            if p.state.flags[flag]:
                return
        # A rule that sets it: its conditions, then a turn for it to fire.
        _fire_rule_doing(p, libworld.A_SET, flag, f"flag {flag}")
    finally:
        p.seeking.discard(key)


def _fire_rule_doing(p: Plan, action: int, arg: int, what: str) -> None:
    w = p.world
    tried = False
    for number, rule in enumerate(w.rules):
        if not any(op == action and a == arg for op, a, _ in rule.then):
            continue
        if rule.once and p.state.fired[number]:
            continue
        tried = True
        try:
            # Rooms last: a condition that moves the player would undo a
            # `C_AT` met first, and nothing else here moves them.
            ordered = sorted(rule.when, key=lambda c: c[0] == libworld.C_AT)
            for op, a in ordered:
                _achieve(p, op, a)
        except Unplannable:
            continue
        _look(p)
        if _done(p, action, arg):
            return
    raise Unplannable(f"nothing the planner can do sets {what}"
                      + ("" if tried else ": no rule or line sets it"))


def _done(p: Plan, action: int, arg: int) -> bool:
    s = p.state
    if action == libworld.A_SET:
        return bool(s.flags[arg])
    if action == libworld.A_SEAL:
        return bool(s.sealed[arg])
    if action == libworld.A_ALTER:
        return bool(s.altered[arg])
    return False


def _clear_flag(p: Plan, flag: int) -> None:
    if not p.state.flags[flag]:
        return
    for number, rule in enumerate(p.world.rules):
        if any(op == libworld.A_CLEAR and a == flag for op, a, _ in rule.then) \
                and not (rule.once and p.state.fired[number]):
            for op, a in rule.when:
                _achieve(p, op, a)
            _look(p)
            if not p.state.flags[flag]:
                return
    raise Unplannable(f"flag {flag} is set and nothing clears it")


def _raise_topic(p: Plan, topic: int) -> None:
    """Asked of a person if there is one anywhere, else of the archive."""
    w = p.world
    word = w.topics[topic].words[0].lower()
    if w.people:
        person = min(range(len(w.people)),
                     key=lambda i: _distance(p, p.state.pwhere[i]))
        try:
            _go(p, p.state.pwhere[person])
            p.do(f"ask {w.people[person].name} about {word}")
            return
        except Unplannable:
            pass
    _archive(p, word)


def _archive(p: Plan, word: str) -> None:
    if p.world.terminal is None:
        raise Unplannable(f"no terminal to ask about {word!r} at")
    if not p.state.at_terminal:
        _go(p, p.world.terminal)
        p.do("use")
    p.do(f"archive {word}")


def _heat(p: Plan, wanted: int) -> None:
    hot = sorted(((t.heat, i) for i, t in enumerate(p.world.topics) if t.heat),
                 reverse=True)
    if not hot:
        raise Unplannable(f"HEAT {wanted}: no topic costs attention")
    guard = 0
    while p.state.heat < wanted:
        _archive(p, p.world.topics[hot[0][1]].words[0].lower())
        guard += 1
        if guard > 64:
            raise Unplannable(f"HEAT {wanted}: attention stopped climbing")


def _log(p: Plan, wanted: int) -> None:
    if not p.world.topics:
        raise Unplannable(f"LOGGED {wanted}: nothing to ask the archive")
    word = p.world.topics[0].words[0].lower()
    guard = 0
    while p.state.logged < wanted:
        _archive(p, word)
        guard += 1
        if guard > 300:
            raise Unplannable(f"LOGGED {wanted}: the log stopped growing")


def _accuse(p: Plan, guilty: bool) -> None:
    w = p.world
    assert w.culprit is not None
    if p.state.accused:
        raise Unplannable("the accusation has been made already")
    culprit = w.culprit.upper()
    person = next((x for x in w.people if x.name.upper() == culprit), None)
    if guilty:
        if person is not None:
            _go_anywhere(p)
            p.do(f"accuse {person.name.lower()}")
            return
        door = next(d for d in w.doors
                    if d.subject is not None and d.subject.upper() == culprit)
        _go(p, door.room)
        p.do(f"accuse {door.name.lower()}")
        return
    innocent = next((x for x in w.people if x.name.upper() != culprit), None)
    if innocent is not None:
        _go_anywhere(p)
        p.do(f"accuse {innocent.name.lower()}")
        return
    other = next((d for d in w.doors if d.subject is not None
                  and d.subject.upper() != culprit), None)
    if other is None:
        raise Unplannable("nobody innocent to accuse")
    _go(p, other.room)
    p.do(f"accuse {other.name.lower()}")


def _go_anywhere(p: Plan) -> None:
    if p.state.at_terminal:
        p.do("leave")


def _distance(p: Plan, room: int) -> int:
    seen = {p.state.here: 0}
    queue = deque([p.state.here])
    while queue:
        here = queue.popleft()
        if here == room:
            return seen[here]
        for target in p.world.rooms[here].exits.values():
            if target not in seen:
                seen[target] = seen[here] + 1
                queue.append(target)
    return 1 << 20
