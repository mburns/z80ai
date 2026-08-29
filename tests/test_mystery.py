"""People to ask, questions that are noticed, and a game that can be won.

`tests/test_if.py` measures the claim that a *turn* costs no card reads. This
measures the four things built on top of that, and the first assertion here is
the same one: a conversation is tables in the image, so talking to somebody is
as free as walking, and `test_a_conversation_reads_nothing_from_the_card` is
what stops that quietly ceasing to be true.

The rest is the shape `IF.md` argues an unreliable oracle needs:

    C_ASKED       what was asked about is state the rules can read
    HEAT          asking is noticed, so a question costs more than time
    people        somebody to ask, whose answer depends on what you know
    the seal      an archive that declines, consistently and on the record
    explore()     and a proof, at build time, that the thing can be won

The last of those is the one worth being loud about. A fair-play mystery makes
a promise no author can check by reading their own source, and the state space
is where the promise actually lives - so `test_the_mystery_can_be_won` walks
the solver's own answer through the emulator, and the two have to agree.
"""

from __future__ import annotations

import pytest

import buildif
import libworld
import worlds_mystery
from libhost import AgonHost
from libworld import Line, Person, Room, Rule, Thing, Topic, World

# --- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    return worlds_mystery.mystery()


@pytest.fixture(scope="module")
def game(world):
    return buildif.build(world).build()


def play(game, *commands: str) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=[*commands, "quit"], files={})
    return host.run(game, max_cycles=50_000_000), host


def said(out: str, phrase: str) -> bool:
    """Whether the game said this, ignoring where `PRWRAP` broke the lines."""
    return " ".join(phrase.split()) in " ".join(out.split())


#: The two moves that reach Marnes, and the three that reach Walk.
TO_MARNES = ("down",)
TO_WALK = ("down", "down", "east")


# --- the claim, again ---------------------------------------------------------


def test_a_conversation_reads_nothing_from_the_card(game):
    """The reason a person is a table and not a query.

    The card costs ~4,600 bytes of I/O to answer one question. A player talks
    to somebody far more often than they consult an archive, so dialogue that
    went through the card would have made talking the expensive thing in a
    game about asking questions.
    """
    _out, host = play(game, *TO_MARNES, "ask marnes about allison",
                      "ask marnes about badge", "ask marnes about pump")
    assert host.io_bytes == 0


def test_asking_costs_a_fraction_of_a_question(game):
    """Three lookups and a linear scan, against the oracle's ~370,000."""
    _out, host = play(game, *TO_MARNES, "ask marnes about allison")
    assert host.cpu.instructions < 60_000


# --- people -------------------------------------------------------------------


def test_a_person_is_listed_where_they_stand(game):
    out, _ = play(game)
    assert said(out, "Mayor Jahns is here")


def test_a_person_is_not_listed_where_they_do_not_stand(game):
    # After the move, not before it: the opening description is the landing,
    # where Jahns is, and that line is supposed to be in the transcript.
    out, _ = play(game, *TO_MARNES)
    arrived = out.split("> down", 1)[1]
    assert not said(arrived, "Mayor Jahns is here")
    assert said(arrived, "Deputy Marnes leans on the end of a table")


def test_asking_a_person_gets_their_line(game):
    out, _ = play(game, *TO_MARNES, "ask marnes about allison")
    assert said(out, "'She was IT,' says Marnes")


def test_a_topic_nobody_wrote_a_line_for_gets_the_default(game):
    """The refuse class, in a voice.

    `IF.md` sets out why this matters for a parser and it is the same failure
    here: a person who answered confidently about a topic the author never
    considered is worse than one who deflects, because the player cannot tell
    the two apart and will act on the invention.
    """
    out, _ = play(game, *TO_MARNES, "ask marnes about hatch")
    assert said(out, "I would not want to say so where it could be written")


def test_a_name_nobody_has_is_named_back(game):
    out, _ = play(game, "ask bernard about allison")
    assert said(out, "You do not know anybody called 'BERNARD'.")


def test_a_topic_word_nobody_wrote_is_named_back(game):
    out, _ = play(game, "ask jahns about zorkmid")
    assert said(out, "I do not know the word 'ZORKMID'.")


def test_asking_somebody_who_is_elsewhere_says_so(game):
    out, _ = play(game, "ask marnes about allison")
    assert said(out, "They are not here.")


def test_asking_with_no_name_says_so(game):
    out, _ = play(game, "ask")
    assert said(out, "Who do you want to ask?")


def test_asking_a_person_about_nothing_says_so(game):
    out, _ = play(game, "ask jahns")
    assert said(out, "What do you want to ask about?")


# --- the splitter -------------------------------------------------------------


def test_the_preposition_is_dropped(game):
    """`ASK MARNES ABOUT ALLISON` and `ASK MARNES ALLISON` are one command.

    Three slots is what the program has, and every natural phrasing of the one
    command that names two things puts a word between them. Dropping it in the
    splitter is what stops that costing every other command a fourth slot.
    """
    with_it, _ = play(game, *TO_MARNES, "ask marnes about allison")
    without, _ = play(game, *TO_MARNES, "ask marnes allison")
    assert with_it.split("about allison", 1)[1] == without.split(
        "marnes allison", 1)[1]


def test_an_article_is_dropped_from_an_ordinary_command(game):
    """The same table, paying for itself on a command that predates it."""
    out, _ = play(game, "down", "take the ledger")
    assert "Taken." in out


def test_a_line_of_nothing_but_noise_is_not_a_turn(game):
    out, _ = play(game, "the about a", "down")
    assert "The Cafeteria" in out


# --- gates, which are the whole of conditional dialogue ------------------------


def test_a_gated_line_is_not_spoken_before_its_flag(game):
    out, _ = play(game, *TO_WALK, "ask walk about allison")
    assert said(out, "'Allison who,' says Walk")
    assert not said(out, "She fitted the landing screen")


def test_a_gated_line_is_spoken_once_its_flag_is_set(game):
    out, _ = play(game, *TO_MARNES, "ask marnes about allison",
                  "down", "east", "ask walk about allison")
    assert said(out, "She fitted the landing screen the week before")


def test_a_line_sets_a_flag_a_rule_can_read(game):
    """`sets` is how a conversation teaches the world something.

    `C_ASKED` records that a subject came up. This records that a particular
    person answered it, which is the difference between having raised a name
    and having been told something.
    """
    out, _ = play(game, *TO_MARNES, "ask marnes about allison",
                  "down", "east", "ask walk about allison")
    assert said(out, "A screen fitted by one person, on the landing")


def test_the_conclusion_needs_both_halves(game):
    out, _ = play(game, *TO_WALK, "ask walk about allison")
    assert not said(out, "A screen fitted by one person")


# --- C_ASKED ------------------------------------------------------------------


def test_a_rule_reads_what_was_asked_about(game):
    """The hook the whole thing was built for: the world reacting to a
    question rather than to a movement."""
    out, _ = play(game, "ask jahns about allison")
    assert said(out, "She looks at you for a moment longer than she needs to")


def test_the_same_rule_does_not_fire_on_a_different_topic(game):
    out, _ = play(game, "ask jahns about hatch")
    assert not said(out, "a moment longer than she needs to")


def test_asked_outlives_the_room_it_was_asked_in(game):
    """Asked in the cafeteria, read on the landing. `ASKED` is overlay, so it
    is the only place this can be remembered."""
    out, _ = play(game, "down", "ask marnes about allison", "up")
    assert said(out, "She looks at you for a moment longer")


def test_asking_is_recorded_even_when_the_answer_was_a_deflection(game):
    """A refusal is a thing the player learned, so it counts as having asked.

    Jahns has a line about Allison and it says nothing. The rule keyed on
    `C_ASKED` still fires, because what the world reacts to is the subject
    being raised rather than the subject being answered.
    """
    out, _ = play(game, "ask jahns about allison")
    assert said(out, "That is the whole of the record")
    assert said(out, "a moment longer than she needs to")


# --- attention ----------------------------------------------------------------


def attention_world() -> World:
    """Four rooms down a line, to walk a counter up and back down again."""
    return World(
        rooms=[Room("A", "a", {"DOWN": 1}),
               Room("B", "b", {"UP": 0, "DOWN": 2}),
               Room("C", "c", {"UP": 1, "DOWN": 3}),
               Room("D", "d", {"UP": 2})],
        things=[],
        messages=["SATURATED", "STILL HIGH", "COOLED TO FIVE"],
        rules=[
            Rule(when=[(libworld.C_AT, 1)], then=[(libworld.A_HEAT, 200, 0)]),
            Rule(when=[(libworld.C_AT, 2)], then=[(libworld.A_HEAT, 200, 0)]),
            Rule(when=[(libworld.C_HEAT, 255)],
                 then=[(libworld.A_PRINT, 0, 0)]),
            Rule(when=[(libworld.C_AT, 3)], then=[(libworld.A_COOL, 250, 0)]),
            Rule(when=[(libworld.C_AT, 3), (libworld.C_HEAT, 200)],
                 then=[(libworld.A_PRINT, 1, 0)]),
            Rule(when=[(libworld.C_AT, 3), (libworld.C_HEAT, 5)],
                 then=[(libworld.A_PRINT, 2, 0)]),
        ])


@pytest.fixture(scope="module")
def attention():
    return buildif.build(attention_world()).build()


def test_attention_saturates_rather_than_wrapping(attention):
    """400 is 255, not 144.

    A counter that rolled over would hand the player an escape from every
    consequence by asking enough questions, which is exactly backwards - the
    whole point of it is that it only goes one way without help.
    """
    out, _ = play(attention, "down", "down")
    assert "SATURATED" in out


def test_attention_comes_back_down(attention):
    """There has to be somewhere to lie low, or the counter is a countdown."""
    out, _ = play(attention, "down", "down", "down")
    assert "COOLED TO FIVE" in out
    assert "STILL HIGH" not in out


def test_attention_floors_at_zero():
    """250 off 5 is 0, not 11."""
    world = World(
        rooms=[Room("A", "a", {"DOWN": 1}), Room("B", "b", {"UP": 0})],
        things=[], messages=["NOT ZERO"],
        rules=[Rule(when=[(libworld.C_AT, 1)],
                    then=[(libworld.A_HEAT, 5, 0), (libworld.A_COOL, 250, 0)]),
               Rule(when=[(libworld.C_HEAT, 1)],
                    then=[(libworld.A_PRINT, 0, 0)])])
    out, _ = play(buildif.build(world).build(), "down", "up", "down")
    assert "NOT ZERO" not in out


def test_a_person_can_be_sent_somewhere_else(game):
    """`A_SEND`, and the reason a person is not a `Thing`: they move through
    `PWHERE` rather than `WHERE`, so nothing can pick one up."""
    out, _ = play(game, "ask jahns about allison", "ask jahns about allison",
                  "ask jahns about allison", "look")
    # Two of Jahns's own topic count three attention; the rest comes from
    # nothing else here, so the deputy stays put and the room is unchanged.
    assert not said(out, "There are boots on the stair")


# --- fair play, as a build-time property --------------------------------------


def test_the_mystery_is_solvable(world):
    assert world.explore().solve() is not None


def test_the_mystery_can_be_won(world, game):
    """The solver's own answer, walked through the emulator.

    This is the assertion the state search exists for. `explore` models the
    device - one pass of the rule table a turn, `LOOK` as the turn that lets a
    cascade finish - and the two agreeing is what makes a walkthrough it finds
    a fact about the binary rather than about the model.
    """
    walkthrough = world.explore().solve()
    out, _ = play(game, *walkthrough)
    assert said(out, "A screen fitted by one person, on the landing")


def test_every_authored_line_can_be_reached(world):
    """The practical fair-play bug is not an unwinnable game - that gets
    noticed. It is a clue behind a door that never opens."""
    assert world.explore().unseen() == {
        "rooms": [], "things": [], "messages": [], "lines": [], "people": [],
        "rules": []}


def test_a_goal_nothing_reaches_is_reported():
    """A game that looks finished and cannot be won."""
    world = World(
        rooms=[Room("A", "a")], things=[Thing("key", "k", 0)],
        goal=[(libworld.C_FLAG, 0)])
    assert world.explore().solve() is None


def test_a_clue_behind_a_door_that_never_opens_is_reported():
    """A gated line whose flag nothing can set. Playable, unwinnable, and
    silent about it - which is the failure this whole search is for."""
    world = World(
        rooms=[Room("A", "a")], things=[],
        people=[Person("marnes", "M is here.", 0, "Nothing to say.")],
        topics=[Topic("allison", ["ALLISON"])],
        lines=[Line(0, 0, "The clue.", gate=3)])
    assert world.explore().unseen()["lines"] == ["marnes on allison"]


def test_a_rule_that_can_never_fire_is_reported():
    """The quietest authoring bug there is: no error, no output, nothing to
    notice. One of these was live in `worlds_mystery` until this found it."""
    world = World(
        rooms=[Room("A", "a")], things=[], messages=["never"],
        rules=[Rule(when=[(libworld.C_FLAG, 9)],
                    then=[(libworld.A_PRINT, 0, 0)])])
    assert world.explore().unseen()["rules"] == ["rule 0"]


def test_look_is_a_move_because_rules_are_one_pass():
    """`RULES_RUN` walks the table once, so a rule made true by a later rule
    does not fire until the next turn. The search has to model that or it
    would find walkthroughs the device cannot follow."""
    world = World(
        rooms=[Room("A", "a")], things=[], messages=["second"],
        # Rule 0 is *before* rule 1 in the table and depends on it, so it
        # cannot fire in the same pass that sets the flag.
        rules=[Rule(when=[(libworld.C_FLAG, 0)],
                    then=[(libworld.A_SET, 1, 0)]),
               Rule(when=[(libworld.C_AT, 0)], then=[(libworld.A_SET, 0, 0)])],
        goal=[(libworld.C_FLAG, 1)])
    assert world.explore().solve() == ["look"]


def test_dropping_is_modelled_where_a_rule_can_see_it():
    """The reduction that makes the search finish, and its boundary.

    Dropping can only ever *open* something through `C_HERE` - `C_HAVE` going
    false and `C_CARRYING` falling can each only stop a rule firing. So a
    world with a `C_HERE` models drops and one without does not.
    """
    def build_world(condition):
        return World(rooms=[Room("A", "a", {"DOWN": 1}), Room("B", "b")],
                     things=[Thing("key", "k", 0)], messages=["m"],
                     rules=[Rule(when=[condition],
                                 then=[(libworld.A_PRINT, 0, 0)])])

    assert build_world((libworld.C_HERE, 0)).droppable() == {0}
    assert build_world((libworld.C_HAVE, 0)).droppable() == set()


def test_a_puzzle_that_needs_putting_something_down_is_solvable():
    """The `C_HERE` case end to end, so the reduction is checked by a world
    that depends on it rather than only by the property it rests on."""
    world = World(
        rooms=[Room("A", "a", {"DOWN": 1}), Room("B", "b", {"UP": 0})],
        things=[Thing("wrench", "w", 0)], messages=["The wrench fits."],
        rules=[Rule(when=[(libworld.C_AT, 1), (libworld.C_HERE, 0)],
                    then=[(libworld.A_PRINT, 0, 0),
                          (libworld.A_SET, 0, 0)])],
        goal=[(libworld.C_FLAG, 0)])
    assert world.explore().solve() == ["take wrench", "down", "drop wrench"]


# --- the card, standing in a room ---------------------------------------------
#
# The other half of `C_ASKED`, and the half that needed a hook rather than a
# table: a question put to the archive has to become state before the answer
# is printed, or the two programs are two programs that happen to share
# `INPBUF`.

#: A card small enough to build in a fixture and large enough to hold the
#: three records `worlds_mystery` names.
TITLES = ["Cleaning Record 218-04",
          "Incident Report 214-11: Cistern Pump Failure",
          "Standing Order 11: Screen Fitting"]
LEADS = ["Allison Becker, IT, Level 34. Sent to clean on the fourth day.",
         "The cistern pump on Level 142 stopped without warning.",
         "A screen is fitted by two people and never by one."]


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """The oracle program carrying the mystery, over a three-article card."""
    import buildwikibin
    import libsearch

    out = tmp_path_factory.mktemp("mystery")
    index = libsearch.build(TITLES, LEADS, {})
    libsearch.write_index(index, out / "W.IDX")
    libsearch.write_text(index, out / "W.DAT")

    world = worlds_mystery.mystery()
    libworld.resolve_topics(world, TITLES)
    builder = buildwikibin.build(index.num_docs, index_name="W.IDX",
                                 text_name="W.DAT", world=world)
    return builder.build(), {"W.IDX": (out / "W.IDX").read_bytes(),
                             "W.DAT": (out / "W.DAT").read_bytes()}


def visit(merged, *commands: str) -> str:
    game, files = merged
    host = AgonHost(stdin=[*commands, "!"], files=files)
    return host.run(game, max_cycles=2_000_000_000)


#: Down to the IT office and sat at the screen.
AT_SCREEN = ("down", "down", "east", "use")


def test_an_open_record_is_printed(merged):
    out = visit(merged, *AT_SCREEN, "pump")
    assert "Incident Report 214-11" in out
    assert said(out, "The cistern pump on Level 142 stopped without warning.")


def test_a_sealed_record_prints_the_seal_instead(merged):
    """The archive declining is the archive being *consistent* about what it
    will not say, which is what makes it a clue rather than noise.
    `data/silo/plant.py` set out the principle for the corpus and this is the
    same one at the terminal."""
    out = visit(merged, *AT_SCREEN, "allison")
    assert said(out, "RECORD SEALED BY ORDER OF JUDICIAL")
    assert "Allison Becker" not in out


def test_a_question_to_the_card_is_recorded_as_asked(merged):
    """The hook. Consulting the archive about her and then standing in front
    of the mayor fires a rule that neither the card nor the map could have
    reached on its own."""
    out = visit(merged, *AT_SCREEN, "allison", "leave", "west", "up", "up")
    assert said(out, "She looks at you for a moment longer than she needs to")


def test_a_sealed_record_still_counts_as_having_asked(merged):
    """It printed a refusal, and the refusal is the thing that was learned."""
    out = visit(merged, *AT_SCREEN, "allison", "leave", "west", "up", "up")
    assert said(out, "RECORD SEALED")
    assert said(out, "a moment longer than she needs to")


def test_asking_the_card_costs_attention(merged):
    """Three for the sealed record and two for raising it in front of Jahns
    is five, which is what the deputy was watching for. Neither source
    reaches it alone, which is the point of a counter rather than a flag."""
    out = visit(merged, *AT_SCREEN, "allison", "leave", "west", "up", "up")
    assert said(out, "There are boots on the stair below you")


def test_a_quiet_question_costs_nothing(merged):
    """The pump is not sealed and not watched, so reading it is free. An
    attention counter that went up on every question would be a turn limit
    wearing a costume."""
    out = visit(merged, *AT_SCREEN, "pump", "leave", "west", "up", "up")
    assert not said(out, "There are boots on the stair")


def test_walking_still_reads_nothing_from_the_card(merged):
    """The card is open the whole time, and a move must still not touch it."""
    game, files = merged
    host = AgonHost(stdin=["down", "down", "up", "up", "!"], files=files)
    host.run(game, max_cycles=2_000_000_000)
    quiet = AgonHost(stdin=["!"], files=files)
    quiet.run(game, max_cycles=2_000_000_000)
    assert host.io_bytes == quiet.io_bytes


def test_talking_to_somebody_reads_nothing_from_the_card(merged):
    """The same claim for the verb this change added. A conversation is
    tables in the image even when a card is mounted beside them."""
    game, files = merged
    talked = AgonHost(stdin=["down", "ask marnes about allison", "!"],
                      files=files)
    talked.run(game, max_cycles=2_000_000_000)
    quiet = AgonHost(stdin=["!"], files=files)
    quiet.run(game, max_cycles=2_000_000_000)
    assert talked.io_bytes == quiet.io_bytes


def test_holding_up_a_piece_of_paper_is_noticed_too(merged):
    """`CONSULT <thing>` and `NOTICE` compose without either knowing about
    the other.

    `CONSULT` types the thing's subject into the same `INPBUF` a player types
    into and jumps to `ML_ASK`, which is where the scan and then `NOTICE`
    already were. So a badge held up to the screen reaches the sealed record
    by the same path the word `allison` does, and is marked and charged the
    same way. That is the join working rather than two features being wired
    to each other.
    """
    out = visit(merged, "down", "down", "take badge", "east", "use", "leave",
                "consult badge")
    assert said(out, "RECORD SEALED BY ORDER OF JUDICIAL")


def test_a_subject_that_is_not_sealed_reads_normally(merged):
    out = visit(merged, "down", "down", "down", "down", "take wrench",
                "up", "up", "east", "consult wrench")
    assert "Incident Report 214-11" in out


def test_a_rule_reading_a_flag_only_dialogue_sets_is_not_called_dead():
    """The integration bug this merge actually had.

    `World.reach` over-approximates by dropping every way the world can go
    backwards, and it is sound only if it knows every way a flag can go
    *forwards*. `Line.sets` is a second one, and until `reach` was told about
    it `_check_impossible` refused `worlds_mystery` outright - every rule that
    reads what somebody said was reported as dead code.

    An over-approximation that has not been told about a mechanism does not
    degrade gracefully. It condemns the mechanism.
    """
    world = World(
        rooms=[Room("A", "a")], things=[], messages=["the conclusion"],
        people=[Person("marnes", "M is here.", 0, "Nothing to say.")],
        topics=[Topic("allison", ["ALLISON"])],
        lines=[Line(0, 0, "She was IT.", sets=2)],
        rules=[Rule(when=[(libworld.C_FLAG, 2)],
                    then=[(libworld.A_PRINT, 0, 0)])])
    world.check()                                # would raise if reported dead
    assert world.dead_rules() == []
    assert world.explore().unseen()["rules"] == []


def test_the_two_programs_define_no_label_twice():
    """The hazard is silent: `label` assigns into a dict, so a name defined
    twice resolves to whichever was emitted last and nothing says so. This
    change added `NOTICE`, `WATCH` and five word tables to a binary that
    already had a few hundred labels in it."""
    import collections

    import buildwikibin
    import libez80

    seen: collections.Counter[str] = collections.Counter()
    original = libez80.EZ80Builder.label

    def spy(self, name: str) -> None:
        seen[name] += 1
        original(self, name)

    libez80.EZ80Builder.label = spy
    try:
        world = worlds_mystery.mystery()
        libworld.resolve_topics(world, TITLES)
        buildwikibin.build(600, world=world)
    finally:
        libez80.EZ80Builder.label = original

    assert [name for name, n in seen.items() if n > 1] == []


def test_an_article_the_card_does_not_hold_is_refused():
    """A mistyped title is invisible at play time - the archive still answers
    and the topic simply never marks itself - so it is an error at build."""
    world = worlds_mystery.mystery()
    world.topics[0].titles = ["Cleaning Record 999-99"]
    with pytest.raises(ValueError, match="not on this card"):
        libworld.resolve_topics(world, TITLES)


# --- what the tables refuse, before anything is emitted -----------------------


def small(**kwargs) -> World:
    return World(rooms=[Room("A", "a")], things=[], **kwargs)


def one_person(**kwargs) -> World:
    return small(people=[Person("marnes", "M.", 0, "Nothing.")],
                 topics=[Topic("allison", ["ALLISON"])], **kwargs)


def test_a_person_standing_in_a_room_that_does_not_exist_is_refused():
    world = small(people=[Person("marnes", "M.", 9, "Nothing.")])
    with pytest.raises(ValueError, match="does not exist"):
        world.check()


def test_two_people_sharing_a_name_are_refused():
    world = small(people=[Person("marnes", "M.", 0, "x"),
                          Person("MARNES", "M.", 0, "x")])
    with pytest.raises(ValueError, match="two people are called"):
        world.check()


def test_a_word_that_is_both_a_person_and_a_thing_is_refused():
    """Not ambiguous to the machine - `TAKE` and `ASK` read different tables -
    and entirely ambiguous to the player, who cannot see the tables."""
    world = World(rooms=[Room("A", "a")], things=[Thing("marnes", "m", 0)],
                  people=[Person("marnes", "M.", 0, "x")])
    with pytest.raises(ValueError, match="both a person and a thing"):
        world.check()


def test_a_person_with_no_default_line_is_refused():
    """Every person needs the refuse class, so it cannot be left out."""
    world = small(people=[Person("marnes", "M.", 0, "")])
    with pytest.raises(ValueError, match="no default line"):
        world.check()


def test_a_topic_no_word_reaches_is_refused():
    world = small(topics=[Topic("allison", [])])
    with pytest.raises(ValueError, match="has no words"):
        world.check()


def test_two_topics_claiming_one_word_are_refused():
    world = small(topics=[Topic("allison", ["SHE"]),
                          Topic("jahns", ["she"])])
    with pytest.raises(ValueError, match="resolves to one topic"):
        world.check()


def test_a_line_spoken_by_nobody_is_refused():
    world = one_person(lines=[Line(4, 0, "x")])
    with pytest.raises(ValueError, match="who does not exist"):
        world.check()


def test_a_line_about_no_topic_is_refused():
    world = one_person(lines=[Line(0, 7, "x")])
    with pytest.raises(ValueError, match="topic 7"):
        world.check()


def test_a_line_that_repeats_another_is_refused():
    """The scan takes the first, so the second is text nobody can hear."""
    world = one_person(lines=[Line(0, 0, "one"), Line(0, 0, "two")])
    with pytest.raises(ValueError, match="can never be spoken"):
        world.check()


def test_an_ungated_line_above_a_gated_one_is_refused():
    """The mistake that actually gets made: the fallback written first, so it
    always wins and the specific line never runs."""
    world = one_person(lines=[Line(0, 0, "fallback"),
                              Line(0, 0, "specific", gate=1)])
    with pytest.raises(ValueError, match="always wins"):
        world.check()


def test_a_line_gating_on_a_flag_the_world_does_not_reserve_is_refused():
    world = one_person(flags=2, lines=[Line(0, 0, "x", gate=9)])
    with pytest.raises(ValueError, match="gates flag 9"):
        world.check()


def test_a_rule_asking_about_a_topic_that_does_not_exist_is_refused():
    world = one_person(rules=[Rule(when=[(libworld.C_ASKED, 5)], then=[])])
    with pytest.raises(ValueError, match="ASKED 5"):
        world.check()


def test_a_rule_naming_a_person_who_does_not_exist_is_refused():
    world = one_person(rules=[Rule(when=[(libworld.C_WITH, 5)], then=[])])
    with pytest.raises(ValueError, match="WITH 5"):
        world.check()


def test_sending_a_person_to_a_room_that_does_not_exist_is_refused():
    world = one_person(rules=[
        Rule(when=[(libworld.C_AT, 0)],
             then=[(libworld.A_SEND, 0, 9)])])
    with pytest.raises(ValueError, match="SEND to room 9"):
        world.check()


def test_watching_for_no_attention_at_all_is_refused():
    """`HEAT 0` holds on the first turn and every turn, which is the same
    mistake as a rule with no conditions and is never what was meant."""
    world = small(rules=[Rule(when=[(libworld.C_HEAT, 0)], then=[])])
    with pytest.raises(ValueError, match="HEAT 0 always holds"):
        world.check()


def test_a_goal_naming_something_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="the goal: HAVE 3"):
        small(goal=[(libworld.C_HAVE, 3)]).check()


# --- the overlay --------------------------------------------------------------


def test_what_was_asked_is_part_of_the_saved_game(world):
    """`ASKED` and `HEAT` are inside the contiguous run, not beside it.

    A restore that put the player back somewhere but forgot what they had been
    told would re-explain everything and fire every `C_ASKED` rule a second
    time - a worse bug than losing the save, because it looks like the game
    working.
    """
    builder = buildif.build(world)
    start, length = buildif.overlay_at(builder, world)
    assert length == world.overlay_bytes
    assert start == builder.labels["HERE"]
    assert start < builder.labels["ASKED"] < start + length
    assert start < builder.labels["HEAT"] < start + length


def test_the_whole_saved_game_is_still_under_a_hundred_bytes(world):
    """Four people, five topics, eight lines and sixty-four flags."""
    assert world.overlay_bytes < 100


def test_nothing_clears_what_has_been_asked():
    """`ASKED` is the one monotone thing in the overlay, and it is monotone
    because there is no opcode that could undo it - not because no rule does.
    A mystery whose record of what the player had been told could be rewound
    would not be fair."""
    assert "ASKED" not in libworld.ACTION_NAMES.values()
    assert not any(name.startswith("UNASK")
                   for name in libworld.ACTION_NAMES.values())
