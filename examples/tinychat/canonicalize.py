#!/usr/bin/env python3
"""
Collapse tinychat's 502 replies onto a vocabulary a 2-bit model can learn.

The hand-written corpus in source-data.txt.gz is good: 2,990 queries in a
consistent voice.  The replies are the problem.  502 distinct ones, 297 of them
used exactly once, is not a vocabulary - it is 502 classes with an average of
six examples each, and the model memorises rather than learns (10.7% macro).

Almost all of the tail is *style*, not meaning: YA / YUP / YE / BET / VALID are
one answer wearing five hats, and so are WAT / WUT / HUH? / ???.  So this maps
rather than deletes.  Every query survives; only the reply it maps to changes.

    python examples/tinychat/canonicalize.py > out.txt
    python data/lint.py out.txt --strict
    python examples/tinychat/canonicalize.py --unmapped   # audit the mapping

Judgement calls are inevitable here and the point is that they are reviewable:
the mapping is data, not a regenerated blob.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libdata import parse_pair

SOURCE = Path(__file__).with_name('source-data.txt.gz')

#: The intermediate vocabulary: what the corpus distinguishes, before asking
#: whether it has the evidence to support the distinction.
FINE = [
    'OK', 'YES', 'NO', 'MAYBE', 'IDK',          # the core four plus a hedge
    'WHAT?', 'WHY?', 'HOW?', 'WHO?',            # asking back
    'IS IT?', 'R U?', 'DO U?', 'AM I?',         # reflecting it at you
    'HI', 'BYE',                                # bookends
    'GOOD', 'NICE', 'SAME', 'OOF', 'LOL',       # reactions
    'GO ON',                                    # a prompt for more
]

#: ...and the further collapse to what 1,817 training pairs can actually carry.
#: Measured on held-out real queries, holding everything else fixed:
#:
#:      replies   per class   overall   macro
#:           21          87     29.9%   16.8%
#:           16         114     28.9%   20.4%
#:           12         151     31.8%   27.0%
#:           11         165     37.8%   41.8%
#:            8         227     39.3%   32.7%
#:
#: Eleven is the turn. Below it overall keeps creeping up but only because the
#: majority class is growing - by eight replies a constant guesser already
#: scores 25.9% - and macro falls away again.
MERGE = {
    'SAME': 'OK', 'LOL': 'OK', 'GOOD': 'OK', 'NICE': 'OK', 'OOF': 'OK',
    'HOW?': 'WHAT?', 'WHO?': 'WHAT?', 'GO ON': 'WHAT?',
    'AM I?': 'R U?',
    'IS IT?': 'DO U?',
}

VOCAB = [r for r in FINE if r not in MERGE]

#: Two-character queries carry almost no trigram signal, and several of them
#: hash to the same bucket vector while wanting different replies - '??' and
#: 'GG' and 'KK' are indistinguishable to a 128-bucket encoder.
MIN_QUERY_LEN = 3

#: Anything matching one of these patterns becomes that reply.  Order matters:
#: the first match wins, so put the specific before the general.  Patterns are
#: matched against the whole reply, uppercased.
RULES: list[tuple[str, str]] = [
    # "Are you a bot" gets a plain yes: a BOT reply of its own drew only ten
    # surviving examples, which is not enough to learn.
    (r'\b(BOT|HUMAN|ECHO)\b', 'YES'),
    # Thanks. Its own reply would be the charset's only X, and one output
    # neuron is a poor trade for sixteen examples.
    (r'\b(THX|TY|NP|THANK)', 'OK'),
    # Sympathy and commiseration.
    (r'\b(OOF|RIP|SRY|SORRY|SOZ|AW|MOOD|BRR|EW|SUS)', 'OOF'),
    # Amusement.
    (r'\b(LOL|HAHA|HEH|HA|LMAO|JK|AYY|POG|HAH)\b', 'LOL'),
    # Agreement with a shared feeling.
    (r'\b(SAME|IKR|ME 2|U 2|U2|MOOD)', 'SAME'),
    # Approval.
    (r'\b(NICE|NEAT|NOICE|SICK|DOPE|LIT|VALID|COOL|CUTE|GG|GJ|BASED|WOW|YAY)',
     'NICE'),
    (r'\b(GOOD|GREAT|FINE|WARM|YUM|BET|EZ|WAY|MOST|GL)\b', 'GOOD'),
    # Greetings and farewells.
    (r'\b(HI|HEY|HELLO|YO|MORNING|AYO)\b', 'HI'),
    (r'\b(BYE|L8R|LATER|TTYL|BAI|CYA|WB|BED|NAP|REST|END|GO BED)', 'BYE'),
    # Not knowing, before the question rules - "IDK WAT" is a shrug, not a
    # question.
    (r'\b(IDK|NVM|WELP|LOST|NOWHERE|TBH|NGL|IMO|I MEAN|I FEEL|HM)', 'IDK'),
    # Hedging.
    (r'\b(MAYBE|PROB|POSS|KINDA|SORTA|MIGHT|COULD|ALMOST|EITHER|IF|I HOPE|'
     r'I TRY|GONNA)', 'MAYBE'),
    # Asking back - specific question words first.
    (r'\b(WHY|Y THO|CUZ|BECAUSE|BC|REASONS)', 'WHY?'),
    (r'\bHOW\b|HOW MANY|HOW MUCH', 'HOW?'),
    (r'\b(WHO|WHOS)\b', 'WHO?'),
    (r'\b(R U|SO R U|U R|UR|ARE YOU|U GOOD|U THINK|U GUESS|U DO|U CAN|U WONT|'
     r'U GOT|U PICK|AT YOU)', 'R U?'),
    (r'\b(DO U|DO YOU|DO WE|DID I|CAN YOU|WILL I|DO I|AM I|IS THERE|ISNT IT|'
     r'WAS IT)', 'DO U?'),
    (r'\b(IS IT|THING|OF\?|ON\?|AT\?|FOR\?|OR\?|THO\?|BTW|WITH WHAT|OF YOU|'
     r'US\?)', 'IS IT?'),
    (r'\b(WAT|WUT|WHAT|HUH|\?\?\?|EXPLAIN|SRS|RLY|O RLY|FR|OH|OOH|AH|UM|'
     r'WELL|SOOOO|O\b)', 'WHAT?'),
    # Prompting for more.
    (r'\b(GO ON|MORE|THEN|BUT|LIKE|AND|TALK|WAIT|STOP|CALM|TRY|DO IT|LETS GO|'
     r'GO GO|SO\?|DONE|NOW|BEFORE)', 'GO ON'),
    # Everything that is really "is it / are they / does it", asked back.
    (r'\b(ARE|WERE|DOES|DOESNT|DID|HAS|MUST|CHANGE|DEFINE|TELL)\b', 'DO U?'),
    (r'\b(ABOUT|THAN|WHICH|WITH|LESS|OR ELSE|TOO MUCH|WHERE|WHEN|SOON|'
     r'EVENING|VERY|PART)', 'IS IT?'),
    (r'\b(ME|YOU|U)\b\??$', 'R U?'),
    # Quantities and frequencies: hedges in disguise.
    (r'\b(ALL|SOME|MANY|FEW|NEVER|ALWAYS|SOMETIMES|NEUTRAL|BOTH)\b', 'MAYBE'),
    # Affirm and negate last: they are the biggest nets.
    (r'\b(NO|NAH|NA|DONT|DO NOT|CAN NOT|NO CAP|NO1|NO U|FALSE|BAD)', 'NO'),
    (r'\b(YES|YA|YE|YUP|YEA|YEET|TRU|RIGHT|I DO|I SAID|SURE|SUR|FR THO|'
     r'I AM|IT IS|IT DOES|I CAN|I WOULD|I DID|WE ARE|I THINK SO|HOPE SO|'
     r'BELIEVE IT|IK|YOU ARE|YOU SAY)', 'YES'),
    (r'\b(OK|K|FAIR|ITS OK|WORD|HERE|IM HERE|YOURS|I SEE|NM|MM|VIBE|EAT)',
     'OK'),
    # Anything still ending in a question mark is asking something back.
    (r'\?$', 'WHAT?'),
]

#: Replies that no rule should try to be clever about.  Numbers and the
#: arithmetic block cannot work at all - the model has no arithmetic - and the
#: animal noises are jokes tied to one prompt each, with no second example to
#: learn from.
DROP = re.compile(r'^[\d ]+$|^(MEOW|WOOF|MOO|QUACK|BEEP|BOOP|LA LA|LA LA LA|'
                  r'PIZZA|TEA|GRAY|MAGIC|CODE|TINY|OLD|DAY|LEFT|UP|BRO|'
                  r'NO SHAPE|NO COLOR|NOT LONG|LOTS|PICK ONE|TRY AGAIN)$')


def canonical(reply: str, merge: bool = True) -> str | None:
    """The vocabulary entry for ``reply``, or None if it should be dropped.

    ``merge=False`` stops at the 21-reply intermediate, which is what the
    ``--fine`` flag reports and what the sweep above was measured against.
    """
    if DROP.match(reply):
        return None
    fine = reply if reply in FINE else None
    if fine is None:
        for pattern, target in RULES:
            if re.search(pattern, reply):
                fine = target
                break
    if fine is None:
        return None
    return MERGE.get(fine, fine) if merge else fine


def load_source() -> list[tuple[str, str]]:
    with gzip.open(SOURCE, 'rt') as fh:
        return [p for p in (parse_pair(line) for line in fh) if p]


def build() -> tuple[list[tuple[str, str]], Counter, Counter, dict[str, int]]:
    """Returns (pairs, dropped replies, unmapped replies, a tally)."""
    pairs: list[tuple[str, str]] = []
    dropped: Counter = Counter()
    unmapped: Counter = Counter()

    too_short = 0
    for query, reply in load_source():
        if len(query) < MIN_QUERY_LEN:
            too_short += 1
            continue
        mapped = canonical(reply)
        if mapped is None:
            (dropped if DROP.match(reply) else unmapped)[reply] += 1
            continue
        pairs.append((query, mapped))

    # A query mapping to two different replies is unlearnable, and collapsing
    # the vocabulary creates some that were not there before. Keep the majority
    # answer; drop the query entirely if it is a genuine tie.
    by_query: dict[str, Counter] = {}
    for query, reply in pairs:
        by_query.setdefault(query, Counter())[reply] += 1

    resolved = []
    ties = 0
    for query, counter in sorted(by_query.items()):     # sorted: reproducible
        (best, n), *rest = counter.most_common()
        if rest and rest[0][1] == n:
            ties += 1                                   # tie: no right answer
            continue
        resolved.append((query, best))

    tally = {
        'source': len(load_source()),
        'too_short': too_short,
        'unlearnable': sum(dropped.values()),
        'unmapped': sum(unmapped.values()),
        'unique_queries': len(by_query),
        'duplicates': len(pairs) - len(by_query),
        'ties': ties,
        'kept': len(resolved),
    }
    return resolved, dropped, unmapped, tally


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--unmapped', action='store_true',
                        help='List replies no rule matched, and exit')
    args = parser.parse_args()

    pairs, _dropped, unmapped, tally = build()

    if args.unmapped:
        print(f"# {len(unmapped)} replies matched no rule "
              f"({sum(unmapped.values())} pairs)", file=sys.stderr)
        for reply, n in unmapped.most_common():
            print(f"{n:5}  {reply}")
        return

    print("# tinychat, canonicalized.")
    print("# Generated by examples/tinychat/canonicalize.py - do not edit by hand.")
    print(f"# {tally['source']} source pairs over 502 replies ->"
          f" {tally['kept']} over {len(VOCAB)}.")
    print(f"#   {tally['too_short']:4} dropped, query under "
          f"{MIN_QUERY_LEN} characters")
    print(f"#   {tally['unlearnable']:4} dropped, cannot be learned "
          f"(arithmetic, one-off jokes)")
    print(f"#   {tally['unmapped']:4} dropped, no rule matched")
    print(f"#   {tally['duplicates']:4} collapsed as duplicate queries")
    print(f"#   {tally['ties']:4} dropped, the same query wanting two replies "
          f"equally often")
    for query, reply in pairs:
        print(f"{query}|{reply}")


if __name__ == '__main__':
    main()
