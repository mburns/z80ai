"""
Training-data handling: parsing, splitting, and the checks worth running first.

Deliberately free of torch, so the test suite and `data/lint.py` can use it
without pulling in a training dependency - the same reason `libinfer.py` is
NumPy-only.  `feedme.py` imports from here rather than keeping its own copies.
"""

from __future__ import annotations

import gzip
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

Pair = tuple[str, str]

#: Longest query and response feedme will keep. Both are truncated, not
#: dropped, so data past these limits silently changes meaning.
MAX_QUERY_LEN = 60
MAX_RESPONSE_LEN = 50


def parse_pair(line: str) -> Pair | None:
    """Parse ``query|response`` into an uppercased pair, or None if unusable.

    Blank lines and ``#`` comments are skipped.  Comments only survived before
    because none of them happened to contain a pipe; a comment that did would
    have become a training pair.
    """
    line = line.strip()
    if not line or line.startswith('#') or '|' not in line:
        return None

    query, response = line.split('|', 1)
    query = query.strip().upper()
    response = response.strip().upper()

    if len(query) < 2 or len(response) < 1:
        return None

    if len(query) > MAX_QUERY_LEN:
        head = query[:MAX_QUERY_LEN]
        query = head.rsplit(' ', 1)[0] if ' ' in query[40:MAX_QUERY_LEN] else head
    if len(response) > MAX_RESPONSE_LEN:
        head = response[:MAX_RESPONSE_LEN]
        response = head.rsplit(' ', 1)[0] if ' ' in response[30:MAX_RESPONSE_LEN] else head
    return query, response


def load_pairs(lines: Iterable[str], limit: int = 0) -> list[Pair]:
    """Parse every line, stopping after ``limit`` pairs (0 = no limit)."""
    pairs: list[Pair] = []
    for line in lines:
        pair = parse_pair(line)
        if pair:
            pairs.append(pair)
            if limit and len(pairs) >= limit:
                break
    return pairs


def read_files(paths: Sequence[str]) -> list[Pair]:
    """Read pairs from files (``.gz`` transparently) or stdin when empty."""
    if not paths:
        return load_pairs(sys.stdin)
    pairs: list[Pair] = []
    for path in paths:
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as fh:
            pairs.extend(load_pairs(fh))
    return pairs


def build_charset(pairs: Sequence[Pair]) -> str:
    """Every character appearing in a response, which is what feedme encodes.

    Queries never contribute: they are hashed into buckets, not spelled out.
    """
    return ''.join(sorted({c for _, r in pairs for c in r}))


def split_pairs(pairs: Sequence[Pair], val_frac: float = 0.1,
                seed: int = 0) -> tuple[list[Pair], list[Pair]]:
    """Hold out a fraction of the *unique queries* for validation.

    Splitting by line instead would put the same query on both sides - 35% of
    the shipped guess data is exact duplicate pairs - and the validation score
    would be the training score under another name.

    Templated paraphrases of a held-out query ("can it cut you" / "can this cut
    you") do still land in the training half, so this is an optimistic estimate
    of generalization.  It is a much better one than measuring on the training
    set, which is what happened before.
    """
    if val_frac <= 0:
        return list(pairs), []
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")

    queries = sorted({q for q, _ in pairs})  # sorted, then seeded: reproducible
    random.Random(seed).shuffle(queries)
    held_out = set(queries[:int(len(queries) * val_frac)])

    train = [p for p in pairs if p[0] not in held_out]
    val = [p for p in pairs if p[0] in held_out]
    return train, val


def accuracy_ceiling(pairs: Sequence[Pair]) -> float:
    """Best pair-level accuracy any model could reach, given contradictions.

    A query that appears with two different responses cannot be got right both
    times.  Worth printing beside the accuracy: otherwise a model that has
    learned everything learnable looks like it is underperforming, and the fix
    looks like "train longer" when it is "fix the data".
    """
    if not pairs:
        return 1.0
    by_query: dict[str, Counter[str]] = defaultdict(Counter)
    for query, response in pairs:
        by_query[query][response] += 1
    return sum(c.most_common(1)[0][1] for c in by_query.values()) / len(pairs)


def score_predictions(pairs: Sequence[Pair],
                      predict: Callable[[str], str]) -> tuple[float, float]:
    """``(overall, macro)`` accuracy of ``predict`` over ``pairs``.

    Overall weights every pair equally, so a dominant answer inflates it:
    always saying NO scores 58% on guess.  Macro averages over distinct
    answers, where the same guesser scores 25%.  Quote both, and compare like
    with like - a model's per-character score is neither of these.
    """
    if not pairs:
        return 1.0, 1.0
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for query, reply in pairs:
        total[reply] += 1
        correct[reply] += predict(query) == reply
    overall = sum(correct.values()) / len(pairs)
    macro = sum(correct[k] / total[k] for k in total) / len(total)
    return overall, macro


@dataclass
class Steadiness:
    """How much of a classifier's answer is the question, and how much is the
    subject it was asked about."""

    #: questions asked, and how many got the label their phrasing means
    asked: int = 0
    right: int = 0
    #: phrasings where every subject got the same answer, right or wrong
    steady: int = 0
    phrasings: int = 0
    #: (label, phrasing, share right, the label it mostly gave instead)
    worst: list[tuple[str, str, float, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.right / self.asked if self.asked else 1.0

    @property
    def steadiness(self) -> float:
        return self.steady / self.phrasings if self.phrasings else 1.0


def name_sensitivity(templates: Mapping[str, Sequence[str]],
                     subjects: Callable[[str, int], Sequence[str]],
                     predict: Callable[[str], str],
                     per_template: int = 40) -> Steadiness:
    """Ask one phrasing about many subjects, and see whether the answer moves.

    Accuracy over a question set cannot separate two failures that need
    different fixes: a phrasing the model never learned, and a phrasing it did
    learn whose answer depends on *who* is being asked about. The second is
    invisible in a per-question score and is the one the encoder causes - a
    query is hashed into 128 trigram buckets and a name is most of a short
    question, so the subject is not something the model steps over on its way
    to the verb.

    ``templates`` maps a label to phrasings containing ``{s}``; ``subjects``
    returns names to substitute for a given label, so a question is only ever
    asked about somebody it makes sense for.

    `steady` counts phrasings where changing only the name never changes the
    answer. It says whether a model can be relied on for a question it has
    already been taught, which accuracy does not.
    """
    out = Steadiness()
    for label, phrasings in templates.items():
        names = subjects(label, per_template * len(phrasings))
        for i, phrasing in enumerate(phrasings):
            block = names[i * per_template:(i + 1) * per_template]
            if not block:
                continue
            got = [predict(phrasing.format(s=name)) for name in block]
            hits = sum(1 for g in got if g.lower() == label.lower())
            out.asked += len(got)
            out.right += hits
            out.phrasings += 1
            out.steady += len(set(got)) == 1
            if hits < len(got):
                instead = Counter(g.lower() for g in got if g.lower() != label.lower())
                out.worst.append((label, phrasing, hits / len(got),
                                  instead.most_common(1)[0][0]))
    out.worst.sort(key=lambda row: row[2])
    return out


def contradictions(pairs: Sequence[Pair]) -> dict[str, set[str]]:
    """Queries that appear with more than one response, and what they map to."""
    by_query: dict[str, set[str]] = defaultdict(set)
    for query, response in pairs:
        by_query[query].add(response)
    return {q: rs for q, rs in by_query.items() if len(rs) > 1}
