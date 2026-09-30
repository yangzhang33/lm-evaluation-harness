"""ilsp/winogrande_greek as a partial-evaluation task, with the translation artifacts filtered out.

ILSP reformulated WinoGrande for Greek: instead of a sentence with a blank plus two fillers, each row carries two
complete Greek sentences (`multiple_choice_targets`) that differ in the substituted entity, and the model must prefer
the plausible one. The two candidates were machine-translated *independently*, which leaves two exploitable
artifacts: (a) the whole sentence retranslated differently, so the pair differs everywhere rather than at the
entity, and (b) a Latin-script name surviving in one branch only. Measured over the full splits (2026-09-21):
validation 18.9% and 0.5%, train 19.4% and 1.1%. Both kinds are dropped here, leaving 1,021 of 1,267 validation
rows and 32,106 of 40,398 train rows (the few-shot pool).

Scoring mirrors upstream `winogrande`'s partial evaluation: the shared word-prefix of the two sentences is the
context and only the differing remainders are scored, so the likelihood of the identical prefix cannot enter
`acc_norm`'s length normalisation.
"""
import re

import datasets

LATIN = re.compile(r"[A-Za-z]")


def _common_prefix_words(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def _clean(doc) -> bool:
    t, sc = doc["multiple_choice_targets"], doc["multiple_choice_scores"]
    if len(t) != 2 or sorted(sc) != [0.0, 1.0]:
        return False
    a, b = t
    if _common_prefix_words(a, b) < 0.5 * min(len(a.split()), len(b.split())):
        return False                                   # retranslated differently: no aligned contrast
    return bool(LATIN.search(a)) == bool(LATIN.search(b))  # a Latin name in one branch only is a giveaway


def _split(doc):
    a, b = doc["multiple_choice_targets"]
    n = _common_prefix_words(a, b)
    wa, wb = a.split(), b.split()
    return {
        "query": " ".join(wa[:n]),
        "choices": [" ".join(wa[n:]), " ".join(wb[n:])],
        "gold": doc["multiple_choice_scores"].index(1.0),
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(_clean).map(_split)
