#!/usr/bin/env python3
"""DW06 prefill prompt: a deterministic word stream used as the 2K prompt.

Words are drawn from a fixed list with a fixed seed, so the prompt is reproducible from
this file alone. The count is asserted against the tokenizer by the caller, not guessed.
"""
import argparse
import random

WORDS = """
add remove update delete create build parse encode decode token matrix vector weight layer
attention module runtime session context buffer memory cache table column row index value
string number double single float integer boolean array list dictionary object class method
function parameter return import export public private static readonly internal sealed record
struct interface namespace transaction commit rollback query join filter group order limit
offset stream file path directory folder permission owner review comment issue branch merge
rebase conflict resolve test verify assert measure report benchmark protocol threshold budget
resident streamed shard tensor quantize dequantize block scale minimum maximum average median
""".split()

PUNCTUATION = [".", ".", ".", ",", ",", ";", ":", ")", "(", "]", "["]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=int, default=2048, help="number of words to draw")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out", help="write the prompt to this file instead of stdout")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pieces = []
    for index in range(args.words):
        word = rng.choice(WORDS)
        if index % 17 == 0:
            word = word.capitalize()
        pieces.append(word)
        if rng.random() < 0.14:
            pieces.append(rng.choice(PUNCTUATION))
        if rng.random() < 0.03:
            pieces.append("\n")
    text = " ".join(pieces)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
