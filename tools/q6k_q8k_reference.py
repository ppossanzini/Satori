#!/usr/bin/env python3
"""Independent FP64 reference for the Q6_K weight block against a Q8_K activation block.

The Q6_K decode is imported from `q6k_reference.py` and the Q8_K quantizer from `q8k_reference.py`,
both of which are themselves checked against the pinned ggml source. Neither shares code with
Satori, so a match here is evidence about the layout and not a restatement of it.

The point of this fixture is the `-32` centring: the reference AVX2 kernel multiplies uncentred
six-bit codes and removes the offset through the activation's group sums, which groups the weights
differently from the format's decoder. Emitting the product through the decoder's own grouping
settles which one the bytes actually mean.
"""
import json
import random
import struct
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
OUT = ROOT / "fixtures" / "operators" / "q6k-q8k-v1.json"
sys.path.insert(0, str(TOOLS))

from q6k_reference import decode as decode_q6k  # noqa: E402
from q8k_reference import QK_K, f32, quantize  # noqa: E402


def make_weight_block(rng):
    """A Q6_K block: ql[128], qh[64], scales[16] signed, FP16 d."""
    ql = [rng.randrange(256) for _ in range(128)]
    qh = [rng.randrange(256) for _ in range(64)]
    scales = [rng.randrange(-128, 128) for _ in range(16)]
    d = rng.choice([0.001953125, 0.0078125, 0.03125, 0.125])
    return bytes(ql) + bytes(qh) + struct.pack("<16b", *scales) + struct.pack("<e", d)


def make_activation(rng, kind):
    if kind == "uniform":
        return [f32(rng.uniform(-1.0, 1.0)) for _ in range(QK_K)]
    if kind == "spike":
        values = [f32(rng.uniform(-0.01, 0.01)) for _ in range(QK_K)]
        values[rng.randrange(QK_K)] = 3.0
        return values
    if kind == "tilted":
        return [f32((index - 128) / 128.0) for index in range(QK_K)]
    raise ValueError(kind)


def main() -> int:
    rng = random.Random(20261014)
    cases = []
    for index, kind in enumerate(["uniform", "uniform", "spike", "tilted", "uniform"]):
        activation_values = make_activation(rng, kind)
        scale, codes, bsums = quantize(activation_values)
        weights = make_weight_block(rng)
        decoded = decode_q6k(weights)  # FP64, 256 weights
        # The product the format's own decoder implies, in FP64, with the FP16 super-scale applied at
        # the end exactly as Q6KCodec does.
        d = struct.unpack_from("<e", weights, 208)[0]
        integer = sum((decoded[i] / d if d else 0.0) * codes[i] for i in range(QK_K))
        cases.append({
            "name": f"{kind}-{index}",
            "weights_hex": weights.hex(),
            "input": activation_values,
            "activation_hex": (struct.pack("<f", scale) + bytes(c & 0xFF for c in codes)
                               + struct.pack("<16h", *bsums)).hex(),
            "activation_scale": scale,
            "dot": scale * d * integer,
        })
    document = {
        "source": "ggml block_q6_K decoder against block_q8_K, both from the pinned source",
        "revision": "bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444",
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
