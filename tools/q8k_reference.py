#!/usr/bin/env python3
"""Independent FP64 reference for the GGML Q8_K activation block.

The quantizer is written from the pinned ggml implementation (`quantize_row_q8_K_ref` in
`ggml-quants.c`, revision recorded in docs/gguf-subset-v1.md) and never calls Satori. It
emits, for each case, the input floats, the quantized block bytes and the dot product of
the block against a companion Q4_K weight block, so the C# integer kernel is checked
against an implementation that shares no code with it.
"""
import json
import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "operators" / "q8k-v1.json"

QK_K = 256


def nearest_int(value):
    """Reproduce ggml's nearest_int: add 1.5*2^23 in float32 and read the low mantissa bits.

    The addition is the rounding step, done by the FPU in single precision with
    round-half-to-even, so this cannot be replaced by Python's own rounding without
    changing which side of a tie an element falls on.
    """
    shifted = struct.unpack("<f", struct.pack("<f", value + 12582912.0))[0]
    bits = struct.unpack("<i", struct.pack("<f", shifted))[0]
    return (bits & 0x007FFFFF) - 0x00400000


def quantize(values):
    """Reproduce quantize_row_q8_K_ref: 292-byte block as (scale, codes, bsums)."""
    assert len(values) == QK_K
    amax = 0.0
    vmax = 0.0
    for value in values:
        absolute = abs(value)
        if absolute > amax:
            amax = absolute
            vmax = value
    if amax == 0.0:
        return 0.0, [0] * QK_K, [0] * (QK_K // 16)
    iscale = -127.0 / vmax
    codes = [min(127, nearest_int(iscale * value)) for value in values]
    bsums = [sum(codes[group * 16:(group + 1) * 16]) for group in range(QK_K // 16)]
    return 1.0 / iscale, codes, bsums


def block_bytes(scale, codes, bsums):
    return struct.pack("<f", scale) + bytes(c & 0xFF for c in codes) + struct.pack("<16h", *bsums)


def q4k_dot(weights, scale, codes, bsums):
    """Q4_K block against a quantized activation, in FP64, straight from the layout."""
    d = struct.unpack_from("<e", weights, 0)[0]
    dmin = struct.unpack_from("<e", weights, 2)[0]
    scales = weights[4:16]
    quants = weights[16:144]

    def scale_min(j):
        if j < 4:
            return scales[j] & 63, scales[j + 4] & 63
        return ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))

    if scale == 0.0:
        return 0.0
    total = 0.0
    for group in range(4):
        s_low, n_low = scale_min(group * 2)
        s_high, n_high = scale_min(group * 2 + 1)
        low = 0
        high = 0
        for l in range(32):
            packed = quants[group * 32 + l]
            low += (packed & 0xF) * codes[group * 64 + l]
            high += (packed >> 4) * codes[group * 64 + 32 + l]
        total += scale * d * (s_low * low + s_high * high)
        low_sum = bsums[group * 4] + bsums[group * 4 + 1]
        high_sum = bsums[group * 4 + 2] + bsums[group * 4 + 3]
        total -= scale * dmin * (n_low * low_sum + n_high * high_sum)
    return total


def f32(value):
    """The single-precision value a C# float would hold, since the reference must see the same input."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_inputs(rng, kind):
    if kind == "uniform":
        return [f32(rng.uniform(-1.0, 1.0)) for _ in range(QK_K)]
    if kind == "spike":
        values = [f32(rng.uniform(-0.01, 0.01)) for _ in range(QK_K)]
        values[rng.randrange(QK_K)] = 4.0
        return values
    if kind == "zero":
        return [0.0] * QK_K
    if kind == "tilted":
        return [f32((index - 128) / 128.0) for index in range(QK_K)]
    raise ValueError(kind)


def make_weight_block(rng):
    scales = [rng.randrange(256) for _ in range(12)]
    quants = [rng.randrange(256) for _ in range(128)]
    d = rng.choice([0.001953125, 0.0078125, 0.03125, 0.125])
    dmin = rng.choice([0.001953125, 0.015625, 0.0625])
    return struct.pack("<e", d) + struct.pack("<e", dmin) + bytes(scales) + bytes(quants)


def main():
    rng = random.Random(20261014)
    cases = []
    kinds = ["uniform", "spike", "zero", "tilted", "uniform", "uniform"]
    for index, kind in enumerate(kinds):
        values = make_inputs(rng, kind)
        scale, codes, bsums = quantize(values)
        weights = make_weight_block(rng)
        cases.append({
            "name": f"{kind}-{index}",
            "input": values,
            "block_hex": block_bytes(scale, codes, bsums).hex(),
            "weights_hex": weights.hex(),
            "scale": scale,
            "codes": codes,
            "bsums": bsums,
            "dot": q4k_dot(weights, scale, codes, bsums),
        })
    document = {
        "source": "ggml quantize_row_q8_K_ref + ggml_vec_dot_q4_K_q8_K",
        "revision": "bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444",
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
