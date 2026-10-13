#!/usr/bin/env python3
"""Independent FP64 reference for the GGML Q4_K super-block.

The decoder is written from the pinned ggml layout (block_q4_K: FP16 `d`, FP16 `dmin`,
`scales[12]`, `qs[128]`) and never calls Satori. It writes a fixture used to check the
C# scalar decoder and dot product.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "operators" / "q4k-v1.json"


def get_scale_min(j, scales):
    """6-bit scale and min of sub-block j (0..7), per ggml get_scale_min_k4."""
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def decode(block):
    """Reconstruct 256 weights (FP64) from a 144-byte Q4_K super-block."""
    assert len(block) == 144
    d = struct.unpack_from("<e", block, 0)[0]
    dmin = struct.unpack_from("<e", block, 2)[0]
    scales = block[4:16]
    qs = block[16:144]
    out = [0.0] * 256
    for group in range(4):
        q = qs[group * 32:(group + 1) * 32]
        s0, m0 = get_scale_min(group * 2, scales)
        s1, m1 = get_scale_min(group * 2 + 1, scales)
        dl, ml = d * s0, dmin * m0
        dh, mh = d * s1, dmin * m1
        for l in range(32):
            out[group * 64 + l] = dl * (q[l] & 0xF) - ml
            out[group * 64 + 32 + l] = dh * (q[l] >> 4) - mh
    return out


def block_bytes(d, dmin, scales, quants):
    return struct.pack("<ee", d, dmin) + bytes(scales) + bytes(quants)


def make_cases():
    rng = __import__("random").Random(20261013)
    cases = []

    # Zero block: both scales zero and all quants zero -> all weights zero.
    cases.append(("zero", block_bytes(0.0, 0.0, [0] * 12, [0] * 128)))

    # Simple positive: d=1, dmin=0, uniform scales and quants.
    cases.append(("positive", block_bytes(1.0, 0.0, [0x3F] * 12, [0x37] * 128)))

    # Negative min: d positive, dmin large.
    cases.append(("negative-min", block_bytes(0.5, 2.0, [0x2A] * 12, [0x18] * 128)))

    # Extremes: maximum scale and full-range quants.
    cases.append(("extremes", block_bytes(8.0, 8.0, [0xFF] * 12, [0xFF] * 128)))

    # Random finite blocks.
    for i in range(4):
        scales = [rng.randrange(256) for _ in range(12)]
        quants = [rng.randrange(256) for _ in range(128)]
        d = rng.choice([0.5, 1.0, 2.0, 0.03125])
        dmin = rng.choice([0.0, 0.5, 1.0])
        cases.append((f"random-{i}", block_bytes(d, dmin, scales, quants)))

    # Deterministic input: irrational-ish so sums do not cancel.
    inputs = [((i * 37) % 101 - 50) / 17.0 for i in range(256)]

    out = []
    for name, block in cases:
        weights = decode(block)
        dot = sum(w * x for w, x in zip(weights, inputs))
        out.append({
            "name": name,
            "block_hex": block.hex(),
            "input": [round(x, 12) for x in inputs],
            "weights": [round(w, 12) for w in weights],
            "dot": dot,
        })
    return out


def main():
    cases = make_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_by": "tools/q4k_reference.py",
        "source": "GGML block_q4_K layout; independent FP64 decode",
        "block_bytes": 144,
        "elements": 256,
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} with {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
