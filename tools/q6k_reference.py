#!/usr/bin/env python3
"""Independent FP64 reference for the GGML Q6_K super-block.

The decoder is written from the pinned ggml layout (block_q6_K: `ql[128]`, `qh[64]`,
`scales[16]` signed int8, FP16 `d`) and never calls Satori.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "operators" / "q6k-v1.json"


def decode(block):
    """Reconstruct 256 weights (FP64) from a 210-byte Q6_K super-block."""
    assert len(block) == 210
    ql = block[0:128]
    qh = block[128:192]
    scales = block[192:208]
    d = struct.unpack_from("<e", block, 208)[0]
    sc = list(struct.unpack_from("<16b", scales))
    out = [0.0] * 256
    for half in range(2):
        lql = ql[half * 64:(half + 1) * 64]
        lqh = qh[half * 32:(half + 1) * 32]
        lsc = sc[half * 8:(half + 1) * 8]
        base = half * 128
        for l in range(32):
            is_ = l // 16
            q1 = ((lql[l] & 0xF) | (((lqh[l] >> 0) & 3) << 4)) - 32
            q2 = ((lql[l + 32] & 0xF) | (((lqh[l] >> 2) & 3) << 4)) - 32
            q3 = ((lql[l] >> 4) | (((lqh[l] >> 4) & 3) << 4)) - 32
            q4 = ((lql[l + 32] >> 4) | (((lqh[l] >> 6) & 3) << 4)) - 32
            out[base + l + 0] = d * lsc[is_ + 0] * q1
            out[base + l + 32] = d * lsc[is_ + 2] * q2
            out[base + l + 64] = d * lsc[is_ + 4] * q3
            out[base + l + 96] = d * lsc[is_ + 6] * q4
    return out


def block_bytes(d, scales, ql, qh):
    return bytes(ql) + bytes(qh) + struct.pack("<16b", *scales) + struct.pack("<e", d)


def make_cases():
    rng = __import__("random").Random(20261014)
    cases = []

    cases.append(("zero", block_bytes(0.0, [0] * 16, [0] * 128, [0] * 64)))
    cases.append(("positive", block_bytes(1.0, [16] * 16, [0x59] * 128, [0xAA] * 64)))
    cases.append(("negative-scales", block_bytes(0.5, [-16] * 16, [0x12] * 128, [0x55] * 64)))
    cases.append(("extremes", block_bytes(8.0, [127, -128] * 8, [0xFF] * 128, [0xFF] * 64)))
    for i in range(4):
        scales = [rng.randrange(-128, 128) for _ in range(16)]
        ql = [rng.randrange(256) for _ in range(128)]
        qh = [rng.randrange(256) for _ in range(64)]
        d = rng.choice([0.5, 1.0, 2.0, 0.03125])
        cases.append((f"random-{i}", block_bytes(d, scales, ql, qh)))

    inputs = [((i * 53) % 97 - 48) / 19.0 for i in range(256)]
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
        "generated_by": "tools/q6k_reference.py",
        "source": "GGML block_q6_K layout; independent FP64 decode",
        "block_bytes": 210,
        "elements": 256,
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} with {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
