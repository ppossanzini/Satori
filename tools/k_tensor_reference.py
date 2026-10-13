#!/usr/bin/env python3
"""DW06.4: real-artifact parity fixture for the K codecs.

Reads a real Q4_K tensor and the real Q6_K output tensor from a pinned GGUF artifact, takes a small window
of raw block bytes, decodes them with the independent FP64 decoders in `q4k_reference.py` / `q6k_reference.py`
(no Satori code), and writes the input/dot the C# codec is compared against.
"""
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import q4k_reference  # noqa: E402
import q6k_reference  # noqa: E402

OUT = ROOT / "fixtures" / "operators" / "k-real-v1.json"

GGUF_SCALARS = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4),
                6: ('f', 4), 7: ('B', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
BLOCKS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 14: (256, 210)}


def parse(path):
    data = Path(path).read_bytes()
    pos = 4
    version, = struct.unpack_from('<I', data, pos); pos += 4
    assert version == 3, version
    ntensors, = struct.unpack_from('<Q', data, pos); pos += 8
    nmeta, = struct.unpack_from('<Q', data, pos); pos += 8
    alignment = 32

    def string():
        nonlocal pos
        n, = struct.unpack_from('<Q', data, pos); pos += 8
        s = data[pos:pos + n].decode('utf-8'); pos += n
        return s

    def value(kind):
        nonlocal pos
        if kind == 8:
            return string()
        if kind == 9:
            elem, = struct.unpack_from('<I', data, pos); pos += 4
            n, = struct.unpack_from('<Q', data, pos); pos += 8
            return [value(elem) for _ in range(n)]
        fmt, size = GGUF_SCALARS[kind]
        v, = struct.unpack_from('<' + fmt, data, pos); pos += size
        return v

    for _ in range(nmeta):
        key = string()
        kind, = struct.unpack_from('<I', data, pos); pos += 4
        v = value(kind)
        if key == 'general.alignment':
            alignment = v
    tensors = {}
    for _ in range(ntensors):
        name = string()
        rank, = struct.unpack_from('<I', data, pos); pos += 4
        dims = struct.unpack_from('<' + 'Q' * rank, data, pos); pos += 8 * rank
        kind, = struct.unpack_from('<I', data, pos); pos += 4
        offset, = struct.unpack_from('<Q', data, pos); pos += 8
        tensors[name] = (dims, kind, offset)
    data_start = pos + (-pos % alignment)
    return data, tensors, data_start


def window(data, tensors, data_start, name, blocks):
    dims, kind, offset = tensors[name]
    block, size = BLOCKS[kind]
    start = data_start + offset
    return data[start:start + blocks * size], dims, kind


def main():
    if len(sys.argv) != 2:
        print("usage: k_tensor_reference.py <qwen2.5-coder-7b-q4_k_m.gguf>")
        return 2
    path = Path(sys.argv[1])
    data, tensors, data_start = parse(path)

    # A real Q4_K projection tensor and the real Q6_K output tensor.
    q4k_name = next(n for n, (d, k, _) in tensors.items() if k == 12 and d[0] % 256 == 0)
    q6k_name = "output.weight"
    assert tensors[q6k_name][1] == 14, tensors[q6k_name][1]

    q4k_bytes, q4k_dims, _ = window(data, tensors, data_start, q4k_name, 2)
    q6k_bytes, q6k_dims, _ = window(data, tensors, data_start, q6k_name, 1)

    inputs = [((i * 41) % 89 - 44) / 13.0 for i in range(256)]

    def case(name, raw, decode, blocks):
        weights = []
        for b in range(blocks):
            weights.extend(decode(raw[b * (len(raw) // blocks):(b + 1) * (len(raw) // blocks)]))
        # dot is computed per 256-element super-block against the same 256-input window.
        dots = [sum(w * x for w, x in zip(weights[b * 256:(b + 1) * 256], inputs)) for b in range(blocks)]
        return {
            "name": name,
            "block_hex": raw.hex(),
            "blocks": blocks,
            "input": [round(x, 12) for x in inputs],
            "weights": [round(w, 12) for w in weights],
            "dots": dots,
        }

    cases = [
        case(f"q4k:{q4k_name}", q4k_bytes, q4k_reference.decode, 2),
        case(f"q6k:{q6k_name}", q6k_bytes, q6k_reference.decode, 1),
    ]
    doc = {
        "generated_by": "tools/k_tensor_reference.py",
        "source": f"real tensors from {path.name}; independent FP64 decode of the same bytes",
        "artifact_sha256": "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
        "q4k_tensor": q4k_name,
        "q6k_tensor": q6k_name,
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}: {q4k_name} (Q4_K) and {q6k_name} (Q6_K)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
