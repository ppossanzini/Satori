"""Generate tiny GGUF golden files independently of the Satori reader."""

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_fixture(name, type_id, dimensions, payload, decoded):
    quantized = type_id in (2, 8)
    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    if quantized:
        metadata += string("general.quantization_version") + struct.pack("<II", 4, 2)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 2 if quantized else 1)
    directory = string("weights") + struct.pack("<I", len(dimensions))
    directory += struct.pack("<" + "Q" * len(dimensions), *dimensions)
    directory += struct.pack("<IQ", type_id, 0)
    prefix = header + metadata + directory
    data_start = (len(prefix) + 31) // 32 * 32
    data = prefix + bytes(data_start - len(prefix)) + payload
    data += bytes((-len(payload)) % 32)
    (ROOT / name).write_bytes(data)
    return {
        "file": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "file_bytes": len(data),
        "version": 3,
        "alignment": 32,
        "metadata_count": 2 if quantized else 1,
        "tensor_name": "weights",
        "tensor_type_id": type_id,
        "dimensions": dimensions,
        "data_start": data_start,
        "relative_offset": 0,
        "payload_bytes": len(payload),
        "decoded_weights": decoded,
        "ones_dot_product": sum(decoded),
    }


def main():
    decoded = [-4.0] * 16 + [3.5] * 16
    fixtures = [
        write_fixture("q4_0.gguf", 2, [32, 1], struct.pack("<e", 0.5) + bytes([0xF0] * 16), decoded),
        write_fixture("q8_0.gguf", 8, [32, 1], struct.pack("<e", 0.5) + struct.pack("<32b", *([-8] * 16 + [7] * 16)), decoded),
        write_fixture("f32.gguf", 0, [3], struct.pack("<3f", 1, 2, 3), [1.0, 2.0, 3.0]),
        write_fixture("f16.gguf", 1, [3], struct.pack("<3e", 0, 1, -1), [0.0, 1.0, -1.0]),
    ]
    manifest = {
        "profile": "Satori GGUF subset v1",
        "provenance": "Synthetic golden data from the fixed GGUF layout; no Satori reader or quantizer used",
        "fixtures": fixtures,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
