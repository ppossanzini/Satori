#!/usr/bin/env python3
"""Download only the GGUF header/descriptor region of candidate artifacts and record a
reproducible descriptor inventory for each one.

The inventory is produced by `Satori.Console inventory <file> --json`, which reads the
header, metadata and tensor directory and never touches the tensor payload. A ranged
download (first `RANGE_BYTES`) is therefore sufficient and no full weights are fetched.

Outputs:
  docs/scouting/inventories/manifest.json       provenance (source, revision, sha256, range)
  docs/scouting/inventories/<name>.json         descriptor inventory per artifact
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RANGE_BYTES = 32 * 1024 * 1024
TEMP = ROOT / ".temp" / "deepseek" / "ds02"
OUT = ROOT / "docs" / "scouting" / "inventories"
CONSOLE = ROOT / "src" / "Satori.Console" / "bin" / "Release" / "net10.0" / "Satori.Console.dll"

# name -> (repo, revision, filename, full-file sha256)
ARTIFACTS = {
    "qwen2.5-coder-1.5b-q4_0": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
        "qwen2.5-coder-1.5b-instruct-q4_0.gguf",
        "aa8353e0d0fca3a0041828701e90db7635197400f040676d11d7798665fa316e"),
    "qwen2.5-coder-1.5b-q8_0": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
        "qwen2.5-coder-1.5b-instruct-q8_0.gguf",
        "507de59046601282ba768a9789900e6ccf60ed93ddf346730b7c68eb0715bc47"),
    "qwen2.5-coder-1.5b-q4_k_m": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"),
    "qwen2.5-coder-7b-q4_0": (
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "13fb94bfda8c8cf22497dc57b78f391a9acb426a",
        "qwen2.5-coder-7b-instruct-q4_0.gguf",
        "8561411b4705cbf2d105cf8e2084c6d6f1bcaba415249ce65fc5f21230d4961b"),
    "qwen2.5-coder-7b-q4_k_m": (
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "13fb94bfda8c8cf22497dc57b78f391a9acb426a",
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"),
    "qwen3-4b-q4_k_m": (
        "Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b",
        "Qwen3-4B-Q4_K_M.gguf",
        "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"),
    "qwen3-4b-q8_0": (
        "Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b",
        "Qwen3-4B-Q8_0.gguf",
        "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300"),
    "bitnet-b1.58-2b-i2_s": (
        "microsoft/bitnet-b1.58-2B-4T-gguf", "a1f2f1c765812aa8af3f6eda4a313707064bba15",
        "ggml-model-i2_s.gguf",
        "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"),
    "apex-i-nanoplus": (
        "IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF", "ef9ff9001e8205458bf842fd13f43e5318eb2d46",
        "Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf",
        "18f6b0bd47dacad208aea82b1d5060309c4b8bca4de2028e5dd9dcdbf7de56f4"),
    "bonsai-ptq1_0": (
        "prism-ml/Ternary-Bonsai-2-27B-gguf", "b072e1d3b35a0a630cece372c2127528e0994386",
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"),
    "bonsai-pq2_0": (
        "prism-ml/Ternary-Bonsai-2-27B-gguf", "b072e1d3b35a0a630cece372c2127528e0994386",
        "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"),
}


def main() -> int:
    TEMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (repo, revision, filename, sha256) in ARTIFACTS.items():
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        head = TEMP / f"{name}.head"
        print(f"[{name}] {repo}@{revision[:12]} {filename}", flush=True)
        subprocess.run(["curl", "-sL", "-r", f"0-{RANGE_BYTES - 1}", "-o", str(head), url], check=True)
        result = subprocess.run(
            ["dotnet", str(CONSOLE), "inventory", str(head), "--json"],
            check=True, capture_output=True, text=True)
        inventory = json.loads(result.stdout)
        inventory["Artifact"] = name
        inventory["Source"] = url
        (OUT / f"{name}.json").write_text(json.dumps(inventory, indent=2) + "\n")
        manifest.append({
            "artifact": name, "repo": repo, "revision": revision, "file": filename,
            "sha256": sha256, "range_bytes": RANGE_BYTES,
            "architecture": inventory["Architecture"],
            "tensor_count": inventory["TensorCount"],
            "unsupported_types": inventory["UnsupportedTypes"],
            "data_offset": inventory["DataOffset"],
        })
        print(f"    arch={inventory['Architecture']} tensors={inventory['TensorCount']} "
              f"unsupported={inventory['UnsupportedTypes']}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(
        {"generated_by": "tools/gguf_header_inventory.py", "verified_on": "2026-10-13",
         "method": "ranged download of the header/descriptor region; no full weights fetched",
         "artifacts": manifest}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
