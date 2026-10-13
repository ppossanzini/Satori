#!/usr/bin/env python3
"""Build a tokenizer-only GGUF fixture and official reference token sequences.

The fixture GGUF carries only the tokenizer metadata (no tensors). The reference ids are produced by the
official Hugging Face tokenizer for the same repository, so the C# tokenizer can be checked for identity
against an independent implementation. Provenance (repo, revisions, hashes, library versions) is written to
reference.json.
"""

import hashlib
import json
import pathlib
import platform
import struct
import sys

REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "tokenizers" / "qwen2.5-coder-1.5b"

CORPUS = {
    "italian": "Ciao, come stai? Oggi \u00e8 una bella giornata, vero?",
    "code_csharp": "public static int Add(int a, int b) => a + b;\n// somma due interi\n",
    "unicode_emoji": "Fatto! \U0001f642\U0001f680 fine.",
    "unicode_cjk": "\u4f60\u597d\u4e16\u754c\uff0c\u3053\u3093\u306b\u3061\u306f\u3002",
    "unicode_accents": "\u00e0\u00e8\u00ec\u00f2\u00f9 \u00c0\u00c8\u00cc\u00d2\u00d9 \u00e7 \u00f1 stra\u00dfe",
    "whitespace": "  \n\n  x \t y   ",
    "special_tokens": "<|im_start|>user\n\u00e8 italiano?<|im_end|>\n",
    "mixed": "Ciao! def f(): return 42  # \u00e8 un numero",
    "empty": "",
}


def build_gguf(tokens, merges, token_types, ids, chat_template):
    def string(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    def array_string(values) -> bytes:
        out = struct.pack("<I", 8) + struct.pack("<Q", len(values))
        for value in values:
            out += string(value)
        return out

    def array_i32(values) -> bytes:
        out = struct.pack("<I", 5) + struct.pack("<Q", len(values))
        for value in values:
            out += struct.pack("<i", value)
        return out

    metadata = [
        ("general.architecture", 8, string("qwen2")),
        ("tokenizer.ggml.model", 8, string("gpt2")),
        ("tokenizer.ggml.pre", 8, string("qwen2")),
        ("tokenizer.ggml.tokens", 9, array_string(tokens)),
        ("tokenizer.ggml.merges", 9, array_string(merges)),
        ("tokenizer.ggml.token_type", 9, array_i32(token_types)),
        ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", ids["bos"])),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", ids["eos"])),
        ("tokenizer.ggml.padding_token_id", 4, struct.pack("<I", ids["pad"])),
    ]
    if chat_template is not None:
        metadata.append(("tokenizer.chat_template", 8, string(chat_template)))

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(metadata))
    for key, mtype, value in metadata:
        out += string(key) + struct.pack("<I", mtype) + value
    while len(out) % 32 != 0:
        out += b"\x00"
    return out


def main() -> int:
    import tokenizers
    import transformers
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer_json = hf_hub_download(REPO, "tokenizer.json")
    try:
        config_json = hf_hub_download(REPO, "tokenizer_config.json")
        chat_template = json.load(open(config_json)).get("chat_template")
    except Exception:
        chat_template = None

    raw = json.load(open(tokenizer_json))
    vocab = raw["model"]["vocab"]
    merges = raw["model"]["merges"]
    added = raw.get("added_tokens", [])
    size = max(max(vocab.values()), max((a["id"] for a in added), default=0)) + 1
    tokens = [None] * size
    for token, index in vocab.items():
        tokens[index] = token
    token_types = [1] * size  # 1 = normal
    for entry in added:
        if tokens[entry["id"]] is None:
            tokens[entry["id"]] = entry["content"]
        token_types[entry["id"]] = 3 if entry.get("special") else 4  # 3 control, 4 user-defined
    if any(t is None for t in tokens):
        raise SystemExit("non-contiguous vocabulary")

    ids = {
        "bos": int(vocab.get("<|endoftext|>", 0)),
        "eos": int(vocab.get("<|im_end|>", 0)),
        "pad": int(vocab.get("<|endoftext|>", 0)),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    gguf = build_gguf(tokens, merges, token_types, ids, chat_template)
    (OUTPUT / "tokenizer.gguf").write_bytes(gguf)

    tokenizer = Tokenizer.from_file(tokenizer_json)
    hf = AutoTokenizer.from_pretrained(REPO)

    cases = [
        {"name": name, "text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids}
        for name, text in CORPUS.items()
    ]

    conversation = [
        ("system", "Sei un assistente utile."),
        ("user", "Scrivi una funzione in C# che somma due interi."),
    ]
    template_text = hf.apply_chat_template(
        [{"role": role, "content": content} for role, content in conversation],
        tokenize=False, add_generation_prompt=True)

    reference = {
        "repo": REPO,
        "tokenizer_json_sha256": hashlib.sha256(open(tokenizer_json, "rb").read()).hexdigest(),
        "fixture_gguf_sha256": hashlib.sha256(gguf).hexdigest(),
        "vocabulary_size": size,
        "merge_count": len(merges),
        "bos_token_id": ids["bos"],
        "eos_token_id": ids["eos"],
        "libraries": {"tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
                      "python": platform.python_version()},
        "cases": cases,
        "chat_template_case": {
            "messages": [{"role": r, "content": c} for r, c in conversation],
            "add_generation_prompt": True,
            "text": template_text,
        },
    }
    (OUTPUT / "reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote tokenizer.gguf ({len(gguf)} bytes), vocab={size}, merges={len(merges)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
