#!/usr/bin/env python3
"""Measure a raw llama.cpp completion with an exact prompt token count and streamed TTFT."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post(base, endpoint, payload):
    request = urllib.request.Request(base + endpoint, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=1200)


def measure(base, tokens, decode):
    started = time.monotonic()
    first_token = None
    final = None
    text = []
    with post(base, "/completion", {"prompt": tokens, "n_predict": decode, "temperature": 0,
                                        "cache_prompt": False, "stream": True, "ignore_eos": True}) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("content"):
                if first_token is None:
                    first_token = time.monotonic() - started
                    print(f"first token at {first_token:.3f}s", flush=True)
                text.append(chunk["content"])
            if chunk.get("stop"):
                final = chunk
    if final is None or first_token is None:
        raise RuntimeError("completion stream did not finish with content and a final result")
    return {"prompt_tokens": len(tokens), "requested_decode": decode,
            "ttft_seconds": first_token, "seconds": time.monotonic() - started,
            "timings": final.get("timings"), "final": final, "text": "".join(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18089")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.decode, args.repeats) < 1 or args.warmup < 0:
        parser.error("token counts and repeats must be positive; warmup must be nonnegative")
    sentence = "public static int Add(int a, int b) => a + b;\n"
    with post(args.url, "/tokenize", {"content": sentence * args.prompt_tokens, "add_special": False}) as response:
        tokens = json.load(response)["tokens"][:args.prompt_tokens]
    if len(tokens) != args.prompt_tokens:
        raise RuntimeError("tokenizer returned fewer tokens than requested")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"prompt_token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
              "warmup": [], "runs": []}
    for index in range(args.warmup + args.repeats):
        warmup = index < args.warmup
        print(f"{'warmup' if warmup else 'sample'} {index + 1}", flush=True)
        result = measure(args.url, tokens, args.decode)
        report["warmup" if warmup else "runs"].append(result)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in ("final", "text")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())