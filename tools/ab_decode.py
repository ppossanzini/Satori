#!/usr/bin/env python3
"""A/B runner for the Satori decode path.

Runs two published builds alternately on the same artifact and reports the per-configuration
minimum and median of the generation time. Alternating matters: running all of A and then all of
B lets a slow patch of thermal state land entirely on one side, which is how a single-run
comparison produced a 3.8x swing on identical code.

Each run is a separate process, and the reported figures are the per-token generation cost taken
from the console's own elapsed_ms, so process start and model validation are excluded.
"""
import argparse
import statistics
import subprocess
import sys
from pathlib import Path


def one_run(binary: Path, model: Path, prompt: str, tokens: int, context: int, threads: int) -> tuple[int, int, int]:
    """Returns (elapsed_ms, prompt_tokens, generated_tokens) for one process run."""
    process = subprocess.run(
        [str(binary), "text", str(model), prompt, str(tokens), "--context", str(context), "--threads", str(threads)],
        capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise SystemExit(f"{binary.name} exited {process.returncode}: {process.stderr.strip()[:400]}")
    line = process.stderr.strip().splitlines()[-1]
    fields = dict(part.split(": ", 1) for part in line.split("; ") if ": " in part)
    return int(fields["elapsed_ms"]), int(fields["prompt"]), int(fields["generated"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--a", required=True, help="first binary (label=path)")
    parser.add_argument("--b", required=True, help="second binary (label=path)")
    parser.add_argument("--prompt", default="def fibonacci(n):")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    def parse(spec: str) -> tuple[str, Path]:
        label, _, path = spec.partition("=")
        return label, Path(path) / "Satori.Console"

    configs = [parse(args.a), parse(args.b)]
    model = Path(args.model)
    samples: dict[str, list[float]] = {label: [] for label, _ in configs}
    prompt_tokens = generated = 0

    for label, binary in configs:
        for _ in range(args.warmup):
            one_run(binary, model, args.prompt, args.tokens, args.context, args.threads)
        print(f"# {label}: {binary}", file=sys.stderr)

    # Alternate A and B within each round so drift is shared, and reverse the order every round so
    # neither side always runs right after the other's cache-warming work.
    for round_index in range(args.repeats):
        order = configs if round_index % 2 == 0 else list(reversed(configs))
        for label, binary in order:
            elapsed, prompt_tokens, generated = one_run(binary, model, args.prompt, args.tokens,
                                                        args.context, args.threads)
            per_token = elapsed / max(generated, 1)
            samples[label].append(per_token)
            print(f"round {round_index + 1:2d} {label:>10}: {elapsed:7d} ms total, {per_token:8.1f} ms/token")

    print()
    print(f"prompt {prompt_tokens} tokens, generated {generated}, context {args.context}, threads {args.threads}")
    print(f"{'config':>10} {'min':>10} {'median':>10} {'max':>10} {'tok/s at min':>13}")
    summaries = {}
    for label, _ in configs:
        values = samples[label]
        minimum = min(values)
        summaries[label] = minimum
        print(f"{label:>10} {minimum:10.1f} {statistics.median(values):10.1f} {max(values):10.1f} {1000 / minimum:13.3f}")

    first, second = configs[0][0], configs[1][0]
    ratio = summaries[first] / summaries[second]
    spread = max(max(samples[first]) / min(samples[first]), max(samples[second]) / min(samples[second]))
    print()
    print(f"{first} / {second} = {ratio:.3f}  (>1 means {second} is faster at its minimum)")
    print(f"within-configuration spread of the worst case: {spread:.2f}x")
    if spread > 1.15:
        print("WARNING: spread above 15%; treat any ratio below that as unresolved on this host.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
