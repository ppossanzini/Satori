#!/usr/bin/env bash
# Alternating A/B for two builds of this repository.
#
#   ab-pinned.sh <rounds> <cpus|none> [tag]
#
# The point of the alternation is that neither revision is systematically measured on a hotter
# machine. Running all of A and then all of B cannot separate the change from the drift between the
# two blocks, which is the mistake docs/measurement-method.md records.
#
# Build the two revisions first, from their own checkouts, into the directories below.
set -euo pipefail

rounds=${1:?usage: ab-pinned.sh <rounds> <cpus|none> [tag]}
cpus=${2:?usage: ab-pinned.sh <rounds> <cpus|none> [tag]}
tag=${3:-ab}

base=${AB_BASE:-/tmp/ab/base}
candidate=${AB_CANDIDATE:-$PWD}
out=${AB_OUT:-/tmp/ab}
filter=${AB_FILTER:-FullyQualifiedName~RealArtifactProjectionsAgainstSynthetic}

pin=()
if [ "$cpus" != "none" ]; then pin=(taskset -c "$cpus"); fi

mkdir -p "$out"
for ((round = 1; round <= rounds; round++)); do
  # Odd rounds run the baseline first, even rounds the candidate, so the order contributes to both.
  if (( round % 2 == 1 )); then order=(base candidate); else order=(candidate base); fi
  for side in "${order[@]}"; do
    if [ "$side" = base ]; then dir=$base; else dir=$candidate; fi
    (cd "$dir" && "${pin[@]}" dotnet test tests/Satori.Numerics.Tests -c Release --no-build -m:1 \
      --filter "$filter" --logger 'console;verbosity=detailed' \
      > "$out/$tag-$round-$side.log" 2>&1)
  done
  echo "round $round done"
done

echo "logs in $out/$tag-*.log"
