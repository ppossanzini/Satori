#!/usr/bin/env bash
set -u
cd /home/paolo/Work/archer/Satori
PY=.temp/deepseek/dw02/measure.py
LL=.temp/deepseek/dw02/llama.cpp/build/bin
PR=.temp/deepseek/dw02/llama-prism/build/bin
M=.temp/deepseek/dw02/models
OUT=.temp/deepseek/dw02/results/memory.tsv
rm -f "$OUT"

LD_LIBRARY_PATH=$LL python3 "$PY" --label qwen-7b-q4km --out "$OUT" -- \
  "$LL/llama-bench" -m "$M/qwen2.5-coder-7b-instruct-q4_k_m.gguf" -p 1 -n 1 -r 1 -t 8 >/dev/null 2>&1

LD_LIBRARY_PATH=$LL python3 "$PY" --label apex-nanoplus --out "$OUT" -- \
  "$LL/llama-bench" -m "$M/Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf" -p 1 -n 1 -r 1 -t 8 >/dev/null 2>&1

LD_LIBRARY_PATH=$PR python3 "$PY" --label bonsai-ptq1_0 --out "$OUT" -- \
  "$PR/llama-bench" -m "$M/Ternary-Bonsai-2-27B-PTQ1_0.gguf" -p 1 -n 1 -r 1 -t 8 >/dev/null 2>&1

echo "=== memory.tsv ==="
cat "$OUT"
