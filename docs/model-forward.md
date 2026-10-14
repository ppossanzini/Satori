# Generic Dense Forward

Status: DS08. A metadata-driven CPU forward pass for the dense `qwen2` and `qwen3` families, checked
against an independent float64 reference on two topologies.

## Contract

- `IModelWeights` is the weight-access boundary in `Satori.Core.Model`. Core depends only on this
  interface; the GGUF module provides `GgufWeightStore`. Projections call the shared quantized MatVec,
  vectors are decoded once and cached, embedding rows are decoded on demand. No payload is copied.
- `Project` multiplies one input vector; `ProjectBatch` multiplies an array of `positions` vectors in one
  pass, so a weight block is decoded once per pass instead of once per position. An implementation may split
  the output rows over `threads` workers, and the result must not depend on that split.
- `DenseForward(ModelShape, IModelWeights).Logits(ReadOnlySpan<int> tokens[, ForwardObserver])` returns the
  logits of the final position for a token sequence. Every cast is checked.
- `DenseForward.LayerBatch` applies one layer to a whole chunk of positions through `ProjectBatch` with the
  batch scratch of `DenseBatchScratch`; `DenseSession.Forward` is its only caller. `Step` stays the
  single-position path, so decoding is unchanged and the two share the operator sequence.
- All dimensions come from the validated `ModelShape`: layers, embedding, feed-forward, query/KV head
  counts, head dimension (from an explicit key/value length when declared), vocabulary, epsilon, RoPE base
  and the tied-embedding flag. Nothing is a nominal constant.
- Operators are the shared `TransformerOps`: RMS norm, half-split RoPE, grouped-query attention, SwiGLU and
  add. Optional Q/K per-head normalization and QKV bias are applied only when the tensors are present, so
  one class serves both families without branching on a model name.
- A new tensor format is supported by registering a kernel in the quantized MatVec, not by editing
  `DenseForward`.

`ForwardObserver(layer, position, stage, values)` reports `attn_norm`, `q`, `attn_ctx`, `attn_residual`,
`ffn_act`, `out` per layer and `final_norm`, `logits` at the end; `layer`/`position` are `-1` for final
stages. The span is valid only inside the callback.

## Per-Layer Order

`x += attn_output(grouped_attention(rope(q), rope(k), v))`, then `x += ffn_down(swiglu(ffn_gate(x), ffn_up(x)))`,
with RMS norm before each block and after the last layer. Incremental/KV state is a later task; the whole
sequence is processed at once.

## Reference

`tools/qwen_forward_reference.py` is a second implementation: its own GGUF parser and Q8_0 decode and the
forward pass in NumPy float64, with no Satori code. Weight tensors are decoded one at a time and released,
except the token embedding (needed as the output projection when tied), so the peak stays bounded. The
fixture records provenance, the model SHA-256 and the NumPy version.

| Fixture | Model | Topology |
|---|---|---|
| `fixtures/qwen/qwen2.5-coder-1.5b-q8_0-forward-v1.json` | Qwen2.5-Coder 1.5B Q8_0 | 28 layers, 12 query / 2 KV heads, head dim 64, QKV bias |
| `fixtures/qwen/qwen3-4b-q8_0-forward-v1.json` | Qwen3 4B Q8_0 | 36 layers, 32 query / 8 KV heads, head dim 128, Q/K norm |

The Qwen3 case is the one that motivates reading the head dimension from `attention.key_length`:
`2560 / 32 = 80`, but the declared head dimension is 128.

```sh
python3 tools/download_qwen_models.py
python3 tools/qwen_forward_reference.py .temp/qwen-corpus/qwen2.5-coder-1.5b-q8_0.gguf fixtures/qwen/qwen2.5-coder-1.5b-q8_0-forward-v1.json
python3 tools/qwen_forward_reference.py .temp/qwen-corpus/qwen3-4b-q8_0.gguf fixtures/qwen/qwen3-4b-q8_0-forward-v1.json
```

## Verification

`dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~QwenForwardTests"`
requires the corpus (`python tools/download_qwen_models.py`). For each topology it asserts the derived shape
equals the reference configuration, that the capability report is empty, that the argmax token matches, that
64 sampled logits agree within `5e-3 + 5e-3 * |reference|`, and that the first- and last-layer intermediates
agree within `2e-3 + 2e-3 * |reference|`. Tolerances follow from the reference being float64 while the
kernel accumulates in FP32 over up to 36 layers.
