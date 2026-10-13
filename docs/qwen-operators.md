# Qwen2/Qwen3 CPU Operators

Status: DS07. The operators required by the dense `qwen2` and `qwen3` families, each checked against an
independent float64 reference. All arithmetic is FP32 with FP64 accumulation on the critical reductions.

## Scope

- Rotary embedding in the half-split (GPT-NeoX) convention, with configurable base and rotary dimension.
- Grouped-query attention with a causal prefix through the current token.
- Per-head RMS normalization for Q/K (`attn_q_norm`, `attn_k_norm`).
- Bias add for the Qwen2 Q/K/V projections.
- Projections whose output width differs from the input width.

Excluded: MoE routing and recurrent/hybrid attention (APEX, Bonsai), covered by later tasks.

## Operators

| Operator | API | Notes |
|---|---|---|
| Half-split RoPE | `TransformerOps.RopeHalf(vector, headDimension, position, frequencyBase, rotaryDimensions)` | Pair `(k, k + half)`; only the first `rotaryDimensions` lanes rotate. Supports partial rotary (Qwen3 `rope.dimension_count` < key length) and the per-model base. |
| Grouped-query attention | `TransformerOps.GroupedAttention(query, keys, values, queryHeads, kvHeads, headDimension, tokenCount, scores, output)` | Query head `h` reads KV head `h / (queryHeads / kvHeads)`; stable softmax; caller-owned cache. |
| Per-head RMS norm | `TransformerOps.RmsNormHeads(vector, weights, epsilon, headDimension)` | Independent normalization per head with a weight of length `headDimension`; exact in-place permitted. |
| Bias add | `TransformerOps.Add(left, right, output)` | Reused unchanged; the Qwen2 Q/K/V bias has the projection output width. |
| Projection | `QuantizedMatrix.Execute` + `TransformerOps.Add` | Reused; the Q8_0/Q4_0 MatVec already supports independent rows and columns. |

Only the first three were new. Bias and the projection reuse existing, already-verified operators, per the
DS07 rule "reuse existing ones when conformant"; no duplicate operator was added.

## Numerical Contract

- Inputs must be finite; dimensions are exact and validated before execution. `RopeHalf` rejects a
  non-positive base, a rotary dimension that is odd, zero or larger than the head, and a negative position.
  `GroupedAttention` rejects `queryHeads % kvHeads != 0`, width mismatches and a score scratch smaller than
  the token count. `RmsNormHeads` rejects a per-head weight that is not `headDimension` long.
- `GroupedAttention` reduces to `TransformerOps.Attention` when `queryHeads == kvHeads`.
- Attention output must not overlap query, key or value; the score scratch must not overlap any of them.
- Tolerance for the reference comparison: `2e-6 + 2e-6 * |reference|`, matching the existing operator suite.

## Reference

`tools/transformer_reference_v2.py` builds `fixtures/operators/transformer-v2.json` with NumPy float64
formulas independent of the Satori code: only inputs are rounded to float32, all arithmetic is float64.
Cases: GQA (4 query / 2 KV heads, head dim 8, 3 tokens), full and partial RoPE, per-head RMS norm and bias
add. The fixture records its provenance and the NumPy version; regeneration is byte-identical.

The 8×32 biased-projection case is checked in the test against a manual float64 decode of the same Q8_0
payload, so the projection and the bias are not validated by Satori's own decoder.

## Verification

`dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~QwenOperatorTests"` runs
eight cases: GQA versus reference, GQA reduced to plain attention, full and partial RoPE versus reference,
per-head RMS norm versus reference, bias add versus reference, the biased projection versus a manual
float64 decode, and rejection of invalid dimensions.
