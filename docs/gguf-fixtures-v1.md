# GGUF Fixtures v1

Profile: [gguf-subset-v1.md](gguf-subset-v1.md). Fixtures describe containers, not complete LLM models.

## Available Golden Fixtures

Directory: `fixtures/gguf/v1/`. The manifest contains SHA-256, offsets, shapes, lengths and expected reconstructed values. The Python generator uses only `struct` and calls no Satori code.

| File | GGML Type and Shape | Block or Values | Result with Input of Ones |
|---|---|---|---:|
| `q4_0.gguf` | Q4_0, `[32, 1]` | Scale 0.5; 16 bytes `F0`: 16 weights −4, then 16 weights 3.5 | −8 |
| `q8_0.gguf` | Q8_0, `[32, 1]` | Scale 0.5; 16 Int8 −8, then 16 Int8 7 | −8 |
| `f32.gguf` | F32, `[3]` | 1, 2, 3 | 6, reference data only |
| `f16.gguf` | F16, `[3]` | 0, 1, −1 | 0, reference data only |

Q4_0 and Q8_0 represent the same weights. The expected result can be computed by hand; this prevents the codec and kernel from sharing an undetected error. Headers use GGUF v3, default alignment, one tensor entry and `general.architecture=llama`. Quantized files declare quantization version 2.

## Cases to Implement in T12

The table remains the acceptance catalog. Implemented checks cover golden fixtures, truncation, scalar/array metadata, UTF-8, alignment, dimensions/overflow, overlaps, windows and disposal; the public corpus also checks mixed catalogs. Not every table combination is a separate test. Large files and corrupted cases are generated under `.temp/`, without adding models to the repository.

| ID | Case | Reader Outcome |
|---|---|---|
| V01 | Four available golden fixtures | Catalog matches manifest; identical bytes |
| V02 | Mixed F32/F16/Q4_0/Q8_0 file; aligned cumulative offsets | All entries indexed; no dequantization |
| V03 | Explicit alignment 8, 64 or 4096 | Correct `dataStart` and absolute offsets |
| V04 | All scalars and flat arrays; empty arrays and multibyte strings | Correct types, counts, values and boundaries |
| V05 | String value with NUL, unknown keys, reordered metadata | Contents preserved |
| V06 | Rank 1, 3, 4 with allowed types | Cataloged; no implicit MatVec |
| V07 | Sparse file with second tensor beyond 2 GiB, large first F32 tensor and canonical offsets | Int64 offsets and windowed reads without full allocation |
| V08 | Payload with negative, zero, negative-zero or nonfinite scale | Cataloged without numerical validation; scale checks deferred to kernel |
| V09 | Additional bytes after last tensor padding | Ignored |
| U01 | Version 1, 2 or greater than 3; recognizable big-endian header | `NotSupportedException` |
| U02 | Disallowed tensor type, even if not selected | `NotSupportedException` |
| U03 | Nested array or unknown metadata type | `NotSupportedException` |
| U04 | Declared shard, quantization version other than 2, `general.type` other than `model` | `NotSupportedException` |
| U05 | Power-of-two alignment beyond 4096, 64-byte name, rank beyond 4 | `NotSupportedException` |
| U06 | Zero tensor count or Satori limit exceeded | `NotSupportedException`, limit identified |
| U07 | Gap or unordered directory with otherwise valid, nonoverlapping ranges | `NotSupportedException`, layout outside profile |
| E01 | Wrong magic; truncated header | `InvalidDataException` |
| E02 | Counter, string or array declared beyond remaining bytes | `InvalidDataException`, without proportional allocation |
| E03 | Wrong type for an interpreted key; missing required metadata | `InvalidDataException` |
| E04 | Bool other than 0/1, invalid UTF-8, illegal key, NUL in name | `InvalidDataException` |
| E05 | Duplicate key or tensor name, empty name | `InvalidDataException` |
| E06 | Zero rank, zero dimension, UInt64 beyond Int64 or overflowing products | `InvalidDataException` |
| E07 | First Q4_0/Q8_0 dimension not divisible by 32 | `InvalidDataException` |
| E08 | Zero or non-power-of-two alignment, unaligned offset | `InvalidDataException` |
| E09 | Overflowing offset/length sum, payload or final padding beyond file | `InvalidDataException` |
| E10 | Overlapping tensor ranges | `InvalidDataException` |
| L01 | Failed opening in every section; callback exception | Resources released; no partial catalog |
| L02 | Read concurrent with Dispose, new read after Dispose | Active access valid; new access rejected |
| L03 | Window within file but beyond selected tensor | Rejected before callback; no padding or adjacent-tensor read |

When an input violates multiple rules, bounds/overflow and structure take precedence over profile checks. In E02, a count already exceeding a Satori limit may be rejected as U06 before reading the rest of the file.

## Numerical Cases for T13/T14

- Golden Q4_0/Q8_0: per-position decoding and expected MatVec −8.
- Positive/negative scales, ±0 and FP16 subnormals; NaN/Infinity rejected on opening the numerical tensor.
- Q8_0 with Int8 −128 and 127; Q4_0 with all nibbles 0–15 in both halves.
- Nonsquare matrices, multiple rows and multiple blocks per row.
- FP64 comparison on the same decoded weights; no confusion with quantization error.
- Mapped/owned identity and no allocations on the hot path.

The four golden fixtures were read by the official gguf-py reader at the pinned revision. Catalogs of the two public regression artifacts (57 tensors each) are compared entry by entry with the same reader. Numerical reference generation remains independent of Satori.
