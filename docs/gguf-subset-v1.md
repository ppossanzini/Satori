# Satori GGUF Profile v1

Status: reader, numerical adapter and scalar/AVX2 Q8_0/Q4_0 kernels implemented.

## Scope

- GGUF version 3, little-endian, single immutable file.
- Metadata and tensor catalog on a read-only mapping.
- Allowed tensor types: F32, F16, Q4_0, Q8_0.
- Initial MatVec operations: two-dimensional Q8_0 and Q4_0 tensors.
- Excluded: other versions, big-endian, shards, adapters, automatic ternary conversion, tokenizer, inference and GPU.

A file accepted by the catalog is not necessarily an executable model. This profile's restrictions do not define the validity of every GGUF file.

## Pinned References

GGML revision: `3612c0af93f2cfc8f048295300a61187723b12ae`, verified on October 12, 2026.

- [GGUF structure](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/docs/gguf.md).
- [Block layouts](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/src/ggml-common.h).
- [Reference decoding](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/src/ggml-quants.c).
- [GGML reader](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/src/gguf.cpp).
- [Quantization version](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/include/ggml.h).
- [Split metadata types in the llama.cpp writer](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/gguf-py/gguf/gguf_writer.py), revision `efe3626b119ff81033e309c3ba07a770bef2c546`.

## Container Structure

24-byte header: ASCII magic `GGUF`, UInt32 version, UInt64 tensor count, UInt64 metadata count. Metadata, tensor directory, padding and data follow.

`dataStart = alignUp(directoryEnd, alignment)`. Directory offsets are relative to `dataStart`. The absolute offset is the checked sum of the two values.

Each entry contains a name prefixed by its UInt64 byte length, UInt32 rank, UInt64 dimensions, UInt32 type and UInt64 offset. Allowed rank: 1–4. Dimensions must be positive and representable as Int64. The product of dimensions and tensor length must remain within Int64.

Unique names, valid UTF-8, no NUL, 1 to 63 bytes. The limit of 63 follows the pinned GGML reader; the specification text states 64.

Default alignment: 32 bytes. `general.alignment`, if present, must be UInt32 and a power of two between 8 and 4096. The specification describes multiples of eight; this profile adopts the stricter GGML reader constraint and adds the 4096 limit.

Allowed layout: directory in physical order, first relative offset zero, each subsequent offset equal to the previous one plus its length rounded up to alignment. The file must also contain final padding for the last tensor. Additional gaps, aliases, overlaps and physical order differing from the directory are rejected. Bytes after the last aligned region are ignored, never interpreted as tensors. Padding is not used to compute weights; its contents are not validated.

## Metadata

Allowed scalar types, with GGUF IDs:

| ID | Type | Bytes |
|---:|---|---:|
| 0 / 1 | UInt8 / Int8 | 1 |
| 2 / 3 | UInt16 / Int16 | 2 |
| 4 / 5 | UInt32 / Int32 | 4 |
| 6 | Float32 | 4 |
| 7 | Bool | 1 |
| 8 | UTF-8 String with UInt64 length | variable |
| 10 / 11 | UInt64 / Int64 | 8 |
| 12 | Float64 | 8 |

Array, ID 9: UInt32 element type, UInt64 element count, consecutive values. Scalar elements from the table are allowed, including strings. Empty arrays are allowed; nested arrays are excluded from the profile. Bool is valid only with byte 0 or 1. Nonfinite Float32/Float64 values are retained as raw metadata; they are not interpreted as inference parameters.

Unique ASCII keys, 1 to 65535 bytes; grammar `[a-z0-9_]+(\.[a-z0-9_]+)*`. Strict UTF-8 string values, with no required terminator. NUL is allowed in values, excluded from keys and tensor names. Do not normalize Unicode or change case or contents.

Unknown keys with known types are indexed. An unknown value type cannot be skipped without knowing its length: the file is rejected.

### Keys Interpreted by the Reader

| Key | Satori Rule |
|---|---|
| `general.architecture` | Required nonempty String `[a-z0-9]+`; identifies the file, does not enable an architecture |
| `general.type` | String if present, only `model`; other variants outside the profile |
| `general.quantization_version` | UInt32, required with Q4_0/Q8_0; allowed value 2 |
| `general.alignment` | Rule from the container section |
| `general.file_type` | UInt32 if present; informational, never replaces each tensor's type |
| `split.count` | UInt16 if present, 0 or 1 (0 is emitted by some single-file writers, for example Qwen2.5-Coder) |
| `split.no` | UInt16 if present, only 0 |
| `split.tensors.count` | Nonnegative Int32 if present, equal to the local tensor count |

Partial `split.*` keys are allowed if they satisfy the constraints. Other model or tokenizer keys are retained; their interpretation belongs to the model profile.

## Tensor Types

| Type | ID | Elements/Block | Bytes/Block | Bytes/Weight |
|---|---:|---:|---:|---:|
| F32 | 0 | 1 | 4 | 4 |
| F16 | 1 | 1 | 2 | 2 |
| Q4_0 | 2 | 32 | 18 | 0.5625 |
| Q8_0 | 8 | 32 | 34 | 1.0625 |

Length: `product(dimensions) / elementsPerBlock × bytesPerBlock`, with checked arithmetic. For Q4_0/Q8_0, the first dimension must also be a multiple of 32: blocks do not cross rows. Do not add logical padding as in internal Ternary2.

Any tensor type outside the table makes the file unsupported, even if the tensor is not selected. This allows validation of the entire data region without estimating unknown lengths. A Q4_K_M file with Q4_K/Q6_K is outside the profile.

### Q4_0/Q8_0 Numerical Contract

First two bytes of the block: little-endian FP16 scale, converted to FP32.

- Q4_0: 16 bytes follow. For `j = 0..15`, the low nibble produces `w[j] = d × ((b[j] & 15) - 8)`; the high nibble produces `w[j+16] = d × ((b[j] >> 4) - 8)`.
- Q8_0: 32 Int8 values follow. `w[j] = d × q[j]`; the entire Int8 range is representable.

Scales may be negative, zero or negative zero. Do not apply Ternary2 validation rules. T12 validates the container and ranges without numerically scanning weights. T13/T14 check scales of the selected tensor once and reject NaN/Infinity before execution. F32/F16 are cataloged; T12 promises no specific MatVec for them.

For a rank-2 matrix: `columns = dimensions[0]`, `rows = dimensions[1]`. The first dimension is contiguous. Initial activations and accumulation use FP32; the numerical reference uses FP64 on the same decoded weights. Q4_0 order is not the order of four consecutive two-bit weights in Ternary2.

## Parser Limits

Fixed Satori limits for the first version; these are not standard limits.

| Resource | Maximum |
|---|---:|
| File | 1 TiB, 64-bit process |
| Tensors | 100000 |
| Metadata pairs | 100000 |
| Value string | 16 MiB of UTF-8 |
| Elements per array | 1048576 |
| Header + metadata + directory + initial padding | 256 MiB |

Zero tensor count is excluded: vocabulary-only files are not needed by the laboratory. No allocations proportional to unchecked counters. Before each read or skip, check the logical limit and remaining bytes. All product, sum, padding and UInt64→Int64 conversion calculations are checked.

The catalog uses Int64 offsets and lengths. Windows exposed to the CPU have Int32 lengths. A tensor larger than `int.MaxValue` bytes can be indexed and read in windows; initial T13/T14 MatVec operations may reject it as unsupported.

## T12 Deliverable

Module: `src/Satori.Formats.Gguf`, depending on `Satori.Core` for mapping. `Satori.Core` does not depend on the GGUF module. Future numerical kernels receive payload and shape; the GGUF adapter selects the type without introducing GGUF into kernels.

- A mapping owner with an immutable tensor catalog.
- Metadata indexed by key, type, position and length; arrays are not automatically materialized.
- Typed scalar and string reads on request; scoped access to array bytes with declared type and count.
- Weight access through windows protected by the `MappedWeights` lifetime.
- No full file copy, no dequantization on opening.

Tensor access specifies a relative offset and window length. The reader checks the tensor boundary before delegating to the mapping. A window cannot read padding or the next tensor's data.

Malformed or truncated file: `InvalidDataException`, with position and context. Excluded version, endianness, type or variant: `NotSupportedException`. Satori limits exceeded: `NotSupportedException`, naming the limit. Filesystem errors: original I/O exceptions. Access after disposal: `ObjectDisposedException`. A failed open releases all resources. No partial catalog is published.

## Planned Verification

The [fixture catalog](gguf-fixtures-v1.md) defines valid, out-of-profile and corrupted cases. The reader is verified on golden fixtures, corrupted and large containers, and the public GGUF verification corpus. MatVec results are compared with independent FP64 decoding; evidence is in `docs/benchmarks/gguf-2026-10-12/`.
