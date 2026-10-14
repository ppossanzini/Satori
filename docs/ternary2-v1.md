# Ternary2 v1

Internal experimental format, distinct from GGUF formats. Row-major matrix with positive row and column counts. Blocks of 64, 128 or 256 weights, without crossing rows. The last block of each row is padded with zeros.

Each block occupies `4 + blockSize / 4` bytes: little-endian IEEE754 FP32 scale, followed by two-bit weights. Within a byte, the first weight occupies bits 0–1; bits 2–3, 4–5, 6–7 follow. `00 = 0`, `01 = +1`, `10 = -1`, `11` is invalid. Reconstructed weight: signed code × scale. The scale is finite and nonnegative; zero blocks have scale +0 and a zero payload. Padding is always zero. With a zero scale, all logical weights must also be zero.

Deterministic quantization: scale = mean absolute value of valid weights in the block only, accumulated in FP64 and converted to FP32. If the scale is zero, the block is zero. Otherwise select +1 for `x > scale / 2`, −1 for `x < −scale / 2`, and zero otherwise, including ties. The threshold is computed in FP64 to avoid underflow on subnormals. This quantization is not calibrated for LLM quality.

Nonfinite inputs are rejected. Dimensions and lengths use checked arithmetic; each matrix payload allows at most `int.MaxValue` bytes. In-memory import copies and validates the payload before making it available to kernels. The raw payload contains no header: shape and block size are explicit arguments. The STQ2 file container below adds this metadata.

Example: block of 64, first four values `[2, -2, 0, 0]`, all others zero. Scale = 0.0625; first four bytes `00 00 80 3D`, first weight byte `09`, remaining 15 weight bytes zero. An all-zero block occupies 20 zero bytes.

MatVec: FP32 activations, FP32 sum of signed contributions per block, multiplication by scale, FP32 accumulation across blocks. No full expansion or allocation in the kernel. Accumulation error must be compared with FP64 on the same reconstructed weights; quantization error must be measured separately against original weights.

## STQ2 v1 Container

Little-endian file: fixed 32-byte header, followed by the Ternary2 v1 payload. No additional trailing data is allowed.

| Offset | Type | Content |
|---:|---|---|
| 0 | 4 ASCII bytes | `STQ2` |
| 4 | UInt16 | Version: 1 |
| 6 | UInt16 | Header length: 32 |
| 8 | Int32 | Positive row count |
| 12 | Int32 | Positive column count |
| 16 | Int32 | Block size: 64, 128 or 256 |
| 20 | UInt32 | Reserved: zero |
| 24 | Int64 | Payload length, equal to the length derived from the shape |

`Ternary2Matrix.Save` creates a new file without overwriting existing files. An I/O error during writing may leave a partial file; the loader rejects truncated headers/payloads. This is neither an atomic save nor a GGUF format.

`MappedTernary2Matrix` validates the header and entire payload once on opening, without copying it into an array. Each MatVec acquires the mapping for the entire computation and calls the same scalar/AVX2 kernels as the in-memory matrix. Closing does not invalidate a computation already started, but prevents subsequent ones.

The generic owner supports files with offsets beyond 2 GiB; the current container is limited to a single matrix payload of at most `int.MaxValue` bytes. Larger individual tensors, tensor collections and checksums are not yet supported. The file must remain immutable after validation and until active accesses finish.
