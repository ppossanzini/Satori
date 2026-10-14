# TernaryBitplane v1

Internal CPU format, independent of GGUF. Same shapes, scales and logical weights as [Ternary2 v1](ternary2-v1.md); no new quantization rule. Blocks of 64, 128 or 256 elements, local to each row; final row/block padded with zeros. Positive dimensions; payload within Int32 using checked arithmetic.

## Block

| Region | Offset | Bytes |
|---|---:|---:|
| Little-endian FP32 scale | 0 | 4 |
| Positive plane P | 4 | blockSize/8 |
| Negative plane N | 4+blockSize/8 | blockSize/8 |

Logical weight i uses bit `i%8` of byte `i/8`, least significant first. `(P,N)`: (0,0) zero, (1,0) +1, (0,1) −1. (1,1) is invalid. Reconstructed weight: `scale × (P−N)`.

Finite, nonnegative scale, excluding negative zero. Zero scale: both planes entirely zero. Padding contains no set bits. Payload per block: `4+blockSize/4` bytes, identical to Ternary2; the layout changes decoding, not the byte count.

Example with blockSize 64, scale 0.5, first eight codes `[+1,0,−1,+1,−1,0,0,+1]`: scale `00 00 00 3F`, first P byte `89`, first N byte `14`; all other bytes zero. Result with an input of ones: 0.5. Equivalent Ternary2 starts with code payload `61 42` after the same scale.

## Lossless Conversion

From validated Ternary2: copy the four scale bytes without recomputing the scale; convert code 01 to P, 10 to N, 00 to no bit. No change to order, blocks or padding. Deterministic reverse conversion: P→01, N→10. Byte-identical round trip to the original Ternary2, including FP32 subnormals and partial blocks.

Do not requantize from reconstructed values: the original threshold could produce different codes. Conversions may allocate; MatVec on an already constructed object does not allocate.

## STB1 Container

32-byte header, analogous to but distinct from the STQ2 container:

| Offset | Field |
|---:|---|
| 0 | ASCII magic `STB1` |
| 4 | UInt16 LE version = 1 |
| 6 | UInt16 LE header size = 32 |
| 8 / 12 / 16 | Int32 LE rows / columns / block size |
| 20 | Reserved UInt32 = 0 |
| 24 | Int64 LE payload length |

Exact file: header + payload. No checksum. Save creates a new file; a failed write may leave an incomplete file. Import copies and validates; load validates header and payload before publishing the object. Mapping of the new container is not included: the layout comparison uses both payloads owned in memory to isolate decoding cost.

## Numerics and Benchmarks

Scalar MatVec: FP32 accumulation of signed contributions within a block, multiplication by scale and accumulation across blocks. AVX2: eight activations at a time, masks from the planes, scalar tail without out-of-bounds input reads. No dependency on FMA or AVX-512; scalar fallback.

FP64 tolerance: same function and corpus as Ternary2. Scalar kernels for both layouts follow the same column order and must match; SIMD may differ within the limit documented in the benchmark method. Finite arithmetic is not guaranteed if FP32 accumulation overflows.

The T17 benchmark measures complete scalar MatVec, including decoding, on matrices of size 1024/2048/4096 and block sizes 64/128/256. T18 adds both SIMD kernels. Scales, codes, input, hardware, process, warm mode and thread count are identical. Conversion and validation are outside measured time. Report zero allocations and bytes/weight too; unpack-only measurements must not be presented as MatVec speed.

The AVX2 implementation may be exposed explicitly; automatic selection only after correctness and a measured advantage. A kernel without an advantage remains experimental. No promises about token/s, quality or model RAM usage.
