# Resident Vulkan Q8_0 MatVec

`VulkanContext.CreateQ8Matrix` copies validated serialized Q8_0 weights into an owned immutable GPU buffer. The mapped file or caller payload may then be released. Each matrix owns input/output device buffers and vector staging/readback buffers; only `(columns + rows) * 4` bytes transfer per execution. No full FP32 weight expansion is retained. Context disposal closes its matrices; operations and disposal serialize on the context lock.

The baseline shader assigns one invocation to a row, with 64 invocations per workgroup and guarded row tails. Each 34-byte block contains a little-endian half scale and 32 signed bytes. Byte extraction from uint storage handles unaligned blocks; zero padding rounds the final buffer to four bytes. Half decoding uses `unpackHalf2x16`, without requiring native FP16 or int8 arithmetic. `precise` preserves sequential FP32 block sums and scale accumulation. Q4_0 and model offload remain outside this implementation.

Creation checks exact payload length, finite scales, storage/dispatch/descriptor/push-constant limits and the device allocation count. Execution checks dimensions, overlap and finite input before modifying output. Buffer barriers connect upload, compute, transfer and host read; a fence precedes query retrieval and output commit. Noncoherent staging uses flush/invalidate. Submission failures fault the context; disposal waits for pending device work before freeing resources. Device-loss injection and noncoherent hardware paths were not exercised.

Optional timestamps bracket dispatch at the compute stage. They report a GPU dispatch window, including timestamp/dispatch overhead, rather than isolated arithmetic. Query availability and timestamp wrap are handled; unsupported timestamps return null. Host timing includes upload, command recording, submission, fence, query retrieval and readback; it excludes input validation, lock acquisition and initial weight upload. See the Vulkan reference for [timestamp ordering](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html) and [query result availability](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html).

## Evidence

[Raw measurements](benchmarks/vulkan-q8-2026-10-12/q8-matvec.json): Intel Arc MTL, .NET 10, Khronos synchronization validation enabled. All 30 synthetic cases and all 44 Stories15M Q8 matrices passed the independent FP64 decode/dot oracle, as did CPU scalar and automatic AVX2. The maximum GPU error/bound ratio was 0.003338. The bound is `gamma(n) * sum(abs(products)) + n * float.Epsilon`, with `n = columns + columns / 32 + 2` and unit roundoff `2^-24`.

Synthetic cases cover signed quantized values, positive/negative/zero/subnormal/maximal half scales, 32–768 columns, unaligned block starts and 1/7/63/64/65/129 rows. Nine GPU tests cover these cases, independent mapped-weight ownership, repeated calls, invalid inputs, zero warm managed allocation and concurrent execution/disposal. The 114 CPU numerical tests also pass.

Three warmups and fifteen sequential samples per real matrix separate CPU scalar, CPU automatic, GPU kernel and host times. Median GPU kernel times range from 0.162 to 18.409 ms; host times from 1.736 to 27.541 ms. CPU AVX2 medians range from 0.021 to 2.692 ms. For `output.weight` (32000 × 288), GPU kernel/host medians are 17.658/26.440 ms versus CPU AVX2 2.338 ms. All measured GPU calls allocate zero managed bytes. These validation-enabled diagnostic measurements establish correctness and expose the slow baseline; they do not demonstrate acceleration or predict whole-model latency. Model integration is documented in [Stories15M offload](stories15m-vulkan-offload.md).

## Reproduction

Build Release and compile shaders using `python3 tools/compile_vulkan_shaders.py` (offline glslang and spirv-val required). Stage Khronos validation layers locally as described in [minimal path](vulkan-minimal-path.md), replacing the temporary directory with `.temp/t29`. Then:

```sh
VK_LAYER_PATH="$PWD/.temp/t29/layer/usr/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$PWD/.temp/t29/layer/usr/lib" VK_VALIDATION_VALIDATE_SYNC=1 \
dotnet run --project validation/Satori.Vulkan.Validation -c Release --no-build -- \
  --q8 .temp/gguf-corpus/stories15M-q8_0.gguf
```

The report records the model SHA256, device, allocation sizes, cold upload, numerical errors and raw timing samples. Temporary validation layers are removed after verification; they must be restaged for subsequent runs.
