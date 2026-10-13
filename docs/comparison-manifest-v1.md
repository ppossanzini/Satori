# Candidate Comparison Manifest v1

Status: DW02, in progress. Pins the three candidates, the reference runtimes, the test hardware and the
workload for the P01–P03 comparison. Candidate tensor capability stays owned by
[the capability matrix](capability-matrix-v1.md); this manifest fixes identity, provenance and environment.

## Candidates

Revision is the Hugging Face commit. Size and SHA-256 are the full LFS file. Types are the tensor-directory
histogram from the DS02 header inventory.

| Role | Artifact | Repository @ revision | File | Size (B) | SHA-256 | License |
|---|---|---|---:|---|---|---|
| Coding reference (Q4) | Qwen2.5-Coder 7B Instruct Q4_K_M | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` @ `13fb94bfda8c8cf22497dc57b78f391a9acb426a` | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | 4683073536 | `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` | Apache-2.0 |
| Hybrid MoE | Tiel-Coder 35B-A3B APEX-I-NanoPlus | `IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF` @ `ef9ff9001e8205458bf842fd13f43e5318eb2d46` | `Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf` | 12554424672 | `18f6b0bd47dacad208aea82b1d5060309c4b8bca4de2028e5dd9dcdbf7de56f4` | MIT (card) |
| Ternary | Ternary-Bonsai-2 27B PTQ1_0 | `prism-ml/Ternary-Bonsai-2-27B-gguf` @ `b072e1d3b35a0a630cece372c2127528e0994386` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5946648928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` | Apache-2.0 |

| Artifact | Architecture | Tensor types | `general.file_type` |
|---|---|---|---|
| Qwen2.5-Coder 7B Q4_K_M | qwen2 | Q4_K 169, F32 141, Q6_K 29 | 15 |
| APEX-I-NanoPlus | qwen35moe | F32 331, Q4_K 241, IQ2_S 50, IQ3_XXS 44, Q8_0 30, IQ2_XXS 22, Q6_K 11, Q3_K 4 | 15 |
| Bonsai PTQ1_0 | qwen35 | unknown(143) 402, F32 353, BF16 96 | 143 |

APEX optional vision projector `mmproj-Q8_0.gguf` (614194176 B) is out of scope for the text comparison.
Bonsai optional projectors `mmproj-BF16.gguf` and `mmproj-Q8_0.gguf` are out of scope.

## APEX provenance

Resolved from the binary descriptor, which is authoritative over the card tags.

- `general.architecture = qwen35moe`, `general.name = Ornith-1.5-35B`, `general.size_label = 256x2.6B`,
  256 experts, 8 active per token.
- Card `base_model = ornith-ai/Ornith-1.5-35B-A3B`, `quantized_by = IsValorum`, `license = mit`.
- Card tags mix `qwen3.6`, `qwen35moe` and `qwen3_5_moe`: aliases of one Qwen3.5-MoE lineage. The base
  identity is Qwen3.5-MoE, not Qwen2 or Qwen3 dense.
- The APEX recipe is mixed, not a single codec: the expert projections use the K/IQ types above. The
  `general.file_type = 15` label is an artifact selector, not a per-tensor claim.

## Reference runtimes

Built from source on the Intel host, CPU only.

| Artifact | Reference runtime | Revision | State |
|---|---|---|---|
| Qwen2.5-Coder 7B Q4_K_M, APEX-I-NanoPlus | upstream llama.cpp | `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444` (0.6.0-dev, ggml 0.26.0) | built |
| Bonsai PTQ1_0 | Prism llama.cpp fork | `e8fd209c975c80aa7fb36a8423571ff3b1b7e955` (ggml 0.21.0) | built |

Build flags for both: `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF
-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF`, targets `llama-completion` and
`llama-bench`. With the server off the combined `llama-app` target does not link; the `llama-completion`
tool is the CLI. Binaries under `.temp/deepseek/dw02/llama.cpp/build/bin` and
`.temp/deepseek/dw02/llama-prism/build/bin`.

Artifact identity confirmed on disk: Qwen2.5-Coder 7B Q4_K_M SHA-256 and size match the manifest row above.

Reference-runtime load confirmed: Qwen2.5-Coder 7B Q4_K_M loads and generates in upstream `llama-completion`
(greedy completion of `def add(a, b):` → `return a + b`; 8 threads, prompt eval 7.84 tok/s, eval 2.67 tok/s
at context 131072). APEX-I-NanoPlus loads and generates in the same runtime (greedy `def add(a, b):` →
`return a + b`; load 4.1 s, prompt eval 4.94 tok/s, eval 1.15 tok/s, context 262144), so upstream supports
the `qwen35moe` architecture and the mixed K/IQ types. Bonsai PTQ1_0 loads and generates in the Prism
runtime (greedy `def add(a, b):` → `return a + b`; load 18.9 s, prompt eval 1.01 tok/s, eval 0.52 tok/s,
context 262144).

Observation, not a measurement: the Bonsai PTQ1_0 process reached ~22 GB RSS during load and generation on
this host, above the 12 GB budget and the 16 GB target. Formal memory figures are produced in the DW04
measurements.

All three artifact SHA-256 values match the manifest rows above.

## Hardware and budget

- Primary test host: Geekom A5, Ryzen 7, 32 GB, Arch Linux (protocol section 4). Not this host.
- Intel comparison host: Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux (this host).
- Physical RAM and the enforced runtime budget are separate. Every report states host RAM and the
  enforcement method. The constrained configuration target is 12 GB (protocol section 8).
- Contexts: 2K and 4K.

## Open decisions

| Item | Required from | Note |
|---|---|---|
| Download authorization | owner | 24.2 GB total for the three text files |
| Test host for the comparison | owner | primary Geekom A5 is not this host |
| C# corpus freeze | owner | task count, prompts, difficulty, compile/run criteria (protocol section 3) |
| Concurrent workload freeze | owner | IDE, browser, build, observation window (protocol section 6) |

## Verify

```sh
python3 tools/download_qwen_models.py     # Qwen corpus (already used by the DS11 baseline)
python3 tools/gguf_header_inventory.py    # regenerate the descriptor manifests
```
