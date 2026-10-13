# Capability Matrix v1

Status: DS02. Derived only from descriptor inventories of the candidate artifacts at pinned revisions.
No weights were downloaded and no numerical claim is made here; the matrix states what must be added for
each artifact to run, not that it runs.

## Method

For each candidate, only the header/descriptor region (first 32 MiB) was fetched over an HTTP range
request and read by `Satori.Console inventory --json`, which touches header, metadata and tensor
directory but never the tensor payload. Full-file SHA-256, Hugging Face revision and source URL are
recorded in `docs/scouting/inventories/manifest.json`; one inventory JSON per artifact sits beside it.
Tool: `tools/gguf_header_inventory.py` (reproducible).

Capability is read from the file's own metadata and tensor directory, never from its name. Evidence: the
official `qwen2.5-coder-1.5b-instruct-q4_0.gguf` declares 1 Q6_K tensor (`output.weight`) besides 197
Q4_0 and 141 F32, so a file named `q4_0` is not executable without Q6_K.

## Artifact Inventories

Revision is the Hugging Face commit. Types are the tensor directory histogram; `unknown(n)` is a type id
Satori does not catalogue.

| Artifact | Architecture | Tensors | Tensor types | `general.file_type` |
|---|---|---:|---|---:|
| qwen2.5-coder-1.5b q4_0 | qwen2 | 339 | Q4_0 197, F32 141, Q6_K 1 | 2 |
| qwen2.5-coder-1.5b q8_0 | qwen2 | 339 | Q8_0 198, F32 141 | 7 |
| qwen2.5-coder-1.5b q4_k_m | qwen2 | 339 | Q4_K 169, F32 141, Q6_K 29 | 15 |
| qwen2.5-coder-7b q4_0 | qwen2 | 339 | Q4_0 197, F32 141, Q6_K 1 | 2 |
| qwen2.5-coder-7b q4_k_m | qwen2 | 339 | Q4_K 169, F32 141, Q6_K 29 | 15 |
| qwen3-4b Q4_K_M | qwen3 | 398 | Q4_K 216, F32 145, Q6_K 37 | 15 |
| qwen3-4b Q8_0 | qwen3 | 398 | Q8_0 253, F32 145 | 7 |
| bitnet-b1.58-2b i2_s | bitnet-b1.58 | 332 | unknown(36) 210, F32 121, F16 1 | 40 |
| apex-i-nanoplus | qwen35moe | 733 | F32 331, Q4_K 241, IQ2_S 50, IQ3_XXS 44, Q8_0 30, IQ2_XXS 22, Q6_K 11, Q3_K 4 | 15 |
| bonsai PTQ1_0 | qwen35 | 851 | unknown(143) 402, F32 353, BF16 96 | 143 |
| bonsai PQ2_0 | qwen35 | 851 | unknown(142) 402, F32 353, BF16 96 | 141 |

## Architecture

| Architecture | Present in | Satori status | Reason |
|---|---|---|---|
| qwen2 | coder 1.5B/7B | Missing runtime | No metadata-driven dense runtime; needs QKV bias, GQA, partial RoPE from metadata |
| qwen3 | 4B | Missing runtime | Adds explicit `key_length`/`value_length` (128) and Q/K norm tensors |
| bitnet-b1.58 | 2B | Missing runtime | Custom tensor type 36, `attention.head_count_kv` 5, ReLU² activation, subln |
| qwen35moe | APEX | Missing runtime | MoE routing, hybrid SSM/full attention, M-RoPE sections |
| qwen35 | Bonsai | Missing runtime | Hybrid SSM/full attention, Hadamard transformations, ternary tensors |

Dense and hybrid shapes are declared by metadata, not derived: `qwen3-4b` declares
`qwen3.attention.key_length = 128` while `embedding_length / head_count = 2560 / 32 = 80`. Deriving head
dimension from `hidden/heads` would be wrong for this artifact.

## Tokenizer

All artifacts use `tokenizer.ggml.model = gpt2` (BPE) with `tokens`, `token_type` and `merges` arrays and
a `tokenizer.chat_template` string.

| Artifact family | `tokenizer.ggml.pre` | Extra | Status |
|---|---|---|---|
| qwen2 / qwen3 / qwen35 / qwen35moe | qwen2, qwen35 | chat template with tool-call markup | Missing BPE tokenizer + template |
| bitnet-b1.58 | absent | Llama-3 style merges, `add_bos_token = true` | Missing BPE tokenizer + template |

The laboratory tokenizer (`StoriesTokenizer`) is a SentencePiece profile and does not cover these
gpt2/BPE files.

## Operators

Operator presence is taken from tensor names per layer.

| Requirement | Evidence | Satori status |
|---|---|---|
| QKV bias | `blk.N.attn_q/k/v.bias` in qwen2 | Missing (RMSNorm/RoPE/attention present) |
| Q/K per-head norm | `blk.N.attn_q_norm/k_norm.weight` in qwen3 | Missing |
| GQA | `head_count_kv < head_count` in every artifact | Attention supports causal prefix; grouped-query path missing |
| Partial / M-RoPE | `rope.dimension_count = 64` with `key_length = 256` (qwen35*); `rope.dimension_sections = [4]` | Missing |
| MoE routing | `ffn_gate_inp`, `ffn_*_exps`, `ffn_*_shexp` (APEX) | Missing |
| Hybrid SSM | `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`, `ssm_a`, `ssm_norm`, `ssm_out` (APEX, Bonsai) | Missing |
| Fusion gate / post-attention norm | `attn_gate`, `post_attention_norm` (APEX, Bonsai) | Missing |
| ReLU² | `hidden_act = relu2` (bitnet) | Missing (SwiGLU present) |
| Packed QKV | `attn_qkv.weight` (APEX, Bonsai) | Missing |

## Codec

Type ids seen in the inventories. `Supported` is the current profile; everything else makes the whole
file unusable today.

| Type | Id | Present in | Satori | Required by |
|---|---:|---|---|---|
| F32 | 0 | all | Supported | n/a |
| F16 | 1 | bitnet | Supported | non-quantized reads |
| Q4_0 | 2 | coder 1.5B/7B q4_0 | Supported | baseline |
| Q8_0 | 8 | coder 1.5B/7B/4B q8_0, APEX | Supported | baseline |
| Q3_K | 11 | APEX | Missing | APEX |
| Q4_K | 12 | all q4_k_m, APEX | Missing | coder Q4_K_M, Qwen3, APEX |
| Q6_K | 14 | all q4_0/q4_k_m, Qwen3, APEX | Missing | every listed non-pure file |
| IQ2_XXS | 16 | APEX | Missing | APEX |
| IQ3_XXS | 18 | APEX | Missing | APEX |
| IQ2_S | 22 | APEX | Missing | APEX |
| BF16 | 30 | Bonsai | Missing | Bonsai |
| unknown | 36 | bitnet (210) | Unverified | BitNet `I2_S`; identity/schema must be confirmed, not guessed |
| unknown | 143 | Bonsai PTQ1_0 (402) | Unverified | Prism `PTQ1_0`; private fork type |
| unknown | 142 | Bonsai PQ2_0 (402) | Unverified | Prism `PQ2_0`; private fork type |

The mixed recipes select a kernel per tensor type; the file prefix (Q4_K_M, UD-IQ, APEX) is an artifact
selector, not a codec.

## Transformations

| Transformation | Evidence | Satori status |
|---|---|---|
| Prism Hadamard on inputs | `prism.hadamard.transform = normalized-sylvester-walsh-hadamard`, `block_size = 1024`, `axis = input-last-dimension`, `sign_mode = explicit`, `weight_names` (401), `sign_values` (28672) | Missing; mandatory for Bonsai |
| Ternary rotation semantics on embedding/output | `prism.hadamard.inverse_weight_names`, `gdn_v_grouped` | Missing |

## Backend

Vulkan exposes Q8_0 MatVec and FP32 operator primitives only, with no model-level offload or planner.
None of the artifacts above can run on either backend until the CPU runtime exists; the Vulkan column is
informational and not a DS02 deliverable.

| Backend | Available now | Status for these artifacts |
|---|---|---|
| CPU scalar/AVX2 | Ternary2, Q4_0, Q8_0 MatVec; RMSNorm, RoPE, attention, SwiGLU | Missing every codec and operator above |
| Vulkan | Q8_0 MatVec, FP32 operators | No model-level path |

## Gaps the Reader Must Close (evidence only)

These are observed from the artifacts and must be resolved before generation. No kernel is promised here.

1. `general.architecture` value `bitnet-b1.58` contains `.` and `-`; the strict reader's `[a-z0-9]` rule
   rejects it.
2. `split.count = 0` and `split.no = 0` (UInt16) appear in real single-file models; the strict reader
   requires `split.count = 1` and would reject them.
3. The strict reader rejects any tensor type outside `{F32, F16, Q4_0, Q8_0}`, so every artifact above
   except the pure q8_0 files fails to open.
4. Types 36, 143 and 142 are reported as `unknown`; they are private/fork types whose layout must be
   verified against their own reference before being catalogued (BitNet `I2_S`, Prism `PTQ1_0`/`PQ2_0`).

## Explicit Rejections

- No capability is inferred from a file name, publisher or size.
- No decoding formula, tolerance or quality claim is made in this matrix.
- Private fork tensor types are not catalogued until their schema is verified.
- Vision/mmproj files are out of scope for these text artifacts.
