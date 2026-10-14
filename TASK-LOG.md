# Verification history

Historical task entries describe their original scope and may mention removed implementations. Current product requirements and executable work are defined only in TASKS.md.

# Stories-specific optimization removal

Removed Stories-specific Vulkan offload, resident model/layer graphs, measured planner, attribution and benchmark runners, CLI backend options and their four dedicated test files. Preserved CPU Stories15M regression fixtures and reusable Vulkan numerical primitives. Removed the model-specific optimization plan; T46-T50 are cancelled in their previous formulation. Published measurements remain historical evidence.

Verification: Release solution build passed. All 34 CPU Stories regression tests passed. Vulkan hardware verification was blocked by missing Khronos validation layer/debug utils before kernel execution.

Skills: capabilities-dev-problem-solving-methodology, phase-development-project-conventions, base-dossier-writing, run-tests.

# DeepSeek Plan Execution

Tasks DS01–DS40 from DeepSeek_todo.md. One task per commit; no push.

## DS01 — 2026-10-13 — Completed

Exposed the descriptor inventory of a GGUF container through `GgufInventory` (reusing `MappedWeights` and `GgufCursor`) and the console command `inventory <file.gguf> [--json]`. It reports architecture, version, alignment, data offset, tensor and metadata counts, every metadata key with its declared type and decoded scalar value or array shape, and every tensor with type, dimensions, declared offset, computed byte length and support status. `GgufTypes` adds the GGML type catalogue with block geometry and profile support.

Unsupported tensor types are reported explicitly instead of rejecting the file, which `GgufFile` still does for execution. The scan and the validated catalogue agree on the four golden fixtures (tensor name, type, dimensions, absolute offset, byte length, data offset, alignment, counts), checked in `GgufInventoryTests`.

Verification: Release build without warnings; 93/93 offline numerics tests passed (5 new); demo on `fixtures/gguf/v1/q8_0.gguf`, `f32.gguf` and the tokenizer fixture (arrays described, not materialised); non-GGUF input rejected with a readable message and exit code 1.

Reproduce: `dotnet run --project src/Satori.Console -c Release -- inventory fixtures/gguf/v1/q8_0.gguf`.

Limits: header, metadata and directory only; no weight decoding or execution; an unknown metadata value type is rejected because it cannot be skipped; private fork types (Prism PTQ1_0/PQ2_0) are not catalogued. Contract in `docs/gguf-inventory.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.


## DS02 — 2026-10-13 — Completed

Pinned the first capability matrix from descriptor inventories of the candidate artifacts, without
downloading weights: a 32 MiB ranged download of each header/descriptor region, read by the DS01
inventory. Tool `tools/gguf_header_inventory.py`; per-artifact JSON and provenance in
`docs/scouting/inventories/`; matrix in `docs/capability-matrix-v1.md`.

Eleven artifacts at pinned revisions were inventoried: Qwen2.5-Coder 1.5B/7B (q4_0, q8_0, q4_k_m),
Qwen3 4B (Q4_K_M, Q8_0), BitNet b1.58 2B (i2_s), Tiel-Coder APEX-I-NanoPlus and Ternary-Bonsai-2 27B
(PTQ1_0, PQ2_0). Each inventory records architecture, tensor types with counts and `general.file_type`.

Findings: the official `q4_0` files carry a Q6_K `output.weight`, so capability is never deduced from the
file name. Unknown types are explicit: 36 (BitNet `I2_S`, 210 tensors), 143 (Prism `PTQ1_0`), 142 (Prism
`PQ2_0`); BF16 and the K/IQ types are listed as missing. The matrix covers architecture, tokenizer,
operators, codec and transformations with supported/missing and the reason, and lists four reader gaps
(`bitnet-b1.58` architecture string, `split.count = 0`, strict type set, private types).

Verification: inventory tool ran end-to-end; eleven inventories and the manifest produced; matrix written.

Reproduce: `python3 tools/gguf_header_inventory.py`.

Limits: descriptor region only; no weights, no decoding, no quality claim; APEX/Bonsai private types remain
unverified.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS03 — 2026-10-13 — Completed

Specified the generic runtime/session and memory-plan contracts in `docs/runtime-contracts-v1.md`: an
immutable `ModelDefinition` with descriptor-only content and metadata-derived dimensions; a capability
check producing explicit gaps before any weight read; a scoped `IWeightSource` with resident and streamed
implementations and shared/tied alias handling; `IInferenceSession` with KV and recurrent state layouts
selected per layer; a `MemoryBudget` with disjoint categories and `Unified`/`Discrete` domains that avoid
double counting on integrated GPUs; residency and prefetch; cancellation; and the error/lifetime
invariants.

Two topologies (dense grouped-query and hybrid MoE) are shown representable by the same definition with no
nominal dimensions; head dimension is taken from explicit `key_length`/`value_length` when present, never
from `embedding_length / head_count`. No runtime implementation was added; the requirement stays in
TASKS.md. Adapters are selected by `ArchitectureId` and never leak a model name into core interfaces.

Verification: two topologies represented without nominal dimensions; budget expressed in bytes with a
single owner per allocation and no iGPU double counting; error and lifetime invariants written.

Limits: specification only; DS05 onward implements it.

Skills: phase-development-project-conventions, base-dossier-writing.

## DS04 — 2026-10-13 — Completed

Fixed the validation protocol in `docs/validation-protocol-v1.md`: the artifact set for correctness,
quality and performance; independent numerical references and per-operator tolerances; a C# coding task set
with compile/run criteria, Italian prompts and long outputs, with quality prompts excluded from
calibration; target hardware and 2K/4K contexts; the startup/prefill/TTFT/decode/memory/I-O measurement
groups; the manifest fields; and the separation of correctness, quality and performance. No advertised
benchmark is accepted as an oracle and no `sleep` or theoretical bandwidth counts as a measurement.

Owner-approved thresholds (2026-10-13): coding C# pass rate 80%; peak RAM under 12 GB. Decode rate and TTFT
at 2K were left open by the owner and are reported as data only. A GPU result must beat CPU by a stated
factor, to be confirmed before DS39.

Verification: manifest fields, commands and provenance defined; the three axes kept separate; regenerable
vs durable output locations defined; thresholds recorded where approved.

Limits: decode/TTFT target and the GPU factor are still open; no product threshold was invented.

Skills: phase-development-project-conventions, base-dossier-writing.

## DS05 — 2026-10-13 — Completed

Implemented the validated model descriptor (`src/Satori.Core/Model`) and the GGUF loader
(`GgufModelLoader`). `ModelDefinition` is immutable and descriptor-only; `ModelConfig` reads scalar metadata
by architecture prefix; `ModelShape` holds metadata-derived dimensions; `ModelPlanner` computes a checked
shape/memory plan (weights, F32 KV state, workspace upper bound) against a byte budget; `IArchitectureAdapter`
and `ArchitectureAdapters` form the capability registry with `DenseTransformerAdapter` for qwen2/qwen3.
`CapabilityGap`/`CapabilityReport` make every missing capability explicit: tensor format, architecture and
shape. Head dimension comes from `attention.key_length` when declared, never forced to
`embedding_length / head_count`. Console command `describe <model.gguf> [context]`.

Demo on two topologies from the DS02 header prefixes: qwen2.5-coder-1.5b q8_0 (28 layers, 12/2 heads,
head_dim 128, executable, plan 2.0 GB at 2K) and qwen3-4b q8_0 (36 layers, 32/8 heads, explicit
key_length 128, tied output, executable, plan 4.9 GB at 2K). qwen2.5-coder-7b q4_k_m derives its shape and
reports Q4_K/Q6_K format gaps; BitNet reports an unknown type 36 format gap and an architecture gap.

Verification: Release build without warnings; 102/102 offline numerics tests passed (9 new in
`ModelDescriptorTests`): shape incompatibility, indivisible head dimension, missing required metadata,
unknown architecture, plan overflow (Int64), budget excess, and unsupported tensor type becoming a format
gap.

Limits: plan is an estimate from metadata; only qwen2/qwen3 adapters exist; the descriptor validates, it
does not decode or execute. Contract in `docs/model-descriptor.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS06 — 2026-10-13 — Completed

Implemented the byte-level BPE tokenizer (`Satori.Core.Tokenization`): `BpeTokenizer` (NFC → pre-tokenizer
split → byte-level map → BPE merges; added/special tokens matched on the raw input), `ByteLevel`,
`PreTokenizers` (patterns keyed by `tokenizer.ggml.pre`, unknown families resolve to null) and a generic
`ChatMlTemplate`. `GgufTokenizerReader`/`GgufBpeTokenizer` read `tokenizer.ggml.tokens/merges/token_type`,
scalar ids and the chat template from the container using the tolerant scan, so a model with unsupported
tensor types still yields its tokenizer.

Fixture `fixtures/tokenizers/qwen2.5-coder-1.5b/` is generated by `tools/qwen_bpe_tokenizer_fixture.py` from
the official `Qwen/Qwen2.5-Coder-1.5B-Instruct` `tokenizer.json`: a tokenizer-only GGUF (5,924,288 bytes,
vocab 151665, merges 151387) plus `reference.json` with the official token ids and the
`apply_chat_template` render, with tokenizer.json/fixture SHA-256 and library versions.

Verification: Release build without warnings; 107/107 offline numerics tests passed (5 new in
`BpeTokenizerTests`). Token-id identity with the official tokenizer on Italian, C#, emoji, CJK, accented
text, irregular whitespace, special tokens and the empty string; decode round-trip; ChatML render equals the
official `apply_chat_template` for an explicit-system conversation. Provenance: tokenizers 0.23.2,
transformers 5.19.0, Python 3.14.

Reproduce: `python3 tools/qwen_bpe_tokenizer_fixture.py` then
`dotnet test tests/Satori.Numerics.Tests -c Release --no-restore -m:1 --filter FullyQualifiedName~BpeTokenizerTests`.

Limits: pre-tokenizer patterns cover the qwen2/qwen3 families (BitNet/Llama-3 without `pre` is rejected
until added); the ChatML renderer does not evaluate the artifact Jinja template (tool calls, multimodal);
the default system message is supplied by the caller. Contract in `docs/tokenizer.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS07 — 2026-10-13 — Completed

Implemented the CPU operators the dense qwen2/qwen3 families need: `TransformerOps.RopeHalf` (half-split
GPT-NeoX rotary with configurable base and rotary dimension, so partial RoPE and per-model base are
honoured), `TransformerOps.RmsNormHeads` (per-head RMS norm for `attn_q_norm`/`attn_k_norm`) and
`TransformerOps.GroupedAttention` (grouped-query attention with a causal prefix). Bias add reuses
`TransformerOps.Add` and projections reuse the existing Q8_0/Q4_0 MatVec, which already supports independent
rows and columns: only the three genuinely missing operators were added, per the DS07 rule to reuse
conformant operators.

Verification: Release build without warnings; 115/115 offline numerics tests passed, 8 in
`QwenOperatorTests`. An independent NumPy float64 oracle (`tools/transformer_reference_v2.py`, seed 70207,
fixture `fixtures/operators/transformer-v2.json`) checks GQA, full and partial RoPE, per-head RMS norm and
bias add against formulas written without Satori; regeneration is byte-identical. GQA is also shown to
reduce to plain attention at equal head counts, and the biased projection is checked against a manual
float64 decode of the same Q8_0 payload. Tolerance 2e-6 + 2e-6 x |reference|.

Reproduce: `python3 tools/transformer_reference_v2.py` then
`dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~QwenOperatorTests"`.

Limits: MoE and recurrent/hybrid attention excluded (later tasks); bias is a plain vector add, not a fused
kernel. Contract in `docs/qwen-operators.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS08 — 2026-10-13 — Completed

Implemented the generic dense CPU forward: `DenseForward` (embedding, per-layer grouped-query attention with
half-split RoPE and optional Q/K per-head norm, SwiGLU feed-forward, final norm, output projection) driven
entirely by the validated `ModelShape`, plus `IModelWeights` in Core and `GgufWeightStore` in the GGUF
module. Projections call the shared quantized MatVec, so a new tensor format is a kernel registration, not a
forward change; QKV bias and Q/K norm are applied from tensor presence, so one class serves qwen2 and qwen3
with no model-name branch. Optional per-layer intermediates are exposed through `ForwardObserver`.

Verification: Release build without warnings; 145/145 numerics tests passed, 2 in `QwenForwardTests`
against an independent NumPy float64 reference (`tools/qwen_forward_reference.py`, its own GGUF parser and
Q8_0 decode) on two topologies: Qwen2.5-Coder 1.5B Q8_0 (12/2 heads, QKV bias) and Qwen3 4B Q8_0 (32/8
heads, head dim 128 via `attention.key_length`, Q/K norm). Shape equals the reference config, capability
report empty, argmax matches, 64 sampled logits within 5e-3 + 5e-3 x |ref|, first/last-layer intermediates
within 2e-3 + 2e-3 x |ref|.

Reproduce: `python3 tools/download_qwen_models.py`, then `python3 tools/qwen_forward_reference.py` for each
model, then `dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~QwenForwardTests"`.

Limits: whole-sequence forward only, no incremental state, sampling or streaming (later tasks); Q4_K/Q6_K
types still unsupported, so only the Q8_0 artifacts run. Contract in `docs/model-forward.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS09 — 2026-10-13 — Completed

Implemented the incremental session from the DS03 contract: `IInferenceSession` and `DenseSession`, which
owns a key/value state dimensioned from the model shape and a chosen context, and reuses the shared
per-layer step of `DenseForward` (extracted as internal `Step`/`Final`, with a reusable `DenseScratch`), so
incremental execution and full recomputation agree by construction. `Reset`, the context limit, token and
buffer validation, and idempotent `Dispose` follow the contract; limits are errors, never silent truncation.

Verification: Release build without warnings; 153/153 numerics tests passed across three consecutive runs.
`DenseSessionTests` (offline, synthetic weights) checks incremental versus full recomputation on a
qwen2-like and a qwen3-like shape, reset identity, plan-consistent state bytes, context/token/buffer
enforcement, dispose, and the capability/shape guards. `QwenForwardTests.IncrementalSessionMatchesFullForward`
checks on Qwen2.5-Coder 1.5B and Qwen3 4B that the session reaches the independent-reference argmax and that
appending one token equals recomputing the prefix through `DenseForward`.

Two pre-existing exact-allocation tests (`MappedTernary2Tests`, `StoriesSessionTests`) were intermittently
failing under parallel execution once the model-integration tests increased load; test-collection
parallelization is now disabled via `xunit.runner.json` so those assertions stay meaningful. No production
code was involved.

Reproduce: `dotnet test tests/Satori.Numerics.Tests -c Release`.

Limits: whole prompt plus one token at a time; no sampling, generation or streaming (later tasks); KV is
F32 only. Contract in `docs/model-session.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS10 — 2026-10-13 — Completed

Connected generation to the session: `ITextTokenizer` (implemented by `BpeTokenizer`) and `TextGenerator`,
which encodes the prompt, forwards it once, then samples each next token with the shared `TokenSampler`
(temperature 0 = greedy) and appends it, stopping at the end-of-sequence token, the token limit or the
session context. Reused rather than reimplemented: `DenseSession`, `TokenSampler`/`GreedyTokenSelector`,
`BpeTokenizer`. Added the generic `text` console command; no model name appears in the generator or command.

Verification: Release build without warnings; 161/161 numerics tests passed. `TextGeneratorTests` (offline,
synthetic weights) cover greedy equals a manual sampler loop, default greedy, EOS stop, `ContextFull`,
seeded sampling equals the shared sampler, and argument validation. `QwenForwardTests` adds greedy parity
against the independent float64 reference: from a code prompt taken from the official tokenizer fixture, the
greedy continuation is token-identical on Qwen2.5-Coder 1.5B (`322 21198 33264 3815 88112 4330 15230 1757 1728 198 322 1531`)
and Qwen3 4B (`322 3815 58179 2691 11422 650 58179 1099 78 11 76608 2477`). The reference tool now emits the
greedy oracle (`--greedy`, `--greedy-prompt`). Console demo on the 1.5B file printed
` // return a + b;\n\n// int` with stop `TokenLimit`.

Reproduce: `python3 tools/download_qwen_models.py`, regenerate the references with
`python3 tools/qwen_forward_reference.py <model> <fixture> --greedy 12 --greedy-prompt <ids>`, then
`dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~QwenForwardTests"`.

Limits: no KV prompt caching, no streaming output, no chat template (later tasks). Seed verification was not
required because the sampling path was not modified. Contract in `docs/text-generation.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

# DeepSeek Work Plan Execution

Tasks DW01–DW10 from DeepSeek_todo.md. One task per commit; no push.

## DW01 — 2026-10-13 — Completed

Inventoried the reusable assets in `docs/reusable-assets-v1.md`: code capabilities (numerical kernels, GGUF
reader and inventory, model descriptor/planner, tokenizer, operators, dense forward, session, generation,
console commands, Vulkan primitives, Stories15M laboratory path), independent reference tools, fixtures,
reports, and the declared-unverified set. Each code capability cites its module, contract document and
verifying test or report.

Verification: Release build without warnings; 165/165 numerics tests passed. The document states which
capabilities are unverified: Vulkan hardware (validation layer unavailable), the removed model-specific
paths, and every codec absent from the profile.

Limits: covers Satori's own assets only; candidate artifact capability stays owned by
`docs/capability-matrix-v1.md`. No runtime change.

Skills: phase-development-task-execution, base-dossier-writing.

## DW02 — 2026-10-13 — Completed

Pinned the comparison in `docs/comparison-manifest-v1.md`: the three candidates with repository, revision,
file, exact size, full-file SHA-256, license, architecture and tensor types; the reference runtimes; the
test hardware; and the open decisions. APEX provenance was resolved: `general.architecture = qwen35moe`,
`general.name = Ornith-1.5-35B`, 256 experts / 8 active; card base `ornith-ai/Ornith-1.5-35B-A3B`, MIT. The
card `qwen3.6`/`qwen3_5_moe`/`qwen35moe` tags are aliases of one Qwen3.5-MoE lineage; the base identity is
Qwen3.5-MoE, not dense Qwen2/Qwen3.

Reference runtimes built CPU-only from source: upstream llama.cpp `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444`
(0.6.0-dev, ggml 0.26.0) for Qwen and APEX; the Prism fork `e8fd209c975c80aa7fb36a8423571ff3b1b7e955`
(ggml 0.21.0) for Bonsai. With the server disabled the combined `llama-app` target does not link; the
`llama-completion` tool is the CLI.

Verification: all three artifacts were downloaded from their pinned revisions and their SHA-256 values match
the manifest rows. Each loads and generates in its reference runtime: Qwen2.5-Coder 7B Q4_K_M
(`def add(a, b):` → `return a + b`; prompt eval 7.84 tok/s, eval 2.67 tok/s), APEX-I-NanoPlus
(load 4.1 s, prompt eval 4.94 tok/s, eval 1.15 tok/s; upstream supports the mixed K/IQ types), Bonsai PTQ1_0
in the Prism runtime (load 18.9 s, prompt eval 1.01 tok/s, eval 0.52 tok/s). Observation, not a measurement:
the Bonsai process reached ~22 GB RSS, above the 12 GB budget and the 16 GB target.

Owner decisions recorded: runtimes built on this Intel host (Intel Core Ultra 7 155H, 32 GB, Arch Linux),
not the primary Geekom A5; the coding corpus and the concurrent workload are proposed in
`docs/corpus-csharp-v1.md` and `docs/comparison-workload-v1.md` for owner freeze.

Reproduce: `hf download <repo> <file> --revision <sha> --local-dir .temp/deepseek/dw02/models`; build the
runtimes with the flags in the manifest; run
`.temp/deepseek/dw02/llama.cpp/build/bin/llama-completion -m <model.gguf> -p "def add(a, b):" -n 8 --temp 0 -t 8 -no-cnv`.

Limits: registry proxy throttled downloads; the corpus and workload are proposals, not yet frozen; no
candidate measurement beyond the load confirmation.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW03 — 2026-10-13 — Completed

Closed the open protocol thresholds from the owner's decisions and recorded them in
`docs/validation-protocol-v1.md` section 8: TTFT at 2K ≤ 60 s and decode at 2K ≥ 3 tok/s; build slowdown
≤ 25% versus the no-inference baseline; maximum cancellation latency ≤ 2 s; no sustained swap during the
observation window; Vulkan benefit deferred until a path is selected. IDE interaction latency stays
reported as data. Recorded the no-inference baseline (`dotnet build Satori.sln -c Release --no-restore
-m:1` = 1.55 s steady state over three runs on the Intel host) and the resource limits (8 inference
threads, context 2048 and 4096, 12 GB runtime ceiling).

Verification: thresholds and baseline written in the protocol section that owns them; no implementation
change.

Limits: thresholds are provisional to this CPU build; they were motivated by the DW02 reference-load
figures, which are not the batched measurements. The corpus and workload proposals are still pending owner
freeze.

Skills: phase-development-task-execution, base-dossier-writing.

## DW04.1–DW04.3 — 2026-10-13 — Completed

Measured the three candidates in their reference runtimes on the Intel comparison host (Intel Core Ultra 7
155H, 22 logical processors, 32 GB, Arch Linux), 8 threads, f16 KV. Report and raw evidence in
`docs/benchmarks/comparison-2026-10-13/` (llama-bench JSON per candidate, `memory.tsv`, `measure.py`,
`run-memory.sh`).

Throughput (`llama-bench -p 2048 -n 128 -r 1 -t 8`): Qwen2.5-Coder 7B Q4_K_M prefill 22.16 tok/s and
decode 2.94 tok/s; APEX-I-NanoPlus prefill 39.70 tok/s and decode 4.88 tok/s; Bonsai PTQ1_0 prefill 1.37
tok/s and decode 0.99 tok/s. TTFT at 2K, derived from the prefill rate: 92 s, 52 s and 1490 s.

Peak RSS (one-token run, minimal context): Qwen 7B 7504 MiB, APEX 12705 MiB, Bonsai 5955 MiB. Bonsai is
close to its 5.95 GB file with no permanent expansion; Qwen exceeds its 4.68 GB file.

Verdict against the DW03 thresholds: Qwen fails TTFT (92>60) and decode (2.94<3), passes memory (7.5<12);
APEX passes TTFT and decode, fails memory (12.7>12); Bonsai fails TTFT and decode, passes memory
(5.95<12). No candidate passes every threshold on this host.

Reproduce: build the runtimes as in `docs/comparison-manifest-v1.md`, then run
`bash docs/benchmarks/comparison-2026-10-13/run-memory.sh` and the `llama-bench` command in the report.

Limits: one host (Intel, not the primary Geekom A5); one repetition; TTFT derived, not measured; peak RSS
is a working set; memory measured at minimal context, so 2K/4K add KV; no quality measurement here.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW04 — 2026-10-13 — Completed

Selected the primary path with the owner: **Qwen2.5-Coder 7B Instruct Q4_K_M** (dense `qwen2`), resident
memory policy, contexts 2048 and 4096. Recorded in `docs/selected-path-v1.md` with the required capabilities
and the memory policy. External baseline: upstream llama.cpp `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444`.

No candidate passed every threshold on the Intel host, so the selection rests on the closest fit: memory
7504 MiB inside the 12 GB budget, decode 2.94 tok/s (-2%) and TTFT 92 s. The owner kept the latency
thresholds and required them met by optimization, not by changing the artifact.

The only missing capability for this path is the Q4_K and Q6_K codecs and their MatVec, with AVX2 kernels.
MoE, hybrid attention, ternary and IQ codecs, vision and the 262144-token context are out of scope.

Verification: the decision names the artifact, the constraints and the exact missing capabilities; the
artifact and runtime are pinned and measured (`docs/benchmarks/comparison-2026-10-13/`). The quality gate
against the frozen C# corpus remains open.

Limits: quality is not yet measured, so the selection is provisional on that axis; the primary Geekom A5 is
not measured.

Skills: phase-development-task-execution, base-dossier-writing.

## DW05 — 2026-10-13 — Completed

Restricted the plan to the selected path in `docs/path-plan-v1.md`: the capability matrix narrowed to
`qwen2` dense with only Q4_K and Q6_K missing; the session and resource contracts narrowed to the resident
policy (no streaming, no expert cache, no SSD KV, so DW08 is not applicable); the memory plan (weights
4.68 GB resident plus KV and workspace inside 12 GB); and the ordered subtasks DW06.1–DW06.8, each with a
dependency, an oracle and a verification criterion.

The Q4_K and Q6_K block layouts are pinned for DW06.1 from the ggml reference at the revision already
recorded in `docs/gguf-subset-v1.md`; the oracle is an independent FP64 reconstruction.

Verification: every subtask names its oracle and criterion; no architecture breadth is required; the only
missing capability is named exactly.

Limits: DW06.6 and DW06.7 depend on DW06.5 because the measured bottleneck is the quantized MatVec; the
quality corpus must be frozen before DW06.8.

Skills: phase-development-task-execution, base-dossier-writing.

## DW06.1 — 2026-10-13 — Completed

Implemented the GGML Q4_K super-block codec in `src/Satori.Core/Q4KCodec.cs` from the pinned ggml layout
(144 bytes per 256 weights: FP16 `d`, FP16 `dmin`, `scales[12]`, `qs[128]`; `GetScaleMin`, `DotBlock`,
`DecodeBlock`, `AreScalesFinite`). Registered the format: `BlockQuantization.Q4_K`, a per-format block
geometry in `QuantizedMatrixShape` (block 256, 144 bytes), a scalar Q4_K branch in `QuantizedMatVec`
(two-scale validation, and an explicit exclusion from the AVX2 path until DW06.5), the type id 12 in
`GgufTypes`, and the type 12 mapping in `GgufQuantizedMatrix`.

Verification: Release build without warnings; 170/170 numerics tests passed. The independent FP64 oracle
`tools/q4k_reference.py` writes `fixtures/operators/q4k-v1.json` from the ggml layout without calling
Satori; its decoder never reuses the C# code. `Q4KCodecTests` checks the reconstruct and the dot on eight
blocks (zero, positive, negative min, extremes, four random) and rejects a non-finite super-scale. Tolerance
1e-3 + 1e-4 x |ref| on weights (FP64 reference vs FP32 decode) and 1e-2 + 1e-4 x |ref| on the dot.

Two existing tests asserted Q4_K was unsupported and were updated for the new profile: `GgufInventoryTests`
now expects only Q6_K as unsupported, and `ModelDescriptorTests` now uses Q6_K as the format gap.

Reproduce: `python3 tools/q4k_reference.py` then
`dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~Q4KCodecTests"`.

Limits: scalar only; a Q4_K_M artifact still needs a Q6_K codec for its output and part of `ffn_down`, so
the path is not executable yet.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.2 — 2026-10-13 — Completed

Implemented the GGML Q6_K super-block codec in `src/Satori.Core/Q6KCodec.cs` from the pinned ggml layout
(210 bytes per 256 weights: `ql[128]`, `qh[64]`, 16 signed 8-bit sub-scales, FP16 `d`). Registered the
format: `BlockQuantization.Q6_K`, block geometry 256/210 in `QuantizedMatrixShape`, the scalar Q6_K branch
in `QuantizedMatVec` (finite super-scale check, still outside the AVX2 path until DW06.5), the type id 14
in `GgufTypes`, and the type 14 mapping in `GgufQuantizedMatrix`.

Verification: Release build without warnings; 174/174 numerics tests passed. The independent FP64 oracle
`tools/q6k_reference.py` writes `fixtures/operators/q6k-v1.json` from the ggml layout without calling
Satori. `Q6KCodecTests` checks the reconstruct and the dot on eight blocks (zero, positive, negative scales,
extremes, four random) and rejects a non-finite super-scale. One defect was found and fixed during the task:
the 16 sub-scales must be read as signed bytes; reading them unsigned flipped the sign of every negative
scale and the reference caught it.

Reproduce: `python3 tools/q6k_reference.py` then
`dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~Q6KCodecTests"`.

Limits: scalar only; the format dispatch accepts Q4_K/Q6_K but end-to-end execution on a Q4_K_M artifact is
DW06.4 and still needs the AVX2 kernels and the forward parity check.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.3 — 2026-10-13 — Completed

Made the Q4_K_M file executable end to end. Extended the strict reader `GgufFile` to accept type 12 (Q4_K,
256/144) and type 14 (Q6_K, 256/210), fixed `quantized` to mean any non-F32/F16 type, and generalized the
row multiple check to the block size. Added `QuantizedRow.DecodeRow` for any supported format and routed
`GgufQuantizedMatrix.ReadRow` (embedding lookup) through it, so a Q4_K/Q6_K embedding tensor decodes.

Verification: Release build without warnings. `describe` on the pinned Qwen2.5-Coder 7B Q4_K_M reports
`Executable: yes` (weights 4677120000, plan 4912439296 at 2048). A real generation on that file succeeds
with a coherent completion (prompt `def add(a, b):` → text ` # defining a function`, stop `TokenLimit`),
proving every tensor type of the artifact is decoded with no silent fallback; the process held 4.68 GB RSS
equal to the compressed weights, with no permanent expansion. `GgufStructureTests` adds a container with a
Q4_K and a Q6_K tensor that the strict reader accepts, and asserts both types are supported in the profile.

Reproduce: `dotnet run --project src/Satori.Console -c Release -- describe .temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf 2048`
and `dotnet run --project src/Satori.Console -c Release -- text <same file> "def add(a, b):" 4 --context 512`.

Limits: the scalar K kernels make the 7B forward slow (~195 s for six prompt tokens); DW06.5–DW06.7 address
this. Numerical parity against an independent reference is DW06.4.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.4 — 2026-10-13 — Completed

Verified K parity on real artifact bytes and end-to-end execution. `tools/k_tensor_reference.py` reads the
two real tensors of the pinned Qwen2.5-Coder 7B Q4_K_M (`token_embd.weight` Q4_K, `output.weight` Q6_K),
takes a window of their raw block bytes, decodes them with the independent FP64 decoders in
`q4k_reference.py` and `q6k_reference.py` (no Satori code), and writes `fixtures/operators/k-real-v1.json`.
`KParityTests` decodes the same bytes with `Q4KCodec`/`Q6KCodec` and checks the reconstruct and the dot
against that reference (tolerance 1e-3 + 1e-4 relative on weights, 1e-2 + 1e-4 on the dot).

End-to-end: the real Q4_K_M file loads through the strict reader, its capability check is empty, and a
generation completes with a coherent completion; the process held 4.68 GB RSS equal to the compressed
weights, with no silent fallback and no permanent expansion.

Verification: Release build without warnings; the parity test passes on the real artifact bytes. The full
numerics suite is green at 175/175 (K parity adds one case on top of the 174 of DW06.2).

Reproduce: `python3 tools/k_tensor_reference.py .temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf`
then `dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~KParityTests"`.

Limits: parity is on a representative window of two tensors, not the whole artifact; a full logits-vs-reference
run is deferred because the scalar K forward is too slow (DW06.5–DW06.7). Quality versus the Q8 reference is
not asserted.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.5 — 2026-10-13 — Completed

Implemented AVX2 kernels for Q4_K and Q6_K in `src/Satori.Core/QuantizedKAvx2Kernel.cs`, mirroring
`Q4KCodec`/`Q6KCodec` block arithmetic exactly. `QuantizedMatVec.Execute` now dispatches every format to
AVX2 when available, and `ExecuteAvx2` routes the K formats to the new kernel; the scalar path remains the
reference and the fallback. The Q4_K/Q6_K super-block scale extraction, the Q4_K 6-bit scale/min unpacking
and the Q6_K nibble/2-bit assembly are unchanged from the scalar codecs.

Verification: Release build without warnings; the full numerics suite is green. `KParityTests` adds two
theory cases (Q4_K, Q6_K) that decode two-block rows with the scalar and the AVX2 paths on the fixture bytes
and require them within 1e-3 + 1e-4 relative. One defect was caught by this parity check and fixed: the
kernel read the first 256 inputs for every block instead of advancing by the block column offset. A real
7B Q4_K_M generation dropped from ~195 s (scalar) to 30.7 s with the AVX2 kernels (about 6.4x) with the same
output, so the kernels are the default rather than experimental.

Reproduce: `dotnet test tests/Satori.Numerics.Tests -c Release --filter "FullyQualifiedName~KParityTests"`.

Limits: AVX2 only, no AVX-512; the speedup is a single artifact on this host, not a general claim. Prefill is
still unbatched and single-threaded (DW06.6).

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.6 — 2026-10-14 — Criterion not met

Implemented the batched prefill: `DenseSession.Forward` walks the prompt in chunks of 256 positions and
hands each chunk to `DenseForward.LayerBatch`, which runs the same operator sequence as the single-position
`Step` but calls every projection once for the whole chunk. Each weight block is therefore decoded once per
chunk instead of once per position, which is the cost a prompt actually pays. The new
`QuantizedMatVec.ExecuteBatch` splits rows over `InferenceOptions.Threads` workers, each worker processing
rows in tiles of 16 with the block loop outside the position loop, so an input window is read once and then
dotted with every row of the tile. `InferenceOptions` fixes the thread budget once (default 8, the
protocol's limit) instead of reading a process-wide setting. The console gained `--threads`.

Two defects were found by the checks and fixed: the projected query was shared across positions between the
two attention phases, and `GgufWeightStore`'s lazy caches were plain dictionaries that the new parallel
prefill corrupted — the process aborted with a collection-corruption exception. The caches are now
`ConcurrentDictionary` of `Lazy<T>` with publication semantics, so each handle is built exactly once.

Verification: Release build without warnings; the numerics suite is green at 191 tests. `DenseSessionTests`
adds `BatchedPrefillMatchesSequentialAppend`, which requires **bit-identical** logits between a 200-position
batched prefill and 200 `Append` calls at one and four threads, plus the continuation-after-prefill and
thread-validation cases. `BatchedMatVecTests` checks the batched kernel against the per-position kernel on
the DW06.1/DW06.2 fixture bytes and that the row split never changes a byte. `BatchedMatVecThroughputTests`
is the measurement guard: it fails if the batched projection is not faster than one projection per position.

Measured on the 7B Q4_K_M artifact: 607 prompt tokens in 166.35 s, peak RSS 4756 MiB, i.e. 3.65 prefill
tok/s against the reference 22.16 tok/s on the same host. Isolated on a 3584x3584 Q4_K projection with 64
positions the batched path is 2.77x the shipped AVX2 per-position path and 10.18x the scalar path, scaling
to 43.9 GMAC/s at four threads and no further at eight. Report:
`docs/benchmarks/prefill-2026-10-14/README.md`.

**The DW06.6 criterion is not met**: TTFT at 2K extrapolates to roughly 560 s against a threshold of 60 s.
The structure is correct and is a real gain, but the remaining gap is inside the dot kernel, not in the
prefill structure: Satori reconstructs each block to F32 and multiplies with FMA, so one AVX2 instruction
performs 8 multiply-adds, while the reference quantizes the activations per block and multiplies integers,
performing 64 per instruction with the activation bytes shared across rows. Closing that is kernel work and
belongs to DW06.7 and DW09; DW06 stays open.

Limits: one host, one artifact, one repetition per figure. The isolated figures use a synthetic payload with
uniform scales. The batched kernel is 2.77x the AVX2 path but still about 3.5x below the reference's
per-thread rate, so the optimization is unfinished, not merely unmeasured.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.7 — 2026-10-14 — Criterion not met

Built the two pieces DW06.6's report named: a Q8_K activation quantizer with the integer weight products it
feeds (`Q8KCodec`, `Q4KCodec.DotProduct`/`MinProduct`), and a row-tiled batched kernel whose per-block
constants are extracted once per weight block instead of once per position
(`QuantizedKAvx2Kernel.ExecuteQuantizedTiled`). The integer batched path is wired for Q4_K only: it is 169 of
the artifact's 198 quantized tensors, and Q6_K's four-codes-per-byte-under-four-scales layout does not map
onto the byte-pair multiply-add that gives the kernel its advantage. A Q6_K integer product was written,
found wrong by its test, and removed rather than shipped; Q6_K stays on the floating-point path.

The decisive change was not the integers. Measuring the real decode shapes showed that the per-position
kernel scalarized its constant extraction inside the innermost loop, so `Step` and `Final` now route through
the batched projection: at one position — the decode case — the batched kernel is 9.38x the per-position one
on gate/up and 3.38x on the output projection.

Verification: Release build without warnings; the numerics suite is green at 205 tests.
`Q8KCodecTests` quantizes the fixture inputs and requires the block bytes to be **identical** to an
independent FP64 reproduction of `quantize_row_q8_K_ref` by `tools/q8k_reference.py`, together with the
integer product against a reproduction of `ggml_vec_dot_q4_K_q8_K`. Reproducing the reference's
`nearest_int` value by value mattered: it rounds through the 12582912 trick in single precision, and a
substitute rounding produced a different byte on one fixture. `BatchedMatVecTests` bounds the integer path
against the per-position path within 1e-2 absolute and 1e-3 relative, which is the 8-bit activation
quantization's own error level and is stated as such; `BatchedMatVecThroughputTests` carries the shape
breakdown and fails if the batched path ever stops beating the per-position one on the large shapes.

Measured, 512 context, 8 threads: 16 tokens in 21.47 s against 28.41 s before, a **1.32x** gain, i.e.
0.81 tok/s against the 3 tok/s threshold. A decode token's 197 projections total 7.07 GMAC and take 1050.5 ms,
a ceiling of 0.95 tok/s before attention, which matches the end-to-end figure. The best single projection
reaches 13.1 GMAC/s at one position against the reference's ~154 GMAC/s, so decode is about 12x below the
reference. Report: `docs/benchmarks/decode-2026-10-14/README.md`.

**The DW06.7 criterion is not met**, and this corrects DW06.6's diagnosis: the integer dot is worth about
30%, while the batched structure and constant hoisting are worth the rest. The remaining factor of 3.7 needs
three things this subtask did not do: a size-aware dispatch, since the batched kernel's fixed per-call cost
makes it 2.2x **slower** on the 512x3584 k/v projection; a kernel that reaches more than 13 GMAC/s on a
single large projection at one position; and attention and the elementwise operators, which account for the
0.95 to 0.81 tok/s gap and have not been touched. DW06 stays open; no protocol threshold was changed.

Limits: one host, one artifact, one repetition per end-to-end figure. Isolated figures use synthetic payloads
with uniform scales. The integer path is lossy by construction, so it differs from the floating-point path in
the last bits; that difference is bounded by a test and not eliminated.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.7-correction — 2026-10-14 — Measurement base rebuilt

The performance figures in DW06.5-DW06.7 were taken with one warmed call and one timed call per
configuration. Two runs of the **same published binary** on the same artifact gave 1390 and 2080 ms per
token, 1.5x apart, so every ratio derived from those figures was inside the noise and one of them was
wrong in direction.

Built `tools/ab_decode.py`, which runs two published revisions alternately within each round, reverses
their order every round, and reports minimum, median, maximum and the within-configuration spread per
configuration, warning above 15%. Built the A/B pair from a Git worktree at the commit, so the working
tree was never disturbed. Wrote `docs/measurement-method.md`: never a single run; alternate revisions;
state the resolution before the comparison; an in-cache benchmark never answers a workload question;
name what each figure measured; revert a change that cannot be shown to help. Corrected
`docs/benchmarks/decode-2026-10-14/README.md` and `docs/benchmarks/prefill-2026-10-14/README.md`, with
the corrections stated as corrections rather than edited silently.

What the corrected measurement changed:

| Claim | Corrected |
|---|---|
| batched kernel is 2.2x slower on the 512x3584 k/v projection | 3.1x **faster** |
| one 3584x3584 projection at one position: 5.2 GMAC/s | 23.6 GMAC/s |
| a size-aware dispatch is required | not required; the fixed cost never dominated |
| decode is ~12x below the reference | ~6x |
| decode 1.32x over DW06.6 (single runs) | not established: 0.980 against a 1.22x spread |

A branch-free Q4_K scale/min unpacking was written to replace eight branchy per-sub-block accesses.
The A/B measured it at 0.980 of the committed revision against a 1.22x spread: **unresolved and
marginally the wrong direction, so it was reverted**, and the kernel returned to `0d4581a`. Only the
measurement fix was kept.

Verified: Release build without warnings and the numerics suite green. `BatchedMatVecThroughputTests`
now takes nine repetitions per figure and reports minimum and median, so a kernel regression shows as a
number rather than as a claim.

Limits: the within-configuration spread is still 1.22x, so ratios below that are unresolved on this
host and resolving tighter needs more rounds or a quieter machine. The synthetic figures remain
in-cache payloads and are documented as bounding the kernel only.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

## DW06.7-diagnosis — 2026-10-14 — The bottleneck is Q6_K, not the Q4_K kernel

Measured the real artifact's own tensors against synthetic payloads of the same shape and format
(`RealArtifactProjectionsAgainstSynthetic`, nine repetitions, minimum). The cache effect is real but small,
1.07x to 2.43x. The dominant fact had never been measured: **`ffn_down` and the output projection are
Q6_K, not Q4_K**, as are 29 of the artifact's 198 quantized tensors and about 2.0 GB of the 4.3 GB a token
reads.

| Tensor | Format | Real GMAC/s | Q4_K integer path |
|---|---|---:|---:|
| blk.0.ffn_gate | Q4_K | 12.0 | 12.0 |
| blk.0.ffn_down | **Q6_K** | **3.3** | — |
| output | **Q6_K** | **3.4** | — |

Q6_K has no integer kernel, so it runs the F32 decode path at **2.5x to 3.6x less per MAC**. One decode
token spends 159 ms on the output projection plus 571 ms on 28 `ffn_down` calls: **730 ms of about
1409 ms, more than half**.

Three hypotheses were tested and discarded rather than assumed: the activation quantization is 1% to 8% of
a call, not its cost; raising the thread-pool floor changes nothing (1.11x and 0.94x); and the thread curve
does not flatten after the fast cores, so the hybrid core mixture is not the ceiling. The thread count is
worth about 7% (1.070x against an 1.11x spread) and is therefore **not established**. The path streams
4.3 GB in 1409 ms, about 3.1 GB/s against a memory system above 50 GB/s, so it is **not bandwidth-bound**.

Also measured: the same code and shape gave 2078 us and 5769 us at one thread in two different test runs,
a 2.8x difference. Repetition removes noise **within** a run, not drift **between** runs, so only the
alternated A/B produces figures to decide on. `tools/ab_decode.py` now compares two **configurations** as
well as two binaries, which is how the thread question was asked.

Verified: `RealArtifactProjectionsAgainstSynthetic` skips when the 4.7 GB artifact is absent, so a checkout
without it still passes. Report: `docs/benchmarks/decode-2026-10-14/README.md`.

The next decode work, in measured order of value: a Q6_K integer kernel (730 ms of a token), prefetching
and software pipelining in the row loop (the rate is at 6% of memory capacity and the pattern streams), and
attention plus the elementwise operators, which are still untouched.

Skills: phase-development-task-execution, capabilities-dev-problem-solving-methodology, base-dossier-writing.

# Execution Log

## T11 — 2026-10-12 — Completed

Skills: `phase-development-project-conventions`, `phase-development-task-execution`, `capabilities-dev-problem-solving-methodology`, `base-dossier-writing`. Scope: GGUF specification, acceptance catalog and synthetic golden fixtures. No runtime changes or new xUnit tests.

Contract pinned in `docs/gguf-subset-v1.md`. Official sources verified at the revisions listed in the contract. The `docs/gguf-fixtures-v1.md` catalog distinguishes valid, out-of-profile, corrupted and lifetime cases, and future numerical checks.

Four golden fixtures independent of Satori code in `fixtures/gguf/v1/`, with a deterministic generator and manifest of hashes, offsets and expected results. Headers, lengths, alignments, SHA-256, identical regeneration and independent payload decoding verified. Q4_0/Q8_0 reconstruct the same 32 weights and produce −8 with an input of ones.

Neither the Satori nor the official GGUF parser has yet run on the fixtures. Catalog variants are criteria for T12–T14, not already passed tests. No runtime build or suite repeated: the task changes only documents and reference data. Diff check passed. Temporary references removed from the task workspace.

Next: T12, within the module and dependency boundaries fixed by the profile.

## T10 — 2026-10-12 — Completed

Skills: `phase-development-project-conventions`, `phase-development-task-execution`, `capabilities-dev-problem-solving-methodology`. The last was used to isolate a syntax error in the corrupted-file test and an uncaught console exception, fixing them with focused checks. Tests limited to binary format, corruption, numerical identity, disposal and allocations.

Implemented the STQ2 v1 container with a 32-byte header, shape, block size, version and payload length. Saving to a new file without overwriting. Mapped loading with full validation once on opening. Shared internal immutable shape; scalar/AVX2 kernels now receive the same payload span from an array or mapping, without duplicating algorithms. A callback with ref struct state keeps the mapping acquired for the entire MatVec without closures or weight copies.

Final checks: full Release build without warnings/errors; 24/24 tests passed with AVX2 enabled and 24/24 with AVX2 disabled. Five new xUnit cases traverse the entire corpus with all three block sizes, check headers against an independent binary fixture, rejection of corrupted or truncated headers/payloads, bitwise identity of mapped/owned results for each kernel, zero allocations after warm-up and rejection after Dispose. T09 tests continue to verify the reused owner's concurrent lifetime.

Console: demo file save and mapped load succeeded; result `[4, -2.6666667]` with AVX2 and scalar. Non-STQ2 file rejected with a readable message and exit code 1, without a stack trace. After correcting the message, the focused console build succeeded again. The first build attempt for the new tests failed; the 19 tests of the old assembly run then were not used as T10 evidence. Final checks above refer to the new assembly with 24 tests.

Benchmark: 6/6 warm measurements on sizes 1024²/2048²/4096² with blocks of 64, 9/9 opening trials with identity confirmed and 72/72 numerical CPU reference checks. Zero managed allocations per MatVec on both paths. For 4096², approximately 3.75 ms mapped and 3.76 ms owned in this run: comparable within variability, without claiming small advantages. Snapshot in `docs/benchmarks/mapped-2026-10-12/`.

Benchmark criterion clarification: newly written files and no page-cache eviction; validation traverses the payload. Opening/validation, first subsequent computation and warm steady state are distinguished; no actual cold SSD read was measured. The T10 criterion was made explicit in the plan to avoid that attribution.

Limits: one matrix per file, maximum payload `int.MaxValue` bytes, immutable file required; no checksum, atomic save or GGUF. Temporary fixtures and benchmarks in unique `.temp/` subdirectories, cleaned by their respective owners. The demo file is a regenerable output in `reports/`. The console formatter could not open the local socket in the sandbox: indentation corrected directly, recompilation succeeded.

Milestone A completed. Next T11: pin the GGUF subset before implementing the reader.

## T09 — 2026-10-12 — Completed

Skills: `phase-development-project-conventions`, `phase-development-task-execution`. Scope: generic mapping owner; tests only for critical pointer, lifetime, concurrency and allocation invariants. No kernel changes.

Implemented `MappedWeights`: read-only opening, 64-bit process, `long` offsets, `int`-length windows, checks before exposing the pointer. Access through a callback with generic state including ref structs; no returned span. SafeHandle acquisition under a short lock, callback outside the lock, pointer release in `finally`. Disposal rejects new reads while active reads retain the mapping until they finish. Partially acquired constructor resources released on error.

Verification: Release build with zero warnings/errors; 19/19 critical tests passed in total. Six new cases cover sparse files with reads beyond 2 GiB, unaligned offsets and invalid ranges, two concurrent readers active during Dispose, Dispose inside the callback, callback exceptions and zero allocations after warm-up with a reused delegate. Byte identity compared with known file contents.

Fixtures created exclusively under `.temp/t09/<id>/` and removed by their respective owners. No agent temporary files outside the project.

Limits: verified on Linux x64; immutable file required, no guarantee of RAM residency, no prefetch or GPU buffers. The callback does not retain spans beyond its completion. Numerical integration and mapping benchmarks deferred to T10. Contract in `docs/mapped-weights.md`.

## T08 — 2026-10-12 — Completed

Skills applied: `phase-development-project-conventions`, `phase-development-task-execution`; new tests limited to SIMD correctness, tail reads and allocations, as requested by the user.

AVX2 kernel: decodes eight weights from two bytes using variable lane shifts, zero/sign masks on activations, FP32 accumulation and per-block reduction. No dequantization buffer. Tails of one to seven elements processed by the scalar kernel; vector loads only within the input boundary. No FMA or AVX-512 dependency. Automatic selection through runtime capabilities, scalar fallback; scalar reference still explicitly callable.

Final checks: Release build without warnings/errors; 13/13 critical tests passed with AVX2 enabled and another 13/13 with `DOTNET_EnableAVX2=0`. Tests include all block sizes, 27 critical cases and dimensions crossing SIMD tails and block boundaries. Thread counter: zero managed allocations after warm-up. Final demo correctly identifies AVX2 and scalar in their respective processes, with the same known output.

Performance: two separate runs, each with 18 benchmarks and 72/72 valid numerical results. Observed advantage 27.27–31.88× over Satori's scalar kernel on Intel Core Ultra 7 155H, for 1024²/2048²/4096² matrices and blocks of 64/128/256. Zero allocations per operation in both kernels. Full snapshots in `docs/benchmarks/avx2-2026-10-12/run1/` and `run2/`.

Limits: short in-process job, normal priority because the system denies high priority; synthetic results specific to this CPU. No comparison with other engines, no token/s measurement. Automatic selection uses AVX2 availability; a future planner may select by shape/hardware. Next task T09: mapping ownership and lifetime.

## T07 — 2026-10-12 — Completed

Skills: `phase-development-project-conventions`, `phase-development-task-execution`; no new tests for simple benchmark configuration steps.

BenchmarkDotNet 0.14.0 with in-process toolchain, one thread, three warm-ups and three measured iterations of approximately 200 ms. Matrices 1024², 2048², 4096²; blocks of 64/128/256. Setup and quantization excluded; decoding included. JSON report with latency, standard deviation, MatVec/s, bytes/weight, effective bandwidth and allocations. Numerical report also extended to the nine measured matrices.

Verification: Release build succeeded without warnings/errors; 36/36 numerical combinations passed; 9/9 benchmarks completed; zero managed allocations per operation reported by the diagnoser. Snapshot in `docs/benchmarks/scalar-2026-10-12/`; method in `docs/benchmark-method.md`. Regenerable reports under `reports/scalar/`, excluded from Git.

Observed limit: high priority denied by the system and considerable variability, with wide confidence intervals in the short job. Preliminary baseline; no block-size selection or performance promise. Comparison with AVX2 will require repeated measurements under the same protocol. No model or GPU tests.

## T06 — 2026-10-12 — Completed

Skills: `phase-development-project-conventions`, `phase-development-task-execution`; continuation authorized by the user, tests limited to critical numerical invariants.

Corpus v1 with xorshift32, seed 1729 and fixed FP32 conversion; nine families per block size: random, zeros, alternating, sparse, small/large scales, subnormal, maximum scale and single weight. Partial rows and blocks included. Separate FP64 references for original and reconstructed weights; report distinguishes weight error, output quantization error and kernel error. Tolerance derived from the FP32 accumulation bound with an underflow term.

Verification: Release build without warnings/errors; 9 total xUnit cases passed, including three cases covering the 27 corpus combinations. No kernel changes. The corpus does not measure linguistic quality or speed.

## Verification Policy Revision — 2026-10-12

At the user's request, removed the three obvious FP32 tests and the Satori.Tests project. The historical T01 result below describes verification performed then, not tests still present. Future checks of simple steps use builds/demos; tests are reserved for critical behavior.

## T02–T05 — 2026-10-12 — Implemented

Skills applied: `phase-development-project-conventions` and `phase-development-task-execution`. The user's explicit instruction to proceed with implementation and critical tests only governs this increment, without further approval gates.

- T02: Ternary2 v1 specification pinned with an independent binary example; per-block scales, padding, threshold, nonfinite values and explicit limits.
- T03: codec with quantization, validated import and dequantization; critical binary and numerical tests passed.
- T04: concrete CPU contract with private owned payload and copy on import. Plan simplification: generic descriptors deferred while there is only one quantized format; no external pointer or lifetime introduced.
- T05: scalar MatVec on compressed blocks, without weight expansion. Independent FP64 comparison on the same quantized weights, on partial rows and blocks for all block sizes. Error against the original FP32 matrix remains a separate measurement to develop in the T06 corpus.

Checks: Release build without warnings/errors; 6 critical xUnit cases passed; zero managed allocations measured over 100 runs after warm-up; console output `[4, -2.6666667]`, 40 bytes including scales/padding; diff check without errors. Restore and test runner executed with authorized network/local-socket access.

Main files: `docs/ternary2-v1.md`, `src/Satori.Core/Ternary2Matrix.cs`, `src/Satori.Core/Ternary2MatVec.cs`, `src/Satori.Console/Program.cs`, `tests/Satori.Numerics.Tests/`.

Limits: owned managed memory, not yet mapping; no speed, SIMD or model-quality benchmark. T06–T07 remain open.

## T01 — 2026-10-12 — Completed

Scope: solution, FP32 library, console, tests, SDK configuration and startup instructions.

Declared skill: `phase-development-project-conventions`, to adopt project structure and checks. No server/API skills: this increment is a local numerical library.

Result: .NET 10/C# 14 solution with row-major FP32 matrix-vector product, dimension validation and prohibition of output overlapping inputs. Console with a known 2×3 case. No quantization implemented in this task.

Checks on Linux x64, SDK 10.0.112:

- `dotnet restore Satori.sln -p:RestoreDisableParallel=true`: succeeded with network access for NuGet.
- `dotnet build Satori.sln -c Release --no-restore -m:1`: succeeded, zero warnings, zero errors.
- `dotnet test Satori.sln -c Release --no-build --no-restore -m:1`: 3 tests passed; required access to the test runner's local socket.
- `dotnet run --project src/Satori.Console -c Release --no-restore -p:BuildInParallel=false`: output `Satori FP32 MatVec: [9, 4]`, exit code 0.

Sandbox limits prevented the first restore and test run; authorized attempts succeeded without reducing project checks.

## T12 — GGUF Reader

Implemented `Satori.Formats.Gguf`: immutable catalogs, typed scalars, indexed arrays, tensor windows and protected mapping. UTF-8, types, dimensions, overflow, alignments, ranges and profile limits validated. Core does not depend on GGUF.

Verification: Release build without warnings; 46 critical tests passed, including golden fixtures, every truncation of the Q8_0 fixture, scalar/array metadata, alignments 8/64/4096, sparse files beyond 2 GiB, overlaps and lifetime. An incorrect offset in the 8-byte-aligned fixture was corrected before completion. Comparison with an official reader remains to be integrated before delivery of the T12–T18 group.

Skills: phase-development-project-conventions and phase-development-task-execution. User authorization: advance through T18 with critical tests only, without intermediate gates. API/server conventions do not apply to this numerical library.

## T13 — Scalar Q8_0 MatVec

Core receives payload and shape without depending on GGUF; the adapter validates scales once, then reads the tensor with a protected lifetime. Signed FP16 scale, per-block FP32 accumulation. Added a separate validator for independent FP64 references.

Verification: clean Release build, 51 tests passed; golden −8, full Int8 range, negative/±0/subnormal scales, NaN/Inf rejection, owned/mapped and zero allocations. Tested the public stories15M-q8_0 GGUF: 57-tensor catalog compared with the pinned official reader and MatVec results from three real tensors compared with NumPy FP64 decoding. Reports in docs/benchmarks/gguf-2026-10-12; no inference executed. The four golden fixtures were also read by the official reader, completing T12's additional verification.

Skills: phase-development-project-conventions, phase-development-task-execution. External reference script only in validation; no runtime dependency on GGML.

## T14 — Scalar Q4_0 MatVec

Added Q4_0 to the numerical contract and adapter: low and high nibbles in two distinct halves of 16 weights. Same shapes and activations as Q8_0 in comparisons; no ternary conversion.

Verification: clean Release build, 55 critical tests passed. Golden fixtures, all nibbles, signed/subnormal and nonfinite scales. Official catalog and three FP64 MatVec results on the stories15M-q4_0 GGUF verified; zero allocations. Six scalar BenchmarkDotNet benchmarks run on the two real files; full reports retained in the gguf-2026-10-12/scalar directory. The validator's quick latencies are diagnostic; the performance reference is BenchmarkDotNet.

Skills: phase-development-project-conventions, phase-development-task-execution.

## T15 — AVX2 Q8_0/Q4_0

Implemented AVX2 kernels separate from the container, with Int8/nibble conversion into eight FP32 lanes, per-block reduction, without FMA or full expansion. Automatic selection by format when AVX2 is available; scalar fallback.

Verification: clean Release build; 59 tests passed with AVX2 and 59 with AVX2 disabled. Independent FP64 references on both public files satisfied. Two BenchmarkDotNet runs, twelve measurements each, zero allocations and an advantage on every measured shape; comparison and environment in docs/benchmarks/gguf-2026-10-12. The advantage supports automatic dispatch on this path, without generalizing to inference or different hardware.

Skills: phase-development-project-conventions, phase-development-task-execution; base-dossier-writing for concise specifications and reports.

## T16 — TernaryBitplane v1 Specification

Pinned P/N planes, FP32 scale, LSB order, invalid overlapping bits, padding, canonical zero, STB1 container and bit-identical conversion to/from Ternary2. Same bytes/weight: the comparison isolates the layout. Independent example verified: P=0x89, N=0x14, Ternary2=0x61/0x42, logical sum 1. Specification in docs/ternary-bitplane-v1.md.

Verification: example recomputed from the eight positions, block lengths 20/36/68 bytes equal to Ternary2. No code or tests added in this specification task.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## T17 — TernaryBitplane Codec and Scalar MatVec

Implemented validated import with copy, lossless conversion to/from Ternary2, STB1 save/load and scalar MatVec. Scales and codes unchanged, without a second quantization; no expansion in the kernel.

Verification: Release build without warnings, 64 tests passed. Independent golden fixture, invalid overlapping bits/padding/scales, byte-identical round trip and identical scalar results across all 27 corpus combinations, zero allocations, every truncation of the golden container rejected. Eighteen complete scalar benchmarks for both layouts, three sizes and three block sizes; snapshot in docs/benchmarks/bitplane-2026-10-12/scalar.

Skills: phase-development-project-conventions, phase-development-task-execution; base-dossier-writing for the specification.

## T18 — TernaryBitplane AVX2

Implemented an eight-lane kernel with combined P/N extraction, sign from masks and scalar tail within input boundaries. AVX2 dispatch with fallback; no FMA/AVX-512 dependency. The layout comparator verifies the same weights and scales before benchmarking.

Final verification: Release build without warnings/errors; 68 critical tests passed with AVX2 and 68 with AVX2 disabled. Corpus extended to tails 1–17 and block boundaries; SIMD identity with Ternary2 and FP64 comparison. Layout validator: 36/36 combinations passed, including nine large matrices. Two BDN runs with 36 measurements each, all without allocations. Bitplane AVX2 20.2–39.2× over its own scalar kernel; against Ternary2 AVX2, latency ratio 0.64–1.54, better in 7/18 pairs: no general advantage demonstrated. Ternary2 remains the adopted layout, bitplane available as an experimental alternative. Reports, environment and comparison in docs/benchmarks/bitplane-2026-10-12.

Final group review: strengthened the expected exception type in GGUF corruption tests; 1 TiB limit checked before mapping and verified on sparse files; GGUF validator checks SHA-256 and the exact reference entry count. Catalogs and MatVec results of both real files already verified. README, specifications, reproduction instructions and plan status updated. Only this group's temporary files removed; public corpus downloadable again with pinned hashes.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. User authorization: sequential completion through T18 with critical tests and no intermediate approvals. No commit requested or created. Next task: T19, first model profile; no inference implemented in this group.

## T19 — First Model Profile

Pinned Stories15M Q8_0 with hash and revision, original author's MIT declaration, SPM tokenizer, topology, no chat template, context 128 and calculated RAM budget. Q8_0 and F32 cover all 57 tensors; T20 includes F32 reads and Q8_0 gather, without new quantized formats. Contract in docs/stories15m-profile-v1.md.

Verification: C++ generator compiled against pinned llama.cpp; 147 fixture files verified and regenerated byte-identically. Sixteen tokenizer cases, three prompts with full logits and 15 intermediate states each, three 12-token greedy sequences. Independent FP64 against official runtime on the same F32-reconstructed weights: maximum logit error <1e-5, intermediates <2.2e-5. Greedy sequences also identical to the official Q8_0 path; its numerical deviations retained separately. Manifest with hashes, provenance, sources and environment in fixtures/models/stories15m-q8-v1.

No Satori inference code added. No obvious tests or new .NET dependencies. Verification concerns fixtures; T20–T24 still need to implement and verify operators, tokenizer, forward pass, cache and generation. Only .temp/t19 removed; shared GGUF corpus retained for subsequent tasks.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. Next: T20.

## T20 — Profile CPU Operators

Implemented RMSNorm with FP64 accumulation, adjacent-pair RoPE/base 10000, causal-prefix attention with stable softmax, stable SwiGLU and residual addition. Q8_0 MatVec reused. Single Q8_0 row gather from mapping; F32 norms copied and validated once. Caller-provided buffers and scratch, explicit aliasing contracts. No tokenizer, forward pass or KV cache owner introduced.

Verification: Release build without warnings/errors; 75 critical tests passed (seven new). Per-component comparison with an independent NumPy FP64 fixture for normalization, RoPE, attention and activation, tolerance 2e-6 + 2e-6 * abs(reference). FP32 extremes, in-place cases, dimensions/aliasing, signed Q8 and negative scale, nonfinite F32 and lifetime. Zero allocations measured after warm-up for operators and mapped gather. Fixture regenerated identically and hashes verified. A GGUF fixture initially missing quantization metadata was corrected; final suite fully passing. No performance benchmark added; these operators are the numerical CPU baseline.

Contract in docs/stories15m-operators.md; generator and manifest versioned in tools/transformer_reference.py and fixtures/operators. Skills applied: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. Execution and critical checks authorized by the T20 request and the user's test policy. Next: T21. No commit requested or created.

## T21 — Profile Tokenizer

Implemented StoriesTokenizer with owned vocabulary, merges by FP32 score and left position, byte fallback, optional BOS and no automatic EOS. Strict UTF-16 encoding with a 16384 UTF-8 byte limit. Full decoding and UTF-8 continuation with incomplete-byte handling. No chat template in the profile. Array hashes, IDs, types and options bound to the Stories15M vocabulary.

Verification: Release build without warnings/errors; 79 critical tests passed, four new tests. Token, piece-byte and decoding identity on the sixteen official cases. Multibyte streaming, continuation whitespace, maximum input, isolated surrogates, invalid IDs, incomplete bytes and modified vocabulary verified. Tokenizer usable after mapping disposal. Total construction allocations after warm-up within 16 MiB. Offline tokenizer fixture of 723488 bytes regenerated identically, with hashes and provenance separate from full oracles.

Scope: StoriesTokenizer and StoriesTokenDecoder in the GGUF project, critical tests, Python standard-library extractor, fixture/manifest and docs/stories15m-tokenizer.md. No new runtime dependency. Only .temp/t21 removed; shared corpus retained. No forward pass or generator introduced. Next: T22.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. The request to proceed and the user's critical-test policy authorize implementation and verification without intermediate gates. No commit requested or created.

## T22 — Single-Token Forward Pass

Implemented StoriesModel: loading the pinned Q8_0 artifact with SHA-256, metadata and 57-tensor checks; embedding, six complete layers, final norm and projection of 32000 logits. Compressed mapped weights and owned F32 norms. Last-token forward pass through prefix recomputation, without persistent state. Temporary K/V; incremental cache and generation excluded. Diagnostic observer with 15 transient states. Calls and disposal serialized.

Verification: Release build without warnings/errors; 88 critical tests passed with AVX2 and 88 with AVX2 disabled. Nine new integration cases. All 15 intermediates and 32000 logits of the three prompts compared with NumPy FP64, in both scalar and AVX2 modes. Maximum absolute error 1.430831e-5; maximum per-tensor RMS 2.654648e-6; matching argmax values. Invalid inputs checked before writing, no residual state, recovery after observer exceptions, disposal and rejection of different artifacts verified. At 128 tokens: finite logits and 1782640 temporary bytes measured after warm-up, without full weight expansion. No throughput or working-set measurement claimed.

Scope: StoriesModel, StoriesLayer and StoriesStateObserver in the GGUF project; StoriesForwardTests and docs/stories15m-forward.md. Integration tests separated by category, real corpus required and never downloaded by tests. Existing fixtures remain unchanged. No persistent temporary files added. Next: T23.

Skills applied: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. The request to proceed and the user's critical-test policy authorize implementation and verification without intermediate gates. No commit requested or created.

## T23 — Incremental KV Cache

Implemented StoriesSession, created by the model and owning reusable FP32 K/V and scratch. One sequence per session, fixed capacity 128, K after RoPE and V without rotation. TryAppend processes only the new token and advances Count after success; at full context it returns false without changing logits or cache. Reset clears and reuses buffers. Dispose clears and releases the workspace without closing the shared model. Full forward pass retained as a separate reference.

Verification: Release build without warnings/errors; 97 critical tests passed with AVX2 and 97 with AVX2 disabled. Nine new integration cases. Incremental logits exactly equal to recomputation for every token of the three prompts, scalar and AVX2. Long comparison at positions 0, 1, 31, 63 and 127; full limit, reset to a different prefix, errors without advancement, lifetime and independent concurrent sessions verified. K/V payload 1769472 bytes; session construction 1782808 measured bytes. Append and reset: zero allocations after warm-up, also verified in the isolated test.

Scope: StoriesSession, StoriesWorkspace, incremental path in StoriesModel, critical tests and docs/stories15m-cache.md. No new format or dependency. Generation and sampling excluded. No persistent temporary files added. Next: T24.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; capabilities-dev-problem-solving-methodology to isolate the first-reset measurement and complete test warm-up without changing the threshold. The request to proceed and the user's critical-test policy authorize implementation and verification without intermediate gates. No commit requested or created.

## T24 — Greedy Generation

Implemented StoriesGenerator, centralized options, immutable result and Eos/TokenLimit/ContextLimit reasons. Prompt encoded with BOS, prefill and new cache per call, argmax with ties resolved by the lowest ID and streaming of the UTF-8 continuation only. Default 32 new tokens, budget bounded by remaining space within 128. Console generate connected to the GGUF runtime; readable model, prompt and limit errors, with exit code 1.

Final verification: Release build without warnings/errors; 109 critical tests passed with AVX2 and 109 with AVX2 disabled, default PGO. Additional standard run: 109 passed. Twelve new cases: nine generation, two argmax and one parser regression. All 12 tokens of the three official oracles identical in both scalar and AVX2 modes; matching text and streaming. Defaults, repeatability, zero budget, full/nearly full context, oversized prompt, failed callback and lifetime verified. Fixtures generate no EOS within 12 tokens: selection and special-piece suppression are verified separately; no end-to-end EOS stopping demonstrated on these prompts.

Console verified on a real completion and negative messages, including return to the Ternary2 demo. Output for Once upon a time with 12 new tokens: , there was a little girl named Lily. She loved. Counts: prompt 5, generated 12, TokenLimit. No performance benchmark or general quality judgment added.

Fixed a JIT crash encountered in the suite: GgufCursor.String uses a dedicated wrapper instead of the temporary covariant array. Diagnosis with a dump and symbols of the installed runtime, discriminating PGO check and regression over 1000 reads. Localized workaround; SDK and default optimizations unchanged. Evidence in docs/diagnostics/dotnet-string-callback.md. Only .temp/t24 removed, including temporary dumps, symbols and tools; shared corpus retained.

Scope: Core selector, GGUF generation and string wrapper, console with GGUF reference, critical tests, docs/stories15m-generation.md and plan status. No new runtime dependency. Skills: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options for generation options, base-dossier-writing; capabilities-dev-problem-solving-methodology for the crash during verification. The request to proceed and the user's critical-test policy authorize implementation and verification without intermediate gates. No commit requested or created. Next: T25.

## T25 — Configurable Sampling

Added SamplingOptions and TokenSampler in Core; integration in StoriesGenerator and console flags for temperature, top-k, top-p and seed. Greedy default preserved, options validated before prefill even at zero budget. Stable FP64 softmax, top-k before nucleus, ties by ID and pinned SplitMix64 PRNG with an unsigned 64-bit seed restarted for each generation. Sampler-owned scratch and O(V log V) sorting; no additional dependency. Contract and cross-platform reproducibility limits in docs/stories15m-generation.md.

Verification: Release build without warnings/errors; 114 tests passed with AVX2 and 114 with AVX2 disabled. Five new critical tests: known PRNG vectors, analytical distributions, filters and numerical limits, invalid configurations/logits and reproducible integration with streaming/scalar. Repeated real console run with seed 123 identical; six negative cases rejected. Previous greedy oracles unchanged. No benchmark or new end-to-end EOS claim. No persistent temporary files created; shared corpus retained. Next: T26.

Skills applied: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing. The T25 request and user policy authorize implementation and critical checks only, without intermediate gates. No commit requested or created.

## T26 — First Model Benchmark

Added the --stories runner with centralized configuration, four actual prompt lengths of 1/5/32/96 tokens, three warm-ups and fifteen samples. Prefill and 11 decode forward passes measured separately; complete Generate API with 12 tokens, first text emission, managed allocations, working set and process peak. Model/tokenizer loading measured separately without a cold-page-cache claim. JSON with samples, statistics, environment, configuration and model/source hashes. Manual Stopwatch runner for correlated phases, without changing previous BenchmarkDotNet benchmarks.

AVX2 and scalar baselines run sequentially on Core Ultra 7 155H, .NET 10.0.12 Release. AVX2 decode 213.04–276.66 token/s; scalar 43.54–66.73. AVX2 first emission 3.79–412.49 ms. Process peak 164.58/164.89 MiB, explicitly including both model owners, runtime, tokenizer, validation and reports; not model-only memory. All 32000 logits of the three cases satisfy the FP64 gate and 36/36 greedy tokens per kernel match llama.cpp F32 on the same Q8_0 weights. No llama.cpp timing benchmark, perplexity, cold storage or general linguistic quality demonstrated. No new end-to-end EOS evidence.

Verification: Release solution build and final runner without warnings/errors; both benchmarks completed successfully; 114 existing critical tests passed after measurements. No tests added for the runner. Units/denominators reviewed, source hash identical in both snapshots, whitespace check and raw samples retained. Report published in docs/stories15m-benchmark.md and docs/benchmarks/stories15m-2026-10-12; README and status updated. No task-owned temporary files created; shared corpus retained. Next: T27, Vulkan backend contract.

Skills applied: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing. The request to continue authorizes T26 and necessary checks without intermediate gates, under the user's critical-test policy. No commit requested or created.

## T27 — Vulkan backend contract — Completed

Fixed docs/vulkan-backend-v1.md: Silk.NET.Vulkan 2.23.0 in an isolated backend project, Vulkan 1.1 / SPIR-V 1.3, offline GLSL compilation with glslang 16.6.0 and validation with SPIRV-Tools 2026.4. Capability gate, options, one compute queue, owned staging/resident buffers, explicit barriers/fence, rooted diagnostic callback and dependency-ordered cleanup. Initialization outcomes, device loss, unchanged output on failure and caller-controlled CPU fallback are specified. Acceptance catalog covers hardware result, tail/reuse, invalid capabilities, synchronization, lifetime and artifact validation. No runtime package/project or GPU implementation introduced.

Verification: official binding/compiler/Khronos sources checked; read-only host probe found Intel Arc MTL (8086:7d55), Vulkan 1.4.354, Mesa 26.2.4 with adequate compute/storage limits. The sandbox probe failed while the host probe succeeded; no false GPU-unavailable conclusion. A temporary two-buffer GLSL transform compiled to valid SPIR-V 1.3 (1432 bytes) and passed spirv-val for vulkan1.1. This attests compilation only, not dispatch. Khronos validation layer absent on the host; its installation/enabling is an explicit prerequisite for closing T28 hardware verification, not a missing T27 contract item. No system configuration changed. No build or numerical suite repeated for a documentation-only task; whitespace checked. Only .temp/t27 removed; shared corpus preserved.

README and plan updated. Existing unrelated edits preserved. Skills applied: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options and base-dossier-writing. The requested decision task authorizes selection within the planned Vulkan scope; no separate architecture workflow or stack reconfirmation invoked. No commit requested or created. Next: T28.

## T28 — Minimal Vulkan path — Completed

Implemented Satori.Backends.Vulkan with pinned Silk.NET.Vulkan/Extensions.EXT 2.23.0. VulkanContext owns loader/instance/device/queue, rooted debug callback, pipeline/descriptors/commands/fence and four reusable dedicated buffers. Capability and allocation-size gates precede dispatch. GLSL FP32 transform runs through staging upload, explicit transfer/compute/host barriers, synchronous fence wait and readback; outputs commit only on success. Empty input is a no-op, aliases/dimension mismatch rejected before mutation, growth replaces completed scratch, and context locking serializes dispatch/disposal. Submission/wait errors fault the context; cleanup unwinds partial initialization. CPU and GGUF have no native dependency. No quantized kernel or model offload added.

Embedded GLSL/SPIR-V/manifest and an explicit offline compiler tool added; byte-identical regeneration and spirv-val passed. Validation executable provides Cpu/Auto/Required policies, actual-backend diagnostics, transfer/scratch/managed-allocation observations and null GPU timestamp duration. Hardware tests are isolated in Satori.Vulkan.Tests and fail when hardware/layer is missing, rather than claiming fallback validation.

Verification: Release solution build, zero warnings/errors. Four hardware tests passed on Intel Arc MTL / Mesa 26.2.4 with Khronos layer 1.4.363 and SyncVal enabled; existing 114 CPU/inference tests passed. Six workloads (0/1/63/64/65/4097 elements) match CPU, including repeated changing input, growth/reuse, concurrent dispatch and disposal. No validation errors or warnings, including teardown. Warm dispatch measured zero managed allocations; maximum diagnostic scratch allocation 65552 bytes. Six expected-negative/policy cases passed: CPU and Auto with absent ICD, Required with absent ICD, Required/Auto with absent layer, and unsupported explicit device selection.

The absent-ICD check exposed misclassification of GENERAL loader errors as specification validation errors. Corrected using Khronos message-type semantics: known initialization rejection preserves GENERAL diagnostics and allows caller-configured fallback; actual validation errors always fail, and GENERAL errors after successful initialization fault operations. Final hardware tests and negative cases passed after the correction. Successful hardware report and negative-case evidence published under docs/diagnostics/vulkan-2026-10-12. Documentation in docs/vulkan-minimal-path.md; contract binding module, README, plan and solution updated. Unrelated existing edits preserved.

Limits: this device uses coherent staging; non-coherent paths reviewed, not exercised. Real device loss/OOM and malformed embedded-artifact failures were not induced; their branches reviewed without production test hooks or driver reset. No GPU timestamp benchmark or acceleration claim. System validation layer was absent: official Arch package 1.4.363.0-1 was downloaded, hash recorded, locally extracted and scoped to child processes. No system package/driver changes. Only .temp/t28 removed after verification; shared corpus and other task directories preserved. Reproduction requires restaging or installing the validation layer.

Skills: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing; capabilities-dev-problem-solving-methodology for the version-matched binding APIs and the discriminating missing-ICD diagnosis. The user's request and critical-test policy authorize implementation and GPU/lifetime verification without intermediate approval gates. No commit requested or created. Next: T29, Q8_0 Vulkan MatVec.

## T29 — Vulkan quantized MatVec (completed)

Implemented owned resident Q8_0 weights, packed byte/half decoding, reusable vector scratch, shared pipeline, optional compute timestamps, allocation limits and context/matrix lifetime serialization. Offline shader binary and ABI/hash manifest are included. Validation CLI `--q8` compares GPU, scalar and AVX2 against an independent FP64 oracle and records separate kernel/host/cold-upload timings.

Release build: zero warnings/errors. Nine hardware tests pass with Khronos SyncVal; 74 corpus workloads (30 critical synthetic, 44 real Stories15M matrices) pass, zero validation errors and zero warm managed allocations. CPU regression: 114 tests. [Published evidence](docs/vulkan-q8-matvec.md). Baseline is slower than AVX2 under validation; no acceleration or whole-model claim. Noncoherent hardware and device-loss injection remain unexercised.

Applied existing project-conventions, task-execution, configuration-options, dossier-writing and problem-solving skills. Only critical numerical/resource tests added, following the user's standing authorization; no separate test approval. Temporary layers staged in `.temp/t29`, no system installation. T30 remains pending.

## T30 — Integrate model offload (completed)

StoriesModel now accepts a Core-only serialized-matrix factory and owns its executions. VulkanStoriesRuntime composes model/context ownership, explicit Cpu/Auto/VulkanRequired policy and initialization-only fallback. Console generation exposes backend/device selection; the generator can borrow the configured model. All 43 MatVec tensors stay compressed and resident; embedding, operators and KV remain CPU. Failed appends clear all cached history and require reset; runtime GPU failures propagate without automatic replay or duplicate emitted text.

Critical verification: 11 hardware tests with Khronos SyncVal, including three independent intermediate/logit oracles, greedy sequences, 128-token incremental CPU comparison, reset, streaming and ownership. CPU suite: 116 tests, including partial resident-construction cleanup and injected failure after committed history plus partial KV updates. One first concurrent-suite run reported a 7752-byte allocation assertion; isolated and full sequential CPU reruns passed unchanged. Root cause is not established. The independent warmed latency run also records zero managed allocation.

Five child-process fallback/required negative cases pass; two positive console GPU runs pass. Release build has zero warnings/errors. Diagnostic report has zero validation errors/messages, including disposal. Local validation layers were staged under `.temp/t30` without system changes and removed after verification.

Published contract and raw evidence: without validation, median decode is 154.032 ms Vulkan versus 3.893 ms CPU AVX2; complete eleven-token session 1680.553 versus 42.050 ms. Maximum CPU/GPU logit difference 9.656e-6, RMS 4.663e-6. GPU matrix buffers 16,658,176 bytes; vector transfers 260,480 bytes per append. This establishes correctness, not acceleration. Device-loss/OOM hardware injection and automatic session recovery remain outside the verified scope. T31 planner remains pending.

Skill set: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing; capabilities-dev-problem-solving-methodology for integration design and allocation diagnosis. Existing C#/.NET and Vulkan stack retained; no new external technology. The user's standing policy authorizes critical tests with implementation and overrides the skill's separate test approval gate. No obvious/bootstrap tests, commits or unrelated refactoring.

## T31 — Measured execution planner (completed)

Auto now calibrates the available whole-model combinations: CPU automatic Q8_0 and Vulkan packed row64. The numerical gate precedes sequential host measurements; three warmups and five samples measure two incremental appends. Median GPU latency must improve CPU by more than 5%. Explicit Cpu/Required policies and initialization fallback remain intact. No new kernels/layouts or per-tensor planner are implied.

The derived cache hashes CPU/features and actual AVX2 support, OS/runtime, GPU UUID/API/driver/capabilities and Core/Formats/backend binary contents (including shader/layout resources). Matching identities require five finite positive timings per backend. Malformed/missing data, identity mismatches and invalid samples trigger calibration. Selection is derived from medians; serialized choice cannot override it. Cache IO errors propagate. Cache hits currently still initialize/upload the GPU candidate; only measurement is skipped.

Published evidence: five real hardware child-process cases pass (fresh, exact reuse, identity change, invalid samples, malformed JSON), all selecting CPU and emitting expected text. Final CPU/GPU medians: 9.1441/305.4129 ms for two appends. Exact cached measurements/choice were checked. Two critical policy/cache tests and 116 CPU regression tests pass. Release build: zero warnings/errors. Native kernels/synchronization unchanged; prior hardware offload tests remain their evidence. Task-local cache/scripts under `.temp/t31` removed after report publication.

Skills retained: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing and capabilities-dev-problem-solving-methodology for measurement/selection design. Standing user authorization includes critical tests with implementation; no separate approval gate. No bootstrap tests, commits, system changes or unrelated refactoring. The baseline through T31 is complete; extension profiles remain tentative in TASKS.md.

## Vulkan optimization planning — T32–T45

User requested planning only. Added a sequential, bounded optimization plan with explicit dependencies, numerical/lifetime gates and exclusions. T32 produces executable diagnostic evidence; T34 is a separate architecture decision before resident-session implementation. Kernel reduction and staging changes precede GPU operators and complete single-submission execution. Planner and benchmark close the sequence. Existing T31 remains completed; T32 is pending.

Applied analysis-piano-task and the existing base-dossier-writing brevity policy. Inspected current shader, offload/planner contracts and measured baseline. No implementation, benchmark execution or new tests in this planning turn. No acceleration estimate is presented as observed evidence.

## T32 — GPU call phase diagnostics (completed)

Added opt-in host intervals for upload (map/copy/cache/unmap), command recording, submission preparation, QueueSubmit, fence wait, post-wait validation, query retrieval and readback. Successful MatVec counters expose submissions, waits, transferred bytes and instrumented-call counts. Explicit model options independently enable GPU timestamps and host phases; diagnostic options do not enter Auto planner selection.

Published six sequential process reports: three per timestamp mode, each with three warmups and seven measured runs on the existing eleven-token workload. Every checked prefix passes CPU/GPU numerical gates. Each measured run has 473 submissions/waits, 2,865,280 vector bytes, zero managed hot allocations and phase deltas summing to host duration within 1e-6 ms. Timestamp-off decode medians span 149.783–157.967 ms versus CPU 3.720–5.646 ms. Fence wait dominates; it overlaps device execution and cannot be equated to pure host overhead or transfer time. No speedup claimed. Report.

Release build: zero warnings/errors. Thirteen existing GPU/policy tests pass with Khronos SyncVal; 116 CPU tests pass separately. No new implementation-mirroring tests. Local validation package and scripts were confined to `.temp/t32` and removed after publication. Native shader/algorithm unchanged.

Skills: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing; capabilities-dev-problem-solving-methodology for diagnostic scope and evidence. User authorization covers critical verification alongside implementation and requests a commit per completed task. Prerequisite code baseline recorded separately as cc5cf8c, excluding pre-existing editorial/personal changes. T33 is next; T34–T45 remain planned.

## T33 — Reuse CPU plans before Vulkan weight uploads (completed)

Separated device/binary identity, cache lookup and calibration. Auto now returns a validated CPU model before constructing GPU matrices when the matching cache selects CPU. Cached GPU choices retain resident construction without recalibration; invalid/missing caches retain the numerical gate and measured selection. Required and unavailable-device behavior remain intact. Initialization snapshots expose candidate device and Q8 resource/work counters without requiring an active GPU model.

Published startup evidence: three independent measurement groups, each three discarded fresh-process warmups plus seven measured processes per version, alternating order. Baseline backend a704883, identical runner scope and separate binary-valid caches. Runtime initialization medians change 432.616→214.766, 418.548→217.648 and 414.409→234.402 ms (43.4–50.4% reduction). Every optimized CPU cache hit has zero matrices, weight/matrix allocation bytes and dispatches/transfers. Context creation remains; no inference speedup is claimed.

Real process checks cover fresh/identical/invalid identity/invalid samples/malformed cache, and an explicitly synthetic cached GPU choice (not performance evidence). Output is Once/token 9038 throughout successful runs. CLI checks Required, unavailable-driver Auto/Required and invalid model failure with a valid cache. Build: zero warnings/errors; two policy tests pass. CPU suite: 116 pass when run alone; first run had the previously observed transient allocation assertion (7,752 bytes), with 115 others passing. No numerical/native synchronization changes; T32 SyncVal evidence retained. No obvious new unit tests. Task-local baseline/caches/scripts removed after publication; shared corpus preserved.

Skills retained: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing and capabilities-dev-problem-solving-methodology. Standing authorization covers critical verification and a commit per completed task. Pre-existing editorial/personal changes excluded. T34 is next: define the resident GPU session contract before implementation.

## T34 — Define the resident GPU session contract (completed)

Pinned ownership, isolated KV/scratch, one submission per append, immutable recording descriptors, buffer budget, workgroup reduction, FP32 numerical gates and failure/reset/disposal transitions. Reviewed all operator dependencies and the six-layer token diagram against the CPU implementation. Documentation-only task; no runtime changes or tests. Implementation: resident execution.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T35 — Add cooperative Q8 workgroup reduction (completed)

Both kernels pass 148 FP64-bound workloads including all 44 real matrices, with zero SyncVal errors. Thirteen GPU/policy tests pass; Release build has zero warnings/errors. Three unvalidated process reports show faster small-matrix device times but a regression for 32000-row output; Row64 remains the default. Evidence: docs/benchmarks/q8-cooperative-2026-10-12. No numerical bounds relaxed.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T36 — Keep staging buffers persistently mapped (completed)

Mapped host-visible allocations once for their lifetime; whole-allocation flush/invalidate retained and unmap follows pending-work completion. Nine critical hardware tests pass with SyncVal, including distinct repeated vectors, allocation and concurrent lifetime cases. Build succeeds without warnings. Three timestamp-off whole-model phase reports published in docs/benchmarks/staging-2026-10-12; no mapping-only speedup claimed; noncoherent hardware not exercised.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T37 — Implement resident RMSNorm recording (completed)

Added immutable FP32 pipeline descriptors, owned arena/staging with context lifetime registration, and RMSNorm recording plus synchronous facade. Independent fixture, zeros, profile ranges and RMSNorm-to-Q8 FP64 checks pass with SyncVal; build has zero warnings/errors. Operator tolerance unchanged. docs/vulkan-transformer-operators.md documents scope and lifecycle.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T38 — Implement resident RoPE recording (completed)

Added in-place paired RoPE using the immutable host-computed rotation table. Independent fixture and profile positions 0,1,31,63,127 pass unchanged operator tolerances. Two operator tests pass with zero SyncVal errors; build without warnings/errors. Production recording performs no readback.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T39 — Implement resident SwiGLU recording (completed)

Added sign-stable FP32 SwiGLU and exact gate alias. Independent fixture, extreme signed gates and real feed-forward block pass unchanged tolerances. Three operator tests pass under SyncVal; build without warnings/errors. No activation approximation or intermediate readback in recording.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T40 — Implement resident residual addition (completed)

Elementwise residual recording allows exact aliases and retains explicit barriers. Critical cases cover both exact input aliases, partial-overlap rejection, one-element boundary and normalization chaining. Four operator tests pass with SyncVal; Release build without warnings/errors.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T41 — Implement incremental GPU attention and private KV (completed)

Added stable FP32 causal attention recording and private synchronous attention sessions. Independent Python binary64 fixtures at 1,2,32,64,128 tokens pass unchanged operator bounds. Isolation, reset, full-context rejection, cache budget and context lifetime pass. Five operator/attention tests pass with zero SyncVal errors; build without warnings/errors. No FP64 or subgroup requirement.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T42 — Compose complete resident transformer layers (completed)

Added shared compressed weights/tables, private KV/activation arenas and immutable per-session matrix descriptors with offset-aware Q8 shaders. Complete layers record without intermediate host transfer. Layer 0 at three independent prefixes and all six fixed-input layer boundaries pass the model gates, with one submission/fence, seven MatVecs and 2304 transfer bytes. SyncVal zero errors; build without warnings. docs/vulkan-resident-layers.md records buffers and ownership.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T43 — Integrate single-submission resident model sessions (completed)

Integrated a backend-independent session factory into Formats and complete resident execution into explicit Vulkan runtime/generation. Productive append transfers 129152 bytes, records 43 MatVecs and submits/waits once. Fifteen independent states/logits, official greedy sequences, streaming, 128-token context, reset, session isolation, zero hot allocations and concurrent disposal pass. All 22 GPU/policy tests pass with zero SyncVal errors; all 116 CPU tests pass separately; build has no warnings/errors. Diagnostic captures are excluded from productive counters. docs/stories15m-vulkan-resident.md documents ownership and use.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T44 — Select CPU legacy and resident paths by measured latency (completed)

Auto now validates and measures all three whole-model paths with timestamps/validation disabled; the graph/layout/operator protocol participates in cache identity. Eight process cache cases pass, including old/missing/invalid resident data and both synthetic cached GPU branches. Three independent short/long-context reports show CPU 4.090–5.754 ms, legacy 142.004–193.679 ms, resident 60.457–80.819 ms per append. CPU remains selected. Three critical policy tests pass; build without warnings/errors. docs/stories15m-planner.md records the exact governing workload and limitations.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## Vulkan contract correction — T27 discovery (recorded)

The T27 contract fixed the minimal compute path before any measurement. It admitted no optional device feature and defined a single in-flight synchronous operation. Measured evidence now shows the residual cost is dominated by the kernel and by per-dispatch latency rather than bandwidth, and that the GPU path stays slower than CPU AVX2 even with one submission per append. The conflict was routed back to architecture instead of being worked around inside a task.

Recorded correction, effective in the current-state view and in the backend contract: optional features are queried and never assumed, each with a verified fallback (8-bit storage with int8 arithmetic, subgroup operations, synchronization2 and timeline semaphores, FP16 arithmetic, timestamp queries); the effective capability set participates in the planner cache identity; each kernel declares its own local size, descriptors and push constants; the resident path must be able to attribute kernel window, barrier cost and dispatch cost. One queue, a synchronous public API, unchanged ownership, unchanged failure transitions and unchanged numerical gates remain binding. A falsifiable hypothesis and a stop rule now precede the optimization tasks.

Discovery on the pinned toolchain: glslangValidator 16.6.0 compiles subgroup operations for Vulkan 1.1 (OpCapability GroupNonUniform, GroupNonUniformArithmetic) and 8-bit storage (OpCapability StorageBuffer8BitAccess, extension SPV_KHR_8bit_storage); spirv-val 2026.4 accepts both.

Sequence extended: T45 residual-cost attribution, T46 capability record, T47 capability-gated Q8 kernel, T48 dispatch and barrier reduction, T49 grouped prefill, T50 final benchmark; the previous final benchmark becomes T50.

Skills: base-project-decision-log, phase-development-deviation-routing, base-dossier-writing, capabilities-dev-problem-solving-methodology.

## T45 — Attribute the residual resident cost (completed)

Added an optional timestamp pool, a per-dispatch accumulator and a diagnostic runner to the resident path. Attribution arrives through VulkanResidentOptions and StoriesBackendOptions, requires an explicit backend and writes reports/vulkan/resident-attribution.json. It never enters the productive counters and never reads activations back; writing the result is the only run mode that publishes numbers.

Three independent measurement processes plus one validation run on Intel Arc MTL, ten appends each. Per append: host median 77.754 ms, device span 73.626 ms, kernel 65.020 ms (87.2%), gap 9.446 ms (12.8%). Windows and gaps sum to the span within 1e-13 ms, managed allocations stay at zero and each append keeps one submission and one fence wait. The validation run reports zero errors with Khronos SyncVal. Q8_0 MatVec totals 47.192 ms (72.6% of kernel), the FP32 attention operator 7.849 ms (12.1%) and the output projection alone 15.303 ms.

The hypothesis is confirmed: kernel time dominates, so the barrier and dispatch ceiling is about 9.4 ms per append and the sequence stays T46 → T47 → T48. Evidence: docs/vulkan-dispatch-attribution.md and docs/benchmarks/vulkan-attribution-2026-10-12.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing, capabilities-dev-problem-solving-methodology for the diagnostic scope and the evidence.

## Model-specific documentation cleanup — 2026-10-13

Removed sixteen model-specific guides, stale optimization narratives and fixture prose. README, TASKS and ProjectInfo now direct development to the multi-model requirements. Removed links to deleted guides; numerical fixtures, raw measurements and production code were preserved. Documentation-only verification: local Markdown link scan; no runtime tests added or run. Skills: phase-development-project-conventions and base-dossier-writing.

## Local quantized-model and kernel scouting — 2026-10-13

Inspected DS4 small, AirLLM-plus and AirLLM at verified revisions, plus official HF artifacts/configs for seven candidate model repositories. Published exact file sizes/hashes, estimated KV budgets and architecture/format gaps. Recommended a candidate portfolio rather than a model-named runtime. DS4 supplies quantized-kernel/residency ideas; AirLLM supplies layer streaming; AirLLM-plus uses mock compute and is not inference performance evidence. No model weights downloaded or external code executed. JSON and local links verified; no runtime tests needed for this research-only task. Skills: phase-scouting-solution-scouting, phase-scouting-technical-feasibility and base-dossier-writing.

## Q4_K_M and UD-IQ scouting — 2026-10-13

Evaluated mixed quantization recipes, verified Unsloth artifact revisions/sizes and documented codec, CPU/Vulkan and quality gaps. Proposed Q4_K/Q6_K first, then an IQ3 comparison on the same model; IQ2 for capacity and IQ1 exploratory. No weights downloaded, binary tensor inventory or performance claims made. Evidence JSON parsed; documentation whitespace checked. Skills: phase-scouting-solution-scouting and base-dossier-writing.

## Repository ideas and APEX/Bonsai applicability — 2026-10-13

Consolidated applicability of DS4, AirLLM and AirLLM-plus for bounded residency, layer/expert streaming and compressed kernels. Added APEX as a MoE streaming/cache candidate and Bonsai PTQ1_0/PQ2_0 as distinct ternary/rotation paths; verified artifact metadata and Prism codec definitions. No weights downloaded or inference performed. JSON and local links checked; no runtime tests for documentation-only work.

## DeepSeek implementation plan — 2026-10-13

Created DeepSeek_todo.md with forty ordered tasks, contract gates, per-task outputs/checks/exclusions and commit policy. Covers generic CPU inference, K/IQ, BitNet, APEX bounded streaming/expert cache, Bonsai codecs/rotations and measured Vulkan. Critical-only testing; performance acceptance requires declared targets. Documentation links and whitespace verified; no runtime tests needed. Skill: analysis-piano-task.

## Q6_K integer path — 2026-10-14

Moved Q6_K off the floating-point batched path onto the integer kernel, with a scalar codec product
written to the already-validated decoder's index mapping. The product is checked against two
independent oracles: a Python reference in `tools/q6k_q8k_reference.py` and a composition of
`Q6KCodec.DecodeBlock` with the Q8_K codes. A/B in a worktree at `dfbe274`, three rounds alternating
the two revisions in one session: the two ranges are disjoint on both real Q6_K tensors, `ffn_down`
1.31x and the output projection 1.23x faster. Report in
`docs/benchmarks/q6k-integer-2026-10-14/README.md`. The same candidate also reported 14-19% worse
figures when run inside the full suite, which is why the comparison is alternated and standalone.
Suite 223/223. The DW06.7 target of 3 tok/s is still unmet; this removes 1.3x from a share that
needs about 12x. Skills: phase-development-task-execution and base-dossier-writing.

## Hardware dependence of the integer path — 2026-10-14

Measured whether the Q6_K integer and floating-point paths rank the same across hardware, by running
the A/B pinned to each core class of this hybrid host with the revisions alternated within each
round. P cores: integer wins 1.26x and 1.36x, disjoint ranges. E cores: the output projection
reverses and the floating-point path is 1.05x faster, disjoint ranges. A same-revision run left over
from an earlier shell mistake calibrates the noise floor: 1.02x at P cores but 1.61x at E cores, so
the inversion is real and smaller than the noise where it occurs. Also found the largest single
effect measured: eight worker tasks on six physical P cores costs 1.7x against the same eight tasks
unpinned. Conclusion recorded in `docs/benchmarks/hardware-dependence-2026-10-14/README.md`: no
install-time probe, an explicit Auto/Integer/Float preference instead, and a thread budget derived
from physical cores. No code changed. Skills: phase-development-task-execution and
base-dossier-writing.
