# Text Generation

Status: DS10. Generation wires a dense session to an artifact's tokenizer and reuses the shared sampler.
Generic over model and tokenizer; no model name appears in the generator.

## Contract

| Type | Role |
|---|---|
| `ITextTokenizer` | `Encode`, `Decode`, `IsSpecial`. `BpeTokenizer` implements it. |
| `TextGenerationOptions(int MaxNewTokens, SamplingOptions? Sampling)` | Token budget and sampling; `Temperature = 0` is greedy. |
| `GenerationStop` | `EndOfSequence`, `TokenLimit`, `ContextFull`. |
| `TextGenerationResult` | prompt ids, generated ids, decoded text, stop reason. |
| `TextGenerator(DenseSession, ITextTokenizer, int? endOfSequenceToken)` | Encodes, appends incrementally and samples. |

Generation is prompt to text: encode the prompt, `Forward` it once, then for each step sample the next id
with `TokenSampler` (the shared implementation reused from the reference path), emit it, and `Append` it.
It stops at the end-of-sequence token, the token limit, or when the session's context is full.

Reused, not reimplemented: `DenseSession` (DS09) for incremental state, `TokenSampler` /
`GreedyTokenSelector` for sampling, `BpeTokenizer` (DS06) for text. The only new arithmetic is the loop.

## Errors

| Condition | Result |
|---|---|
| Empty prompt | `ArgumentException` |
| `MaxNewTokens < 0` | `ArgumentOutOfRangeException` |
| Prompt token outside the vocabulary | `ArgumentOutOfRangeException` |
| End-of-sequence id outside the vocabulary | `ArgumentOutOfRangeException` |
| Invalid sampling options | `ArgumentOutOfRangeException` (from `SamplingOptions.Validate`) |

## Verification

Offline (`TextGeneratorTests`, synthetic weights):

- greedy generation equals a manual `GreedyTokenSelector` loop over a fresh session, and the default
  sampling is greedy (`Temperature = 0`);
- the end-of-sequence token stops generation immediately after the first produced token;
- a small context stops with `ContextFull` before the token limit;
- seeded sampling equals `TokenSampler` driven directly;
- invalid arguments are rejected.

Model integration (`QwenForwardTests.GreedyGenerationMatchesIndependentReference`, requires
`python tools/download_qwen_models.py`):

- the greedy continuation from a realistic code prompt (ids from the official tokenizer fixture) is
  token-identical to the independent float64 reference, on Qwen2.5-Coder 1.5B and Qwen3 4B;
- the decoded text and stop reason match, and the session position advances by the prompt plus the produced
  tokens.

Seed verification is out of scope: the sampling path itself was not modified, so the DS10 criterion
"seed verified only if sampling changed" does not apply.

## Console

```sh
dotnet run --project src/Satori.Console -c Release -- text <model.gguf> "<prompt>" [max-new-tokens] \
    [--context N] [--temperature N] [--top-k N] [--top-p N] [--seed N]
```

Prints the decoded text to stdout and the stop reason, prompt/generated counts, context and elapsed
milliseconds to stderr. A non-executable model fails with the capability report before generation.
