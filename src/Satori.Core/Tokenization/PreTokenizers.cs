namespace Satori.Core.Tokenization;

/// <summary>
/// Pre-tokenizer split patterns keyed by the artifact's <c>tokenizer.ggml.pre</c> value. The pattern is a
/// capability choice per artifact family, not part of the tokenizer algorithm. Unknown families resolve to
/// <c>null</c> so the caller reports a capability gap instead of guessing.
/// </summary>
public static class PreTokenizers
{
    /// <summary>GPT-2 style split used by the qwen2 and qwen3 families.</summary>
    public const string Gpt2 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    public static string? Resolve(string? preTokenizer) => preTokenizer switch
    {
        "qwen2" => Gpt2,
        "qwen35" => Gpt2,
        _ => null,
    };
}
