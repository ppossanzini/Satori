using Satori.Core.Tokenization;

namespace Satori.Formats.Gguf;

/// <summary>
/// Builds a <see cref="BpeTokenizer"/> from a GGUF container. The tokenizer model and pre-tokenizer are
/// selected from the artifact metadata; an unsupported model or pre-tokenizer is rejected explicitly.
/// </summary>
public static class GgufBpeTokenizer
{
    public static BpeTokenizer Open(string path)
    {
        var tokenization = GgufTokenizerReader.Read(path);
        if (tokenization.Model != "gpt2")
            throw new NotSupportedException($"Tokenizer model '{tokenization.Model}' is unsupported.");
        string? pattern = PreTokenizers.Resolve(tokenization.PreTokenizer);
        if (pattern is null)
            throw new NotSupportedException($"Pre-tokenizer '{tokenization.PreTokenizer}' is unsupported.");
        return new BpeTokenizer(tokenization.Tokens, tokenization.Merges, pattern,
            tokenization.AddedTokens.Select(a => (a.Content, a.Id)));
    }

    /// <summary>Returns the tokenizer metadata without building a tokenizer, for inspection.</summary>
    public static GgufTokenization Read(string path) => GgufTokenizerReader.Read(path);
}
