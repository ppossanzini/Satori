namespace Satori.Core.Tokenization;

/// <summary>
/// Text tokenizer contract used by generation. The implementation owns the vocabulary and the special-token
/// policy; the generator only encodes a prompt and decodes produced ids.
/// </summary>
public interface ITextTokenizer
{
    /// <summary>Encodes text to token ids, adding no special token the caller did not include.</summary>
    IReadOnlyList<int> Encode(string text);

    /// <summary>Decodes ids to text. Special tokens are skipped unless requested.</summary>
    string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = true);

    /// <summary>Whether the id is a special (control or user-defined) token.</summary>
    bool IsSpecial(int id);
}
