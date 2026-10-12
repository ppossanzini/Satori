using System.Text;

namespace Satori.Formats.Gguf;

/// <summary>Sequence-local UTF-8 continuation decoder. Flush completes an unfinished byte sequence.</summary>
public sealed class StoriesTokenDecoder
{
    private readonly StoriesTokenizer _tokenizer;
    private readonly Decoder _utf8 = Encoding.UTF8.GetDecoder();
    private int _previous;

    internal StoriesTokenDecoder(StoriesTokenizer tokenizer, int previous)
    {
        _ = tokenizer.Piece(previous);
        _tokenizer = tokenizer;
        _previous = previous;
    }

    public string Append(int token)
    {
        ReadOnlySpan<byte> piece = _tokenizer.Piece(token);
        if (_previous == StoriesTokenizer.Bos && !piece.IsEmpty && piece[0] == 32) piece = piece[1..];
        _previous = token;
        return Convert(piece, false);
    }

    public string Flush() => Convert([], true);

    private string Convert(ReadOnlySpan<byte> bytes, bool flush)
    {
        Span<char> chars = stackalloc char[bytes.Length+2];
        int count = _utf8.GetChars(bytes, chars, flush);
        return new string(chars[..count]);
    }
}
