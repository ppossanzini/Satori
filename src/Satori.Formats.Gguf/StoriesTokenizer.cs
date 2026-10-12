using System.Buffers.Binary;
using System.Text;
using System.Security.Cryptography;

namespace Satori.Formats.Gguf;

/// <summary>Owned SentencePiece vocabulary for the Stories15M profile; no chat template.</summary>
public sealed class StoriesTokenizer
{
    public const int Bos = 1, Eos = 2, VocabularySize = 32000, MaxInputBytes = 16384;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    private readonly string[] _tokens = new string[VocabularySize];
    private readonly float[] _scores = new float[VocabularySize];
    private readonly byte[][] _pieces = new byte[VocabularySize][];
    private readonly Dictionary<string, int> _ids = new(StringComparer.Ordinal);

    public StoriesTokenizer(GgufFile file)
    {
        foreach (string key in file.Metadata.Keys)
            if (key.StartsWith("tokenizer.", StringComparison.Ordinal) && key is not
                ("tokenizer.ggml.model" or "tokenizer.ggml.tokens" or "tokenizer.ggml.scores" or
                "tokenizer.ggml.token_type" or "tokenizer.ggml.bos_token_id" or "tokenizer.ggml.eos_token_id" or
                "tokenizer.ggml.unknown_token_id" or "tokenizer.ggml.seperator_token_id" or "tokenizer.ggml.padding_token_id"))
                throw new NotSupportedException("Tokenizer options differ from the Stories profile.");
        if (!Equals(file.ReadScalar("tokenizer.ggml.model"), "llama") ||
            !Equals(file.ReadScalar("tokenizer.ggml.bos_token_id"), 1u) ||
            !Equals(file.ReadScalar("tokenizer.ggml.eos_token_id"), 2u) ||
            !Equals(file.ReadScalar("tokenizer.ggml.unknown_token_id"), 0u) ||
            !Equals(file.ReadScalar("tokenizer.ggml.seperator_token_id"), uint.MaxValue) ||
            !Equals(file.ReadScalar("tokenizer.ggml.padding_token_id"), uint.MaxValue))
            throw new NotSupportedException("Unsupported Stories tokenizer metadata.");
        Array(file, "tokenizer.ggml.tokens", 8, 466931, "b216f8436a10dcace29328040152aeb69cc6f1fdfd0ee4ab2860060e168ac7c7");
        Array(file, "tokenizer.ggml.scores", 6, 128012, "6f6c89a3ae314e3cfc618db4a11956c98cd0f80d49d50c8ad147685da6e487f8");
        Array(file, "tokenizer.ggml.token_type", 5, 128012, "957abc3bb8f042ad9cf6b7f248b6255ec1fff38f1e8fb9f2bac38df4f000f780");
        file.ReadMetadata("tokenizer.ggml.tokens", static (bytes, self) =>
        {
            int offset = 12;
            for (int i = 0; i < VocabularySize; i++)
            {
                int length = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(bytes[offset..])); offset += 8;
                string text = StrictUtf8.GetString(bytes.Slice(offset,length)); offset += length;
                self._tokens[i] = text;
                if (!self._ids.TryAdd(text,i)) throw new InvalidDataException("Duplicate vocabulary entry.");
            }
        }, this);
        file.ReadMetadata("tokenizer.ggml.scores", static (bytes, self) =>
        {
            for (int i = 0; i < VocabularySize; i++)
            {
                float score = BinaryPrimitives.ReadSingleLittleEndian(bytes[(12+i*4)..]);
                if (!float.IsFinite(score)) throw new InvalidDataException("Non-finite tokenizer score.");
                self._scores[i] = score;
            }
        }, this);
        file.ReadMetadata("tokenizer.ggml.token_type", static (bytes, self) =>
        {
            for (int i = 0; i < VocabularySize; i++)
            {
                int type = BinaryPrimitives.ReadInt32LittleEndian(bytes[(12+i*4)..]);
                int expected = i == 0 ? 2 : i < 3 ? 3 : i < 259 ? 6 : 1;
                if (type != expected || (i >= 3 && i < 259 && self._tokens[i] != $"<0x{i-3:X2}>"))
                    throw new NotSupportedException("Unsupported Stories vocabulary types.");
                self._pieces[i] = type == 1 ? StrictUtf8.GetBytes(self._tokens[i].Replace("▁", " ", StringComparison.Ordinal)) :
                    type == 6 ? [(byte)(i-3)] : [];
            }
        }, this);
    }

    private static void Array(GgufFile file, string key, uint element, int byteLength, string hash)
    {
        var entry = file.Metadata[key];
        if (entry.Type != 9 || entry.ElementType != element || entry.ElementCount != VocabularySize || entry.ByteLength != byteLength)
            throw new NotSupportedException("Unsupported vocabulary array.");
        file.ReadMetadata(key, static (bytes, expected) =>
        {
            Span<byte> actual = stackalloc byte[32];
            SHA256.HashData(bytes, actual);
            if (!actual.SequenceEqual(Convert.FromHexString(expected)))
                throw new NotSupportedException("Vocabulary differs from the Stories profile.");
        }, hash);
    }

    public int[] Encode(string text, bool addSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (StrictUtf8.GetByteCount(text) > MaxInputBytes) throw new ArgumentOutOfRangeException(nameof(text), "Input exceeds 16384 UTF-8 bytes.");
        if (text.Length == 0) return addSpecial ? [Bos] : [];
        string normalized = (" " + text).Replace(" ", "▁", StringComparison.Ordinal);
        var symbols = normalized.EnumerateRunes().Select(r => r.ToString()).ToArray();
        int[] next = new int[symbols.Length], previous = new int[symbols.Length];
        for (int i = 0; i < symbols.Length; i++) { next[i] = i+1 == symbols.Length ? -1 : i+1; previous[i] = i-1; }
        var queue = new PriorityQueue<(int Left, int Right, string Text), (float Score, int Left)>();
        void Pair(int left, int right)
        {
            if (left < 0 || right < 0) return;
            string combined = symbols[left]+symbols[right];
            if (_ids.TryGetValue(combined,out int id)) queue.Enqueue((left,right,combined),(-_scores[id],left));
        }
        for (int i = 1; i < symbols.Length; i++) Pair(i-1,i);
        while (queue.TryDequeue(out var pair,out _))
        {
            if (next[pair.Left] != pair.Right || symbols[pair.Right].Length == 0 || symbols[pair.Left]+symbols[pair.Right] != pair.Text) continue;
            symbols[pair.Left] = pair.Text; symbols[pair.Right] = "";
            next[pair.Left] = next[pair.Right];
            if (next[pair.Right] >= 0) previous[next[pair.Right]] = pair.Left;
            Pair(previous[pair.Left],pair.Left); Pair(pair.Left,next[pair.Left]);
        }
        var result = new List<int>();
        if (addSpecial) result.Add(Bos);
        for (int i = 0; i >= 0; i = next[i])
        {
            if (_ids.TryGetValue(symbols[i],out int id)) result.Add(id);
            else foreach (byte b in StrictUtf8.GetBytes(symbols[i])) result.Add(b+3);
        }
        return result.ToArray();
    }

    public ReadOnlySpan<byte> Piece(int token)
    {
        if ((uint)token >= VocabularySize) throw new ArgumentOutOfRangeException(nameof(token));
        return _pieces[token];
    }

    public byte[] DecodeBytes(ReadOnlySpan<int> tokens)
    {
        var bytes = new List<byte>();
        bool removeSpace = tokens.IsEmpty || tokens[0] != Bos;
        foreach (int token in tokens)
        {
            var piece = Piece(token);
            if (removeSpace && !piece.IsEmpty && piece[0] == 32) piece = piece[1..];
            removeSpace = false;
            foreach (byte b in piece) bytes.Add(b);
        }
        return bytes.ToArray();
    }

    public string Decode(ReadOnlySpan<int> tokens) => Encoding.UTF8.GetString(DecodeBytes(tokens));

    public StoriesTokenDecoder CreateDecoder(int previousToken = Bos) => new(this, previousToken);
}
