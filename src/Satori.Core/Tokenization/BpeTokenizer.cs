using System.Text;
using System.Text.RegularExpressions;

namespace Satori.Core.Tokenization;

/// <summary>
/// Byte-level BPE tokenizer equivalent to a GPT-2 style tokenizer with a configurable pre-tokenizer regex.
/// The vocabulary, merge ranks and added tokens come from the artifact; the regex distinguishes families and
/// lives in the capability registry, not in this class.
/// </summary>
/// <remarks>
/// Pipeline: optional NFC normalization, pre-tokenizer split (isolated), byte-level mapping, BPE merges.
/// Added (special) tokens are matched on the raw input before the pipeline. Decoding reverses the byte-level
/// mapping and decodes the accumulated bytes as UTF-8.
/// </remarks>
public sealed class BpeTokenizer
{
    private static readonly UTF8Encoding Utf8 = new(false, false);

    private readonly Dictionary<string, int> _vocabulary;
    private readonly string[] _idToToken;
    private readonly Dictionary<(string Left, string Right), int> _ranks = new();
    private readonly Dictionary<string, int> _added = new(StringComparer.Ordinal);
    private readonly HashSet<int> _specialIds = new();
    private readonly Regex _preTokenizer;

    public BpeTokenizer(
        IReadOnlyList<string> tokens,
        IReadOnlyList<string> merges,
        string preTokenPattern,
        IEnumerable<(string Content, int Id)> addedTokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(merges);

        _idToToken = [.. tokens];
        _vocabulary = new Dictionary<string, int>(tokens.Count, StringComparer.Ordinal);
        for (int id = 0; id < tokens.Count; id++) _vocabulary[tokens[id]] = id;

        for (int rank = 0; rank < merges.Count; rank++)
        {
            string merge = merges[rank];
            int space = merge.IndexOf(' ');
            if (space <= 0 || space == merge.Length - 1)
                throw new InvalidDataException($"Invalid merge at rank {rank}: '{merge}'.");
            _ranks[(merge[..space], merge[(space + 1)..])] = rank;
        }

        foreach (var (content, id) in addedTokens)
        {
            if (content.Length == 0) continue;
            _added.TryAdd(content, id);
            _specialIds.Add(id);
        }

        _preTokenizer = new Regex(preTokenPattern, RegexOptions.CultureInvariant);
    }

    public int VocabularySize => _idToToken.Length;

    public bool IsSpecial(int id) => _specialIds.Contains(id);

    /// <summary>Encodes text to token ids without adding any special token that the caller did not include.</summary>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var result = new List<int>();
        int position = 0;
        var normal = new StringBuilder();
        while (position < text.Length)
        {
            int matchLength = MatchAddedAt(text, position);
            if (matchLength > 0)
            {
                EncodeSegment(normal, result);
                result.Add(_added[text.Substring(position, matchLength)]);
                position += matchLength;
            }
            else
            {
                normal.Append(text[position]);
                position++;
            }
        }
        EncodeSegment(normal, result);
        return result;
    }

    /// <summary>Decodes ids. Special tokens are skipped unless requested.</summary>
    public string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (id < 0 || id >= _idToToken.Length)
                throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {id} is outside the vocabulary.");
            if (_specialIds.Contains(id))
            {
                if (!skipSpecialTokens)
                    bytes.AddRange(Utf8.GetBytes(_idToToken[id]));
                continue;
            }
            foreach (char mapped in _idToToken[id])
                bytes.Add(ByteLevel.CharToByte.TryGetValue(mapped, out byte value)
                    ? value
                    : throw new InvalidDataException($"Token '{_idToToken[id]}' contains a non byte-level character."));
        }
        return Utf8.GetString([.. bytes]);
    }

    private int MatchAddedAt(string text, int position)
    {
        if (_added.Count == 0) return 0;
        int best = 0;
        foreach (string content in _added.Keys)
        {
            if (content.Length > best && position + content.Length <= text.Length
                && string.CompareOrdinal(text, position, content, 0, content.Length) == 0)
                best = content.Length;
        }
        return best;
    }

    private void EncodeSegment(StringBuilder normal, List<int> result)
    {
        if (normal.Length == 0) return;
        string text = normal.ToString().Normalize(NormalizationForm.FormC);
        normal.Clear();
        foreach (Match match in _preTokenizer.Matches(text))
        {
            if (match.Length == 0) continue;
            EncodePiece(ByteLevel.Encode(match.Value), result);
        }
    }

    private void EncodePiece(string mapped, List<int> result)
    {
        if (mapped.Length == 0) return;
        var symbols = new List<string>(mapped.Length);
        foreach (char c in mapped) symbols.Add(c.ToString());

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;
            symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        foreach (string symbol in symbols)
        {
            if (!_vocabulary.TryGetValue(symbol, out int id))
                throw new InvalidDataException($"Symbol '{symbol}' is not in the vocabulary.");
            result.Add(id);
        }
    }
}
