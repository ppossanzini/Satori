using System.Buffers.Binary;
using System.Text;
using Satori.Core;

namespace Satori.Formats.Gguf;

/// <summary>An added token declared by the container, with its id and whether it is special.</summary>
public sealed record GgufAddedToken(string Content, int Id, bool Special);

/// <summary>Tokenizer metadata read from a GGUF container, independent of tensor types.</summary>
public sealed record GgufTokenization(
    string Model,
    string? PreTokenizer,
    IReadOnlyList<string> Tokens,
    IReadOnlyList<string> Merges,
    IReadOnlyList<int> TokenTypes,
    IReadOnlyList<GgufAddedToken> AddedTokens,
    int? BosTokenId,
    int? EosTokenId,
    int? UnknownTokenId,
    int? PaddingTokenId,
    string? ChatTemplate);

/// <summary>
/// Reads the tokenizer arrays (<c>tokenizer.ggml.tokens/merges/token_type</c>) and scalar ids from a GGUF
/// container. It uses the tolerant descriptor scan, so a model whose tensor types are unsupported still
/// yields its tokenizer.
/// </summary>
public static class GgufTokenizerReader
{
    private static readonly UTF8Encoding Utf8 = new(false, true);

    // GGML tokenizer token types: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
    private const int ControlType = 3;
    private const int UserDefinedType = 4;

    public static GgufTokenization Read(string path)
    {
        var inventory = GgufInventory.Scan(path);
        using var mapping = new MappedWeights(path);

        IReadOnlyList<string> tokens = ReadStringArray(inventory, mapping, "tokenizer.ggml.tokens");
        IReadOnlyList<string> merges = ReadStringArray(inventory, mapping, "tokenizer.ggml.merges");
        IReadOnlyList<int> tokenTypes = ReadInt32Array(inventory, mapping, "tokenizer.ggml.token_type");
        string model = ScalarString(inventory, "tokenizer.ggml.model") ?? "";
        string? pre = ScalarString(inventory, "tokenizer.ggml.pre");

        var added = new List<GgufAddedToken>();
        for (int id = 0; id < tokens.Count && id < tokenTypes.Count; id++)
        {
            int type = tokenTypes[id];
            if (type == ControlType || type == UserDefinedType)
                added.Add(new GgufAddedToken(tokens[id], id, Special: true));
        }

        return new GgufTokenization(model, pre, tokens, merges, tokenTypes, added,
            ScalarInt(inventory, "tokenizer.ggml.bos_token_id"),
            ScalarInt(inventory, "tokenizer.ggml.eos_token_id"),
            ScalarInt(inventory, "tokenizer.ggml.unknown_token_id"),
            ScalarInt(inventory, "tokenizer.ggml.padding_token_id"),
            ScalarString(inventory, "tokenizer.chat_template"));
    }

    private static string? ScalarString(GgufInventory inventory, string key) =>
        inventory.Metadata.FirstOrDefault(m => m.Key == key)?.Value as string;

    private static int? ScalarInt(GgufInventory inventory, string key)
    {
        var value = inventory.Metadata.FirstOrDefault(m => m.Key == key)?.Value;
        return value switch
        {
            uint v => v > int.MaxValue ? null : (int)v,
            int v => v,
            null => null,
            _ => null,
        };
    }

    private static GgufMetadataDescriptor Require(GgufInventory inventory, string key)
    {
        var entry = inventory.Metadata.FirstOrDefault(m => m.Key == key)
            ?? throw new InvalidDataException($"GGUF: tokenizer metadata '{key}' is absent.");
        if (entry.Type != 9) throw new InvalidDataException($"GGUF: '{key}' is not an array.");
        return entry;
    }

    private static byte[] ReadArrayPayload(GgufInventory inventory, MappedWeights mapping, string key,
        uint expectedElementType, out long count)
    {
        var entry = Require(inventory, key);
        if (entry.ElementType != expectedElementType)
            throw new InvalidDataException($"GGUF: '{key}' has element type {entry.ElementType}.");
        count = entry.ElementCount ?? throw new InvalidDataException($"GGUF: '{key}' has no element count.");
        long payload = entry.ByteLength - 12;
        if (payload < 0 || payload > int.MaxValue)
            throw new NotSupportedException($"GGUF: '{key}' payload cannot be materialized.");
        var buffer = new byte[payload];
        mapping.Read(entry.Offset + 12, (int)payload, static (bytes, destination) => bytes.CopyTo(destination), buffer);
        return buffer;
    }

    private static string[] ReadStringArray(GgufInventory inventory, MappedWeights mapping, string key)
    {
        byte[] payload = ReadArrayPayload(inventory, mapping, key, 8, out long count);
        var result = new string[count];
        int position = 0;
        for (long i = 0; i < count; i++)
        {
            ulong length = BinaryPrimitives.ReadUInt64LittleEndian(payload.AsSpan(position));
            position += 8;
            if (length > (ulong)(payload.Length - position))
                throw new InvalidDataException($"GGUF: '{key}' string overruns the array.");
            result[i] = Utf8.GetString(payload, position, (int)length);
            position += (int)length;
        }
        return result;
    }

    private static int[] ReadInt32Array(GgufInventory inventory, MappedWeights mapping, string key)
    {
        byte[] payload = ReadArrayPayload(inventory, mapping, key, 5, out long count);
        var result = new int[count];
        for (long i = 0; i < count; i++)
            result[i] = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan((int)i * 4));
        return result;
    }
}
