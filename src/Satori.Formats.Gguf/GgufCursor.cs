using System.Buffers.Binary;
using System.Text;
using Satori.Core;

namespace Satori.Formats.Gguf;

internal sealed class GgufCursor(MappedWeights mapping)
{
    private static readonly UTF8Encoding Utf8 = new(false, true);
    public long Position { get; private set; }

    public void Skip(long length)
    {
        if (length < 0 || length > mapping.Length - Position)
            throw Error("Truncated value");
        if (length > 256L * 1024 * 1024 - Position)
            throw new NotSupportedException("Header limit: 256 MiB.");
        Position += length;
    }

    public ulong Unsigned(int size)
    {
        long start = Position;
        Skip(size);
        Span<ulong> result = stackalloc ulong[1];
        mapping.Read(start, size, static (bytes, result) => result[0] = bytes.Length switch
        {
            1 => bytes[0],
            2 => BinaryPrimitives.ReadUInt16LittleEndian(bytes),
            4 => BinaryPrimitives.ReadUInt32LittleEndian(bytes),
            8 => BinaryPrimitives.ReadUInt64LittleEndian(bytes),
            _ => throw new InvalidOperationException()
        }, result);
        return result[0];
    }

    public long Count(ulong value)
    {
        if (value > long.MaxValue) throw Error("Integer exceeds Int64");
        return (long)value;
    }

    public string String(int limit)
    {
        long length = Count(Unsigned(8));
        if (length > limit) throw new NotSupportedException($"String byte limit: {limit} at {Position}.");
        long start = Position;
        Skip(length);
        var result = new GgufStringResult();
        try { mapping.Read(start, (int)length, static (bytes, result) => result.Value = Utf8.GetString(bytes), result); }
        catch (DecoderFallbackException) { throw Error("Invalid UTF-8"); }
        return result.Value;
    }

    public InvalidDataException Error(string message) => new($"GGUF at byte {Position}: {message}.");
}
