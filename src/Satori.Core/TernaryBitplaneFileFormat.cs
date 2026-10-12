namespace Satori.Core;

internal static class TernaryBitplaneFileFormat
{
    internal static void WriteHeader(Span<byte> header, Ternary2Shape shape)
    {
        Ternary2FileFormat.WriteHeader(header, shape);
        "STB1"u8.CopyTo(header);
    }

    internal static Ternary2Shape ReadHeader(ReadOnlySpan<byte> header)
    {
        if (!header[..4].SequenceEqual("STB1"u8)) throw new InvalidDataException("Invalid bitplane magic.");
        Span<byte> commonHeader = stackalloc byte[32];
        header.CopyTo(commonHeader);
        "STQ2"u8.CopyTo(commonHeader);
        try { return Ternary2FileFormat.ReadHeader(commonHeader); }
        catch (InvalidDataException exception) { throw new InvalidDataException("Invalid bitplane header fields.", exception); }
    }
}
