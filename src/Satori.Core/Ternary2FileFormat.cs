using System.Buffers.Binary;

namespace Satori.Core;

internal static class Ternary2FileFormat
{
    internal const int HeaderSize = 32;

    internal static void WriteHeader(Span<byte> header, Ternary2Shape shape)
    {
        header.Clear();
        "STQ2"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt16LittleEndian(header[4..], 1);
        BinaryPrimitives.WriteUInt16LittleEndian(header[6..], HeaderSize);
        BinaryPrimitives.WriteInt32LittleEndian(header[8..], shape.Rows);
        BinaryPrimitives.WriteInt32LittleEndian(header[12..], shape.Columns);
        BinaryPrimitives.WriteInt32LittleEndian(header[16..], shape.BlockSize);
        BinaryPrimitives.WriteInt64LittleEndian(header[24..], shape.ByteLength);
    }

    internal static Ternary2Shape ReadHeader(ReadOnlySpan<byte> header)
    {
        if (!header[..4].SequenceEqual("STQ2"u8)
            || BinaryPrimitives.ReadUInt16LittleEndian(header[4..]) != 1
            || BinaryPrimitives.ReadUInt16LittleEndian(header[6..]) != HeaderSize
            || BinaryPrimitives.ReadInt32LittleEndian(header[20..]) != 0)
            throw new InvalidDataException("Unsupported Ternary2 header.");
        var shape = new Ternary2Shape(
            BinaryPrimitives.ReadInt32LittleEndian(header[8..]),
            BinaryPrimitives.ReadInt32LittleEndian(header[12..]),
            BinaryPrimitives.ReadInt32LittleEndian(header[16..]));
        if (BinaryPrimitives.ReadInt64LittleEndian(header[24..]) != shape.ByteLength)
            throw new InvalidDataException("Payload length disagrees with matrix shape.");
        return shape;
    }
}
